#!/usr/bin/env python3
"""Backfill provenance for v1 campaigns (docs/PROVENANCE_SPEC.md, "Backfill").

Applies create_tables.PROVENANCE_SCHEMA, upserts the seed sources from
sources_seed.yaml, and maps every campaign without a campaign_sources row to a seed
source. Never updates v1 rows or sets measurements.evaluation_id. Refuses to run on
the frozen databases roar_v1 and roar_full.

Usage:
    python backfill_provenance.py <DATABASE_URL>
"""

import sys
from pathlib import Path
from urllib.parse import urlsplit

import psycopg
import yaml
from psycopg.conninfo import conninfo_to_dict

from create_tables import PROVENANCE_SCHEMA
from provenance import FROZEN_DATABASES, ProvenanceError, contributor_prefix

SEED_PATH = Path(__file__).parent / "sources_seed.yaml"
SEED_COLUMNS = ["slug", "title", "version", "year", "url", "license_spdx", "redistribution",
                "attribution", "authors", "bibtex", "consent_recorded_at"]

IBM = "ibm-ai-native-systems@v1"
ELLIS_CASTRO = "ellis-castro-2026@v1"
CHENG = "cheng2025barbarians@v1"


def _refuse_frozen(dbname: str | None) -> None:
    if dbname in FROZEN_DATABASES:
        raise ProvenanceError(f"refusing to write to frozen database {dbname!r}")


def _dbname(database_url: str) -> str | None:
    try:
        return conninfo_to_dict(database_url).get("dbname")
    except Exception:
        return urlsplit(database_url).path.lstrip("/") or None


def load_seed() -> list[dict]:
    with open(SEED_PATH) as f:
        return yaml.safe_load(f)["sources"]


def seed_source_for(author: str, name: str | None) -> str | None:
    """The seed source a v1 campaign belongs to, or None if unmapped."""
    if author == "Roar Authors":
        return IBM if contributor_prefix(name) is not None else ELLIS_CASTRO
    if author == "Roar Authors 2":
        return CHENG
    return None


def backfill(database_url: str) -> dict:
    """Run the backfill in one transaction. Returns campaign counts per seed slug."""
    _refuse_frozen(_dbname(database_url))  # by name, before connecting
    seed = load_seed()

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            _refuse_frozen(cur.fetchone()[0])

            cur.execute(PROVENANCE_SCHEMA)

            cols = ", ".join(SEED_COLUMNS)
            updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in SEED_COLUMNS if c != "slug")
            changed = " OR ".join(f"source_datasets.{c} IS DISTINCT FROM EXCLUDED.{c}"
                                  for c in SEED_COLUMNS if c != "slug")
            for s in seed:
                cur.execute(
                    f"INSERT INTO source_datasets ({cols}) VALUES ({', '.join(['%s'] * len(SEED_COLUMNS))}) "
                    f"ON CONFLICT (slug) DO UPDATE SET {updates} WHERE {changed}",
                    [s.get(c) for c in SEED_COLUMNS],
                )

            cur.execute(
                "SELECT c.id, c.author, c.name FROM campaigns c "
                "LEFT JOIN campaign_sources cs ON cs.campaign_id = c.id WHERE cs.campaign_id IS NULL"
            )
            rows = []
            for cid, author, name in cur.fetchall():
                slug = seed_source_for(author, name)
                if slug is None:
                    raise ProvenanceError(f"campaign {cid} cannot be mapped to a seed source")
                rows.append((cid, slug))
            if rows:
                cur.executemany(
                    "INSERT INTO campaign_sources (campaign_id, source_slug, submitted_by) VALUES (%s, %s, NULL)",
                    rows,
                )

            cur.execute(
                "SELECT sd.slug, count(cs.campaign_id) FROM source_datasets sd "
                "LEFT JOIN campaign_sources cs ON cs.source_slug = sd.slug "
                "WHERE sd.slug = ANY(%s) GROUP BY sd.slug",
                ([s["slug"] for s in seed],),
            )
            counts = {slug: n for slug, n in cur.fetchall()}
        conn.commit()
    return dict(sorted(counts.items()))


def main():
    if len(sys.argv) != 2:
        print("Usage: python backfill_provenance.py <DATABASE_URL>", file=sys.stderr)
        sys.exit(2)
    try:
        counts = backfill(sys.argv[1])
    except ProvenanceError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    for slug, n in counts.items():
        print(f"{slug}: {n} campaigns")


if __name__ == "__main__":
    main()
