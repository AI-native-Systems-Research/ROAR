"""Spec "Backfill (backfill_provenance.py)" on synthetic v1-like throwaway DBs:
refusals for roar_v1 / roar_full by name (before doing anything), seed upsert, mapping
rules, exact counts, idempotency, v1 immutability, evaluation_id stays NULL, CLI."""

from __future__ import annotations

import subprocess
import sys

import pytest

from conftest import (
    SEED_SLUGS,
    SRC_DIR,
    apply_provenance_schema,
    fetchall,
    fetchone,
    import_backfill,
    insert,
    allocate,
    anon_source_block,
    build_parsed,
    insert_sql_campaign,
    unique,
    v1_columns,
    v1_row_fingerprint,
)

IBM = "ibm-ai-native-systems@v1"
ELLIS = "ellis-castro-2026@v1"
CHENG = "cheng2025barbarians@v1"
GUPTA = "gupta2026harness@v1"


def _nonzero(d):
    return {k: v for k, v in d.items() if v}


# Port 1 on loopback: nothing listens, so any connection attempt fails immediately. A
# compliant backfill refuses by *name* before connecting, raising ProvenanceError.
@pytest.mark.parametrize(
    "url",
    [
        "postgresql://postgres:postgres@127.0.0.1:1/roar_v1",
        "postgresql://postgres:postgres@127.0.0.1:1/roar_full",
        "postgresql://postgres:postgres@127.0.0.1:1/roar_v1?sslmode=disable",
        "postgresql://postgres:postgres@127.0.0.1:1/roar_full?connect_timeout=1",
        "postgres://postgres@127.0.0.1:1/roar_v1",
    ],
)
def test_backfill_refuses_frozen_databases_by_name(url, ProvenanceError):
    bf = import_backfill()
    with pytest.raises(ProvenanceError):
        bf.backfill(url)


def test_backfill_cli_refuses_frozen_database(ProvenanceError):
    import_backfill()
    p = subprocess.run(
        [sys.executable, str(SRC_DIR / "backfill_provenance.py"), "postgresql://postgres:postgres@127.0.0.1:1/roar_v1"],
        cwd=SRC_DIR, capture_output=True, text=True, timeout=60,
    )
    assert p.returncode != 0
    out = (p.stderr + p.stdout).lower()
    # Refused by name, not by failing to reach the (unreachable) server.
    assert "connection refused" not in out and "connection failed" not in out


@pytest.fixture
def v1_like(fresh_db):
    """A DB with SCHEMA only (no provenance) holding v1-style campaigns:
    3 ellis, 2 ibm, 1 cheng."""
    url = fresh_db(provenance=False)
    ids = {
        "ibm1": insert_sql_campaign(url, "alice-test__run1"),
        "ibm2": insert_sql_campaign(url, "bob-test/run_2"),
        "ellis1": insert_sql_campaign(url, "eb1-preview_evox_p11_run1_abc"),
        "ellis2": insert_sql_campaign(url, "cbl/x"),
        "ellis3": insert_sql_campaign(url, "plainname"),
        "cheng1": insert_sql_campaign(url, "alice-test__other", author="Roar Authors 2"),
    }
    return url, ids


EXPECTED = {ELLIS: 3, IBM: 2, CHENG: 1}


def test_backfill_counts_and_mapping(v1_like):
    url, ids = v1_like
    bf = import_backfill()
    counts = bf.backfill(url)
    assert isinstance(counts, dict)
    assert _nonzero(counts) == EXPECTED
    assert counts.get(GUPTA, 0) == 0
    got = dict(fetchall(url, "SELECT campaign_id::text, source_slug FROM campaign_sources"))
    assert got == {
        ids["ibm1"]: IBM, ids["ibm2"]: IBM,
        ids["ellis1"]: ELLIS, ids["ellis2"]: ELLIS, ids["ellis3"]: ELLIS,
        ids["cheng1"]: CHENG,
    }
    assert fetchone(url, "SELECT count(*) FROM campaign_sources WHERE submitted_by IS NOT NULL")[0] == 0


def test_backfill_applies_schema_and_seeds_sources(v1_like):
    url, _ = v1_like
    import_backfill().backfill(url)
    slugs = [r[0] for r in fetchall(url, "SELECT slug FROM source_datasets ORDER BY slug")]
    assert slugs == sorted(SEED_SLUGS)
    assert fetchone(url, "SELECT count(*) FROM source_datasets WHERE consent_recorded_at IS NULL")[0] == 0
    assert fetchone(url, "SELECT bibtex IS NULL FROM source_datasets WHERE slug=%s", (IBM,))[0] is True
    # Seed consent: permission granted 2026-11-17.
    for slug, utc_date, local_date in fetchall(
        url, "SELECT slug, (consent_recorded_at AT TIME ZONE 'UTC')::date::text, "
             "consent_recorded_at::date::text FROM source_datasets"):
        assert "2026-11-17" in (utc_date, local_date), slug
    assert fetchone(url, "SELECT count(*) FROM information_schema.columns WHERE table_name='measurements' "
                         "AND column_name='evaluation_id'")[0] == 1


def test_backfill_idempotent(v1_like):
    url, _ = v1_like
    bf = import_backfill()
    from golden_support import provenance_tables_hash

    first = bf.backfill(url)
    h1 = provenance_tables_hash(url)
    second = bf.backfill(url)
    assert second == first
    assert provenance_tables_hash(url) == h1


def test_backfill_never_changes_v1_rows(v1_like):
    url, _ = v1_like
    import psycopg

    with psycopg.connect(url) as conn:
        cols = v1_columns(conn)
    before = v1_row_fingerprint(url, cols)
    bf = import_backfill()
    bf.backfill(url)
    bf.backfill(url)
    assert v1_row_fingerprint(url, cols) == before
    assert fetchone(url, "SELECT count(*) FROM measurements WHERE evaluation_id IS NOT NULL")[0] == 0
    assert fetchone(url, "SELECT count(*) FROM evaluations")[0] == 0


def test_backfill_unmapped_campaign_raises(v1_like):
    url, _ = v1_like
    insert_sql_campaign(url, "orphan-run", author="someone-else@example.com")
    with pytest.raises(Exception) as ei:
        import_backfill().backfill(url)
    assert not isinstance(ei.value, (ImportError, AttributeError, TypeError)), repr(ei.value)


def test_backfill_skips_campaigns_already_mapped(v1_like):
    """New-style uploads already have campaign_sources rows and are left alone."""
    url, ids = v1_like
    apply_provenance_schema(url)
    slug, title = allocate(url)
    p = build_parsed(author="alice-test@example.com", source=anon_source_block())
    assert insert(url, p, assigned_anon_slug=slug, submitted_by="alice-test@example.com") is True
    counts = import_backfill().backfill(url)
    for k, v in EXPECTED.items():
        assert counts[k] == v
    row = fetchone(url, "SELECT cs.source_slug, cs.submitted_by, c.author FROM campaigns c "
                        "JOIN campaign_sources cs ON cs.campaign_id=c.id WHERE c.name=%s", (p.campaign.name,))
    assert row == (slug, "alice-test@example.com", title)
    # Its measurements keep their generation-time evaluation.
    assert fetchone(url, "SELECT count(*) FROM measurements WHERE evaluation_id IS NOT NULL")[0] == 4


def test_backfill_seed_upsert_noop_on_identical_row(v1_like):
    url, _ = v1_like
    bf = import_backfill()
    bf.backfill(url)
    before = fetchall(url, "SELECT * FROM source_datasets ORDER BY slug")
    bf.backfill(url)
    assert fetchall(url, "SELECT * FROM source_datasets ORDER BY slug") == before


def test_backfill_cli(v1_like):
    url, _ = v1_like
    import_backfill()
    p = subprocess.run([sys.executable, str(SRC_DIR / "backfill_provenance.py"), url],
                       cwd=SRC_DIR, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    assert dict(fetchall(url, "SELECT source_slug, count(*) FROM campaign_sources GROUP BY 1")) == EXPECTED


def test_backfill_empty_db(fresh_db):
    url = fresh_db(provenance=False)
    counts = import_backfill().backfill(url)
    assert _nonzero(counts) == {}
    assert len(fetchall(url, "SELECT slug FROM source_datasets")) == 4
