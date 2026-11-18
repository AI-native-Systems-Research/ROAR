"""Shared fixtures for the provenance / assessor-versioning test suite.

Spec: ``docs/PROVENANCE_SPEC.md`` (normative).

Design notes
------------
* New public names (``provenance``, ``create_tables.PROVENANCE_SCHEMA``,
  ``backfill_provenance`` ...) are imported lazily, inside fixtures or tests, so a
  missing implementation fails each test individually instead of breaking collection.
* Every database a test writes to is a throwaway database named ``roar_provtest_*``
  (or ``roar_v1_provcopy_*`` / ``roar_full_provcopy_*`` for the golden tests). The
  frozen shared databases ``roar_v1`` and ``roar_full`` are only ever *read*, and only
  by the golden and name-leak tests. ``guard_not_frozen`` enforces this for every URL
  this suite creates, drops or writes to.
* Synthetic names only (``alice-test`` ...). No contributor personal names.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import psycopg  # noqa: E402
from psycopg import sql  # noqa: E402

PG_HOST = os.environ.get("ROAR_TEST_PG_HOST", "localhost")
PG_PORT = int(os.environ.get("ROAR_TEST_PG_PORT", "55432"))
PG_USER = os.environ.get("ROAR_TEST_PG_USER", "postgres")
PG_PASSWORD = os.environ.get("ROAR_TEST_PG_PASSWORD", "postgres")
DOCKER_CONTAINER = os.environ.get("ROAR_TEST_PG_CONTAINER", "roar-pg")

FROZEN_DBS = frozenset({"roar_v1", "roar_full"})

# Six original (v1) tables. The migration and backfill must never change their rows.
V1_TABLES = ("systems", "campaigns", "candidates", "candidate_edges", "measurements", "artifacts")

SEED_SLUGS = (
    "ellis-castro-2026@v1",
    "cheng2025barbarians@v1",
    "gupta2026harness@v1",
    "ibm-ai-native-systems@v1",
)

DEFAULT_ASSESSOR = {
    "name": "synthetic-assessor",
    "version": "1.0.0",
    "platform": "linux-x86_64",
    "code_ref": "https://example.com/assessor/commit/abc123",
}


# ---------------------------------------------------------------------------
# Database URL helpers
# ---------------------------------------------------------------------------

def db_url(dbname: str) -> str:
    return f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{dbname}"


ADMIN_URL = db_url("postgres")


def guard_not_frozen(name_or_url: str) -> None:
    """Refuse (hard error) to touch a frozen shared database."""
    name = name_or_url.rsplit("/", 1)[-1].split("?", 1)[0]
    if name in FROZEN_DBS:
        raise RuntimeError(f"test suite refused to write to frozen database {name!r}")


def pg_available() -> bool:
    try:
        with psycopg.connect(ADMIN_URL, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


_PG_OK: bool | None = None


def require_pg() -> None:
    global _PG_OK
    if _PG_OK is None:
        _PG_OK = pg_available()
    if not _PG_OK:
        pytest.skip(
            f"Postgres+pgvector (container {DOCKER_CONTAINER!r}) is unreachable at "
            f"{PG_HOST}:{PG_PORT}; DB-backed provenance tests need it"
        )


def unique(prefix: str = "t") -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


def create_database(name: str, *, provenance: bool) -> str:
    """Create a throwaway DB with create_tables.SCHEMA (+ PROVENANCE_SCHEMA)."""
    guard_not_frozen(name)
    import create_tables

    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = db_url(name)
    with psycopg.connect(url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(create_tables.SCHEMA)
        conn.commit()
    if provenance:
        apply_provenance_schema(url)
    return url


def apply_provenance_schema(url: str) -> None:
    guard_not_frozen(url)
    import create_tables

    schema = getattr(create_tables, "PROVENANCE_SCHEMA", None)
    if schema is None:
        pytest.fail("create_tables.PROVENANCE_SCHEMA is not defined (spec: Schema)")
    with psycopg.connect(url) as conn:
        conn.execute(schema)
        conn.commit()


def drop_database(name: str) -> None:
    guard_not_frozen(name)
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
        )


def fetchall(url: str, query: str, params: Any = None) -> list[tuple]:
    with psycopg.connect(url) as conn:
        return conn.execute(query, params).fetchall()


def fetchone(url: str, query: str, params: Any = None):
    with psycopg.connect(url) as conn:
        return conn.execute(query, params).fetchone()


def execute(url: str, query: str, params: Any = None) -> None:
    guard_not_frozen(url)
    with psycopg.connect(url) as conn:
        conn.execute(query, params)
        conn.commit()


def table_counts(url: str) -> dict[str, int]:
    """Row counts of every v1 + provenance table that exists in the DB."""
    tables = list(V1_TABLES) + ["source_datasets", "campaign_sources", "evaluations"]
    out: dict[str, int] = {}
    with psycopg.connect(url) as conn:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
            ).fetchall()
        }
        for t in tables:
            if t in existing:
                out[t] = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(t))).fetchone()[0]
    return out


def v1_row_fingerprint(url: str, columns_by_table: dict[str, list[str]] | None = None) -> dict[str, tuple[int, str]]:
    """(row count, md5 over per-row md5s ordered by id) for each v1 table.

    Only the *original* v1 columns are hashed (captured from information_schema
    before any migration when ``columns_by_table`` is None), so the additive
    ``measurements.evaluation_id`` column does not perturb the fingerprint.
    """
    with psycopg.connect(url) as conn:
        if columns_by_table is None:
            columns_by_table = v1_columns(conn)
        out = {}
        for t in V1_TABLES:
            cols = columns_by_table[t]
            row_expr = sql.SQL("ROW({})::text").format(
                sql.SQL(", ").join(sql.Identifier(c) for c in cols)
            )
            q = sql.SQL(
                "SELECT count(*), coalesce(md5(string_agg(md5({row}), '' ORDER BY id)), '') FROM {t}"
            ).format(row=row_expr, t=sql.Identifier(t))
            n, h = conn.execute(q).fetchone()
            out[t] = (n, h)
        return out


def v1_columns(conn) -> dict[str, list[str]]:
    import create_tables  # noqa: F401  (documented dependency)

    cols: dict[str, list[str]] = {}
    for t in V1_TABLES:
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position",
            (t,),
        ).fetchall()
        cols[t] = [r[0] for r in rows if not (t == "measurements" and r[0] == "evaluation_id")]
    return cols


# ---------------------------------------------------------------------------
# Lazy import helpers for the new public names
# ---------------------------------------------------------------------------

def import_provenance():
    try:
        import provenance
    except ImportError as e:  # pragma: no cover - red phase
        pytest.fail(f"spec module `provenance` is not importable: {e}")
    return provenance


def provenance_error():
    prov = import_provenance()
    err = getattr(prov, "ProvenanceError", None)
    if err is None:
        pytest.fail("provenance.ProvenanceError is not defined")
    return err


def import_backfill():
    try:
        import backfill_provenance
    except ImportError as e:  # pragma: no cover - red phase
        pytest.fail(f"spec module `backfill_provenance` is not importable: {e}")
    return backfill_provenance


def require_attr(module, name: str):
    obj = getattr(module, name, None)
    if obj is None:
        pytest.fail(f"{module.__name__}.{name} is not defined (spec)")
    return obj


# ---------------------------------------------------------------------------
# Parsed-campaign and source-block builders
# ---------------------------------------------------------------------------

def named_source_block(slug: str | None = None, **overrides) -> dict:
    slug = slug or f"synthetic-{uuid.uuid4().hex[:8]}@v1"
    block = {
        "slug": slug,
        "title": f"Synthetic dataset {slug}",
        "version": slug.rsplit("@", 1)[-1] if "@" in slug else "v1",
        "year": 2026,
        "url": f"https://example.com/datasets/{slug.replace('@', '/')}",
        "license_spdx": "CC-BY-4.0",
        "redistribution": "full",
        "attribution": "named",
        "authors": "Alice Test and Bob Test",
        "bibtex": None,
        "consent": True,
    }
    block.update(overrides)
    return {k: v for k, v in block.items() if v is not None or k in overrides}


def anon_source_block(**overrides) -> dict:
    block = {
        "attribution": "anonymous",
        "redistribution": "full",
        "license_spdx": "CC-BY-4.0",
        "consent": True,
    }
    block.update(overrides)
    return block


def build_parsed_dict(
    name: str | None = None,
    *,
    author: str = "alice-test@example.com",
    assessor: dict | None | str = "default",
    source: dict | None = None,
    n_candidates: int = 2,
    system_name: str | None = None,
) -> dict:
    """A minimal valid ADRSParsedCampaign as a JSON-able dict."""
    name = name or unique("campaign-")
    cands = []
    meas: dict[str, list[dict]] = {}
    for i in range(n_candidates):
        ext = f"{name}-p{i}"
        cands.append({"iteration_index": i, "external_id": ext, "candidate_type": "program"})
        meas[ext] = [
            {"name": "combined_score", "value": str(0.1 * (i + 1))},
            {"name": "status", "value": "VALID"},
        ]
    edges = [
        {"source_external_id": f"{name}-p{i - 1}", "target_external_id": f"{name}-p{i}", "edge_type": "parent"}
        for i in range(1, n_candidates)
    ]
    d: dict[str, Any] = {
        "campaign": {
            "system": {"name": system_name or "synthetic-system", "version": "0.0.1"},
            "author": author,
            "name": name,
            "research_question": "Synthetic research question",
            "algorithm_used": "evox",
            "models_used": ["synthetic-model"],
        },
        "candidates": cands,
        "measurements": meas,
        "artifacts": [{"iteration_index": 0, "external_id": f"{name}-p0", "uri": "x/program.py"}],
        "candidate_edges": edges,
    }
    if assessor == "default":
        d["assessor"] = dict(DEFAULT_ASSESSOR)
    elif assessor is not None:
        d["assessor"] = assessor
    if source is not None:
        d["source"] = source
    return d


def build_parsed(name: str | None = None, **kw):
    from adrs_models import ADRSParsedCampaign

    return ADRSParsedCampaign.model_validate(build_parsed_dict(name, **kw))


# ---------------------------------------------------------------------------
# Campaign folders (real SkyDiscover layout, parsed by the real parser)
# ---------------------------------------------------------------------------

def write_skydiscover_campaign(folder: Path, author_input: dict | None = None, n_programs: int = 2) -> Path:
    """Create a minimal SkyDiscover run that system_detection + parse_skydiscover accept.

    The campaign name the parser records is ``folder.name``.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text("research_question: Synthetic research question\nsearch:\n  type: evox\n")
    progs = folder / "checkpoints" / "checkpoint_1" / "programs"
    progs.mkdir(parents=True, exist_ok=True)
    for i in range(n_programs):
        rec = {
            "id": f"p{i}",
            "iteration": i,
            "metrics": {"combined_score": 0.1 * (i + 1)},
            "solution": f"def f(): return {i}",
        }
        if i:
            rec["parent_id"] = f"p{i - 1}"
        (progs / f"p{i}.json").write_text(json.dumps(rec))
    if author_input is not None:
        write_author_input(folder, author_input)
    return folder


def write_author_input(folder: Path, data: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "author_input.json"
    p.write_text(json.dumps(data))
    return p


def make_targz(src_dir: Path, members: list[str] | None = None) -> bytes:
    """tar.gz the *contents* of src_dir (paths relative to src_dir)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        names = members if members is not None else sorted(p.name for p in src_dir.iterdir())
        for n in names:
            tf.add(src_dir / n, arcname=n)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# pytest hooks / fixtures
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "golden: slow end-to-end checks against copies of the frozen roar_v1/roar_full DBs",
    )


@pytest.fixture(scope="session")
def pg() -> None:
    require_pg()


@pytest.fixture(scope="session")
def base_db_name(pg) -> Iterator[str]:
    """Session throwaway DB ``roar_provtest_<rand>`` with create_tables.SCHEMA only."""
    name = unique("roar_provtest_")
    create_database(name, provenance=False)
    try:
        yield name
    finally:
        drop_database(name)


@pytest.fixture(scope="session")
def db_url_session(base_db_name) -> str:
    """Session throwaway DB with SCHEMA + PROVENANCE_SCHEMA applied."""
    url = db_url(base_db_name)
    apply_provenance_schema(url)
    return url


@pytest.fixture
def db(db_url_session, monkeypatch) -> str:
    """URL of the session provenance DB, also wired into DATABASE_URL for callers
    that read it (CLI helpers, CampaignInserter default, API)."""
    monkeypatch.setenv("DATABASE_URL", db_url_session)
    import insert_adrs_campaign

    monkeypatch.setattr(insert_adrs_campaign, "DATABASE_URL", db_url_session, raising=False)
    return db_url_session


@pytest.fixture
def fresh_db(pg) -> Iterator[Callable[..., str]]:
    """Factory for function-scoped throwaway DBs; all dropped at teardown."""
    created: list[str] = []

    def _make(provenance: bool = True) -> str:
        name = unique("roar_provtest_")
        created.append(name)
        return create_database(name, provenance=provenance)

    yield _make
    for n in created:
        drop_database(n)


@pytest.fixture(scope="session")
def golden_copies(pg):
    """Factory: golden_copies("roar_v1") -> GoldenCopy (created once per session, dropped
    at session end). Skips if the frozen source DB does not exist."""
    from golden_support import GoldenCopy, database_exists

    cache: dict[str, GoldenCopy] = {}

    def _get(src: str):
        if src not in cache:
            if not database_exists(src):
                pytest.skip(f"frozen source database {src!r} not present on {PG_HOST}:{PG_PORT}")
            cache[src] = GoldenCopy(src)
        return cache[src]

    try:
        yield _get
    finally:
        for gc in cache.values():
            gc.drop()


@pytest.fixture
def make_parsed():
    return build_parsed


@pytest.fixture
def make_campaign_folder():
    return write_skydiscover_campaign


@pytest.fixture
def prov():
    return import_provenance()


@pytest.fixture
def ProvenanceError():
    return provenance_error()


def insert(url: str, parsed, **kw):
    """Run CampaignInserter(url).insert(parsed, **kw) with a fresh inserter."""
    from insert_adrs_campaign import CampaignInserter

    return CampaignInserter(database_url=url).insert(parsed, **kw)


def campaign_row(url: str, name: str):
    return fetchone(url, "SELECT id, author FROM campaigns WHERE name=%s", (name,))


def allocate(url: str) -> tuple[str, str]:
    import insert_adrs_campaign

    fn = require_attr(insert_adrs_campaign, "allocate_anon_slug")
    with psycopg.connect(url) as conn:
        out = fn(conn)
        conn.commit()
    return out


def label_to_int(label: str) -> int:
    n = 0
    for ch in label:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def insert_sql_campaign(url: str, name: str, author: str = "Roar Authors", n_candidates: int = 2) -> str:
    """Insert a v1-style campaign directly with SQL (no provenance). Returns id (str)."""
    guard_not_frozen(url)
    with psycopg.connect(url) as conn:
        sys_id = conn.execute(
            "INSERT INTO systems (name, version) VALUES ('skydiscover', 'sqlv1') "
            "ON CONFLICT (name, version) DO UPDATE SET name = EXCLUDED.name RETURNING id"
        ).fetchone()[0]
        cid = conn.execute(
            "INSERT INTO campaigns (system_id, author, name, research_question, algorithm_used, models_used) "
            "VALUES (%s, %s, %s, 'Synthetic research question', 'evox', ARRAY['synthetic-model']) RETURNING id",
            (sys_id, author, name),
        ).fetchone()[0]
        prev = None
        for i in range(n_candidates):
            cand = conn.execute(
                "INSERT INTO candidates (campaign_id, iteration_index, external_id, candidate_type) "
                "VALUES (%s, %s, %s, 'program') RETURNING id",
                (cid, i, f"p{i}"),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO measurements (candidate_id, name, value) VALUES (%s, 'combined_score', %s)",
                (cand, str(0.1 * (i + 1))),
            )
            conn.execute(
                "INSERT INTO artifacts (campaign_id, candidate_id, iteration_index, uri) VALUES (%s, %s, %s, 'a.py')",
                (cid, cand, i),
            )
            if prev is not None:
                conn.execute(
                    "INSERT INTO candidate_edges (source_candidate_id, target_candidate_id) VALUES (%s, %s)",
                    (prev, cand),
                )
            prev = cand
        conn.commit()
    return str(cid)


def insert_sql_source(url: str, slug: str, **cols) -> None:
    """Insert a source_datasets row directly (bypassing the inserter)."""
    guard_not_frozen(url)
    row = {
        "slug": slug,
        "title": f"Synthetic dataset {slug}",
        "version": "v1",
        "year": 2026,
        "url": "https://example.com/d",
        "license_spdx": "CC-BY-4.0",
        "redistribution": "full",
        "attribution": "named",
        "authors": "Alice Test",
        "bibtex": None,
        "consent_recorded_at": None,
        "owner": None,
    }
    row.update(cols)
    keys = list(row)
    q = sql.SQL("INSERT INTO source_datasets ({}) VALUES ({})").format(
        sql.SQL(", ").join(sql.Identifier(k) for k in keys),
        sql.SQL(", ").join(sql.Placeholder() for _ in keys),
    )
    with psycopg.connect(url) as conn:
        conn.execute(q, [row[k] for k in keys])
        conn.commit()


def map_campaign(url: str, campaign_id: str, slug: str, submitted_by: str | None = None) -> None:
    execute(
        url,
        "INSERT INTO campaign_sources (campaign_id, source_slug, submitted_by) VALUES (%s, %s, %s)",
        (campaign_id, slug, submitted_by),
    )
