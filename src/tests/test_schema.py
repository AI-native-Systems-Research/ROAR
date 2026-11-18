"""Spec: "Hard constraints" (SCHEMA byte-identical, PROVENANCE_SCHEMA separate) and
"Schema (create_tables.PROVENANCE_SCHEMA)" incl. CHECKs, FKs, ON DELETE CASCADE,
idempotency, create_tables.main() and drop_tables.py drop order."""

from __future__ import annotations

import hashlib
import uuid

import psycopg
import pytest
from psycopg import errors, sql

from conftest import (
    ADMIN_URL,
    apply_provenance_schema,
    db_url,
    drop_database,
    fetchall,
    fetchone,
    insert_sql_campaign,
    insert_sql_source,
    map_campaign,
    unique,
)

PROV_TABLES = {"source_datasets", "campaign_sources", "evaluations"}


# sha256 of the original (pre-feature) create_tables.SCHEMA string, UTF-8 encoded.
ORIGINAL_SCHEMA_SHA256 = "4835f823fe80fc200ef0a4cd6f1ff1505ada7c6911f145470d393d899312635e"


def test_schema_constant_is_byte_identical_to_original():
    import create_tables

    assert hashlib.sha256(create_tables.SCHEMA.encode("utf-8")).hexdigest() == ORIGINAL_SCHEMA_SHA256


def test_provenance_schema_is_separate_constant():
    import create_tables

    ps = getattr(create_tables, "PROVENANCE_SCHEMA", None)
    assert isinstance(ps, str) and ps.strip(), "create_tables.PROVENANCE_SCHEMA must be a non-empty str"
    assert "source_datasets" not in create_tables.SCHEMA
    assert "evaluation_id" not in create_tables.SCHEMA
    for needle in ("source_datasets", "campaign_sources", "evaluations", "anon_contributor_seq", "evaluation_id"):
        assert needle in ps


def test_provenance_schema_uses_if_not_exists_everywhere():
    import create_tables

    ps = getattr(create_tables, "PROVENANCE_SCHEMA", None)
    assert ps is not None
    upper = " ".join(ps.upper().split())
    for stmt in ("CREATE TABLE ", "CREATE SEQUENCE ", "ADD COLUMN ", "CREATE INDEX ", "CREATE UNIQUE INDEX "):
        start = 0
        while (i := upper.find(stmt, start)) != -1:
            assert upper[i + len(stmt):].startswith("IF NOT EXISTS"), f"{stmt.strip()} without IF NOT EXISTS"
            start = i + 1


def _public_tables(url):
    return {r[0] for r in fetchall(url, "SELECT table_name FROM information_schema.tables WHERE table_schema='public'")}


def test_provenance_schema_is_idempotent(fresh_db):
    url = fresh_db(provenance=False)
    apply_provenance_schema(url)
    tables_once = _public_tables(url)
    cid = insert_sql_campaign(url, unique("idem-"))
    slug = f"idem-{uuid.uuid4().hex[:6]}@v1"
    insert_sql_source(url, slug)
    map_campaign(url, cid, slug, "alice-test@example.com")
    apply_provenance_schema(url)
    apply_provenance_schema(url)
    assert _public_tables(url) == tables_once
    assert PROV_TABLES <= tables_once
    assert fetchone(url, "SELECT count(*) FROM campaign_sources")[0] == 1
    assert fetchone(url, "SELECT count(*) FROM source_datasets WHERE slug=%s", (slug,))[0] == 1


def test_provenance_schema_objects(db):
    cols = {
        (r[0], r[1]): (r[2], r[3])
        for r in fetchall(
            db,
            "SELECT table_name, column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema='public'",
        )
    }
    expected_sd = ["slug", "title", "version", "year", "url", "license_spdx", "redistribution",
                   "attribution", "authors", "bibtex", "consent_recorded_at", "owner"]
    for c in expected_sd:
        assert ("source_datasets", c) in cols, c
    assert cols[("source_datasets", "year")][0] == "integer"
    assert cols[("source_datasets", "consent_recorded_at")][0] == "timestamp with time zone"
    assert cols[("source_datasets", "redistribution")][1] == "NO"
    assert cols[("source_datasets", "attribution")][1] == "NO"
    for c in ("campaign_id", "source_slug", "submitted_by"):
        assert ("campaign_sources", c) in cols
    assert cols[("campaign_sources", "campaign_id")][0] == "uuid"
    assert cols[("campaign_sources", "source_slug")][1] == "NO"
    for c in ("id", "assessor_name", "assessor_version", "platform", "code_ref", "kind", "created_at"):
        assert ("evaluations", c) in cols
    for c in ("assessor_name", "assessor_version", "platform", "kind", "created_at"):
        assert cols[("evaluations", c)][1] == "NO", c
    assert cols[("evaluations", "code_ref")][1] == "YES"
    assert cols[("measurements", "evaluation_id")] == ("uuid", "YES")
    seq = fetchone(db, "SELECT start_value FROM pg_sequences WHERE sequencename='anon_contributor_seq'")
    assert seq is not None and seq[0] == 1


def _insert_source_expect(url, ok: bool, **cols):
    slug = cols.pop("slug", f"chk-{uuid.uuid4().hex[:8]}@v1")
    if ok:
        insert_sql_source(url, slug, **cols)
        assert fetchone(url, "SELECT count(*) FROM source_datasets WHERE slug=%s", (slug,))[0] == 1
    else:
        with pytest.raises((errors.CheckViolation, errors.NotNullViolation)):
            insert_sql_source(url, slug, **cols)


def test_check_anonymous_rejects_authors(db):
    _insert_source_expect(db, False, attribution="anonymous", title="Independent contributor Q",
                          authors="Alice Test", bibtex=None)


def test_check_anonymous_rejects_bibtex(db):
    _insert_source_expect(db, False, attribution="anonymous", title="Independent contributor Q",
                          authors=None, bibtex="@misc{x, title={y}}")


def test_check_anonymous_ok_without_authors_and_bibtex(db):
    _insert_source_expect(db, True, attribution="anonymous", title="Independent contributor Q",
                          authors=None, bibtex=None, owner="alice-test@example.com")


def test_check_named_rejects_null_title(db):
    _insert_source_expect(db, False, attribution="named", title=None)


def test_check_named_rejects_null_authors(db):
    _insert_source_expect(db, False, attribution="named", authors=None)


@pytest.mark.parametrize(
    "title",
    ["independent Contributor X", "Independent contributor A", "INDEPENDENT CONTRIBUTOR B", "independent contributor"],
)
def test_check_named_rejects_independent_contributor_title_any_case(db, title):
    _insert_source_expect(db, False, attribution="named", title=title)


def test_check_named_ok(db):
    _insert_source_expect(db, True, attribution="named", title="A real dataset title")


@pytest.mark.parametrize("col,val", [("redistribution", "other"), ("attribution", "other"),
                                     ("redistribution", None), ("attribution", None)])
def test_check_enumerations(db, col, val):
    _insert_source_expect(db, False, **{col: val})


def test_pointer_only_and_null_license_allowed_by_schema(db):
    _insert_source_expect(db, True, redistribution="pointer_only", license_spdx=None)


def test_evaluations_kind_check(db):
    with psycopg.connect(db) as conn:
        conn.execute(
            "INSERT INTO evaluations (assessor_name, assessor_version, platform, kind) "
            "VALUES ('a', '1', 'unknown', 'regrade')"
        )
        conn.rollback()
        with pytest.raises(errors.CheckViolation):
            conn.execute(
                "INSERT INTO evaluations (assessor_name, assessor_version, platform, kind) "
                "VALUES ('a', '1', 'unknown', 'bogus')"
            )
        conn.rollback()
        with pytest.raises(errors.NotNullViolation):
            conn.execute(
                "INSERT INTO evaluations (assessor_name, assessor_version, platform) VALUES ('a', '1', 'x')"
            )


def test_evaluation_defaults(db):
    with psycopg.connect(db) as conn:
        row = conn.execute(
            "INSERT INTO evaluations (assessor_name, assessor_version, platform, kind) "
            "VALUES ('a', '1', 'unknown', 'generation_time') RETURNING id, created_at"
        ).fetchone()
        assert isinstance(row[0], uuid.UUID) and row[1] is not None
        conn.rollback()


def test_measurement_evaluation_fk(db):
    cid = insert_sql_campaign(db, unique("fk-"), n_candidates=1)
    with psycopg.connect(db) as conn:
        mid = conn.execute(
            "SELECT m.id FROM measurements m JOIN candidates c ON c.id=m.candidate_id WHERE c.campaign_id=%s",
            (cid,),
        ).fetchone()[0]
        with pytest.raises(errors.ForeignKeyViolation):
            conn.execute("UPDATE measurements SET evaluation_id=%s WHERE id=%s", (uuid.uuid4(), mid))
        conn.rollback()
    assert fetchone(db, "SELECT evaluation_id FROM measurements WHERE id=%s", (mid,))[0] is None


def test_campaign_sources_fk_and_primary_key(db):
    cid = insert_sql_campaign(db, unique("cs-"), n_candidates=1)
    with pytest.raises(errors.ForeignKeyViolation):
        map_campaign(db, cid, f"no-such-{uuid.uuid4().hex[:6]}@v1")
    slug = f"cs-{uuid.uuid4().hex[:6]}@v1"
    insert_sql_source(db, slug)
    map_campaign(db, cid, slug, "alice-test@example.com")
    with pytest.raises(errors.UniqueViolation):
        map_campaign(db, cid, slug, "alice-test@example.com")
    with pytest.raises(errors.ForeignKeyViolation):
        map_campaign(db, str(uuid.uuid4()), slug)


def test_campaign_sources_on_delete_cascade(db):
    cid = insert_sql_campaign(db, unique("cascade-"), n_candidates=1)
    slug = f"cascade-{uuid.uuid4().hex[:6]}@v1"
    insert_sql_source(db, slug)
    map_campaign(db, cid, slug, "alice-test@example.com")
    with psycopg.connect(db) as conn:
        conn.execute("DELETE FROM campaigns WHERE id=%s", (cid,))
        conn.commit()
    assert fetchone(db, "SELECT count(*) FROM campaign_sources WHERE campaign_id=%s", (cid,))[0] == 0
    # The source itself is not cascaded away.
    assert fetchone(db, "SELECT count(*) FROM source_datasets WHERE slug=%s", (slug,))[0] == 1


def test_create_tables_main_applies_schema_then_provenance(pg, monkeypatch):
    import create_tables

    name = unique("roar_provtest_")
    monkeypatch.setattr(create_tables, "DATABASE_URL", db_url(name))
    monkeypatch.setattr(create_tables, "POSTGRES_URL", ADMIN_URL)
    try:
        create_tables.main()
        tables = _public_tables(db_url(name))
        assert {"systems", "campaigns", "candidates", "measurements", "artifacts", "candidate_edges"} <= tables
        assert PROV_TABLES <= tables
        assert fetchone(db_url(name), "SELECT count(*) FROM pg_sequences WHERE sequencename='anon_contributor_seq'")[0] == 1
        # Re-running is harmless (idempotent).
        create_tables.main()
    finally:
        drop_database(name)


def test_drop_tables_drops_provenance_objects(fresh_db, monkeypatch):
    import builtins

    import drop_tables

    url = fresh_db(provenance=True)
    cid = insert_sql_campaign(url, unique("drop-"))
    slug = f"drop-{uuid.uuid4().hex[:6]}@v1"
    insert_sql_source(url, slug)
    map_campaign(url, cid, slug, None)
    with psycopg.connect(url) as conn:
        conn.execute(
            "INSERT INTO evaluations (assessor_name, assessor_version, platform, kind) "
            "VALUES ('a', '1', 'unknown', 'generation_time')"
        )
        conn.commit()
    monkeypatch.setattr(drop_tables, "DATABASE_URL", url)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "yes")
    drop_tables.main()
    assert _public_tables(url) == set()
    assert fetchone(url, "SELECT count(*) FROM pg_sequences WHERE sequencename='anon_contributor_seq'")[0] == 0
    # A clean re-create works after the drop.
    import create_tables

    with psycopg.connect(url) as conn:
        conn.execute(create_tables.SCHEMA)
        conn.execute(create_tables.PROVENANCE_SCHEMA)
        conn.commit()
    assert PROV_TABLES <= _public_tables(url)


def test_regrade_representable_and_generation_time_predicate(db):
    """Spec: the schema can represent a regrade; a measurement is generation-time iff
    evaluation_id IS NULL OR evaluations.kind = 'generation_time'."""
    cid = insert_sql_campaign(db, unique("regrade-"), n_candidates=3)
    with psycopg.connect(db) as conn:
        mids = [r[0] for r in conn.execute(
            "SELECT m.id FROM measurements m JOIN candidates c ON c.id=m.candidate_id "
            "WHERE c.campaign_id=%s ORDER BY c.iteration_index", (cid,)).fetchall()]
        assert len(mids) == 3
        gen = conn.execute("INSERT INTO evaluations (assessor_name, assessor_version, platform, code_ref, kind) "
                           "VALUES ('ev', '1', 'unknown', NULL, 'generation_time') RETURNING id").fetchone()[0]
        reg = conn.execute("INSERT INTO evaluations (assessor_name, assessor_version, platform, code_ref, kind) "
                           "VALUES ('ev', '2', 'linux', 'git:abc', 'regrade') RETURNING id").fetchone()[0]
        conn.execute("UPDATE measurements SET evaluation_id=%s WHERE id=%s", (gen, mids[1]))
        # A regrade: a new score for an existing candidate from a different assessor.
        cand = conn.execute("SELECT candidate_id FROM measurements WHERE id=%s", (mids[2],)).fetchone()[0]
        regraded = conn.execute("INSERT INTO measurements (candidate_id, name, value, evaluation_id) "
                                "VALUES (%s, 'combined_score', '0.99', %s) RETURNING id", (cand, reg)).fetchone()[0]
        conn.commit()
        gen_time = {r[0] for r in conn.execute(
            "SELECT m.id FROM measurements m LEFT JOIN evaluations e ON e.id=m.evaluation_id "
            "JOIN candidates c ON c.id=m.candidate_id WHERE c.campaign_id=%s "
            "AND (m.evaluation_id IS NULL OR e.kind='generation_time')", (cid,)).fetchall()}
    assert gen_time == set(mids)  # v1 (NULL) and generation_time rows included
    assert regraded not in gen_time  # the regrade excluded
