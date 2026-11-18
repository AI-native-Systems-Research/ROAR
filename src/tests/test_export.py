"""Spec "Export (hf_export/export.py)": export_provenance(cur, output_dir) writes exactly
sources.parquet (every source_datasets column except owner) and campaign_sources.parquet
(campaign_id, source_slug), never owner/submitted_by, and leaves the six v1 parquet
files alone."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
import pytest

from conftest import fetchone, insert_sql_campaign, insert_sql_source, map_campaign, unique

OWNER = "owner-test@example.com"
SUBMITTER = "submitter-test@example.com"
SOURCE_COLS = {"slug", "title", "version", "year", "url", "license_spdx", "redistribution", "attribution",
               "authors", "bibtex", "consent_recorded_at"}
V1_FILES = ["systems", "campaigns", "candidates", "measurements", "candidate_edges", "artifacts"]


def _export_fn():
    try:
        from hf_export.export import export_provenance
    except ImportError as e:
        pytest.fail(f"hf_export.export.export_provenance is not importable: {e}")
    return export_provenance


@pytest.fixture
def populated(db):
    tag = uuid.uuid4().hex[:6]
    named, anon = f"exp-{tag}@v1", f"contributor-X{tag.upper()}@v1"
    insert_sql_source(db, named, owner=OWNER, consent_recorded_at="2026-11-17T00:00:00+00:00")
    insert_sql_source(db, anon, attribution="anonymous", title="Independent contributor X", authors=None,
                      owner=OWNER)
    c1 = insert_sql_campaign(db, unique("exp-"), n_candidates=1)
    c2 = insert_sql_campaign(db, unique("exp-"), n_candidates=1)
    map_campaign(db, c1, named, SUBMITTER)
    map_campaign(db, c2, anon, SUBMITTER)
    return {"named": named, "anon": anon, "c1": c1, "c2": c2}


def _run(db, out_dir: Path):
    fn = _export_fn()
    with psycopg.connect(db) as conn:
        with conn.cursor() as cur:
            return fn(cur, out_dir)


def test_export_writes_exactly_two_files_with_spec_columns(db, populated, tmp_path):
    import pyarrow.parquet as pq

    paths = _run(db, tmp_path)
    assert isinstance(paths, list) and all(isinstance(p, Path) for p in paths)
    assert {Path(p).resolve() for p in paths} == {(tmp_path / "sources.parquet").resolve(),
                                                  (tmp_path / "campaign_sources.parquet").resolve()}
    assert sorted(os.listdir(tmp_path)) == ["campaign_sources.parquet", "sources.parquet"]
    src = pq.read_table(tmp_path / "sources.parquet")
    cs = pq.read_table(tmp_path / "campaign_sources.parquet")
    assert set(src.column_names) == SOURCE_COLS
    assert set(cs.column_names) == {"campaign_id", "source_slug"}
    assert src.num_rows == fetchone(db, "SELECT count(*) FROM source_datasets")[0]
    assert cs.num_rows == fetchone(db, "SELECT count(*) FROM campaign_sources")[0]
    rows = {r["slug"]: r for r in src.to_pylist()}
    assert rows[populated["anon"]]["authors"] is None
    assert rows[populated["anon"]]["title"] == "Independent contributor X"
    mapping = {str(r["campaign_id"]): r["source_slug"] for r in cs.to_pylist()}
    assert mapping[populated["c1"]] == populated["named"]
    assert mapping[populated["c2"]] == populated["anon"]


def test_export_never_contains_owner_or_submitted_by(db, populated, tmp_path):
    import pyarrow.parquet as pq

    for p in _run(db, tmp_path):
        raw = Path(p).read_bytes()
        assert b"submitted_by" not in raw and b"owner" not in raw, p
        assert OWNER.encode() not in raw and SUBMITTER.encode() not in raw, p
        t = pq.read_table(p)
        assert "owner" not in t.column_names and "submitted_by" not in t.column_names
        flat = str(t.to_pylist())
        assert OWNER not in flat and SUBMITTER not in flat


def test_export_does_not_touch_v1_parquet_files(db, populated, tmp_path):
    before = {}
    for i, n in enumerate(V1_FILES):
        p = tmp_path / f"{n}.parquet"
        p.write_bytes(f"sentinel-{n}".encode())
        os.utime(p, ns=(1_000_000_000 + i, 1_000_000_000 + i))
        before[n] = (p.read_bytes(), p.stat().st_mtime_ns)
    other = tmp_path / "unrelated.txt"
    other.write_text("keep me")
    _run(db, tmp_path)
    for n in V1_FILES:
        p = tmp_path / f"{n}.parquet"
        assert (p.read_bytes(), p.stat().st_mtime_ns) == before[n], n
    assert other.read_text() == "keep me"
    assert set(os.listdir(tmp_path)) == {f"{n}.parquet" for n in V1_FILES} | {
        "unrelated.txt", "sources.parquet", "campaign_sources.parquet"}

