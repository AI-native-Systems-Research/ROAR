"""Golden tests (marker ``golden``; slow; opt-in). Spec "Hard constraints" (v1 rows
immutable; analytics return identical results after migration + backfill) and
"Backfill" (exact counts, idempotent, evaluation_id stays NULL).

Each source DB named in ROAR_GOLDEN_DBS is copied to ``<src>_provcopy_<rand>`` with
``pg_dump | psql`` in ROAR_TEST_PG_CONTAINER, backfilled, compared, and dropped at
session end. See tests/README.md."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from conftest import SEED_SLUGS, fetchall, fetchone, golden_dbs, v1_row_fingerprint

pytestmark = pytest.mark.golden

# Spec "Backfill > Return value": exact counts for the two corpora the spec names,
# identified by their campaign count (928 = the v1 corpus, 1,072 = the full corpus).
EXPECTED_COUNTS_BY_CORPUS_SIZE = {
    928: {"ellis-castro-2026@v1": 914, "ibm-ai-native-systems@v1": 14},
    1072: {"ellis-castro-2026@v1": 914, "ibm-ai-native-systems@v1": 14, "cheng2025barbarians@v1": 144},
}

DBS = golden_dbs()


def _nonzero(d: dict) -> dict:
    return {k: v for k, v in d.items() if v}


def _expected(gc) -> dict | None:
    return EXPECTED_COUNTS_BY_CORPUS_SIZE.get(gc.n_campaigns)


@pytest.mark.parametrize("src", DBS)
def test_backfill_exact_counts(golden_copies, src):
    gc = golden_copies(src)
    gc.require_backfill()
    assert all(isinstance(v, int) for v in gc.counts.values())
    assert gc.counts.get("gupta2026harness@v1", 0) == 0
    by_slug = dict(fetchall(gc.url, "SELECT source_slug, count(*) FROM campaign_sources GROUP BY 1"))
    assert _nonzero(gc.counts) == by_slug
    assert fetchone(gc.url, "SELECT count(*) FROM campaign_sources")[0] == gc.n_campaigns
    assert fetchone(gc.url, "SELECT count(*) FROM campaigns c LEFT JOIN campaign_sources cs "
                            "ON cs.campaign_id=c.id WHERE cs.campaign_id IS NULL")[0] == 0
    assert fetchone(gc.url, "SELECT count(*) FROM campaign_sources WHERE submitted_by IS NOT NULL")[0] == 0
    expected = _expected(gc)
    if expected is None:
        pytest.skip(f"no spec counts for a {gc.n_campaigns}-campaign corpus; generic checks passed")
    assert _nonzero(gc.counts) == expected
    assert by_slug == expected


@pytest.mark.parametrize("src", DBS)
def test_backfill_mapping_rule(golden_copies, src):
    gc = golden_copies(src)
    gc.require_backfill()
    from test_name_leak import local_contributor_prefix

    rows = fetchall(gc.url, "SELECT c.author, c.name, cs.source_slug FROM campaigns c "
                            "JOIN campaign_sources cs ON cs.campaign_id=c.id")
    bad = 0
    for author, name, slug in rows:
        if author == "Roar Authors":
            want = "ibm-ai-native-systems@v1" if local_contributor_prefix(name) is not None else "ellis-castro-2026@v1"
        elif author == "Roar Authors 2":
            want = "cheng2025barbarians@v1"
        else:
            want = None
        bad += slug != want
    assert bad == 0, f"{bad} campaigns mapped against the spec rule"


@pytest.mark.parametrize("src", DBS)
def test_backfill_seeds_sources(golden_copies, src):
    gc = golden_copies(src)
    gc.require_backfill()
    rows = fetchall(gc.url, "SELECT slug, attribution, redistribution, license_spdx, consent_recorded_at "
                            "FROM source_datasets ORDER BY slug")
    assert [r[0] for r in rows] == sorted(SEED_SLUGS)
    for slug, attr, redist, lic, consent in rows:
        assert (attr, redist, lic) == ("named", "full", "CC-BY-4.0"), slug
        assert consent is not None, slug
    assert fetchone(gc.url, "SELECT bibtex IS NULL FROM source_datasets WHERE slug='ibm-ai-native-systems@v1'")[0]
    # Seed consent: permission granted 2026-11-17.
    dates = fetchall(gc.url, "SELECT slug, (consent_recorded_at AT TIME ZONE 'UTC')::date::text, "
                             "consent_recorded_at::date::text FROM source_datasets")
    for slug, utc_date, local_date in dates:
        assert "2026-11-17" in (utc_date, local_date), slug


@pytest.mark.parametrize("src", DBS)
def test_backfill_idempotent(golden_copies, src):
    gc = golden_copies(src)
    gc.require_backfill()
    assert gc.counts_second == gc.counts
    assert gc.prov_hash_second == gc.prov_hash_first


@pytest.mark.parametrize("src", DBS)
def test_v1_rows_unchanged_and_evaluation_id_null(golden_copies, src):
    gc = golden_copies(src)
    gc.require_backfill()
    assert gc.fp_after == gc.fp_before
    assert v1_row_fingerprint(gc.src_url, gc.columns) == gc.fp_after  # equals the source DB
    assert fetchone(gc.url, "SELECT count(*) FROM measurements WHERE evaluation_id IS NOT NULL")[0] == 0
    assert fetchone(gc.url, "SELECT count(*) FROM evaluations")[0] == 0


@pytest.fixture(scope="module")
def originals():
    """Reader outputs and tie sets on the source DBs (read-only), cached."""
    from golden_support import best_embedding_ties, iteration_ties, run_readers
    from conftest import db_url

    cache: dict = {}

    def _get(src):
        if src not in cache:
            u = db_url(src)
            cache[src] = {"out": run_readers(u), "emb": best_embedding_ties(u), "iter": iteration_ties(u)}
        return cache[src]

    return _get


def _reader_names():
    from golden_support import READER_NAMES

    return READER_NAMES


@pytest.mark.parametrize("reader", _reader_names())
@pytest.mark.parametrize("src", DBS)
def test_analytics_reader_identical(golden_copies, originals, src, reader):
    from golden_support import best_embedding_ties, compare_reader, iteration_ties

    gc = golden_copies(src)
    gc.require_backfill()
    orig = originals(src)
    # The tie sets are data, so they must match between original and copy too.
    if reader == "q3.query_campaigns_with_best_embeddings":
        assert best_embedding_ties(gc.url) == orig["emb"]
    if reader == "q3.query_mutation_factors":
        assert iteration_ties(gc.url) == orig["iter"]
    kw = {"emb_ties": orig["emb"], "iter_ties": orig["iter"]}
    compare_reader(reader, orig["out"][reader], gc.post[reader], **kw)  # original vs migrated copy
    compare_reader(reader, gc.pre[reader], gc.post[reader], **kw)  # same copy, before vs after


@pytest.mark.parametrize("src", DBS)
def test_citations_for_whole_corpus(golden_copies, src):
    import psycopg

    gc = golden_copies(src)
    gc.require_backfill()
    import provenance

    with psycopg.connect(gc.url) as conn:
        cs, srcs = provenance.load_provenance(conn)
    ids = [r[0] for r in fetchall(gc.url, "SELECT id FROM campaigns")]
    slugs = [c["slug"] for c in provenance.required_citations(ids, cs, srcs)]
    used = sorted(r[0] for r in fetchall(gc.url, "SELECT DISTINCT source_slug FROM campaign_sources"))
    assert slugs == ["roar"] + used
    expected = _expected(gc)
    if expected is not None:
        assert slugs == ["roar"] + sorted(expected)


def _sha256_manifest() -> dict[str, str]:
    path = os.environ.get("ROAR_PARQUET_SHA256_FILE")
    if not path or not Path(path).is_file():
        pytest.skip("set ROAR_PARQUET_SHA256_FILE (sha256sum-format manifest) to check exported parquet files")
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        out[name.lstrip("*").strip()] = digest.lower()
    assert out, "empty sha256 manifest"
    return out


def test_parquet_files_unchanged():
    """The previously exported v1 parquet files keep their hashes (export_provenance
    must not rewrite them)."""
    d = os.environ.get("ROAR_PARQUET_DIR")
    if not d or not Path(d).is_dir():
        pytest.skip("set ROAR_PARQUET_DIR to the directory of exported v1 parquet files")
    manifest = _sha256_manifest()
    for name, want in manifest.items():
        p = Path(d) / name
        assert p.is_file(), f"missing {name}"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == want, name
