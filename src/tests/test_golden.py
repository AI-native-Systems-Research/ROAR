"""Golden tests (marker ``golden``; slow). Spec "Hard constraints" (v1 immutable;
analytics return identical results after migration + backfill) and "Backfill"
(exact counts on copies of roar_v1 / roar_full; idempotent; evaluation_id stays NULL).

Copies are ``<src>_provcopy_<rand>`` (TEMPLATE when the source has no other
connections, else pg_dump | psql) and are dropped at session end."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from conftest import SEED_SLUGS, fetchall, fetchone, v1_row_fingerprint

pytestmark = pytest.mark.golden

EXPECTED_COUNTS = {
    "roar_v1": {"ellis-castro-2026@v1": 914, "ibm-ai-native-systems@v1": 14},
    "roar_full": {"ellis-castro-2026@v1": 914, "ibm-ai-native-systems@v1": 14, "cheng2025barbarians@v1": 144},
}
TOTAL_CAMPAIGNS = {"roar_v1": 928, "roar_full": 1072}

HF_DATA_DIR = Path("/Users/mmibm/Documents/GitHub/claudes/gamble-workspace/data/roar_hf/data")
HF_SHA256 = {  # from rebuttal_compute/CONTRACT.md
    "artifacts.parquet": "206889ad70ee56d16d0aa1e4ccecd52fed8b3d7803e729636ef4ec979f3dff57",
    "campaigns.parquet": "6dca8718d68c14306c5e20ec418778bd6f14595f9f9205a736baa4e742638031",
    "candidate_edges.parquet": "741144eeb126cf780149f2cec596cd22e47534142cf3c45d27066f618a8e890e",
    "candidates.parquet": "1f2a0cd56f5b763582cfc4a07efd9b681fd827634cc971e4c0f7ee5e93370ae0",
    "measurements.parquet": "ca0c5d810190a8c475ba1ffcc5b44663b50e6298df5289e7fed581a9503d11d6",
    "systems.parquet": "4df3006c02899aab2525bb78939f9a16e0a7ee7f2276580eb149ab5d62d5fe0e",
}

DBS = ["roar_v1", "roar_full"]


def _nonzero(d: dict) -> dict:
    return {k: v for k, v in d.items() if v}


@pytest.mark.parametrize("src", DBS)
def test_backfill_exact_counts(golden_copies, src):
    gc = golden_copies(src)
    gc.require_backfill()
    assert all(isinstance(v, int) for v in gc.counts.values())
    assert _nonzero(gc.counts) == EXPECTED_COUNTS[src]
    assert gc.counts.get("gupta2026harness@v1", 0) == 0
    if src == "roar_v1":
        assert gc.counts.get("cheng2025barbarians@v1", 0) == 0
    by_slug = dict(fetchall(gc.url, "SELECT source_slug, count(*) FROM campaign_sources GROUP BY 1"))
    assert by_slug == EXPECTED_COUNTS[src]
    assert fetchone(gc.url, "SELECT count(*) FROM campaign_sources")[0] == TOTAL_CAMPAIGNS[src]
    assert fetchone(gc.url, "SELECT count(*) FROM campaigns c LEFT JOIN campaign_sources cs "
                            "ON cs.campaign_id=c.id WHERE cs.campaign_id IS NULL")[0] == 0
    assert fetchone(gc.url, "SELECT count(*) FROM campaign_sources WHERE submitted_by IS NOT NULL")[0] == 0


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
    assert v1_row_fingerprint(gc.src_url, gc.columns) == gc.fp_after  # equals the frozen original
    assert fetchone(gc.url, "SELECT count(*) FROM measurements WHERE evaluation_id IS NOT NULL")[0] == 0
    assert fetchone(gc.url, "SELECT count(*) FROM evaluations")[0] == 0


@pytest.fixture(scope="module")
def originals():
    """Reader outputs and tie sets on the frozen originals (read-only), cached."""
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


def test_rq2_published_reproduction_on_v1_copy(golden_copies):
    gc = golden_copies("roar_v1")
    gc.require_backfill()
    m = gc.post["q2.fit_escape_trend_model"]
    assert m["odds_ratio_per_10k"] == pytest.approx(0.896, abs=5e-4)
    assert m["or_ci_lower"] == pytest.approx(0.849, abs=5e-4)
    assert m["or_ci_upper"] == pytest.approx(0.946, abs=5e-4)
    assert m["n_observations"] == 6606
    assert m["n_runs"] == 836


def test_v1_copy_citations_are_roar_ellis_ibm(golden_copies):
    import psycopg

    gc = golden_copies("roar_v1")
    gc.require_backfill()
    import provenance

    with psycopg.connect(gc.url) as conn:
        cs, srcs = provenance.load_provenance(conn)
    ids = [r[0] for r in fetchall(gc.url, "SELECT id FROM campaigns")]
    cites = provenance.required_citations(ids, cs, srcs)
    assert [c["slug"] for c in cites] == ["roar", "ellis-castro-2026@v1", "ibm-ai-native-systems@v1"]


def test_hf_parquet_files_unchanged():
    if not HF_DATA_DIR.is_dir():
        pytest.skip(f"{HF_DATA_DIR} absent")
    missing = [n for n in HF_SHA256 if not (HF_DATA_DIR / n).exists()]
    if len(missing) == len(HF_SHA256):
        pytest.skip("no HF parquet files present")
    assert not missing, f"missing parquet files: {missing}"
    for n, want in HF_SHA256.items():
        h = hashlib.sha256((HF_DATA_DIR / n).read_bytes()).hexdigest()
        assert h == want, n
