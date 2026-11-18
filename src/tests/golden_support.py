"""Support code for the golden tests: copying source DBs (ROAR_GOLDEN_DBS), running
the unchanged analytics readers, and comparing their outputs.

Source DBs are only *read* here (by pg_dump and by the analytics readers). Copies are
named ``<src>_provcopy_<rand>``.
"""

from __future__ import annotations

import hashlib
import json
import math
import shlex
import subprocess
import uuid
from decimal import Decimal
from typing import Any, Callable

import psycopg
from psycopg import sql

from conftest import (
    ADMIN_URL,
    DOCKER_CONTAINER,
    PG_USER,
    db_url,
    drop_database,
    guard_not_frozen,
    v1_columns,
    v1_row_fingerprint,
)

FLOAT_TOL = 1e-9  # absolute floor and relative tolerance (PG aggregation order: a few ulp)


# ---------------------------------------------------------------------------
# Copying
# ---------------------------------------------------------------------------

def database_exists(dbname: str) -> bool:
    with psycopg.connect(ADMIN_URL) as conn:
        return conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,)).fetchone() is not None


def copy_database(src: str) -> tuple[str, str]:
    """Copy a source DB with ``pg_dump | psql`` inside ROAR_TEST_PG_CONTAINER.

    Never ``CREATE DATABASE ... TEMPLATE``: that requires (and briefly enforces) zero
    other connections to a shared source. ``src`` is only read. Returns (copy_name, method).
    """
    name = f"{src}_provcopy_{uuid.uuid4().hex[:8]}"
    guard_not_frozen(name)
    assert DOCKER_CONTAINER, "ROAR_TEST_PG_CONTAINER must be set"
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    cmd = (
        f"docker exec {shlex.quote(DOCKER_CONTAINER)} pg_dump -U {shlex.quote(PG_USER)} {shlex.quote(src)} | "
        f"docker exec -i {shlex.quote(DOCKER_CONTAINER)} psql -q -v ON_ERROR_STOP=1 -U {shlex.quote(PG_USER)} "
        f"{shlex.quote(name)}"
    )
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        drop_database(name)
        raise RuntimeError(f"pg_dump copy of {src} failed (exit {p.returncode})")
    method = "pg_dump"
    # Fresh planner statistics (a restored DB has none; the analytics queries need them).
    with psycopg.connect(db_url(name), autocommit=True) as conn:
        conn.execute("ANALYZE")
    return name, method


# ---------------------------------------------------------------------------
# Readers (unchanged analytics code; imported lazily)
# ---------------------------------------------------------------------------

def readers() -> dict[str, Callable[[str], Any]]:
    import analytics.heatmap as hm
    import analytics.q1_variance as q1
    import analytics.q2_calibration as q2
    import analytics.q3_diversity._common as cm
    import analytics.q3_diversity.early_diversity as ed
    import analytics.q3_diversity.factor_importance as fi

    return {
        "q1.query_skydiscover_scores": lambda u: q1.query_skydiscover_scores(u),
        "q1.get_variance_analysis": lambda u: q1.get_variance_analysis(u, min_runs=10, group_by="model_algorithm"),
        "q2.query_iteration_scores": lambda u: q2.query_iteration_scores(u),
        "q3.query_campaign_diversity_and_scores": lambda u: cm.query_campaign_diversity_and_scores(u),
        "q3.query_campaigns_with_best_embeddings": lambda u: cm.query_campaigns_with_best_embeddings(u),
        "q3.early_diversity_campaign_data": lambda u: ed._compute_early_diversity_campaign_data(database_url=u),
        "q3.query_mutation_factors": lambda u: fi.query_mutation_factors(u),
        "heatmap.query_campaign_scores": lambda u: hm.query_campaign_scores(u),
        "heatmap.query_nous_iteration_statuses": lambda u: hm.query_nous_iteration_statuses(u),
        "heatmap.query_campaign_scores_with_embeddings": lambda u: hm.query_campaign_scores_with_embeddings(u),
    }


READER_NAMES = list(
    [
        "q1.query_skydiscover_scores",
        "q1.get_variance_analysis",
        "q2.query_iteration_scores",
        "q2.compute_stagnation_observations",
        "q2.fit_escape_trend_model",
        "q3.query_campaign_diversity_and_scores",
        "q3.query_campaigns_with_best_embeddings",
        "q3.early_diversity_campaign_data",
        "q3.query_mutation_factors",
        "heatmap.query_campaign_scores",
        "heatmap.query_nous_iteration_statuses",
        "heatmap.query_campaign_scores_with_embeddings",
    ]
)


def run_readers(url: str) -> dict[str, Any]:
    import analytics.q2_calibration as q2

    out = {name: fn(url) for name, fn in readers().items()}
    out["q2.compute_stagnation_observations"] = q2.compute_stagnation_observations(out["q2.query_iteration_scores"])
    out["q2.fit_escape_trend_model"] = q2.fit_escape_trend_model(out["q2.compute_stagnation_observations"])
    return out


# ---------------------------------------------------------------------------
# Tie information (readers whose SQL breaks ties arbitrarily)
# ---------------------------------------------------------------------------

def best_embedding_ties(url: str) -> dict[str, set[str]]:
    """campaign_id -> md5s of direct_code_embedding::text of every max-score candidate
    (the set query_campaigns_with_best_embeddings' DISTINCT ON may pick from)."""
    ties: dict[str, set[str]] = {}
    with psycopg.connect(url) as conn:
        for cid, h in conn.execute(
            r"""
            WITH s AS (
                SELECT c.campaign_id, c.direct_code_embedding AS e, CAST(m.value AS DOUBLE PRECISION) AS v
                FROM candidates c JOIN measurements m ON m.candidate_id = c.id AND m.name = 'combined_score'
                WHERE c.direct_code_embedding IS NOT NULL AND m.value ~ '^-?[0-9]+(\.[0-9]+)?$'
            ), mx AS (SELECT campaign_id, max(v) AS v FROM s GROUP BY campaign_id)
            SELECT s.campaign_id::text, md5(s.e::text) FROM s JOIN mx USING (campaign_id) WHERE s.v = mx.v
            """
        ):
            ties.setdefault(cid, set()).add(h)
    return ties


def iteration_ties(url: str) -> set[tuple[str, int]]:
    """(campaign_id, iteration_index) with >1 scored candidate: the running-best window in
    query_mutation_factors orders by iteration_index only, so ties are arbitrary there."""
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            r"""
            SELECT c.campaign_id::text, c.iteration_index FROM candidates c
            JOIN measurements m ON m.candidate_id = c.id AND m.name = 'combined_score'
            WHERE c.iteration_index IS NOT NULL AND m.value ~ '^-?[0-9]+(\.[0-9]+)?$'
            GROUP BY 1, 2 HAVING count(*) > 1
            """
        ).fetchall()
    return {(r[0], r[1]) for r in rows}


def embedding_md5(emb: list[float]) -> str:
    return hashlib.md5(("[" + ",".join(repr(float(x)) for x in emb) + "]").encode()).hexdigest()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def canon(x: Any) -> Any:
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return [canon(v) for v in x.tolist()]
        if isinstance(x, np.generic):
            return canon(x.item())
    except ImportError:  # pragma: no cover
        pass
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, uuid.UUID):
        return str(x)
    if isinstance(x, dict):
        return {str(k): canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    return x


class Mismatch(AssertionError):
    pass


def _fail(path: str, what: str) -> None:
    # Values are deliberately not echoed: reader rows contain campaign names.
    raise Mismatch(f"{path}: {what}")


def assert_close(a: Any, b: Any, path: str = "$") -> None:
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            _fail(path, "bool differs")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) or isinstance(b, float):
            fa, fb = float(a), float(b)
            if math.isnan(fa) or math.isnan(fb):
                if not (math.isnan(fa) and math.isnan(fb)):
                    _fail(path, "NaN vs number")
            else:
                tol = max(FLOAT_TOL, FLOAT_TOL * max(abs(fa), abs(fb)))
                if abs(fa - fb) > tol:
                    _fail(path, f"numbers differ by {abs(fa - fb):.3g} (tol {tol:.3g})")
        elif a != b:
            _fail(path, "integers differ")
        return
    if isinstance(a, dict):
        if not isinstance(b, dict):
            _fail(path, f"type {type(b).__name__} != dict")
        if set(a) != set(b):
            _fail(path, f"keys differ ({len(set(a) ^ set(b))} keys)")
        for k in a:
            assert_close(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, list):
        if not isinstance(b, list):
            _fail(path, f"type {type(b).__name__} != list")
        if len(a) != len(b):
            _fail(path, f"len {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, f"{path}[{i}]")
        return
    if type(a) is not type(b):
        _fail(path, f"type {type(a).__name__} != {type(b).__name__}")
    if a != b:
        _fail(path, "values differ")


def _row_key(r: Any) -> str:
    return json.dumps(r, sort_keys=True, default=str)


def sort_rows(rows: list) -> list:
    return sorted(rows, key=_row_key)


def compare_reader(name: str, a: Any, b: Any, *, emb_ties=None, iter_ties=None) -> None:
    """Assert reader outputs a (reference) and b (candidate) are identical.

    Row-list outputs are sorted canonically (several readers have no ORDER BY, or ORDER
    BY a non-unique key). Two readers break SQL ties arbitrarily even on the *same* DB;
    for them, the tie-dependent fields are checked against the set of valid tie values:
      * query_campaigns_with_best_embeddings.best_embedding must be the embedding of a
        max-score candidate of that campaign;
      * query_mutation_factors.{best_score_so_far, score_delta_global} are compared only
        for candidates whose (campaign, iteration_index) is not tied.
    """
    a, b = canon(a), canon(b)
    if name == "q3.query_campaigns_with_best_embeddings":
        assert emb_ties is not None
        for rows in (a, b):
            for r in rows:
                if embedding_md5(r["best_embedding"]) not in emb_ties.get(r["campaign_id"], set()):
                    _fail(name, "best_embedding is not a max-score candidate's")
        strip = lambda rows: sorted(({k: v for k, v in r.items() if k != "best_embedding"} for r in rows),
                                    key=lambda r: r["campaign_id"])
        assert_close(strip(a), strip(b), name)
        return
    if name == "q3.query_mutation_factors":
        assert iter_ties is not None
        tie_fields = ("best_score_so_far", "score_delta_global")

        def strip(rows):
            out = []
            for r in rows:
                if (r["campaign_id"], r["iteration_index"]) in iter_ties:
                    r = {k: v for k, v in r.items() if k not in tie_fields}
                out.append(r)
            return sorted(out, key=lambda r: r["candidate_id"])

        assert_close(strip(a), strip(b), name)
        return
    if isinstance(a, list) and isinstance(b, list):
        assert_close(sort_rows(a), sort_rows(b), name)
        return
    assert_close(a, b, name)


# ---------------------------------------------------------------------------
# Golden context: copy -> fingerprint -> readers (pre) -> backfill -> readers (post)
# ---------------------------------------------------------------------------

class GoldenCopy:
    def __init__(self, src: str):
        self.src = src
        self.src_url = db_url(src)
        self.name, self.method = copy_database(src)
        self.url = db_url(self.name)
        try:
            self._build()
        except BaseException:
            self.drop()
            raise

    def _build(self):
        with psycopg.connect(self.url) as conn:
            self.n_campaigns = conn.execute("SELECT count(*) FROM campaigns").fetchone()[0]
            self.columns = v1_columns(conn)
        self.fp_before = v1_row_fingerprint(self.url, self.columns)
        self.backfill_error: BaseException | None = None
        self.counts: dict | None = None
        self.counts_second: dict | None = None
        self.prov_hash_first = self.prov_hash_second = None
        self.fp_after = None
        self.pre = self.post = None
        try:
            import backfill_provenance
        except BaseException as e:  # red phase: don't spend time running readers
            self.backfill_error = e
            self.fp_after = v1_row_fingerprint(self.url, self.columns)
            return
        self.pre = run_readers(self.url)
        try:
            self.counts = backfill_provenance.backfill(self.url)
            self.prov_hash_first = provenance_tables_hash(self.url)
            self.counts_second = backfill_provenance.backfill(self.url)
            self.prov_hash_second = provenance_tables_hash(self.url)
        except BaseException as e:  # recorded; each test reports it
            self.backfill_error = e
        self.fp_after = v1_row_fingerprint(self.url, self.columns)
        if self.backfill_error is None:
            self.post = run_readers(self.url)

    def require_backfill(self):
        import pytest

        if self.backfill_error is not None:
            # Type only: a message could quote campaign names (contributor prefixes).
            pytest.fail(f"backfill on {self.name} failed with {type(self.backfill_error).__name__}")

    def drop(self):
        drop_database(self.name)


def provenance_tables_hash(url: str) -> dict[str, tuple[int, str]]:
    out = {}
    with psycopg.connect(url) as conn:
        for t, key in (("source_datasets", "slug"), ("campaign_sources", "campaign_id"), ("evaluations", "id")):
            out[t] = conn.execute(
                sql.SQL("SELECT count(*), coalesce(md5(string_agg(md5(t::text), '' ORDER BY {k})), '') FROM {t} t").format(
                    k=sql.Identifier(key), t=sql.Identifier(t)
                )
            ).fetchone()
    return out
