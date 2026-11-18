"""Pandas mirrors of ROAR's database query functions, over the Hugging Face parquet tables.

`install()` replaces, in every module that bound them, the four ROAR query functions the
paper's analyses call, so ROAR's own process_*/fit/figure code then runs unchanged with
no Postgres:

  analytics.q2_calibration.query_iteration_scores            (RQ2, RQ4 load_trajectories)
  analytics.q1_variance.query_skydiscover_scores             (RQ1)
  analytics.q3_diversity.factor_importance.query_mutation_factors   (RQ3 context factors)
  analytics.q3_diversity.early_diversity._compute_early_diversity_campaign_data (RQ3 early)

Each mirror reproduces the SQL's joins (including the multiplicity of duplicated
measurement rows), filters, the numeric regex on measurement values, and its ORDER BY.
Where the SQL leaves an order unspecified, the mirror fixes one and says so:
  * q1 has no ORDER BY: campaigns are returned sorted by id.
  * The running best in query_mutation_factors orders tied iteration_index values
    arbitrarily in Postgres; the mirror keeps the parquet row order (stable sort).
    This can flip the improvement flag of a handful of rows (2 of 45,947 on the
    submitted corpus), so RQ3 GLMM coefficients can differ from a Postgres run in the
    third decimal. Tests compare at that tolerance.

Scope: by default only the submitted corpus (author "Roar Authors", 928 campaigns), which
is what the paper reports. `set_corpus("full")` uses all 1,072 mirror campaigns.
"""
import functools
import os

import numpy as np
import pandas as pd

from . import paths

NUM_RE = r"^-?[0-9]+(\.[0-9]+)?$"
_CORPUS = {"author": paths.SUBMITTED_AUTHOR}


def set_corpus(which):
    _CORPUS["author"] = {"submitted": paths.SUBMITTED_AUTHOR, "full": None}[which]
    tables.cache_clear()


def _parse_array(s):
    """Postgres TEXT[] literal as exported to parquet ('{a,b}') -> list, NULL -> None."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = s.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return [s]
    body = s[1:-1]
    if not body:
        return []
    return [x.strip().strip('"') for x in body.split(",")]


@functools.lru_cache(maxsize=1)
def tables():
    """Load the mirror tables restricted to the selected corpus."""
    paths.verify_parquet()
    d = paths.ROAR_DATA
    camp = pd.read_parquet(os.path.join(d, "campaigns.parquet"),
                           columns=["id", "author", "name", "research_question", "algorithm_used",
                                    "models_used", "system_id"])
    if _CORPUS["author"] is not None:
        camp = camp[camp.author == _CORPUS["author"]]
    camp = camp.reset_index(drop=True)
    camp["models_list"] = camp.models_used.map(_parse_array)
    cand = pd.read_parquet(os.path.join(d, "candidates.parquet"),
                           columns=["id", "campaign_id", "iteration_index", "context_code_diversity",
                                    "direct_code_embedding"])
    cand = cand[cand.campaign_id.isin(set(camp.id))].reset_index(drop=True)
    cand["has_emb"] = cand.direct_code_embedding.notna()
    ids = set(cand.id)
    meas = pd.read_parquet(os.path.join(d, "measurements.parquet"))
    meas = meas[meas.candidate_id.isin(ids)].reset_index(drop=True)
    edges = pd.read_parquet(os.path.join(d, "candidate_edges.parquet"))
    edges = edges[edges.target_candidate_id.isin(ids) | edges.source_candidate_id.isin(ids)].reset_index(drop=True)
    return {"campaigns": camp, "candidates": cand, "measurements": meas, "edges": edges}


def numeric_scores():
    """combined_score rows whose text matches ROAR's numeric regex (duplicates kept)."""
    m = tables()["measurements"]
    m = m[(m.name == "combined_score") & m.value.str.match(NUM_RE, na=False)][["candidate_id", "value"]].copy()
    m["score"] = m.value.astype(float)
    return m[["candidate_id", "score"]]


# --------------------------------------------------------------------------- q2
def query_iteration_scores(database_url=None):
    t = tables()
    camp = t["campaigns"]
    camp = camp[camp.algorithm_used.notna()]
    cand = t["candidates"][["id", "campaign_id", "iteration_index"]]
    x = cand.merge(numeric_scores(), left_on="id", right_on="candidate_id")
    x = x[x.campaign_id.isin(set(camp.id))]
    per = x.groupby(["campaign_id", "iteration_index"], sort=True)["score"].max()
    meta = camp.set_index("id")
    out = []
    for cid, s in per.groupby(level=0, sort=True):
        r = meta.loc[cid]
        out.append({
            "campaign_id": str(cid),
            "models_used": r.models_list,
            "algorithm_used": r.algorithm_used,
            "research_question": r.research_question,
            "iterations": [{"index": int(i), "score": float(v)} for (_, i), v in s.items()],
        })
    return out


# --------------------------------------------------------------------------- q1
def query_skydiscover_scores(database_url=None):
    t = tables()
    camp = t["campaigns"]
    camp = camp[camp.research_question.notna()]
    x = t["candidates"][["id", "campaign_id"]].merge(numeric_scores(), left_on="id", right_on="candidate_id")
    best = x.groupby("campaign_id")["score"].max()
    camp = camp[camp.id.isin(best.index)].sort_values("id")
    return [{"campaign_id": str(r.id), "models_used": r.models_list, "algorithm_used": r.algorithm_used,
             "research_question": r.research_question, "primary_score": float(best[r.id])}
            for r in camp.itertuples()]


# --------------------------------------------------------------------------- q3 factors
def mutation_factor_frame():
    """The rows of query_mutation_factors' SQL, before ROAR's Python post-processing."""
    t = tables()
    cand = t["candidates"]
    e = t["edges"]
    sc = numeric_scores()
    pe = e[e.edge_type == "parent"][["target_candidate_id", "source_candidate_id"]]
    pe.columns = ["child_id", "parent_id"]
    ce = e[e.edge_type == "context"][["target_candidate_id", "source_candidate_id"]]
    ce.columns = ["child_id", "context_id"]

    # context_stats: ce JOIN pe JOIN ctx_m JOIN parent_m, grouped by child
    x = ce.merge(pe, on="child_id")
    x = x.merge(sc.rename(columns={"candidate_id": "context_id", "score": "cv"}), on="context_id")
    x = x.merge(sc.rename(columns={"candidate_id": "parent_id", "score": "pv"}), on="parent_id")
    x["better"] = (x.cv > x.pv).astype(int)
    x["worse"] = (x.cv < x.pv).astype(int)
    g = x.groupby("child_id")
    cs = pd.DataFrame({
        "context_size": g.cv.size(),
        "max_context_score": g.cv.max(),
        "context_score_variance": g.cv.agg(lambda s: float(np.var(s.to_numpy()))),   # VAR_POP
        "better_count": g.better.sum(),
        "worse_count": g.worse.sum(),
    }).reset_index()

    # running_best: max over strictly preceding rows of the campaign, ordered by iteration_index
    rb = cand[cand.iteration_index.notna()][["id", "campaign_id", "iteration_index"]].merge(
        sc, left_on="id", right_on="candidate_id")
    rb = rb.sort_values(["campaign_id", "iteration_index"], kind="mergesort")
    rb["best_before"] = rb.groupby("campaign_id")["score"].transform(lambda s: s.cummax().shift(1))
    rb = rb[["id", "best_before"]]

    camp = t["campaigns"][["id", "research_question", "models_list", "algorithm_used"]].rename(
        columns={"id": "campaign_id"})
    c = cand[["id", "campaign_id", "iteration_index", "context_code_diversity", "has_emb"]]
    q = c.merge(camp, on="campaign_id").merge(pe, left_on="id", right_on="child_id")
    q = q.merge(sc.rename(columns={"candidate_id": "id", "score": "child_score"}), on="id")
    q = q.merge(sc.rename(columns={"candidate_id": "parent_id", "score": "parent_score"}), on="parent_id")
    q = q.merge(cs, left_on="id", right_on="child_id", how="left", suffixes=("", "_cs"))
    q = q.merge(rb, on="id", how="left")
    q = q[q.has_emb & q.research_question.notna() & (q.context_size > 0)]
    q = q.sort_values(["campaign_id", "iteration_index"], kind="mergesort").reset_index(drop=True)
    q["context_diversity"] = q.context_code_diversity.fillna(0.0)
    return q


def query_mutation_factors(database_url=None):
    from analytics.q3_diversity._common import _truncate_problem
    from analytics.utils import normalize_model_name
    results = []
    for r in mutation_factor_frame().itertuples(index=False):
        problem = _truncate_problem(r.research_question)
        if problem is None:
            continue
        child_score = float(r.child_score)
        parent_score = float(r.parent_score)
        max_context_score = float(r.max_context_score)
        context_size = int(r.context_size)
        best_before = None if pd.isna(r.best_before) else float(r.best_before)
        baseline = max(parent_score, max_context_score)
        if best_before is not None:
            score_delta_global = child_score - best_before
        else:
            score_delta_global = child_score if child_score > 0 else 0.0
        better_ratio = r.better_count / context_size if context_size > 0 else 0
        worse_ratio = r.worse_count / context_size if context_size > 0 else 0
        results.append({
            "candidate_id": str(r.id),
            "campaign_id": str(r.campaign_id),
            "problem": problem,
            "model": normalize_model_name(r.models_list[0]) if r.models_list else "unknown",
            "algorithm": r.algorithm_used or "unknown",
            "iteration_index": int(r.iteration_index),
            "parent_score": round(parent_score, 4),
            "child_score": round(child_score, 4),
            "max_context_score": round(max_context_score, 4),
            "best_score_so_far": round(best_before, 4) if best_before is not None else None,
            "score_delta": round(child_score - baseline, 4),
            "score_delta_parent_only": round(child_score - parent_score, 4),
            "score_delta_global": round(score_delta_global, 4),
            "context_size": context_size,
            "context_diversity": round(float(r.context_diversity), 4),
            "score_diversity": round(float(r.context_score_variance), 4),
            "better_count": int(r.better_count),
            "worse_count": int(r.worse_count),
            "better_ratio": round(better_ratio, 4),
            "worse_ratio": round(worse_ratio, 4),
        })
    return results


# --------------------------------------------------------------------------- q3 early diversity
def early_candidates(early_fraction=0.25):
    """The early_candidates CTE: embedded candidates with iteration_index <=
    round(max_iter * early_fraction) in campaigns with >= 10 embedded candidates.
    Postgres casts double -> int with rint (round half to even); np.rint matches."""
    cand = tables()["candidates"]
    e = cand[cand.has_emb & cand.iteration_index.notna()]
    st = e.groupby("campaign_id").agg(max_iter=("iteration_index", "max"), n_candidates=("id", "size"))
    st = st[st.n_candidates >= 10]
    e = e.merge(st, left_on="campaign_id", right_index=True)
    lim = np.rint(e.max_iter.to_numpy(dtype=float) * early_fraction)
    return e[e.iteration_index.to_numpy() <= lim], st


def _mean_pairwise_cosine_distance(vectors):
    m = np.asarray(np.stack(vectors), dtype=np.float64)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    s = m @ m.T
    iu = np.triu_indices(len(m), k=1)
    return float(np.mean(1.0 - s[iu]))


def _compute_early_diversity_campaign_data(candidates=None, early_fraction=0.25, database_url=None):
    from analytics.q3_diversity._common import _truncate_problem
    from analytics.utils import normalize_model_name
    t = tables()
    early, st = early_candidates(early_fraction)
    camp = t["campaigns"]
    x = t["candidates"][["id", "campaign_id"]].merge(numeric_scores(), left_on="id", right_on="candidate_id")
    best = x.groupby("campaign_id")["score"].max()
    out = []
    groups = {cid: g for cid, g in early.groupby("campaign_id")}
    for r in camp[camp.research_question.notna()].sort_values("id").itertuples():
        g = groups.get(r.id)
        if g is None or len(g) < 2 or r.id not in best.index or r.id not in st.index:
            continue
        problem = _truncate_problem(r.research_question)
        if problem is None:
            continue
        out.append({
            "campaign_id": str(r.id),
            "problem": problem,
            "model": normalize_model_name(r.models_list[0]) if r.models_list else "unknown",
            "algorithm": r.algorithm_used or "unknown",
            "early_diversity": round(_mean_pairwise_cosine_distance(list(g.direct_code_embedding)), 4),
            "best_score": round(float(best[r.id]), 4),
            "n_early_candidates": int(len(g)),
            "n_total_candidates": int(st.loc[r.id, "n_candidates"]),
        })
    return out


# --------------------------------------------------------------------------- install
def install(corpus="submitted"):
    """Route ROAR's query functions to the parquet mirrors, unless ROAR_BACKEND=postgres."""
    if os.environ.get("ROAR_BACKEND", "parquet") == "postgres":
        return "postgres"
    set_corpus(corpus)
    import analytics.q1_variance as q1
    import analytics.q2_calibration as q2
    import analytics.q3_diversity.early_diversity as ed
    import analytics.q3_diversity.factor_importance as fi
    import analytics.q5.recommend as rec
    q2.query_iteration_scores = query_iteration_scores
    rec.query_iteration_scores = query_iteration_scores
    q1.query_skydiscover_scores = query_skydiscover_scores
    fi.query_mutation_factors = query_mutation_factors
    ed._compute_early_diversity_campaign_data = _compute_early_diversity_campaign_data
    return "parquet"
