#!/usr/bin/env python3
"""Model-problem heatmap analytics for ADRS campaign data.

Module structure:
- query_*: Database queries
- process_*: Data transformation
- get_*: High-level API
"""

import os
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import dict_row
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances


DATABASE_URL_DEFAULT = "postgresql://postgres:postgres@localhost:5432/adrs"


def _get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DATABASE_URL_DEFAULT)


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def query_campaign_scores(database_url: str | None = None) -> list[dict[str, Any]]:
    """Query campaign scores with embeddings from the database.

    Fetches SkyDiscover numeric metrics.

    Returns:
        List of dicts with campaign_id, models_used, research_question,
        embedding, metric_name, and metric_value.
    """
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH campaign_best_scores AS (
                    SELECT
                        c.id as campaign_id,
                        c.models_used,
                        c.research_question,
                        c.research_question_embedding::text as embedding,
                        m.name as metric_name,
                        MAX(CAST(m.value AS DOUBLE PRECISION)) as metric_value
                    FROM campaigns c
                    JOIN candidates cand ON cand.campaign_id = c.id
                    JOIN measurements m ON m.candidate_id = cand.id
                    WHERE c.research_question IS NOT NULL
                      AND c.models_used IS NOT NULL
                      AND c.research_question_embedding IS NOT NULL
                      AND m.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
                    GROUP BY c.id, c.models_used, c.research_question,
                             c.research_question_embedding, m.name
                )
                SELECT
                    campaign_id,
                    models_used,
                    research_question,
                    embedding,
                    metric_name,
                    metric_value
                FROM campaign_best_scores
                ORDER BY campaign_id, metric_name
            """)
            rows = cur.fetchall()

    return [_parse_score_row(row) for row in rows]


def query_nous_iteration_statuses(database_url: str | None = None) -> list[dict[str, Any]]:
    """Query NOUS campaign iteration statuses for confirmation rate calculation.

    With the new schema, arm statuses are stored as measurements:
    h-main_status, h-control_status, h-ablation_status, h-robustness_status

    Returns:
        List of dicts with campaign_id, models_used, research_question,
        embedding, iteration_index, and status value.
    """
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # Query all arm status measurements for NOUS campaigns
            cur.execute("""
                SELECT
                    c.id as campaign_id,
                    c.models_used,
                    c.research_question,
                    c.research_question_embedding::text as embedding,
                    cand.iteration_index,
                    m.name as measurement_name,
                    m.value as status
                FROM campaigns c
                JOIN systems s ON c.system_id = s.id
                JOIN candidates cand ON cand.campaign_id = c.id
                JOIN measurements m ON m.candidate_id = cand.id
                WHERE s.name = 'nous'
                  AND c.research_question IS NOT NULL
                  AND c.models_used IS NOT NULL
                  AND c.research_question_embedding IS NOT NULL
                  AND m.name LIKE '%_status'
                ORDER BY c.id, cand.iteration_index
            """)
            rows = cur.fetchall()

    results = []
    for row in rows:
        embedding = None
        if row["embedding"]:
            embedding_str = row["embedding"].strip("[]")
            embedding = [float(x) for x in embedding_str.split(",")]
        results.append({
            "campaign_id": str(row["campaign_id"]),
            "models_used": row["models_used"],
            "research_question": row["research_question"],
            "embedding": embedding,
            "iteration_index": row["iteration_index"],
            "status": row["status"],
        })
    return results




def _parse_score_row(row: dict) -> dict[str, Any]:
    """Parse a score row, converting embedding string to list."""
    embedding = None
    if row.get("embedding"):
        embedding_str = row["embedding"].strip("[]")
        embedding = [float(x) for x in embedding_str.split(",")]

    return {
        "campaign_id": str(row["campaign_id"]),
        "models_used": row["models_used"],
        "research_question": row["research_question"],
        "embedding": embedding,
        "metric_name": row["metric_name"],
        "metric_value": row["metric_value"],
    }


# ---------------------------------------------------------------------------
# Process functions
# ---------------------------------------------------------------------------

def process_nous_confirmation_rates(
    nous_statuses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compute confirmation rates for NOUS campaigns.

    For each NOUS campaign, calculates the confirmation rate per iteration
    (CONFIRMED=1.0, PARTIALLY_CONFIRMED=0.5, REFUTED=0) and returns the
    MAX across all iterations.

    Returns:
        List of dicts in the same format as query_campaign_scores.
    """
    if not nous_statuses:
        return []

    # Group by (campaign_id, iteration_index)
    iteration_scores: dict[tuple[str, int], dict] = {}
    for row in nous_statuses:
        key = (row["campaign_id"], row["iteration_index"])
        if key not in iteration_scores:
            iteration_scores[key] = {
                "models_used": row["models_used"],
                "research_question": row["research_question"],
                "embedding": row["embedding"],
                "arms_total": 0,
                "arms_score": 0.0,
            }
        iteration_scores[key]["arms_total"] += 1
        status = row["status"]
        if status == "CONFIRMED":
            iteration_scores[key]["arms_score"] += 1.0
        elif status == "PARTIALLY_CONFIRMED":
            iteration_scores[key]["arms_score"] += 0.5

    # Find best rate per campaign
    best_per_campaign: dict[str, dict] = {}
    for (campaign_id, _), data in iteration_scores.items():
        rate = 100.0 * data["arms_score"] / data["arms_total"]
        if campaign_id not in best_per_campaign or rate > best_per_campaign[campaign_id]["metric_value"]:
            best_per_campaign[campaign_id] = {
                "campaign_id": campaign_id,
                "models_used": data["models_used"],
                "research_question": data["research_question"],
                "embedding": data["embedding"],
                "metric_name": "confirmation_rate",
                "metric_value": rate,
            }

    return list(best_per_campaign.values())


def process_heatmap(
    raw_data: list[dict[str, Any]],
    distance_threshold: float = 0.1,
) -> dict[str, Any]:
    """Process raw campaign data into heatmap format.

    Clusters research questions by embedding similarity, then aggregates
    scores by (generator, cluster) pairs.

    Args:
        raw_data: List of campaign score dicts.
        distance_threshold: Threshold for clustering research questions.

    Returns:
        Dictionary with generators, problems, metrics, and matrix.
    """
    if not raw_data:
        return {"generators": [], "problems": [], "metrics": [], "matrix": {}}

    # Group by campaign
    campaigns_by_id: dict[str, dict] = {}
    for row in raw_data:
        cid = row["campaign_id"]
        if cid not in campaigns_by_id:
            campaigns_by_id[cid] = {
                "models_used": row["models_used"],
                "research_question": row["research_question"],
                "embedding": row["embedding"],
                "metrics": {},
            }
        campaigns_by_id[cid]["metrics"][row["metric_name"]] = row["metric_value"]

    # Add primary_score
    for cid, campaign in campaigns_by_id.items():
        metrics = campaign["metrics"]
        if "combined_score" in metrics:
            metrics["primary_score"] = metrics["combined_score"]
        elif "confirmation_rate" in metrics:
            metrics["primary_score"] = metrics["confirmation_rate"]

    if not campaigns_by_id:
            return {
                "generators": [],
                "problems": [],
                "metrics": [],
                "matrix": {},
                "evaluator_filtered": True,
            }

    # Cluster research questions
    campaigns_list = list(campaigns_by_id.values())
    question_to_cluster, cluster_labels = _cluster_questions(campaigns_list, distance_threshold)

    # Aggregate scores by (generator, cluster, metric)
    aggregated: dict[tuple[str, int, str], list[float]] = {}
    all_metrics: set[str] = set()

    for c in campaigns_list:
        rq = c["research_question"]
        if rq not in question_to_cluster:
            continue
        cluster_id = question_to_cluster[rq]
        generator = ", ".join(sorted(c["models_used"])) if c["models_used"] else "unknown"

        for metric_name, metric_value in c["metrics"].items():
            all_metrics.add(metric_name)
            key = (generator, cluster_id, metric_name)
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(metric_value)

    # Compute means
    means: dict[tuple[str, int, str], float] = {}
    for key, values in aggregated.items():
        means[key] = sum(values) / len(values)

    # Build response
    generators = sorted(set(k[0] for k in means.keys()))
    problems = [
        {"cluster_id": cid, "label": cluster_labels[cid]}
        for cid in sorted(cluster_labels.keys())
    ]
    metrics = sorted(all_metrics)

    matrix: dict[str, dict[int, dict[str, float]]] = {}
    for generator in generators:
        matrix[generator] = {}
        for problem in problems:
            cid = problem["cluster_id"]
            matrix[generator][cid] = {}
            for metric in metrics:
                key = (generator, cid, metric)
                if key in means:
                    matrix[generator][cid][metric] = means[key]

    return {
        "generators": generators,
        "problems": problems,
        "metrics": metrics,
        "matrix": matrix,
    }


def _cluster_questions(
    campaigns: list[dict],
    threshold: float,
) -> tuple[dict[str, int], dict[int, str]]:
    """Cluster research questions by embedding similarity."""
    embeddings_map: dict[str, list[float]] = {}
    for c in campaigns:
        rq = c["research_question"]
        if rq and c.get("embedding") and rq not in embeddings_map:
            embeddings_map[rq] = c["embedding"]

    unique_questions = list(embeddings_map.keys())
    question_to_cluster: dict[str, int] = {}
    cluster_labels: dict[int, str] = {}

    if len(unique_questions) < 2:
        for i, q in enumerate(unique_questions):
            question_to_cluster[q] = i
            cluster_labels[i] = q
        return question_to_cluster, cluster_labels

    embeddings = np.array([embeddings_map[q] for q in unique_questions])
    distance_matrix = cosine_distances(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance_matrix)

    for idx, label in enumerate(labels):
        label_int = int(label)
        question_to_cluster[unique_questions[idx]] = label_int
        if label_int not in cluster_labels:
            cluster_labels[label_int] = unique_questions[idx]

    return question_to_cluster, cluster_labels


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def get_model_problem_heatmap(
    database_url: str | None = None,
    distance_threshold: float = 0.1,
) -> dict[str, Any]:
    """Get heatmap data for model vs problem performance.

    Queries database and processes into heatmap format.

    Args:
        database_url: Database URL (uses env var if not provided).
        distance_threshold: Threshold for clustering research questions.
    """
    # Query all scores
    scores = query_campaign_scores(database_url)

    # Add NOUS confirmation rates
    nous_statuses = query_nous_iteration_statuses(database_url)
    nous_rates = process_nous_confirmation_rates(nous_statuses)
    scores.extend(nous_rates)

    return process_heatmap(scores, distance_threshold)


# ---------------------------------------------------------------------------
# Backwards compatibility (deprecated)
# ---------------------------------------------------------------------------

def query_campaign_scores_with_embeddings(database_url: str | None = None) -> list[dict[str, Any]]:
    """Deprecated: Use query_campaign_scores() + process_nous_confirmation_rates()."""
    scores = query_campaign_scores(database_url)
    nous_statuses = query_nous_iteration_statuses(database_url)
    nous_rates = process_nous_confirmation_rates(nous_statuses)
    scores.extend(nous_rates)
    return scores




def get_variance_heatmap(
    raw_data: list[dict[str, Any]],
    distance_threshold: float = 0.1,
    min_runs: int = 2,
) -> dict[str, Any]:
    """Deprecated: Use analytics.q1_variance.get_variance_analysis()."""
    from analytics.q1_variance import get_variance_analysis
    return get_variance_analysis(None, distance_threshold, min_runs)
