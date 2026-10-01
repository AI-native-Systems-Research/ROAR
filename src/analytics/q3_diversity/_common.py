"""Shared infrastructure for Q3 diversity analytics."""

import io
import os
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psycopg
from psycopg.rows import dict_row
from sklearn.metrics.pairwise import cosine_similarity

from analytics.utils import abbreviate_problem, display_mechanism, normalize_model_name

DATABASE_URL_DEFAULT = "postgresql://postgres:postgres@localhost:5432/adrs"

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'high': '#22c55e',
    'low': '#ef4444',
}

PALETTE = [
    '#648FFF',
    '#785EF0',
    '#DC267F',
    '#FE6100',
    '#FFB000',
    '#000000',
    '#808080',
]

COLOR_PALETTE = [
    '#e41a1c',
    '#377eb8',
    '#4daf4a',
    '#984ea3',
    '#ff7f00',
    '#a65628',
    '#f781bf',
    '#17becf',
    '#bcbd22',
    '#1f1f1f',
]

MARKER_PALETTE = ['o', 's', '^', 'D', 'v', 'P', 'X']

LINESTYLE_PALETTE = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

GROUP_BYS = ["algorithm", "model", "model_algorithm"]


def _get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DATABASE_URL_DEFAULT)


def get_problems(database_url: str | None = None) -> list[str]:
    """Get list of unique problems that have diversity data."""
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT camp.research_question
                FROM campaigns camp
                JOIN candidates c ON c.campaign_id = camp.id
                WHERE c.direct_code_embedding IS NOT NULL
                  AND camp.research_question IS NOT NULL
                ORDER BY camp.research_question
            """)
            rows = cur.fetchall()

    problems = [_truncate_problem(r["research_question"]) for r in rows]
    return [p for p in problems if p]


def _truncate_problem(rq: str | None) -> str | None:
    """Truncate research question to a reasonable display length."""
    if not rq:
        return None
    max_len = 50
    truncated = rq[:max_len] + "..." if len(rq) > max_len else rq
    return truncated.replace(",", ";")


def _parse_embedding(embedding_str: str | None) -> list[float] | None:
    """Parse embedding from PostgreSQL vector string format."""
    if not embedding_str:
        return None
    cleaned = embedding_str.strip("[]")
    return [float(x) for x in cleaned.split(",")]


def _generate_empty_figure(message: str) -> bytes:
    """Generate placeholder figure for empty data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _compute_cross_diversity(embeddings: list[list[float]]) -> float:
    """Compute pairwise diversity among a set of embeddings."""
    if len(embeddings) < 2:
        return 0.0

    emb_array = np.array(embeddings)
    sim_matrix = cosine_similarity(emb_array)
    np.fill_diagonal(sim_matrix, 0)
    n = len(embeddings)
    avg_similarity = sim_matrix.sum() / (n * (n - 1))
    return 1 - avg_similarity


def query_campaign_diversity_and_scores(database_url: str | None = None) -> list[dict[str, Any]]:
    """Query campaign-level diversity scores and best scores using direct code embeddings."""
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH campaign_scores AS (
                    SELECT
                        c.campaign_id,
                        MAX(CAST(m.value AS DOUBLE PRECISION)) as best_score
                    FROM candidates c
                    JOIN measurements m ON m.candidate_id = c.id AND m.name = 'combined_score'
                    WHERE m.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
                    GROUP BY c.campaign_id
                ),
                campaign_diversity AS (
                    SELECT
                        c1.campaign_id,
                        COUNT(*) as n_pairs,
                        AVG(c1.direct_code_embedding <=> c2.direct_code_embedding) as diversity_score
                    FROM candidates c1
                    JOIN candidates c2 ON c1.campaign_id = c2.campaign_id AND c1.id < c2.id
                    WHERE c1.direct_code_embedding IS NOT NULL
                      AND c2.direct_code_embedding IS NOT NULL
                    GROUP BY c1.campaign_id
                ),
                campaign_counts AS (
                    SELECT campaign_id, COUNT(*) as n_candidates
                    FROM candidates
                    WHERE direct_code_embedding IS NOT NULL
                    GROUP BY campaign_id
                    HAVING COUNT(*) >= 2
                )
                SELECT
                    camp.id as campaign_id,
                    camp.name as campaign_name,
                    camp.research_question,
                    camp.models_used,
                    camp.algorithm_used,
                    cs.best_score,
                    cd.diversity_score,
                    cc.n_candidates
                FROM campaigns camp
                JOIN campaign_scores cs ON cs.campaign_id = camp.id
                JOIN campaign_diversity cd ON cd.campaign_id = camp.id
                JOIN campaign_counts cc ON cc.campaign_id = camp.id
                WHERE cs.best_score IS NOT NULL
                  AND camp.research_question IS NOT NULL
                ORDER BY camp.id
            """)
            rows = cur.fetchall()

    results = []
    for row in rows:
        problem = _truncate_problem(row["research_question"])
        if problem is None:
            continue

        results.append({
            "campaign_id": str(row["campaign_id"]),
            "campaign_name": row["campaign_name"],
            "problem": problem,
            "research_question": row["research_question"],
            "models_used": row["models_used"],
            "algorithm_used": row["algorithm_used"],
            "n_candidates": row["n_candidates"],
            "diversity_score": round(float(row["diversity_score"]), 4),
            "best_score": float(row["best_score"]),
        })

    return results


def query_campaigns_with_best_embeddings(
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    """Query campaigns with the direct code embedding of their best-scoring candidate."""
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH best_candidates AS (
                    SELECT DISTINCT ON (c.campaign_id)
                        c.campaign_id,
                        c.id as candidate_id,
                        c.direct_code_embedding::text as embedding,
                        m.value as score
                    FROM candidates c
                    JOIN measurements m ON m.candidate_id = c.id AND m.name = 'combined_score'
                    WHERE c.direct_code_embedding IS NOT NULL
                      AND m.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
                    ORDER BY c.campaign_id, CAST(m.value AS DOUBLE PRECISION) DESC
                )
                SELECT
                    bc.campaign_id,
                    bc.candidate_id,
                    bc.embedding,
                    bc.score as best_score,
                    camp.research_question,
                    camp.models_used,
                    camp.algorithm_used
                FROM best_candidates bc
                JOIN campaigns camp ON bc.campaign_id = camp.id
                WHERE camp.research_question IS NOT NULL
                ORDER BY bc.campaign_id
            """)
            rows = cur.fetchall()

    results = []
    for row in rows:
        emb = _parse_embedding(row["embedding"])
        if emb:
            problem = _truncate_problem(row["research_question"])
            if problem is None:
                continue

            results.append({
                "campaign_id": str(row["campaign_id"]),
                "problem": problem,
                "models_used": row["models_used"],
                "algorithm_used": row["algorithm_used"],
                "best_score": float(row["best_score"]),
                "best_embedding": emb,
            })

    return results
