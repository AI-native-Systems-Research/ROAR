#!/usr/bin/env python3
"""Summary statistics analytics for ADRS campaign data.

Module structure:
- query_*: Database queries
- process_*: Data transformation
- get_*: High-level API
"""

import os
from typing import Any

import psycopg
from psycopg.rows import dict_row


DATABASE_URL_DEFAULT = "postgresql://postgres:postgres@localhost:5432/adrs"


def _get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DATABASE_URL_DEFAULT)


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def query_summary_stats(database_url: str | None = None) -> dict[str, Any]:
    """Query summary statistics from the database.

    Returns:
        Dictionary with campaign_count, candidate_count (one per iteration),
        models list, and algorithms list.
    """
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM campaigns")
            campaign_count = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(*) as count FROM candidates")
            candidate_count = cur.fetchone()["count"]

            # Count distinct iterations (max iteration_index per campaign)
            cur.execute("""
                SELECT COALESCE(SUM(max_iter + 1), 0) as count
                FROM (
                    SELECT MAX(iteration_index) as max_iter
                    FROM candidates
                    GROUP BY campaign_id
                ) sub
            """)
            iteration_count = cur.fetchone()["count"]

            cur.execute(
                "SELECT DISTINCT unnest(models_used) as model FROM campaigns ORDER BY model"
            )
            models = [row["model"] for row in cur.fetchall()]

            cur.execute(
                "SELECT DISTINCT algorithm_used FROM campaigns "
                "WHERE algorithm_used IS NOT NULL ORDER BY algorithm_used"
            )
            algorithms = [row["algorithm_used"] for row in cur.fetchall()]

    return {
        "campaign_count": campaign_count,
        "iteration_count": iteration_count,
        "candidate_count": candidate_count,
        "models": models,
        "algorithms": algorithms,
    }


# ---------------------------------------------------------------------------
# Process functions
# ---------------------------------------------------------------------------

def process_summary_stats(raw_stats: dict[str, Any]) -> dict[str, Any]:
    """Process raw summary statistics into final format.

    Args:
        raw_stats: Raw statistics from query_summary_stats.

    Returns:
        Dictionary with counts and lists.
    """
    return {
        "campaign_count": raw_stats["campaign_count"],
        "iteration_count": raw_stats["iteration_count"],
        "candidate_count": raw_stats["candidate_count"],
        "model_count": len(raw_stats["models"]),
        "algorithm_count": len(raw_stats["algorithms"]),
        "models": raw_stats["models"],
        "algorithms": raw_stats["algorithms"],
    }


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def get_summary_stats(database_url: str | None = None) -> dict[str, Any]:
    """Get summary statistics.

    Queries database and processes into final format.
    """
    raw = query_summary_stats(database_url)
    return process_summary_stats(raw)
