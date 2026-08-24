#!/usr/bin/env python3
"""Clustering analytics for ADRS campaign data.

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

def query_campaign_embeddings(database_url: str | None = None) -> list[dict[str, Any]]:
    """Query campaigns from the database.

    Returns:
        List of campaigns with id, name, and research_question.
    """
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, research_question
                FROM campaigns
                WHERE research_question IS NOT NULL
            """)
            rows = cur.fetchall()

    return [
        {
            "id": str(row["id"]),
            "name": row["name"],
            "research_question": row["research_question"],
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Process functions
# ---------------------------------------------------------------------------

def process_clusters(
    campaigns: list[dict[str, Any]],
    distance_threshold: float = 0.1,
) -> dict[str, Any]:
    """Group campaigns by research question.

    Args:
        campaigns: List of campaign dicts with id, name, research_question.
        distance_threshold: Unused, kept for API compatibility.

    Returns:
        Dictionary with cluster_count and clusters list.
    """
    clusters_dict: dict[str, list[dict]] = {}
    for c in campaigns:
        rq = c.get("research_question") or "Unknown"
        if rq not in clusters_dict:
            clusters_dict[rq] = []
        clusters_dict[rq].append({
            "id": c["id"],
            "name": c["name"],
            "research_question": c["research_question"],
        })

    clusters = [
        {
            "cluster_id": idx,
            "campaign_count": len(members),
            "research_questions": [rq],
            "campaigns": members,
        }
        for idx, (rq, members) in enumerate(sorted(clusters_dict.items(), key=lambda x: -len(x[1])))
    ]

    return {
        "cluster_count": len(clusters),
        "clusters": clusters,
        "campaigns_without_embeddings": 0,
    }


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def get_clusters(
    database_url: str | None = None,
    distance_threshold: float = 0.1,
) -> dict[str, Any]:
    """Get campaign clusters by research question similarity.

    Queries database and clusters campaigns.
    """
    campaigns = query_campaign_embeddings(database_url)
    return process_clusters(campaigns, distance_threshold)


# Backwards compatibility alias
cluster_by_research_question = process_clusters
