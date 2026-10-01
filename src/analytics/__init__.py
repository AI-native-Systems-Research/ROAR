#!/usr/bin/env python3
"""Analytics package for ADRS campaign data.

Each module follows this structure:
- query_*: Database queries only
- process_*: Data transformation/aggregation
- get_*: High-level API combining query + process

Modules:
- summary.py: Summary statistics
- clustering.py: Research question clustering
- q1_variance.py: Q1 Basin Structure / Multimodality analysis
"""

# Summary
from analytics.summary import (
    query_summary_stats,
    process_summary_stats,
    get_summary_stats,
)

# Clustering
from analytics.clustering import (
    query_campaign_embeddings,
    process_clusters,
    get_clusters,
    cluster_by_research_question,  # backwards compat alias
)

# Q1: Basin Structure / Multimodality
from analytics.q1_variance import (
    query_skydiscover_scores,
    process_variance_data,
    get_variance_analysis
)

__all__ = [
    # Summary
    "query_summary_stats",
    "process_summary_stats",
    "get_summary_stats",
    # Clustering
    "query_campaign_embeddings",
    "process_clusters",
    "get_clusters",
    "cluster_by_research_question",
    # Q1: Variance
    "query_skydiscover_scores",
    "process_variance_data",
    "get_variance_analysis",
]


class CampaignAnalytics:
    """Backwards-compatible wrapper for analytics functions.

    Deprecated: Use module functions directly instead.
    """

    def __init__(self, database_url: str | None = None):
        self._database_url = database_url

    def get_summary_stats(self) -> dict:
        return get_summary_stats(self._database_url)

    def cluster_by_research_question(self, distance_threshold: float = 0.1) -> dict:
        return get_clusters(self._database_url, distance_threshold)

    def get_recommendation(
        self,
        research_question: str,
        budget: int = 50,
        algorithm: str | None = None,
        model: str | None = None,
    ) -> dict:
        # Imported lazily: analytics.q5 imports q2_calibration, which pulls in
        # matplotlib and numpy that most callers of this package do not need.
        from analytics.q5 import get_recommendation

        return get_recommendation(
            self._database_url,
            research_question,
            budget,
            algorithm=algorithm,
            model=model,
        )

