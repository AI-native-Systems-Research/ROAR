#!/usr/bin/env python3
"""Parser for coding_agent campaign format.

Reads:
- summary.json: campaign metadata
- iterations.jsonl: per-iteration metrics
- candidates/iter_N/: source files per iteration
- author_input.json: version info
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from adrs_models import (
    ADRSArtifact,
    ADRSCampaign,
    ADRSCandidate,
    ADRSCandidateEdge,
    ADRSMeasurement,
    ADRSParsedCampaign,
    ADRSSystem,
)


def parse_coding_agent_campaign(folder_path: Path) -> ADRSParsedCampaign | None:
    """Parse a coding_agent campaign directory into ADRSParsedCampaign."""
    folder_path = Path(folder_path)

    summary_path = folder_path / "summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    # Read author_input.json for version and author
    author_input_path = folder_path / "author_input.json"
    version = None
    author_email = ""
    if author_input_path.exists():
        with open(author_input_path) as f:
            author_input = json.load(f)
            version = author_input.get("version")
            author_email = author_input.get("author", "")

    # Read iterations.jsonl
    iterations_path = folder_path / "iterations.jsonl"
    if not iterations_path.exists():
        return None

    iterations = []
    with open(iterations_path) as f:
        for line in f:
            if line.strip():
                iterations.append(json.loads(line))

    if not iterations:
        return None

    # Parse timestamps
    started_at = _parse_timestamp(iterations[0].get("timestamp"))
    ended_at = _parse_timestamp(iterations[-1].get("timestamp"))

    # Build campaign
    research_question = summary.get("target", "unknown")
    config_label = summary.get("config", "")
    name = folder_path.name

    campaign = ADRSCampaign(
        system=ADRSSystem(name="coding_agent", version=version),
        author=author_email,
        name=name,
        research_question=research_question,
        started_at=started_at,
        ended_at=ended_at,
        algorithm_used="coding_agent",
        models_used=None,
        final_metrics={
            "best_score": summary.get("result", {}).get("best_score"),
            "iterations_completed": summary.get("result", {}).get("iterations_completed"),
            "elapsed_s": summary.get("elapsed_s"),
        },
        config_used={"config": summary.get("config")} if summary.get("config") else None,
    )

    # Build candidates, measurements, edges
    candidates: list[ADRSCandidate] = []
    measurements: dict[str, list[ADRSMeasurement]] = {}
    artifacts: list[ADRSArtifact] = []
    candidate_edges: list[ADRSCandidateEdge] = []

    for iter_data in iterations:
        iter_idx = iter_data["iteration"]
        external_id = f"iter_{iter_idx}"

        candidates.append(ADRSCandidate(
            iteration_index=iter_idx,
            external_id=external_id,
            candidate_type="program",
            created_at=_parse_timestamp(iter_data.get("timestamp")),
        ))

        # Measurements
        iter_measurements: list[ADRSMeasurement] = []

        if iter_data.get("score") is not None:
            iter_measurements.append(ADRSMeasurement(
                name="combined_score", value=str(iter_data["score"])
            ))

        if iter_data.get("passed") and iter_data.get("build_success"):
            status = "VALID"
        elif iter_data.get("build_success"):
            status = "BUILD_OK"
        else:
            status = "INVALID"
        iter_measurements.append(ADRSMeasurement(name="status", value=status))

        metrics = iter_data.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                iter_measurements.append(ADRSMeasurement(
                    name=metric_name, value=str(metric_value)
                ))

        if iter_data.get("build_duration_s") is not None:
            iter_measurements.append(ADRSMeasurement(
                name="build_duration_s", value=str(iter_data["build_duration_s"])
            ))

        measurements[external_id] = iter_measurements

        # Parent edge
        if iter_idx > 0:
            candidate_edges.append(ADRSCandidateEdge(
                source_external_id=f"iter_{iter_idx - 1}",
                target_external_id=external_id,
                edge_type="parent",
            ))

    return ADRSParsedCampaign(
        campaign=campaign,
        candidates=candidates,
        measurements=measurements,
        artifacts=artifacts,
        candidate_edges=candidate_edges,
    )


def _parse_timestamp(ts) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None
    return None
