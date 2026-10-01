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
import re
from pathlib import Path
from datetime import datetime, timezone

from model_dirname import models_from_path
from adrs_models import (
    ADRSArtifact,
    ADRSCampaign,
    ADRSCandidate,
    ADRSCandidateEdge,
    ADRSMeasurement,
    ADRSParsedCampaign,
    ADRSSystem,
)


def matches_coding_agent(folder_path: Path) -> bool:
    """True if folder_path is a coding_agent run: summary.json + iterations.jsonl at the
    run root, with summary.json's framework == "coding_agent". The framework gate separates
    it from the curated-GEPA bundle (which also has summary.json + iterations.jsonl). Used
    by system_detection."""
    folder_path = Path(folder_path)
    summary_path = folder_path / "summary.json"
    if not summary_path.exists() or not (folder_path / "iterations.jsonl").exists():
        return False
    try:
        with open(summary_path) as f:
            return json.load(f).get("framework") == "coding_agent"
    except (OSError, json.JSONDecodeError):
        return False


def _extract_objective(folder_path: Path) -> str | None:
    """First non-empty line under the `## Objective` heading of effective_context.md, verbatim."""
    context_path = folder_path / "effective_context.md"
    if not context_path.exists():
        return None
    in_objective = False
    with open(context_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                in_objective = stripped.lstrip("#").strip().lower() == "objective"
            elif in_objective and stripped:
                return stripped
    return None


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
    # The run's declared objective; summary.json's "target" is only the workbench target id.
    research_question = _extract_objective(folder_path) or summary.get("target")
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
        models_used=_extract_models(folder_path, summary),
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

    # Artifacts: the evolved source per iteration (candidates/iter_N/*.rs), linked to that
    # iteration's candidate, plus the final state. Stored by reference (uri), like the
    # other parsers. Without this the coding_agent format's code would be dropped.
    candidates_dir = folder_path / "candidates"
    if candidates_dir.is_dir():
        for iter_dir in sorted(candidates_dir.iterdir()):
            if not iter_dir.is_dir():
                continue
            m = re.match(r"iter_(\d+)$", iter_dir.name)
            if not m:
                continue
            iter_num = int(m.group(1))
            for code_file in sorted(iter_dir.glob("*.rs")):
                if code_file.name.startswith("."):
                    continue
                artifacts.append(ADRSArtifact(
                    iteration_index=iter_num,
                    external_id=f"iter_{iter_num}",
                    uri=str(code_file),
                    content_hash=_compute_file_hash(code_file),
                    size_bytes=code_file.stat().st_size,
                    mime_type="text/x-rust",
                ))

    final_dir = folder_path / "final_state"
    if final_dir.is_dir():
        for final_file in sorted(final_dir.glob("*.rs")):
            if final_file.name.startswith("."):
                continue
            artifacts.append(ADRSArtifact(
                iteration_index=None,
                uri=str(final_file),
                content_hash=_compute_file_hash(final_file),
                size_bytes=final_file.stat().st_size,
                mime_type="text/x-rust",
            ))

    return ADRSParsedCampaign(
        campaign=campaign,
        candidates=candidates,
        measurements=measurements,
        artifacts=artifacts,
        candidate_edges=candidate_edges,
    )


def _extract_models(folder_path: Path, summary: dict) -> list[str] | None:
    """Model(s) for the run, verbatim. coding_agent's summary.json carries no model field,
    so check a structured `run_config.json`/summary key if present, else the run dir name."""
    rc = folder_path / "run_config.json"
    if rc.exists():
        try:
            with open(rc) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
        for key in ("model_name", "model", "model_id"):
            v = data.get(key)
            if isinstance(v, str) and v:
                return [v]
    for key in ("model", "model_name", "model_id", "llm"):
        v = summary.get(key)
        if isinstance(v, str) and v:
            return [v]
    return models_from_path(folder_path)


def _compute_file_hash(file_path: Path) -> str | None:
    """SHA256 of a file, or None if it can't be read."""
    try:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except OSError:
        return None


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
