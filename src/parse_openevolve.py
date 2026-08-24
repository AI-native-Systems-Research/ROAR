#!/usr/bin/env python3
"""Parser for OpenEvolve campaign outputs.

OpenEvolve uses island-based MAP-Elites with evolution traces.
Structure:
- evolution_trace.jsonl: Line-by-line mutation events (parent/child code, metrics, prompts)
- checkpoints/checkpoint_N/programs/*.json: Program snapshots with full metadata
- checkpoints/checkpoint_N/metadata.json: Island/archive state
- best/best_program.py: Best solution code
- best/best_program_info.json: Best solution metadata
- logs/: Execution logs
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from adrs_models import (
    ADRSArtifact,
    ADRSCampaign,
    ADRSCandidate,
    ADRSCandidateEdge,
    ADRSMeasurement,
    ADRSParsedCampaign,
    ADRSSystem,
)


def parse_timestamp(ts: float | str | None) -> datetime | None:
    """Parse timestamp (Unix epoch or ISO string) to datetime."""
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts)
        elif isinstance(ts, str):
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            return datetime.fromisoformat(ts)
    except (ValueError, TypeError, OSError):
        return None
    return None


def compute_file_hash(file_path: Path) -> str | None:
    """Compute SHA256 hash of a file."""
    if not file_path.exists():
        return None
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_mime_type(file_path: Path) -> str:
    """Get MIME type based on file extension."""
    suffix = file_path.suffix.lower()
    mime_map = {
        ".py": "text/x-python",
        ".json": "application/json",
        ".jsonl": "application/x-jsonlines",
        ".md": "text/markdown",
        ".yaml": "application/x-yaml",
        ".yml": "application/x-yaml",
        ".log": "text/plain",
        ".txt": "text/plain",
    }
    return mime_map.get(suffix, "application/octet-stream")


def load_json(file_path: Path, warn_on_error: bool = True) -> dict | None:
    """Load JSON file if it exists."""
    if not file_path.exists():
        return None
    try:
        with open(file_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        if warn_on_error:
            print(f"Warning: Could not parse {file_path}: {e}")
        return None


def load_jsonl(file_path: Path) -> list[dict]:
    """Load JSONL file, return list of records."""
    if not file_path.exists():
        return []
    records = []
    try:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not parse {file_path}: {e}")
    return records


class OpenEvolveParser:
    """Parse an OpenEvolve campaign folder into ADRS models."""

    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        self.evolution_trace: list[dict] = []
        self.program_db: dict[str, dict] = {}
        self.best_info: dict = {}
        self.author_input: dict = {}

    def parse(self) -> ADRSParsedCampaign | None:
        """Parse all campaign data. Returns ADRSParsedCampaign or None."""
        if not self.folder_path.is_dir():
            print(f"Error: {self.folder_path} is not a directory")
            return None

        # Load author input (optional, contains system version)
        self.author_input = load_json(self.folder_path / "author_input.json", warn_on_error=False) or {}

        # Load evolution trace
        self.evolution_trace = load_jsonl(self.folder_path / "evolution_trace.jsonl")
        if not self.evolution_trace:
            print(f"Error: No evolution_trace.jsonl found in {self.folder_path}")
            return None

        # Load best program info
        self.best_info = load_json(self.folder_path / "best" / "best_program_info.json") or {}

        # Build program database from checkpoints
        self._build_program_database()

        if not self.program_db:
            print(f"Error: No programs found in checkpoints in {self.folder_path}")
            return None

        campaign = self._parse_campaign()
        candidates, edges = self._parse_candidates_and_edges()
        measurements = self._parse_measurements(candidates)
        artifacts = self._collect_artifacts()

        return ADRSParsedCampaign(
            campaign=campaign,
            candidates=candidates,
            measurements=measurements,
            artifacts=artifacts,
            candidate_edges=edges,
        )

    def _build_program_database(self):
        """Build database of all programs from checkpoints (deduplicated by ID)."""
        checkpoints_dir = self.folder_path / "checkpoints"
        if not checkpoints_dir.exists():
            return

        for checkpoint_dir in sorted(checkpoints_dir.iterdir()):
            if not checkpoint_dir.is_dir() or checkpoint_dir.name.startswith("."):
                continue

            programs_dir = checkpoint_dir / "programs"
            if not programs_dir.exists():
                continue

            for program_file in programs_dir.glob("*.json"):
                if program_file.name.startswith("."):
                    continue
                program_info = load_json(program_file, warn_on_error=False)
                if program_info and "id" in program_info:
                    prog_id = program_info["id"]
                    # Keep first occurrence (earlier checkpoint)
                    if prog_id not in self.program_db:
                        self.program_db[prog_id] = program_info

    def _parse_campaign(self) -> ADRSCampaign:
        """Parse campaign-level information."""
        name = self.folder_path.name

        # Extract research question from prompts if available
        research_question = self._extract_research_question()

        # Get timestamps from evolution trace
        started_at = None
        ended_at = None
        if self.evolution_trace:
            first_ts = self.evolution_trace[0].get("timestamp")
            last_ts = self.evolution_trace[-1].get("timestamp")
            started_at = parse_timestamp(first_ts)
            ended_at = parse_timestamp(last_ts)

        # Total iterations
        total_iterations = len(self.evolution_trace)

        # Best metrics from best_info or last evolution entry
        final_metrics = {}
        if self.best_info:
            metrics = self.best_info.get("metrics", {})
            final_metrics = {
                "best_score": metrics.get("combined_score"),
                "best_iteration": self.best_info.get("iteration"),
                "best_generation": self.best_info.get("generation"),
                "total_iterations": total_iterations,
            }

        # Config from prompts (extract blocked approaches, etc.)
        config_used = self._extract_config()

        system = ADRSSystem(
            name="openevolve",
            version=self.author_input.get("version"),
        )

        return ADRSCampaign(
            system=system,
            author=self.author_input.get("author", ""),
            name=name,
            research_question=research_question,
            started_at=started_at,
            ended_at=ended_at,
            config_used=config_used,
            algorithm_used="map-elites",
            models_used=None,
            total_cost_usd=None,
            total_tokens=None,
            final_summary=None,
            final_metrics=final_metrics,
        )

    def _extract_research_question(self) -> str | None:
        """Extract research question/objective from prompts."""
        # Try to get from first program's prompts
        for prog in self.program_db.values():
            prompts = prog.get("prompts")
            if prompts:
                # Look in full_rewrite_user prompt
                full_rewrite = prompts.get("full_rewrite_user", {})
                system_prompt = full_rewrite.get("system", "")
                if system_prompt:
                    # Extract problem description - look for key phrases
                    if "KV cache eviction" in system_prompt:
                        return "Optimize KV cache eviction policy for vLLM inference server"
                    # Return first meaningful line
                    lines = system_prompt.strip().split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith("#") and len(line) > 20:
                            return line[:200]
        return None

    def _extract_config(self) -> dict:
        """Extract configuration from prompts and metadata."""
        config = {
            "total_iterations": len(self.evolution_trace),
        }

        # Get island count from checkpoint metadata
        checkpoints_dir = self.folder_path / "checkpoints"
        if checkpoints_dir.exists():
            for checkpoint_dir in sorted(checkpoints_dir.iterdir()):
                metadata = load_json(checkpoint_dir / "metadata.json", warn_on_error=False)
                if metadata:
                    config["num_islands"] = len(metadata.get("islands", []))
                    config["archive_size"] = len(metadata.get("archive", []))
                    break

        return config

    def _parse_candidates_and_edges(
        self,
    ) -> tuple[list[ADRSCandidate], list[ADRSCandidateEdge]]:
        """Parse candidates from program database and create edges."""
        candidates = []
        edges = []
        valid_ids: set[str] = set()

        # Sort programs by iteration_found to assign iteration_index
        sorted_progs = sorted(
            self.program_db.items(),
            key=lambda x: (x[1].get("iteration_found") or 0, x[0])
        )

        for idx, (prog_id, prog) in enumerate(sorted_progs):
            # Get code from program JSON
            code = prog.get("code")

            # Get metrics
            metrics = prog.get("metrics", {})
            score = metrics.get("combined_score")

            # Create candidate
            candidate = ADRSCandidate(
                iteration_index=idx,
                external_id=prog_id,
                code=code,
                score=score,
                generation=prog.get("generation"),
                created_at=parse_timestamp(prog.get("timestamp")),
                metadata={
                    "iteration_found": prog.get("iteration_found"),
                    "language": prog.get("language"),
                    "complexity": prog.get("complexity"),
                    "diversity": prog.get("diversity"),
                },
            )
            candidates.append(candidate)
            valid_ids.add(prog_id)

        # Create parent edges
        for prog_id, prog in self.program_db.items():
            parent_id = prog.get("parent_id")
            if parent_id and parent_id in valid_ids and prog_id in valid_ids:
                edge = ADRSCandidateEdge(
                    source_external_id=parent_id,
                    target_external_id=prog_id,
                    edge_type="parent",
                )
                edges.append(edge)

            # Check for context edges (other_context_ids)
            context_ids = prog.get("other_context_ids") or []
            for ctx_id in context_ids:
                if ctx_id in valid_ids and ctx_id != parent_id:
                    edge = ADRSCandidateEdge(
                        source_external_id=ctx_id,
                        target_external_id=prog_id,
                        edge_type="context",
                    )
                    edges.append(edge)

        return candidates, edges

    def _parse_measurements(
        self, candidates: list[ADRSCandidate]
    ) -> dict[str, list[ADRSMeasurement]]:
        """Extract measurements from program metrics, keyed by external_id."""
        measurements: dict[str, list[ADRSMeasurement]] = {}

        # Build set of valid external_ids
        valid_ids = {c.external_id for c in candidates if c.external_id}

        for prog_id, prog in self.program_db.items():
            if prog_id not in valid_ids:
                continue

            metrics = prog.get("metrics", {})
            if not metrics:
                continue

            candidate_measurements = []

            # Top-level metrics
            metric_names = [
                "combined_score",
                "cpu_hit_rate",
                "ttft_ratio",
                "throughput_ratio",
                "eviction_rate",
                "evictions",
                "lookup_total",
                "lookup_hits",
                "lookup_misses",
                "engine_external_prefix_hit_rate",
                "engine_gpu_prefix_hit_rate",
            ]

            for name in metric_names:
                value = metrics.get(name)
                if value is not None:
                    candidate_measurements.append(
                        ADRSMeasurement(
                            name=name,
                            value=str(value),
                        )
                    )

            # Nested metrics (metrics.metrics)
            nested = metrics.get("metrics", {})
            nested_names = [
                "request_throughput",
                "output_token_throughput",
                "mean_ttft_ms",
                "p99_ttft_ms",
                "mean_request_latency_ms",
                "total_requests",
                "failures",
            ]

            for name in nested_names:
                value = nested.get(name)
                if value is not None:
                    candidate_measurements.append(
                        ADRSMeasurement(
                            name=name,
                            value=str(value),
                        )
                    )

            if candidate_measurements:
                measurements[prog_id] = candidate_measurements

        return measurements

    def _get_metric_unit(self, name: str) -> str | None:
        """Get unit for a metric name."""
        units = {
            "combined_score": "score",
            "cpu_hit_rate": "ratio",
            "ttft_ratio": "ratio",
            "throughput_ratio": "ratio",
            "eviction_rate": "ratio",
            "request_throughput": "req/s",
            "output_token_throughput": "tokens/s",
            "mean_ttft_ms": "ms",
            "p99_ttft_ms": "ms",
            "mean_request_latency_ms": "ms",
        }
        return units.get(name)

    def _collect_artifacts(self) -> list[ADRSArtifact]:
        """Collect artifact files from the campaign."""
        artifacts = []

        # Best program
        best_program = self.folder_path / "best" / "best_program.py"
        if best_program.exists():
            artifacts.append(
                ADRSArtifact(
                    uri=str(best_program),
                    content_hash=compute_file_hash(best_program),
                    size_bytes=best_program.stat().st_size,
                    mime_type="text/x-python",
                )
            )

        # Best program info
        best_info = self.folder_path / "best" / "best_program_info.json"
        if best_info.exists():
            artifacts.append(
                ADRSArtifact(
                    uri=str(best_info),
                    content_hash=compute_file_hash(best_info),
                    size_bytes=best_info.stat().st_size,
                    mime_type="application/json",
                )
            )

        # Evolution trace
        trace_file = self.folder_path / "evolution_trace.jsonl"
        if trace_file.exists():
            artifacts.append(
                ADRSArtifact(
                    uri=str(trace_file),
                    content_hash=compute_file_hash(trace_file),
                    size_bytes=trace_file.stat().st_size,
                    mime_type="application/x-jsonlines",
                )
            )

        # Logs directory
        logs_dir = self.folder_path / "logs"
        if logs_dir.exists():
            for log_file in logs_dir.glob("*"):
                if log_file.is_file() and not log_file.name.startswith("."):
                    artifacts.append(
                        ADRSArtifact(
                            uri=str(log_file),
                            content_hash=compute_file_hash(log_file),
                            size_bytes=log_file.stat().st_size,
                            mime_type=get_mime_type(log_file),
                        )
                    )

        # Per-candidate program files from checkpoints
        seen_program_ids: set[str] = set()
        checkpoints_dir = self.folder_path / "checkpoints"
        if checkpoints_dir.exists():
            for checkpoint_dir in sorted(checkpoints_dir.iterdir()):
                if not checkpoint_dir.is_dir() or checkpoint_dir.name.startswith("."):
                    continue
                programs_dir = checkpoint_dir / "programs"
                if not programs_dir.exists():
                    continue
                for program_file in programs_dir.glob("*.json"):
                    if program_file.name.startswith("."):
                        continue
                    external_id = program_file.stem
                    if external_id in seen_program_ids:
                        continue
                    seen_program_ids.add(external_id)
                    program_info = self.program_db.get(external_id, {})
                    iteration_index = program_info.get("iteration_found")
                    artifacts.append(
                        ADRSArtifact(
                            iteration_index=iteration_index,
                            external_id=external_id,
                            uri=str(program_file),
                            content_hash=compute_file_hash(program_file),
                            size_bytes=program_file.stat().st_size,
                            mime_type="application/json",
                        )
                    )

        return artifacts


def parse_openevolve_campaign(folder_path: Path) -> ADRSParsedCampaign | None:
    """Convenience function to parse an OpenEvolve campaign folder."""
    parser = OpenEvolveParser(folder_path)
    return parser.parse()


def main():
    """CLI entry point for testing."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: parse_openevolve.py <campaign_folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    result = parse_openevolve_campaign(folder)

    if result:
        print(f"Campaign: {result.campaign.name}")
        print(f"System: {result.campaign.system.name} v{result.campaign.system.version}")
        print(f"Research question: {result.campaign.research_question}")
        print(f"Algorithm: {result.campaign.algorithm_used}")
        print(f"Candidates: {len(result.candidates)}")
        print(f"Edges: {len(result.candidate_edges)}")
        print(f"Measurements: {len(result.measurements)}")
        print(f"Artifacts: {len(result.artifacts)}")

        # Show edge breakdown
        parent_edges = [e for e in result.candidate_edges if e.edge_type == "parent"]
        context_edges = [e for e in result.candidate_edges if e.edge_type == "context"]
        print(f"  Parent edges: {len(parent_edges)}")
        print(f"  Context edges: {len(context_edges)}")
    else:
        print("Failed to parse campaign")
        sys.exit(1)


if __name__ == "__main__":
    main()
