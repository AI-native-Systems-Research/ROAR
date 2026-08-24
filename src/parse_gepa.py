#!/usr/bin/env python3
"""Parser for GEPA (GPU-Enhanced Program Analysis) campaign outputs.

GEPA uses a sequential hill-climbing approach where each iteration mutates
the previous best. Unlike AdaEvolve/EvoX, GEPA has no context/BoN sampling.

Expected structure:
- summary.json: Campaign summary with best score and config
- iterations.jsonl: Per-iteration metrics (score, throughput, latency, etc.)
- iterations_analysis.jsonl: Per-iteration analysis (result, strategy, delta)
- implementation_analysis.json: Rich per-iteration analysis (change_type, lever, summary)
- effective_context.md: Problem context and objective
- candidates/iter_N/*.rs: Source code for each iteration
- seeds/*.rs: Initial seed programs
- final_state/*.rs: Final best program
"""

import hashlib
import json
import re
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


def parse_timestamp(ts: str | None) -> datetime | None:
    """Parse ISO timestamp string to datetime."""
    if ts is None:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
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
        ".rs": "text/x-rust",
        ".json": "application/json",
        ".jsonl": "application/x-jsonlines",
        ".md": "text/markdown",
        ".yaml": "application/x-yaml",
        ".yml": "application/x-yaml",
        ".py": "text/x-python",
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


class GEPAParser:
    """Parse a GEPA campaign folder into ADRS models."""

    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        self.summary: dict = {}
        self.iterations: list[dict] = []
        self.iterations_analysis: list[dict] = []
        self.implementation_analysis: dict = {}
        self.effective_context: str = ""
        self.author_input: dict = {}

    def parse(self) -> ADRSParsedCampaign | None:
        """Parse all campaign data. Returns ADRSParsedCampaign or None."""
        if not self.folder_path.is_dir():
            print(f"Error: {self.folder_path} is not a directory")
            return None

        if not self._load_data_files():
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

    def _load_data_files(self) -> bool:
        """Load all GEPA output files. Returns False if required files missing."""
        # Load author input (optional, contains system version)
        self.author_input = load_json(self.folder_path / "author_input.json", warn_on_error=False) or {}

        self.summary = load_json(self.folder_path / "summary.json") or {}
        if not self.summary:
            print(f"Error: No summary.json found in {self.folder_path}")
            return False

        self.iterations = load_jsonl(self.folder_path / "iterations.jsonl")
        if not self.iterations:
            print(f"Error: No iterations.jsonl found in {self.folder_path}")
            return False

        self.iterations_analysis = load_jsonl(
            self.folder_path / "iterations_analysis.jsonl"
        )
        self.implementation_analysis = (
            load_json(self.folder_path / "implementation_analysis.json") or {}
        )

        context_path = self.folder_path / "effective_context.md"
        if context_path.exists():
            try:
                self.effective_context = context_path.read_text()
            except OSError:
                pass

        return True

    def _merge_iteration_data(self) -> dict[int, dict]:
        """Merge data from all sources into a single dict keyed by iteration."""
        merged: dict[int, dict] = {}

        for item in self.iterations:
            iter_num = item.get("iteration", 0)
            merged[iter_num] = {"metrics": item}

        for item in self.iterations_analysis:
            iter_num = item.get("iteration", 0)
            if iter_num not in merged:
                merged[iter_num] = {}
            merged[iter_num]["analysis"] = item

        impl_iterations = self.implementation_analysis.get("iterations", [])
        for item in impl_iterations:
            iter_num = item.get("iteration", 0)
            if iter_num not in merged:
                merged[iter_num] = {}
            merged[iter_num]["implementation"] = item

        return merged

    def _parse_campaign(self) -> ADRSCampaign:
        """Build campaign from summary.json and effective_context.md."""
        name = self.folder_path.name

        research_question = self._extract_research_question()
        started_at, ended_at = self._extract_timestamps()

        final_metrics = None
        if self.summary:
            result = self.summary.get("result", {})
            final_metrics = {
                "best_score": self.summary.get("result", {}).get("best_score"),
                "iterations_completed": result.get("iterations_completed"),
                "elapsed_s": self.summary.get("elapsed_s"),
                "budget": self.summary.get("budget"),
            }
            summary_data = result.get("summary", {})
            if summary_data:
                final_metrics["build_success_rate"] = summary_data.get(
                    "build_success_rate"
                )
                final_metrics["pass_rate"] = summary_data.get("pass_rate")
                final_metrics["best_iteration"] = summary_data.get("best_iteration")

        run_summary = self.implementation_analysis.get("run_summary", {})
        final_summary = run_summary.get("strategy_description")

        target = self.summary.get("target", "")
        config = self.summary.get("config", "")
        config_used = {
            "target": target,
            "config": config,
            "budget": self.summary.get("budget"),
            "worktree": self.summary.get("worktree"),
        }

        system = ADRSSystem(
            name="gepa",
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
            algorithm_used="gepa",
            models_used=None,
            total_cost_usd=None,
            total_tokens=None,
            final_summary=final_summary,
            final_metrics=final_metrics,
        )

    def _extract_research_question(self) -> str | None:
        """Parse Objective section from effective_context.md."""
        if not self.effective_context:
            return None

        match = re.search(
            r"## Objective\s*\n(.+?)(?=\n##|\Z)", self.effective_context, re.DOTALL
        )
        if match:
            return match.group(1).strip()

        target = self.summary.get("target", "")
        config = self.summary.get("config", "")
        if target:
            return f"{target} optimization ({config})" if config else target

        return None

    def _extract_timestamps(self) -> tuple[datetime | None, datetime | None]:
        """Get start/end times from first/last iteration timestamps."""
        if not self.iterations:
            return None, None

        started_at = parse_timestamp(self.iterations[0].get("timestamp"))
        ended_at = parse_timestamp(self.iterations[-1].get("timestamp"))

        return started_at, ended_at

    def _parse_candidates_and_edges(
        self,
    ) -> tuple[list[ADRSCandidate], list[ADRSCandidateEdge]]:
        """Create one candidate per iteration with sequential parent edges only."""
        candidates: list[ADRSCandidate] = []
        edges: list[ADRSCandidateEdge] = []

        merged_data = self._merge_iteration_data()

        for iter_num in sorted(merged_data.keys()):
            data = merged_data[iter_num]
            metrics = data.get("metrics", {})

            external_id = f"iter-{iter_num}"
            timestamp = parse_timestamp(metrics.get("timestamp"))

            candidate = ADRSCandidate(
                iteration_index=iter_num,
                external_id=external_id,
                candidate_type="program",
                created_at=timestamp,
            )
            candidates.append(candidate)

            if iter_num > 0:
                parent_external_id = f"iter-{iter_num - 1}"
                edges.append(
                    ADRSCandidateEdge(
                        source_external_id=parent_external_id,
                        target_external_id=external_id,
                        edge_type="parent",
                    )
                )

        return candidates, edges

    def _parse_measurements(
        self, candidates: list[ADRSCandidate]
    ) -> dict[str, list[ADRSMeasurement]]:
        """Extract measurements from all data sources."""
        measurements: dict[str, list[ADRSMeasurement]] = {}
        merged_data = self._merge_iteration_data()

        for candidate in candidates:
            external_id = candidate.external_id
            if not external_id:
                continue

            iter_num = candidate.iteration_index
            data = merged_data.get(iter_num, {})
            metrics_data = data.get("metrics", {})
            analysis_data = data.get("analysis", {})
            impl_data = data.get("implementation", {})

            meas_list: list[ADRSMeasurement] = []

            score = metrics_data.get("score")
            if score is not None:
                meas_list.append(
                    ADRSMeasurement(name="combined_score", value=str(score))
                )

            passed = metrics_data.get("passed", True)
            status = "VALID" if passed else "INVALID"
            meas_list.append(ADRSMeasurement(name="status", value=status))

            build_success = metrics_data.get("build_success")
            if build_success is not None:
                meas_list.append(
                    ADRSMeasurement(name="build_success", value=str(build_success))
                )

            inner_metrics = metrics_data.get("metrics", {})
            for metric_name, value in inner_metrics.items():
                if value is not None:
                    meas_list.append(
                        ADRSMeasurement(name=metric_name, value=str(value))
                    )

            if score is not None:
                metric_parts = [f"score={score:.4f}"]
                for k, v in inner_metrics.items():
                    if isinstance(v, float):
                        metric_parts.append(f"{k}={v:.2f}")
                    elif v is not None:
                        metric_parts.append(f"{k}={v}")
                meas_list.append(
                    ADRSMeasurement(name="result_summary", value=", ".join(metric_parts))
                )

            result = analysis_data.get("result")
            if result:
                meas_list.append(ADRSMeasurement(name="result", value=result))

            strategy = analysis_data.get("strategy")
            if strategy:
                meas_list.append(ADRSMeasurement(name="strategy", value=strategy))

            delta = analysis_data.get("delta")
            if delta is not None:
                meas_list.append(ADRSMeasurement(name="delta", value=str(delta)))

            change_type = impl_data.get("change_type")
            if change_type:
                meas_list.append(ADRSMeasurement(name="change_type", value=change_type))

            lever = impl_data.get("lever")
            if lever:
                meas_list.append(ADRSMeasurement(name="lever", value=lever))

            summary = impl_data.get("summary")
            if summary:
                meas_list.append(ADRSMeasurement(name="change_summary", value=summary))

            failure_reason = impl_data.get("failure_reason")
            if failure_reason:
                meas_list.append(
                    ADRSMeasurement(name="failure_reason", value=failure_reason)
                )

            detail = impl_data.get("detail")
            if detail:
                meas_list.append(ADRSMeasurement(name="change_detail", value=detail))

            if meas_list:
                measurements[external_id] = meas_list

        return measurements

    def _collect_artifacts(self) -> list[ADRSArtifact]:
        """Collect code files from candidates/iter_N/, seeds/, final_state/."""
        artifacts: list[ADRSArtifact] = []

        summary_path = self.folder_path / "summary.json"
        if summary_path.exists():
            artifacts.append(
                ADRSArtifact(
                    iteration_index=None,
                    uri=str(summary_path),
                    content_hash=compute_file_hash(summary_path),
                    size_bytes=summary_path.stat().st_size,
                    mime_type="application/json",
                )
            )

        context_path = self.folder_path / "effective_context.md"
        if context_path.exists():
            artifacts.append(
                ADRSArtifact(
                    iteration_index=None,
                    uri=str(context_path),
                    content_hash=compute_file_hash(context_path),
                    size_bytes=context_path.stat().st_size,
                    mime_type="text/markdown",
                )
            )

        impl_path = self.folder_path / "implementation_analysis.json"
        if impl_path.exists():
            artifacts.append(
                ADRSArtifact(
                    iteration_index=None,
                    uri=str(impl_path),
                    content_hash=compute_file_hash(impl_path),
                    size_bytes=impl_path.stat().st_size,
                    mime_type="application/json",
                )
            )

        candidates_dir = self.folder_path / "candidates"
        if candidates_dir.exists():
            for iter_dir in sorted(candidates_dir.iterdir()):
                if not iter_dir.is_dir():
                    continue
                iter_match = re.match(r"iter_(\d+)", iter_dir.name)
                if not iter_match:
                    continue
                iter_num = int(iter_match.group(1))
                external_id = f"iter-{iter_num}"

                for code_file in iter_dir.glob("*.rs"):
                    if code_file.name.startswith("."):
                        continue
                    artifacts.append(
                        ADRSArtifact(
                            iteration_index=iter_num,
                            external_id=external_id,
                            uri=str(code_file),
                            content_hash=compute_file_hash(code_file),
                            size_bytes=code_file.stat().st_size,
                            mime_type="text/x-rust",
                        )
                    )

        seeds_dir = self.folder_path / "seeds"
        if seeds_dir.exists():
            for seed_file in seeds_dir.glob("*.rs"):
                if seed_file.name.startswith("."):
                    continue
                artifacts.append(
                    ADRSArtifact(
                        iteration_index=None,
                        uri=str(seed_file),
                        content_hash=compute_file_hash(seed_file),
                        size_bytes=seed_file.stat().st_size,
                        mime_type="text/x-rust",
                    )
                )

        final_dir = self.folder_path / "final_state"
        if final_dir.exists():
            for final_file in final_dir.glob("*.rs"):
                if final_file.name.startswith("."):
                    continue
                artifacts.append(
                    ADRSArtifact(
                        iteration_index=None,
                        uri=str(final_file),
                        content_hash=compute_file_hash(final_file),
                        size_bytes=final_file.stat().st_size,
                        mime_type="text/x-rust",
                    )
                )

        best_file = self.folder_path / "best_candidate.rs"
        if best_file.exists():
            artifacts.append(
                ADRSArtifact(
                    iteration_index=None,
                    uri=str(best_file),
                    content_hash=compute_file_hash(best_file),
                    size_bytes=best_file.stat().st_size,
                    mime_type="text/x-rust",
                )
            )

        return artifacts


def parse_gepa_campaign(folder_path: Path) -> ADRSParsedCampaign | None:
    """Parse a GEPA campaign into ADRS models.

    Args:
        folder_path: Path to the campaign directory

    Returns:
        ADRSParsedCampaign or None if parsing fails
    """
    parser = GEPAParser(folder_path)
    return parser.parse()


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse GEPA campaign into ADRS models (test mode)"
    )
    parser.add_argument("path", type=Path, help="Path to campaign folder")
    args = parser.parse_args()

    parsed = parse_gepa_campaign(args.path)
    if parsed:
        print(f"\nSuccessfully parsed campaign: {parsed.campaign.name}")
        print(f"  System: {parsed.campaign.system.name}")
        print(f"  Research question: {parsed.campaign.research_question}")
        print(f"  Candidates: {len(parsed.candidates)}")
        total_measurements = sum(len(m) for m in parsed.measurements.values())
        print(
            f"  Measurements: {total_measurements} (for {len(parsed.measurements)} candidates)"
        )
        print(f"  Artifacts: {len(parsed.artifacts)}")
        print(f"  Edges: {len(parsed.candidate_edges)} (parent only, no context)")
        if parsed.campaign.final_metrics:
            print(f"  Best score: {parsed.campaign.final_metrics.get('best_score')}")
    else:
        print("Failed to parse campaign")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
