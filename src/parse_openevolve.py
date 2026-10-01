#!/usr/bin/env python3
"""Parser for OpenEvolve NATIVE output.

OpenEvolve (github.com/algorithmicsuperintelligence/openevolve) uses island-based
MAP-Elites. Its `database.py` writes, per checkpoint, `programs/{id}.json` (one file per
program, with code, metrics, parent_id, generation, timestamp, prompts) plus
`metadata.json` (island/archive state); the controller also writes a `best/` dir.

Native structure (checkpoints/ and best/ may sit directly under the run dir OR nested
under a `results/` subdir — both are handled):
- checkpoints/checkpoint_N/programs/*.json: Program snapshots with full metadata
- checkpoints/checkpoint_N/metadata.json: Island/archive state
- best/best_program.py: Best solution code
- best/best_program_info.json: Best solution metadata
- logs/: Execution logs

`evolution_trace.jsonl` is NOT native OpenEvolve output — it is written by a separate
instrumentation layer (EvoTrace) and is present only for some contributors. It is used
when present but is OPTIONAL; timestamps then come from the programs themselves.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

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


def load_yaml(file_path: Path) -> dict | None:
    """Load a YAML file into a dict; None if missing/unparseable (yaml is optional)."""
    if not file_path.exists():
        return None
    try:
        import yaml

        with open(file_path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
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
        # `base` is the dir that actually holds checkpoints/ and best/. The corpus nests
        # them under results/, so accept either the run root or the results/ dir.
        self.base = self._resolve_base(folder_path)
        self.evolution_trace: list[dict] = []
        self.program_db: dict[str, dict] = {}
        self.best_info: dict = {}
        self.author_input: dict = {}

    @staticmethod
    def _resolve_base(folder_path: Path) -> Path:
        """Find the dir containing checkpoints/ (run root or a results/ subdir)."""
        if (folder_path / "checkpoints").is_dir():
            return folder_path
        if (folder_path / "results" / "checkpoints").is_dir():
            return folder_path / "results"
        return folder_path

    @classmethod
    def matches(cls, folder_path: Path) -> bool:
        """True if folder_path is a native OpenEvolve run: checkpoints/ (at the run root
        or under results/) plus a native OpenEvolve signal — a best/best_program_info.json
        or an openevolve_*.log. The native signal keeps this from claiming a SkyDiscover
        run, which also has checkpoints/ but no such log. Used by system_detection."""
        base = cls._resolve_base(folder_path)
        if not (base / "checkpoints").is_dir():
            return False
        if (base / "best" / "best_program_info.json").exists():
            return True
        for logs in (base / "logs", folder_path / "logs", base.parent / "logs"):
            if logs.is_dir() and any(logs.glob("openevolve_*.log")):
                return True
        return False

    def _logs_dir(self) -> Path | None:
        """Locate the run's logs/ dir (under base, run root, or base's parent)."""
        for cand in (self.base / "logs", self.folder_path / "logs", self.base.parent / "logs"):
            if cand.is_dir():
                return cand
        return None

    def _extract_models(self) -> list[str] | None:
        """Model(s) for the run, verbatim. OpenEvolve records the model only in its run
        log (`Initialized ... LLM ... with model: X`); fall back to the run dir name."""
        logs = self._logs_dir()
        if logs:
            log_files = sorted(logs.glob("*.log"))
            if log_files:
                try:
                    text = log_files[-1].read_text(errors="ignore")
                except OSError:
                    text = ""
                found: list[str] = []
                seen: set[str] = set()
                for m in re.finditer(r"[Ii]nitialized .*?LLM .*?models?:\s*([\w./:-]+)", text):
                    tok = m.group(1)
                    if tok not in seen:
                        seen.add(tok)
                        found.append(tok)
                if found:
                    return found
        return models_from_path(self.folder_path)

    def parse(self) -> ADRSParsedCampaign | None:
        """Parse all campaign data. Returns ADRSParsedCampaign or None."""
        if not self.folder_path.is_dir():
            print(f"Error: {self.folder_path} is not a directory")
            return None

        # Load author input (optional, contains system version). May sit at run root,
        # the results/ dir, or run root parent.
        for cand in (self.base, self.folder_path, self.base.parent):
            data = load_json(cand / "author_input.json", warn_on_error=False)
            if data:
                self.author_input = data
                break

        # Load evolution trace if present (OPTIONAL — an EvoTrace add-on, not native).
        self.evolution_trace = load_jsonl(self.base / "evolution_trace.jsonl")

        # Load best program info
        self.best_info = load_json(self.base / "best" / "best_program_info.json") or {}

        # Build program database from checkpoints
        self._build_program_database()

        if not self.program_db:
            print(f"Error: No programs found in checkpoints in {self.base}")
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
        checkpoints_dir = self.base / "checkpoints"
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

        # Get timestamps from the evolution trace when present, else from the programs.
        started_at = None
        ended_at = None
        if self.evolution_trace:
            started_at = parse_timestamp(self.evolution_trace[0].get("timestamp"))
            ended_at = parse_timestamp(self.evolution_trace[-1].get("timestamp"))
        else:
            prog_ts = [
                p.get("timestamp")
                for p in self.program_db.values()
                if p.get("timestamp") is not None
            ]
            if prog_ts:
                started_at = parse_timestamp(min(prog_ts))
                ended_at = parse_timestamp(max(prog_ts))

        # Total iterations (trace length when available, else number of programs)
        total_iterations = len(self.evolution_trace) or len(self.program_db)

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
            algorithm_used="openevolve",
            models_used=self._extract_models(),
            total_cost_usd=None,
            total_tokens=None,
            final_summary=None,
            final_metrics=final_metrics,
        )

    def _extract_research_question(self) -> str | None:
        """Research question, read verbatim from a declared `research_question` key in the
        run's research_question.yaml. OpenEvolve persists no config into the run dir, so this
        is author-supplied (run root, results/ dir, or run root parent). A dedicated filename
        (not config.yaml) is used so the author-supplied file can't trip SkyDiscover detection,
        which treats config.yaml as one of its markers."""
        for cand in (self.base, self.folder_path, self.base.parent):
            cfg = load_yaml(cand / "research_question.yaml")
            if isinstance(cfg, dict) and cfg.get("research_question"):
                return cfg["research_question"]
        return None

    def _extract_config(self) -> dict:
        """Extract configuration from prompts and metadata."""
        config = {
            "total_iterations": len(self.evolution_trace) or len(self.program_db),
        }

        # Get island count from checkpoint metadata
        checkpoints_dir = self.base / "checkpoints"
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
            if not isinstance(metrics, dict) or not metrics:
                continue

            candidate_measurements = []

            # Generic: emit every scalar metric the program reports (works for any
            # benchmark, not just a fixed list). One level of nesting (metrics.metrics)
            # is flattened; deeper structures / lists are skipped.
            def _is_scalar(v) -> bool:
                return isinstance(v, (int, float, str, bool))

            for name, value in metrics.items():
                if value is None:
                    continue
                if _is_scalar(value):
                    candidate_measurements.append(
                        ADRSMeasurement(name=name, value=str(value))
                    )
                elif isinstance(value, dict):
                    for nested_name, nested_value in value.items():
                        if nested_value is not None and _is_scalar(nested_value):
                            candidate_measurements.append(
                                ADRSMeasurement(
                                    name=nested_name, value=str(nested_value)
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
        best_program = self.base / "best" / "best_program.py"
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
        best_info = self.base / "best" / "best_program_info.json"
        if best_info.exists():
            artifacts.append(
                ADRSArtifact(
                    uri=str(best_info),
                    content_hash=compute_file_hash(best_info),
                    size_bytes=best_info.stat().st_size,
                    mime_type="application/json",
                )
            )

        # Evolution trace (optional; only present when EvoTrace instrumentation ran)
        trace_file = self.base / "evolution_trace.jsonl"
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
        logs_dir = self._logs_dir()
        if logs_dir:
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
        checkpoints_dir = self.base / "checkpoints"
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
        print(f"Models: {result.campaign.models_used}")
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
