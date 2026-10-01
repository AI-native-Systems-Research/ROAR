#!/usr/bin/env python3
"""Parser for SkyDiscover campaign outputs.

SkyDiscover uses an output structure:
- config.yaml: Campaign configuration
- output/summary.json: Final summary with best score
- output/best/: Best program found
- output/logs/: Execution logs
- output_old/checkpoints/: Checkpoints from previous runs (if available)
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import yaml

from adrs_models import (
    ADRSArtifact,
    ADRSCampaign,
    ADRSCandidate,
    ADRSCandidateEdge,
    ADRSMeasurement,
    ADRSParsedCampaign,
    ADRSSystem,
)

# SkyDiscover search types that re-implement another framework's mechanism, mapped to that
# mechanism's name so algorithm_used means the same thing across systems.
SEARCH_TYPE_TO_MECHANISM = {
    "openevolve_native": "openevolve",
    "gepa_native": "gepa",
    "shinkaevolve": "shinka",
}


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
        ".yaml": "application/x-yaml",
        ".yml": "application/x-yaml",
        ".json": "application/json",
        ".md": "text/markdown",
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


def load_yaml(file_path: Path) -> dict | None:
    """Load YAML file if it exists."""
    if not file_path.exists():
        return None
    try:
        with open(file_path) as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return None


class SkyDiscoverParser:
    """Parse a SkyDiscover campaign folder into ADRS models."""

    # Files/dirs a SkyDiscover run produces but OpenEvolve does not. Their presence
    # alongside a shared checkpoints/ dir is what identifies a SkyDiscover run.
    MARKERS = ("config.yaml", "skydiscover_config.yaml", "run_metadata.json", "output_old")

    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        self.output_dir: Path = folder_path / "output"  # may be overridden in parse()
        self.config: dict = {}
        self.summary: dict = {}
        self.run_metadata: dict = {}
        self.author_input: dict = {}
        self.program_db: dict[str, dict] = {}  # program_id -> program_info
        self.copy_to_original: dict[str, str] = {}  # copy_id -> original_id (migrations + spawn copies)

    @classmethod
    def _has_native_signal(cls, folder_path: Path) -> bool:
        """True for a *bare* SkyDiscover run (odellia-style: checkpoints/ at the run root,
        no output/ wrapper and none of MARKERS). Keyed on traits SkyDiscover emits that a
        native OpenEvolve run never does, so this can't steal an OpenEvolve run."""
        if (folder_path / "search").is_dir():  # evox backend
            return True
        if any(folder_path.glob("*_iteration_stats_*.jsonl")):  # adaevolve backend
            return True
        # Definitive discriminator: SkyDiscover program records key the source as
        # "solution"; OpenEvolve uses "code". Peek the first checkpoint program JSON.
        for pj in sorted((folder_path / "checkpoints").glob("*/programs/*.json")):
            if pj.name.startswith("._"):
                continue
            try:
                with open(pj) as f:
                    obj = json.load(f)
            except (OSError, ValueError):
                return False
            return isinstance(obj, dict) and "solution" in obj and "code" not in obj
        return False

    @classmethod
    def matches(cls, folder_path: Path) -> bool:
        """True if folder_path is a SkyDiscover run: an output/ wrapper (layout 1), or a
        root checkpoints/ dir plus either a SkyDiscover-specific marker file or a native
        SkyDiscover signal (layout 2, incl. bare dumps). The marker/signal requirement
        disambiguates from OpenEvolve, which also has checkpoints/ but neither. Used by
        system_detection."""
        if (folder_path / "output").is_dir():
            return True
        if (folder_path / "checkpoints").is_dir():
            return any((folder_path / m).exists() for m in cls.MARKERS) or cls._has_native_signal(folder_path)
        return False

    def parse(self) -> ADRSParsedCampaign | None:
        """Parse all campaign data. Returns ADRSParsedCampaign or None."""
        if not self.folder_path.is_dir():
            print(f"Error: {self.folder_path} is not a directory")
            return None

        # Load configuration (optional - not present in all layouts)
        # Try config.yaml first, then skydiscover_config.yaml
        self.config = load_yaml(self.folder_path / "config.yaml") or {}
        if not self.config:
            self.config = load_yaml(self.folder_path / "skydiscover_config.yaml") or {}

        # Load run metadata
        self.run_metadata = load_json(self.folder_path / "run_metadata.json") or {}

        # Load author input (optional, contains system version)
        self.author_input = load_json(self.folder_path / "author_input.json", warn_on_error=False) or {}

        # Determine output directory layout:
        # Layout 1: output/ subdirectory exists (gamble-data-transformed style)
        # Layout 2: checkpoints/, logs/ directly in root (odellia style, best/ optional)
        output_dir = self.folder_path / "output"
        if not output_dir.exists():
            # Check for direct layout (no output/ wrapper)
            if (self.folder_path / "checkpoints").exists():
                output_dir = self.folder_path
            else:
                print(f"Error: No output directory or checkpoints found in {self.folder_path}")
                return None

        self.output_dir = output_dir
        self.summary = load_json(output_dir / "summary.json") or {}

        # Build program database from all available sources
        self._build_program_database()

        # Parse campaign-level data
        campaign = self._parse_campaign()

        # Parse candidates and edges (no separate iterations table)
        candidates, edges = self._parse_candidates_and_edges()

        # Parse measurements
        measurements = self._parse_measurements(candidates)

        # Collect artifacts
        artifacts = self._collect_artifacts()

        return ADRSParsedCampaign(
            campaign=campaign,
            candidates=candidates,
            measurements=measurements,
            artifacts=artifacts,
            candidate_edges=edges,
        )

    def _build_program_database(self):
        """Build a database of all programs from checkpoints and best program.

        Checkpoint programs are loaded first (they have complete data including
        other_context_ids), then best_program_info is used as fallback only if
        the program wasn't found in checkpoints.

        Also builds copy_to_original map to resolve duplicate copies back
        to their original programs. Two types of copies are identified:
        - Migration copies: 'migrated_from' in metadata
        - Spawn copies: 'seeded_to_spawned_island' in metadata
        """
        output_dir = self.output_dir

        # Load from checkpoints FIRST (they have complete data including context)
        # (output_old if available, otherwise output_dir)
        for output_base in [self.folder_path / "output_old", output_dir]:
            checkpoints_dir = output_base / "checkpoints"
            if not checkpoints_dir.exists():
                continue

            for checkpoint_dir in sorted(checkpoints_dir.iterdir()):
                if not checkpoint_dir.is_dir():
                    continue

                # Extract checkpoint iteration number
                checkpoint_name = checkpoint_dir.name
                checkpoint_iter = None
                if checkpoint_name.startswith("checkpoint_"):
                    try:
                        checkpoint_iter = int(checkpoint_name.split("_")[1])
                    except (IndexError, ValueError):
                        pass

                # Load all program JSONs in this checkpoint
                programs_dir = checkpoint_dir / "programs"
                if programs_dir.exists():
                    for program_file in programs_dir.glob("*.json"):
                        if program_file.name.startswith("._"):
                            continue
                        program_info = load_json(program_file, warn_on_error=False)
                        if program_info and "id" in program_info:
                            prog_id = program_info["id"]
                            if prog_id not in self.program_db:
                                self.program_db[prog_id] = program_info
                        else:
                            # Create stub entry from filename for unparseable files
                            prog_id = program_file.stem
                            if prog_id not in self.program_db:
                                self.program_db[prog_id] = {
                                    "id": prog_id,
                                    "iteration": checkpoint_iter,
                                }

        # Load best program info as fallback (it may lack other_context_ids)
        best_info = load_json(output_dir / "best" / "best_program_info.json")
        if best_info and "id" in best_info:
            if best_info["id"] not in self.program_db:
                self.program_db[best_info["id"]] = best_info

        # Build copy_to_original map
        # Copies have their parent_id pointing to the original program.
        # Two types: migration copies ('migrated_from') and spawn copies ('seeded_to_spawned_island')
        for prog_id, program in self.program_db.items():
            metadata = program.get("metadata", {})
            is_copy = (
                metadata.get("migrated_from") is not None
                or metadata.get("seeded_to_spawned_island") is not None
            )
            if is_copy:
                original_id = program.get("parent_id")
                if original_id:
                    self.copy_to_original[prog_id] = original_id

    def _resolve_through_copies(self, program_id: str | None) -> str | None:
        """Resolve a program ID through copy chains to find the original.

        Both migration and spawn copies point to their source via parent_id.
        This follows that chain until we find a non-copied program.
        """
        if program_id is None:
            return None
        visited = set()
        while program_id in self.copy_to_original:
            if program_id in visited:
                break  # Cycle detection
            visited.add(program_id)
            program_id = self.copy_to_original[program_id]
        return program_id

    def _parse_campaign(self) -> ADRSCampaign:
        """Parse campaign-level information."""
        # Generate name from folder name or summary
        name = self.summary.get("output_dir", str(self.folder_path))
        name = Path(name).name if name else self.folder_path.name

        # Determine start/end times from logs
        started_at, ended_at = self._extract_timestamps_from_logs()

        # Research question: try config first, then extract from logs
        research_question = self.config.get("research_question")
        if not research_question:
            research_question = self._extract_research_question_from_logs()

        # Store the entire config
        config_used = self.config if self.config else None

        # Extract final metrics from summary
        final_metrics = None
        if self.summary:
            final_metrics = {
                "best_score": self.summary.get("best_score"),
                "initial_score": self.summary.get("initial_score"),
                "improvement_percent": self.summary.get("improvement_percent"),
            }
            metrics = self.summary.get("metrics", {})
            if metrics:
                final_metrics.update(metrics)

        # Detect algorithm from config, run_metadata, or logs
        algorithm = self.config.get("search", {}).get("type")
        if not algorithm:
            algorithm = self.run_metadata.get("search")
        if not algorithm:
            algorithm = self._extract_algorithm_from_logs()
        # The system field already distinguishes SkyDiscover from standalone OpenEvolve, so
        # SkyDiscover's port of a search is recorded under the mechanism's shared name. The
        # verbatim search.type is still kept in config_used.
        algorithm = SEARCH_TYPE_TO_MECHANISM.get(algorithm, algorithm)

        # Extract models used from config
        models_used = self._extract_models_used()

        # System version from author_input.json if available
        system_version = self.author_input.get("version")

        system = ADRSSystem(
            name="skydiscover",
            version=system_version,
        )

        return ADRSCampaign(
            system=system,
            author=self.author_input.get("author", ""),
            name=name,
            research_question=research_question,
            started_at=started_at,
            ended_at=ended_at,
            config_used=config_used,
            algorithm_used=algorithm,
            models_used=models_used,
            total_cost_usd=None,
            total_tokens=None,
            final_summary=None,
            final_metrics=final_metrics,
        )

    def _extract_models_used(self) -> list[str] | None:
        """Extract model names from SkyDiscover config.

        SkyDiscover configs typically have model info in:
        - llm.model or llm.name
        - llm.provider + llm.model combined
        - generator.model, evaluator.model for multi-model setups
        """
        models: set[str] = set()

        def extract_from_dict(d: dict, depth: int = 0):
            """Recursively extract model names from a dict."""
            if depth > 5:
                return
            for key, value in d.items():
                # Common keys that hold model names
                if key in ("model", "model_name", "model_id", "name") and isinstance(value, str) and value:
                    # Skip if it looks like a project name or path
                    if "/" not in value or value.count("/") == 1:
                        models.add(value)
                elif isinstance(value, dict):
                    extract_from_dict(value, depth + 1)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            extract_from_dict(item, depth + 1)

        if self.config:
            extract_from_dict(self.config)

        return sorted(models) if models else None

    def _extract_research_question_from_logs(self) -> str | None:
        """Extract research question from log files when config.yaml is not available.

        Looks for evaluator class names like 'LayerNormEvaluator' and converts to
        a human-readable research question.
        """
        logs_dir = self.output_dir / "logs"
        if not logs_dir.exists():
            return None

        log_files = sorted(logs_dir.glob("*.log"))
        if not log_files:
            return None

        try:
            with open(log_files[0]) as f:
                # Only check first 100 lines for performance
                for i, line in enumerate(f):
                    if i > 100:
                        break
                    # Look for evaluator initialization pattern
                    match = re.search(r"Initialized.*?for\s+(\w+Evaluator)", line)
                    if match:
                        evaluator_name = match.group(1)
                        # Convert CamelCase to readable: LayerNormEvaluator -> Layer Norm
                        name = re.sub(r"Evaluator$", "", evaluator_name)
                        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
                        return f"CUDA {name} kernel optimization"
        except OSError:
            pass

        return None

    def _extract_algorithm_from_logs(self) -> str | None:
        """Extract search algorithm from log files.

        Looks for 'Runner ready: search=...' pattern.
        """
        logs_dir = self.output_dir / "logs"
        if not logs_dir.exists():
            return None

        log_files = sorted(logs_dir.glob("*.log"))
        if not log_files:
            return None

        try:
            with open(log_files[0]) as f:
                for i, line in enumerate(f):
                    if i > 100:
                        break
                    # Look for "Runner ready: search=adaevolve" or similar
                    match = re.search(r"Runner ready:\s*search=(\w+)", line)
                    if match:
                        return match.group(1)
        except OSError:
            pass

        return None

    def _extract_timestamps_from_logs(self) -> tuple[datetime | None, datetime | None]:
        """Extract start and end times from log files."""
        logs_dir = self.output_dir / "logs"

        if not logs_dir.exists():
            return None, None

        log_files = sorted(logs_dir.glob("*.log"))
        if not log_files:
            return None, None

        log_file = log_files[-1]

        started_at = None
        ended_at = None

        try:
            with open(log_file) as f:
                lines = f.readlines()

                if lines:
                    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+", lines[0])
                    if match:
                        started_at = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")

                if lines:
                    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+", lines[-1])
                    if match:
                        ended_at = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        except (OSError, ValueError):
            pass

        return started_at, ended_at

    def _parse_candidates_and_edges(
        self,
    ) -> tuple[list[ADRSCandidate], list[ADRSCandidateEdge]]:
        """Parse candidates and edges from program database.

        Creates one candidate per unique program. iteration_index indicates when
        the program was first discovered (iteration_found). Iteration 0 contains
        seed programs (initial population before search begins).

        Copies (programs with 'migrated_from' or 'seeded_to_spawned_island' in
        metadata) are skipped since they duplicate existing programs. References
        to copies in parent_id and other_context_ids are resolved to the original.

        Edges capture:
        - parent_id: which program was mutated (edge_type='parent')
        - other_context_ids: programs shown as history in BoN (edge_type='context')
        """
        candidates: list[ADRSCandidate] = []
        edges: list[ADRSCandidateEdge] = []

        # Create one candidate per unique program (skip copies)
        for program in self.program_db.values():
            prog_id = program["id"]

            # Skip migration/spawn copies - they duplicate existing programs
            if prog_id in self.copy_to_original:
                continue

            iter_idx = program.get("iteration_found", program.get("iteration", 0))
            timestamp = parse_timestamp(program.get("timestamp"))

            candidate = ADRSCandidate(
                iteration_index=iter_idx,
                external_id=prog_id,
                candidate_type="program",
                created_at=timestamp,
            )
            candidates.append(candidate)

            # Create edge from parent if available (edge_type='parent')
            # Resolve through copies to find the original program
            parent_id = self._resolve_through_copies(program.get("parent_id"))
            if parent_id and parent_id in self.program_db and parent_id not in self.copy_to_original:
                edges.append(
                    ADRSCandidateEdge(
                        source_external_id=parent_id,
                        target_external_id=prog_id,
                        edge_type="parent",
                    )
                )

            # Create edges from context programs (edge_type='context')
            # Resolve through copies to find the original programs
            context_ids = program.get("other_context_ids") or []
            for context_id in context_ids:
                resolved_id = self._resolve_through_copies(context_id)
                if resolved_id and resolved_id in self.program_db and resolved_id not in self.copy_to_original:
                    edges.append(
                        ADRSCandidateEdge(
                            source_external_id=resolved_id,
                            target_external_id=prog_id,
                            edge_type="context",
                        )
                    )

        # Sort by iteration_index for consistent ordering
        candidates.sort(key=lambda c: (c.iteration_index, c.external_id or ""))

        return candidates, edges

    def _parse_measurements(
        self, candidates: list[ADRSCandidate]
    ) -> dict[str, list[ADRSMeasurement]]:
        """Parse measurements from candidates - keyed by candidate external_id."""
        measurements: dict[str, list[ADRSMeasurement]] = {}

        for candidate in candidates:
            external_id = candidate.external_id
            if not external_id:
                continue

            # Get program info from database
            program = self.program_db.get(external_id, {})
            program_metrics = program.get("metrics", {})

            if external_id not in measurements:
                measurements[external_id] = []

            # Determine status from validity
            validity = program_metrics.get("validity", 1.0)
            status = "VALID" if validity == 1.0 else "INVALID"

            measurements[external_id].append(
                ADRSMeasurement(name="status", value=status)
            )

            if status == "INVALID":
                measurements[external_id].append(
                    ADRSMeasurement(name="error_type", value="invalid_program")
                )

            # Add combined score as conclusion
            combined_score = program_metrics.get("combined_score", 0)
            measurements[external_id].append(
                ADRSMeasurement(name="conclusion", value=f"Combined score: {combined_score:.2f}")
            )

            # Add result summary
            result_summary = ", ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in program_metrics.items()
            )
            measurements[external_id].append(
                ADRSMeasurement(name="result_summary", value=result_summary)
            )

            # Add each metric as a measurement
            for metric_name, value in program_metrics.items():
                if isinstance(value, (int, float)):
                    measurements[external_id].append(
                        ADRSMeasurement(name=metric_name, value=str(value))
                    )

        return measurements

    def _collect_artifacts(self) -> list[ADRSArtifact]:
        """Collect all artifacts from the campaign."""
        artifacts: list[ADRSArtifact] = []

        # Campaign-level config (if present)
        config_path = self.folder_path / "config.yaml"
        if config_path.exists():
            artifacts.append(
                ADRSArtifact(
                    iteration_index=None,
                    uri=str(config_path),
                    content_hash=compute_file_hash(config_path),
                    size_bytes=config_path.stat().st_size,
                    mime_type=get_mime_type(config_path),
                )
            )

        # Output artifacts
        output_dir = self.output_dir
        if output_dir.exists():
            # Summary
            summary_file = output_dir / "summary.json"
            if summary_file.exists():
                artifacts.append(
                    ADRSArtifact(
                        iteration_index=None,
                        uri=str(summary_file),
                        content_hash=compute_file_hash(summary_file),
                        size_bytes=summary_file.stat().st_size,
                        mime_type="application/json",
                    )
                )

            # Best program files
            best_dir = output_dir / "best"
            if best_dir.exists():
                for item in best_dir.iterdir():
                    if item.is_file():
                        artifacts.append(
                            ADRSArtifact(
                                iteration_index=None,
                                uri=str(item),
                                content_hash=compute_file_hash(item),
                                size_bytes=item.stat().st_size,
                                mime_type=get_mime_type(item),
                            )
                        )

            # Log files
            logs_dir = output_dir / "logs"
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    artifacts.append(
                        ADRSArtifact(
                            iteration_index=None,
                            uri=str(log_file),
                            content_hash=compute_file_hash(log_file),
                            size_bytes=log_file.stat().st_size,
                            mime_type="text/plain",
                        )
                    )

        # Checkpoint program files (solution code for each candidate)
        # Only keep first occurrence per external_id to avoid duplicates from cumulative checkpoints
        seen_program_ids: set[str] = set()
        for output_base in [self.folder_path / "output_old", output_dir]:
            checkpoints_dir = output_base / "checkpoints"
            if not checkpoints_dir.exists():
                continue
            for checkpoint_dir in sorted(checkpoints_dir.iterdir()):
                if not checkpoint_dir.is_dir():
                    continue
                programs_dir = checkpoint_dir / "programs"
                if not programs_dir.exists():
                    continue
                for program_file in programs_dir.glob("*.json"):
                    if program_file.name.startswith("._"):
                        continue
                    external_id = program_file.stem
                    if external_id in seen_program_ids:
                        continue
                    seen_program_ids.add(external_id)
                    program_info = self.program_db.get(external_id, {})
                    iteration_index = program_info.get("iteration_found", program_info.get("iteration"))
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


def parse_skydiscover_campaign(folder_path: Path) -> ADRSParsedCampaign | None:
    """Parse a SkyDiscover campaign into ADRS models.

    Args:
        folder_path: Path to the campaign directory

    Returns:
        ADRSParsedCampaign or None if parsing fails
    """
    parser = SkyDiscoverParser(folder_path)
    return parser.parse()


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse SkyDiscover campaign into ADRS models (test mode)"
    )
    parser.add_argument("path", type=Path, help="Path to campaign folder")
    args = parser.parse_args()

    parsed = parse_skydiscover_campaign(args.path)
    if parsed:
        print(f"\nSuccessfully parsed campaign: {parsed.campaign.name}")
        num_iterations = len(set(c.iteration_index for c in parsed.candidates))
        print(f"  Iterations: {num_iterations}")
        print(f"  Candidates: {len(parsed.candidates)}")
        total_measurements = sum(len(m) for m in parsed.measurements.values())
        print(f"  Measurements: {total_measurements} (for {len(parsed.measurements)} candidates)")
        print(f"  Artifacts: {len(parsed.artifacts)}")
        print(f"  Edges: {len(parsed.candidate_edges)}")
    else:
        print("Failed to parse campaign")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
