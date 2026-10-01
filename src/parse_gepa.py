#!/usr/bin/env python3
"""Parser for GEPA (Genetic-Pareto reflective optimization) NATIVE output.

GEPA (github.com/gepa-ai/gepa) optimizes prompts/programs by reflective evolution.
`optimize()` returns a `GEPAResult`; runs persist a `gepa_state.bin` checkpoint and JSON
artifacts. This parser reads the serialized `GEPAResult` (the corpus dumps it as
`gepa_result.json`; GEPA's own filenames `candidates.json` / `run_log.json` are accepted as
a fallback).

Native `GEPAResult` fields used:
- candidates:            list; candidates[i] is a dict of named text components, e.g. {"program": code}
- parents:               list; parents[i] is a list of parent indices (or [None] for a root) -> lineage
- val_aggregate_scores:  list; per-candidate aggregate score on the validation set
- val_subscores:         list; per-candidate list of per-instance validation scores
- best_idx:              int; index of the best candidate
- total_metric_calls, num_full_val_evals, run_dir, seed: run-level bookkeeping

On-disk layout (per run):
- results/gepa_result.json        (serialized GEPAResult; primary input)
- results/candidates/candidate_NNN.py  (mirror of each candidate's program code)
- results/best_program.py, results/gepa_state.bin, results/test_metrics.json
- logs/                           (run logs; sibling of results/)

Note: the earlier curated (non-native) GEPA bundle parser now lives at
`oneoff_parsers/parse_gepa_curated.py`.
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
    """Parse a timestamp (Unix epoch or ISO string) to datetime."""
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts)
        if isinstance(ts, str):
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            return datetime.fromisoformat(ts)
    except (ValueError, TypeError, OSError):
        return None
    return None


def compute_file_hash(file_path: Path) -> str | None:
    """Compute SHA256 hash of a file."""
    if not file_path.exists() or not file_path.is_file():
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
        ".bin": "application/octet-stream",
        ".log": "text/plain",
        ".txt": "text/plain",
    }
    return mime_map.get(suffix, "application/octet-stream")


def load_json(file_path: Path, warn_on_error: bool = True) -> dict | list | None:
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


# Marker files that identify a native GEPA run (any one present at the run root or
# under results/). Shared by _resolve_base and matches() so the two can't drift.
GEPA_MARKERS = ("gepa_result.json", "candidates.json", "gepa_state.bin", "best_program.py")


class GEPAParser:
    """Parse a native GEPA run folder into ADRS models."""

    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        # `base` is the directory that actually holds the GEPA result artifacts. The corpus
        # nests them under results/, so accept either the run root or the results/ dir.
        self.base = self._resolve_base(folder_path)
        self.result: dict = {}
        self.author_input: dict = {}
        # Minimal mode: some runs never dumped gepa_result.json and only left a
        # gepa_state.bin checkpoint + best_program.py (+ test_metrics.json). We still ingest
        # the winner + final metrics, without per-candidate lineage.
        self.minimal = False

    @staticmethod
    def _resolve_base(folder_path: Path) -> Path:
        """Find the dir holding the GEPA artifacts (run root or a results/ subdir)."""
        for name in GEPA_MARKERS:
            if (folder_path / name).exists():
                return folder_path
            if (folder_path / "results" / name).exists():
                return folder_path / "results"
        return folder_path

    @classmethod
    def matches(cls, folder_path: Path) -> bool:
        """True if folder_path is a native GEPA run (any GEPA_MARKERS file at the run
        root or under results/). Used by system_detection."""
        base = cls._resolve_base(folder_path)
        return any((base / name).exists() for name in GEPA_MARKERS)

    def _logs_dir(self) -> Path | None:
        """Locate the run's logs/ dir (sibling of results/, or under the run root)."""
        for cand in (self.base.parent / "logs", self.folder_path / "logs", self.base / "logs"):
            if cand.is_dir():
                return cand
        return None

    def _extract_research_question(self) -> str | None:
        """Research question, read verbatim from a declared `research_question` key in the
        run's research_question.yaml. GEPA persists no config of its own, so this is
        author-supplied (run root, results/ dir, or run root parent). A dedicated filename
        (not config.yaml) is used so the author-supplied file can't trip SkyDiscover detection,
        which treats config.yaml as one of its markers."""
        for cand in (self.base, self.folder_path, self.base.parent):
            cfg = load_yaml(cand / "research_question.yaml")
            if isinstance(cfg, dict) and cfg.get("research_question"):
                return cfg["research_question"]
        return None

    def _extract_models(self) -> list[str] | None:
        """Model(s) for the run, verbatim: read from the GEPA result if present, else
        fall back to the run dir name."""
        for key in ("model", "model_name", "models", "llm_model"):
            v = self.result.get(key)
            if isinstance(v, str) and v:
                return [v]
            if isinstance(v, list) and v:
                return [str(x) for x in v]
        return models_from_path(self.folder_path)

    def parse(self) -> ADRSParsedCampaign | None:
        """Parse all campaign data. Returns ADRSParsedCampaign or None."""
        if not self.folder_path.is_dir():
            print(f"Error: {self.folder_path} is not a directory")
            return None

        # author_input.json may sit at the run root, the results/ dir, or run root parent.
        for cand in (self.base, self.folder_path, self.base.parent):
            data = load_json(cand / "author_input.json", warn_on_error=False)
            if data:
                self.author_input = data
                break

        if not self._load_result():
            return None

        if self.minimal:
            candidates = self._minimal_candidates()
            edges = []
            measurements = self._minimal_measurements()
            campaign = self._minimal_campaign()
        else:
            candidates = self._parse_candidates()
            edges = self._parse_edges()
            measurements = self._parse_measurements()
            campaign = self._parse_campaign(len(candidates))
        artifacts = self._collect_artifacts()

        return ADRSParsedCampaign(
            campaign=campaign,
            candidates=candidates,
            measurements=measurements,
            artifacts=artifacts,
            candidate_edges=edges,
        )

    def _load_result(self) -> bool:
        """Load the serialized GEPAResult (gepa_result.json, or candidates.json fallback)."""
        result = load_json(self.base / "gepa_result.json")
        if isinstance(result, dict) and result.get("candidates"):
            self.result = result
            return True

        # Fallback: GEPA's own native filenames.
        candidates = load_json(self.base / "candidates.json", warn_on_error=False)
        if isinstance(candidates, list) and candidates:
            run_log = load_json(self.base / "run_log.json", warn_on_error=False)
            self.result = {"candidates": candidates}
            if isinstance(run_log, dict):
                for key in (
                    "parents",
                    "val_aggregate_scores",
                    "val_subscores",
                    "best_idx",
                ):
                    if key in run_log:
                        self.result[key] = run_log[key]
            return True

        # Minimal mode: no serialized result, but a best program / checkpoint exists.
        if (self.base / "best_program.py").exists() or (self.base / "gepa_state.bin").exists():
            self.minimal = True
            self.result = {}
            return True

        print(f"Error: No gepa_result.json / candidates.json / best_program.py found in {self.base}")
        return False

    @staticmethod
    def _candidate_code(candidate) -> str | None:
        """Extract program text from a candidate dict of named components."""
        if isinstance(candidate, dict):
            if "program" in candidate:
                return candidate["program"]
            # Multi-component candidate: join the text pieces.
            parts = [f"### {k}\n{v}" for k, v in candidate.items() if isinstance(v, str)]
            return "\n\n".join(parts) if parts else json.dumps(candidate)
        if isinstance(candidate, str):
            return candidate
        return None

    def _parse_candidates(self) -> list[ADRSCandidate]:
        """One candidate per entry in candidates[]."""
        candidates: list[ADRSCandidate] = []
        for i, _cand in enumerate(self.result.get("candidates", [])):
            candidates.append(
                ADRSCandidate(
                    iteration_index=i,
                    external_id=f"cand-{i:03d}",
                    candidate_type="program",
                )
            )
        return candidates

    def _parse_edges(self) -> list[ADRSCandidateEdge]:
        """Lineage from parents[]: parents[i] is a list of parent indices (or [None])."""
        edges: list[ADRSCandidateEdge] = []
        parents = self.result.get("parents") or []
        n = len(self.result.get("candidates", []))
        for i, parent_list in enumerate(parents):
            if parent_list is None:
                continue
            if not isinstance(parent_list, list):
                parent_list = [parent_list]
            for p in parent_list:
                if p is None or not isinstance(p, int):
                    continue
                if 0 <= p < n and p != i:
                    edges.append(
                        ADRSCandidateEdge(
                            source_external_id=f"cand-{p:03d}",
                            target_external_id=f"cand-{i:03d}",
                            edge_type="parent",
                        )
                    )
        return edges

    def _parse_measurements(self) -> dict[str, list[ADRSMeasurement]]:
        """Per-candidate: aggregate validation score, is_best flag, #val instances."""
        measurements: dict[str, list[ADRSMeasurement]] = {}
        agg = self.result.get("val_aggregate_scores") or []
        sub = self.result.get("val_subscores") or []
        best_idx = self.result.get("best_idx")
        n = len(self.result.get("candidates", []))

        for i in range(n):
            ext = f"cand-{i:03d}"
            meas: list[ADRSMeasurement] = []
            if i < len(agg) and agg[i] is not None:
                meas.append(ADRSMeasurement(name="combined_score", value=str(agg[i])))
            if i < len(sub) and isinstance(sub[i], list):
                meas.append(
                    ADRSMeasurement(name="val_instance_count", value=str(len(sub[i])))
                )
            if best_idx is not None and i == best_idx:
                meas.append(ADRSMeasurement(name="is_best", value="true"))
            if meas:
                measurements[ext] = meas
        return measurements

    def _parse_campaign(self, num_candidates: int) -> ADRSCampaign:
        """Campaign-level metadata from the GEPAResult bookkeeping fields."""
        agg = self.result.get("val_aggregate_scores") or []
        best_idx = self.result.get("best_idx")
        best_score = None
        if isinstance(best_idx, int) and 0 <= best_idx < len(agg):
            best_score = agg[best_idx]

        config_used = {
            "run_dir": self.result.get("run_dir"),
            "seed": self.result.get("seed"),
            "total_metric_calls": self.result.get("total_metric_calls"),
            "num_full_val_evals": self.result.get("num_full_val_evals"),
            "num_candidates": num_candidates,
        }
        final_metrics = {
            "best_idx": best_idx,
            "best_score": best_score,
            "num_candidates": num_candidates,
        }

        system = ADRSSystem(name="gepa", version=self.author_input.get("version"))

        return ADRSCampaign(
            system=system,
            author=self.author_input.get("author", ""),
            name=self.folder_path.name,
            research_question=self._extract_research_question(),
            started_at=None,
            ended_at=None,
            config_used=config_used,
            algorithm_used="gepa",
            models_used=self._extract_models(),
            total_cost_usd=None,
            total_tokens=None,
            final_summary=None,
            final_metrics=final_metrics,
        )

    # --- Minimal mode (no gepa_result.json; only best_program.py / gepa_state.bin) ---

    def _minimal_candidates(self) -> list[ADRSCandidate]:
        """A single 'best' candidate (the winner); no lineage is recoverable."""
        return [
            ADRSCandidate(
                iteration_index=0,
                external_id="best",
                candidate_type="program",
            )
        ]

    def _minimal_measurements(self) -> dict[str, list[ADRSMeasurement]]:
        """Attach scalar entries from test_metrics.json to the 'best' candidate."""
        tm = load_json(self.base / "test_metrics.json", warn_on_error=False)
        meas: list[ADRSMeasurement] = []
        if isinstance(tm, dict):
            for name, value in tm.items():
                if isinstance(value, (int, float, bool, str)):
                    meas.append(ADRSMeasurement(name=name, value=str(value)))
        return {"best": meas} if meas else {}

    def _minimal_campaign(self) -> ADRSCampaign:
        """Campaign metadata for a run with no serialized GEPAResult."""
        tm = load_json(self.base / "test_metrics.json", warn_on_error=False) or {}
        best_score = tm.get("optimized_test_score")
        if best_score is None:
            best_score = tm.get("optimized_test_score_per_run")

        system = ADRSSystem(name="gepa", version=self.author_input.get("version"))
        return ADRSCampaign(
            system=system,
            author=self.author_input.get("author", ""),
            name=self.folder_path.name,
            research_question=self._extract_research_question(),
            started_at=None,
            ended_at=None,
            config_used={"result_available": False},
            algorithm_used="gepa",
            models_used=self._extract_models(),
            total_cost_usd=None,
            total_tokens=None,
            final_summary=None,
            final_metrics={
                "best_score": best_score,
                "num_candidates": 1,
                "result_available": False,
            },
        )

    def _collect_artifacts(self) -> list[ADRSArtifact]:
        """Collect the result JSON, candidate code files, best program, state, logs."""
        artifacts: list[ADRSArtifact] = []

        def add(path: Path, iteration_index=None, external_id=None):
            if path.is_file() and not path.name.startswith("."):
                artifacts.append(
                    ADRSArtifact(
                        iteration_index=iteration_index,
                        external_id=external_id,
                        uri=str(path),
                        content_hash=compute_file_hash(path),
                        size_bytes=path.stat().st_size,
                        mime_type=get_mime_type(path),
                    )
                )

        for name in ("gepa_result.json", "gepa_state.bin", "test_metrics.json", "best_program.py"):
            add(self.base / name)

        # Candidate program files (candidate_NNN.py), linked to their candidate by index.
        candidates_dir = self.base / "candidates"
        if candidates_dir.is_dir():
            for code_file in sorted(candidates_dir.glob("candidate_*.py")):
                m = re.search(r"candidate_(\d+)", code_file.name)
                idx = int(m.group(1)) if m else None
                ext = f"cand-{idx:03d}" if idx is not None else None
                add(code_file, iteration_index=idx, external_id=ext)

        logs_dir = self._logs_dir()
        if logs_dir:
            for log_file in sorted(logs_dir.glob("*")):
                add(log_file)

        return artifacts


def parse_gepa_campaign(folder_path: Path) -> ADRSParsedCampaign | None:
    """Parse a native GEPA run into ADRS models.

    Args:
        folder_path: Path to the run directory (run root or its results/ subdir).

    Returns:
        ADRSParsedCampaign or None if parsing fails.
    """
    parser = GEPAParser(folder_path)
    return parser.parse()


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse a native GEPA run into ADRS models (test mode)"
    )
    parser.add_argument("path", type=Path, help="Path to run folder (root or results/)")
    args = parser.parse_args()

    parsed = parse_gepa_campaign(args.path)
    if parsed:
        print(f"\nSuccessfully parsed campaign: {parsed.campaign.name}")
        print(f"  System: {parsed.campaign.system.name}")
        print(f"  Models: {parsed.campaign.models_used}")
        print(f"  Candidates: {len(parsed.candidates)}")
        total_measurements = sum(len(m) for m in parsed.measurements.values())
        print(
            f"  Measurements: {total_measurements} (for {len(parsed.measurements)} candidates)"
        )
        print(f"  Artifacts: {len(parsed.artifacts)}")
        print(f"  Edges: {len(parsed.candidate_edges)} (parent lineage, multi-parent capable)")
        if parsed.campaign.final_metrics:
            print(f"  Best idx: {parsed.campaign.final_metrics.get('best_idx')}")
            print(f"  Best score: {parsed.campaign.final_metrics.get('best_score')}")
    else:
        print("Failed to parse campaign")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
