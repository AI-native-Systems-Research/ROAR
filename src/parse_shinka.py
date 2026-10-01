#!/usr/bin/env python3
"""Parser for ShinkaEvolve NATIVE output.

ShinkaEvolve (github.com/SakanaAI/ShinkaEvolve) does sample-efficient program evolution
with islands and an archive of good solutions. Its `shinka/database/dbase.py` writes a
SQLite database (the source of truth) whose `programs` table stores every program with its
code, direct `parent_id`, the "inspiration" programs that also informed it
(`archive_inspiration_ids`, `top_k_inspiration_ids`), generation, island, scores, metrics,
and whether it passed. An `archive` table lists the kept programs; `metadata_store` holds
run-level key/values.

Native structure (per run; the DB & friends may sit directly under the run dir OR nested
under a `results/` subdir — both are handled):
- evolution_db.sqlite            (SQLite; primary input)
- gen_N/{main.py, original.py, edit.diff, search_replace.txt, results/}
- meta_memory.json, experiment_config.yaml, evolution_run.log
- logs/                          (run logs; sibling of results/)
"""

import hashlib
import json
import re
import sqlite3
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

DB_NAME = "evolution_db.sqlite"


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
        ".sqlite": "application/vnd.sqlite3",
        ".diff": "text/x-diff",
        ".log": "text/plain",
        ".txt": "text/plain",
    }
    return mime_map.get(suffix, "application/octet-stream")


def _json_or_none(text):
    """Parse a JSON string field; return None on failure."""
    if not text or not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _is_scalar(v) -> bool:
    return isinstance(v, (int, float, bool))


class ShinkaParser:
    """Parse a native ShinkaEvolve run folder into ADRS models."""

    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        self.base = self._resolve_base(folder_path)
        self.db_path = self.base / DB_NAME
        self.author_input: dict = {}
        self.programs: list[dict] = []
        self.archive_ids: set[str] = set()
        self.config: dict = {}

    @staticmethod
    def _resolve_base(folder_path: Path) -> Path:
        """Find the dir containing evolution_db.sqlite (run root or a results/ subdir)."""
        if (folder_path / DB_NAME).exists():
            return folder_path
        if (folder_path / "results" / DB_NAME).exists():
            return folder_path / "results"
        return folder_path

    @classmethod
    def matches(cls, folder_path: Path) -> bool:
        """True if folder_path is a native ShinkaEvolve run (has evolution_db.sqlite
        at the run root or under results/). Used by system_detection."""
        return (cls._resolve_base(folder_path) / DB_NAME).exists()

    def _logs_dir(self) -> Path | None:
        for cand in (self.base.parent / "logs", self.folder_path / "logs", self.base / "logs"):
            if cand.is_dir():
                return cand
        return None

    def parse(self) -> ADRSParsedCampaign | None:
        """Parse all campaign data. Returns ADRSParsedCampaign or None."""
        if not self.folder_path.is_dir():
            print(f"Error: {self.folder_path} is not a directory")
            return None

        for cand in (self.base, self.folder_path, self.base.parent):
            data = _json_or_none(_read_text(cand / "author_input.json"))
            if data:
                self.author_input = data
                break

        if not self.db_path.exists():
            print(f"Error: No {DB_NAME} found in {self.base}")
            return None

        if not self._load_db():
            return None

        self.config = self._load_config()

        candidates = self._parse_candidates()
        edges = self._parse_edges()
        measurements = self._parse_measurements()
        campaign = self._parse_campaign()
        artifacts = self._collect_artifacts()

        return ADRSParsedCampaign(
            campaign=campaign,
            candidates=candidates,
            measurements=measurements,
            artifacts=artifacts,
            candidate_edges=edges,
        )

    def _load_db(self) -> bool:
        """Load the programs + archive tables from the SQLite DB (read-only)."""
        try:
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        except sqlite3.Error as e:
            print(f"Error: Could not open {self.db_path}: {e}")
            return False
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM programs ORDER BY generation, timestamp, id"
            ).fetchall()
            self.programs = [dict(r) for r in rows]
            try:
                arch = conn.execute("SELECT program_id FROM archive").fetchall()
                self.archive_ids = {r[0] for r in arch}
            except sqlite3.Error:
                self.archive_ids = set()
        except sqlite3.Error as e:
            print(f"Error: Could not read programs table in {self.db_path}: {e}")
            return False
        finally:
            conn.close()

        if not self.programs:
            print(f"Error: programs table is empty in {self.db_path}")
            return False
        return True

    def _load_config(self) -> dict:
        """Load experiment_config.yaml if possible (yaml optional; regex fallback)."""
        cfg_path = self.base / "experiment_config.yaml"
        text = _read_text(cfg_path)
        if not text:
            return {}
        try:
            import yaml  # optional dependency

            data = yaml.safe_load(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            # Minimal fallback: pull the fields we care about. Needed for OmegaConf-serialized
            # configs (`!!python/object:` tags) that safe_load can't construct.
            cfg: dict = {}
            m = re.search(r"num_islands:\s*(\d+)", text)
            if m:
                cfg["num_islands"] = int(m.group(1))
            # research_question is a top-level author-supplied scalar; recover it directly so a
            # required field isn't lost just because the rest of the config is untagged-unreadable.
            rq = re.search(r"^research_question:\s*(.+?)\s*$", text, re.MULTILINE)
            if rq:
                cfg["research_question"] = rq.group(1).strip().strip("\"'")
            # `llm_models` may be a plain YAML list, or an OmegaConf-serialized block
            # (`llm_models: &id !!python/object:...` with names nested as `_val: <model>`).
            lm = re.search(r"llm_models:\s*\n((?:\s*-\s*.+\n)+)", text)
            if lm:
                cfg["llm_models"] = re.findall(r"-\s*([\w./:-]+)", lm.group(1))
            else:
                # Isolate the top-level `llm_models:` block (excludes meta_/novelty_llm_models
                # by anchoring the key at its own indent) and read every nested `_val:`.
                block = re.search(
                    r"^(?P<ind> *)llm_models:.*\n(?P<body>(?:(?P=ind) +.*\n|[ \t]*\n)*)",
                    text,
                    re.MULTILINE,
                )
                vals = re.findall(r"_val:\s*([\w./:-]+)", block.group("body")) if block else []
                if vals:
                    cfg["llm_models"] = vals
            return cfg

    def _config_get(self, *keys):
        """Fetch a key from config, searching nested dicts one level deep."""
        for k in keys:
            if k in self.config:
                return self.config[k]
        for v in self.config.values():
            if isinstance(v, dict):
                for k in keys:
                    if k in v:
                        return v[k]
        return None

    def _parse_candidates(self) -> list[ADRSCandidate]:
        """One candidate per program row (ordered by generation, timestamp)."""
        candidates: list[ADRSCandidate] = []
        for idx, prog in enumerate(self.programs):
            candidates.append(
                ADRSCandidate(
                    iteration_index=idx,
                    external_id=prog.get("id"),
                    candidate_type="program",
                    created_at=parse_timestamp(prog.get("timestamp")),
                )
            )
        return candidates

    def _parse_edges(self) -> list[ADRSCandidateEdge]:
        """`parent` edges from parent_id; `context` edges from inspiration id lists."""
        edges: list[ADRSCandidateEdge] = []
        valid_ids = {p.get("id") for p in self.programs}

        for prog in self.programs:
            pid = prog.get("id")
            parent_id = prog.get("parent_id")
            if parent_id and parent_id in valid_ids and parent_id != pid:
                edges.append(
                    ADRSCandidateEdge(
                        source_external_id=parent_id,
                        target_external_id=pid,
                        edge_type="parent",
                    )
                )

            context_ids: set[str] = set()
            for field in ("archive_inspiration_ids", "top_k_inspiration_ids"):
                ids = _json_or_none(prog.get(field)) or []
                if isinstance(ids, list):
                    context_ids.update(x for x in ids if isinstance(x, str))
            for cid in context_ids:
                if cid in valid_ids and cid != pid and cid != parent_id:
                    edges.append(
                        ADRSCandidateEdge(
                            source_external_id=cid,
                            target_external_id=pid,
                            edge_type="context",
                        )
                    )
        return edges

    def _parse_measurements(self) -> dict[str, list[ADRSMeasurement]]:
        """Per-program scores, correctness, island/gen, and public/private metrics."""
        measurements: dict[str, list[ADRSMeasurement]] = {}
        for prog in self.programs:
            pid = prog.get("id")
            if not pid:
                continue
            meas: list[ADRSMeasurement] = []

            if prog.get("combined_score") is not None:
                meas.append(
                    ADRSMeasurement(name="combined_score", value=str(prog["combined_score"]))
                )
            if prog.get("correct") is not None:
                meas.append(ADRSMeasurement(name="correct", value=str(prog["correct"])))
            if prog.get("generation") is not None:
                meas.append(ADRSMeasurement(name="generation", value=str(prog["generation"])))
            if prog.get("island_idx") is not None:
                meas.append(ADRSMeasurement(name="island_idx", value=str(prog["island_idx"])))
            if prog.get("children_count") is not None:
                meas.append(
                    ADRSMeasurement(name="children_count", value=str(prog["children_count"]))
                )
            meas.append(
                ADRSMeasurement(
                    name="in_archive", value="true" if pid in self.archive_ids else "false"
                )
            )

            # Scalar entries from public_metrics / private_metrics JSON dicts.
            for field in ("public_metrics", "private_metrics"):
                metrics = _json_or_none(prog.get(field))
                if isinstance(metrics, dict):
                    for name, value in metrics.items():
                        if _is_scalar(value):
                            meas.append(ADRSMeasurement(name=name, value=str(value)))

            if meas:
                measurements[pid] = meas
        return measurements

    def _parse_campaign(self) -> ADRSCampaign:
        """Campaign metadata: timestamps, best program, config, models."""
        timestamps = [p.get("timestamp") for p in self.programs if p.get("timestamp") is not None]
        started_at = parse_timestamp(min(timestamps)) if timestamps else None
        ended_at = parse_timestamp(max(timestamps)) if timestamps else None

        scored = [p for p in self.programs if p.get("combined_score") is not None]
        best = max(scored, key=lambda p: p["combined_score"]) if scored else None

        num_islands = self._config_get("num_islands")
        llm_models = self._config_get("llm_models")
        models_used = None
        if isinstance(llm_models, list):
            models_used = [str(m) for m in llm_models]
        elif isinstance(llm_models, str):
            models_used = [llm_models]
        # Fallback when the config has no model: the run dir name (verbatim).
        if not models_used:
            models_used = models_from_path(self.folder_path)

        config_used = {
            "num_islands": num_islands,
            "num_programs": len(self.programs),
            "archive_size": len(self.archive_ids),
        }
        final_metrics = {
            "best_program_id": best.get("id") if best else None,
            "best_score": best.get("combined_score") if best else None,
            "num_programs": len(self.programs),
            "num_generations": max((p.get("generation") or 0) for p in self.programs) + 1,
        }

        system = ADRSSystem(name="shinka", version=self.author_input.get("version"))

        return ADRSCampaign(
            system=system,
            author=self.author_input.get("author", ""),
            name=self.folder_path.name,
            research_question=self._config_get("research_question"),
            started_at=started_at,
            ended_at=ended_at,
            config_used=config_used,
            algorithm_used="shinka",
            models_used=models_used,
            total_cost_usd=None,
            total_tokens=None,
            final_summary=None,
            final_metrics=final_metrics,
        )

    def _collect_artifacts(self) -> list[ADRSArtifact]:
        """DB, run-level files, per-generation files, and logs."""
        artifacts: list[ADRSArtifact] = []

        def add(path: Path, iteration_index=None):
            if path.is_file() and not path.name.startswith("."):
                artifacts.append(
                    ADRSArtifact(
                        iteration_index=iteration_index,
                        uri=str(path),
                        content_hash=compute_file_hash(path),
                        size_bytes=path.stat().st_size,
                        mime_type=get_mime_type(path),
                    )
                )

        add(self.db_path)
        for name in ("meta_memory.json", "experiment_config.yaml", "evolution_run.log"):
            add(self.base / name)
        for meta_txt in sorted(self.base.glob("meta_*.txt")):
            add(meta_txt)

        # Per-generation code/diff files.
        for gen_dir in sorted(self.base.glob("gen_*")):
            if not gen_dir.is_dir():
                continue
            m = re.search(r"gen_(\d+)", gen_dir.name)
            gen_num = int(m.group(1)) if m else None
            for fname in ("main.py", "original.py", "edit.diff", "search_replace.txt"):
                add(gen_dir / fname, iteration_index=gen_num)

        logs_dir = self._logs_dir()
        if logs_dir:
            for log_file in sorted(logs_dir.glob("*")):
                add(log_file)

        return artifacts


def _read_text(path: Path) -> str | None:
    """Read a text file, returning None if missing/unreadable."""
    try:
        if path.is_file():
            return path.read_text()
    except OSError:
        return None
    return None


def parse_shinka_campaign(folder_path: Path) -> ADRSParsedCampaign | None:
    """Parse a native ShinkaEvolve run into ADRS models.

    Args:
        folder_path: Path to the run directory (run root or its results/ subdir).

    Returns:
        ADRSParsedCampaign or None if parsing fails.
    """
    parser = ShinkaParser(folder_path)
    return parser.parse()


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse a native ShinkaEvolve run into ADRS models (test mode)"
    )
    parser.add_argument("path", type=Path, help="Path to run folder (root or results/)")
    args = parser.parse_args()

    parsed = parse_shinka_campaign(args.path)
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
        parent_edges = [e for e in parsed.candidate_edges if e.edge_type == "parent"]
        context_edges = [e for e in parsed.candidate_edges if e.edge_type == "context"]
        print(f"  Edges: {len(parsed.candidate_edges)} "
              f"({len(parent_edges)} parent, {len(context_edges)} context)")
        if parsed.campaign.final_metrics:
            print(f"  Best score: {parsed.campaign.final_metrics.get('best_score')}")
            print(f"  Generations: {parsed.campaign.final_metrics.get('num_generations')}")
    else:
        print("Failed to parse campaign")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
