#!/usr/bin/env python3
"""Insert ADRS campaign data into the PostgreSQL database.

This script accepts ADRSParsedCampaign objects (the common intermediate format)
and inserts them into the database. It's system-agnostic and works with any
parser that produces ADRSParsedCampaign objects.

Usage:
    # Insert a single campaign (auto-detects system type)
    python insert_adrs_campaign.py /path/to/campaign_folder

    # Insert all campaigns in a directory
    python insert_adrs_campaign.py --all /path/to/campaigns_directory

    # Specify system type explicitly
    python insert_adrs_campaign.py --system nous /path/to/campaign_folder
    python insert_adrs_campaign.py --system skydiscover /path/to/campaign_folder
"""

import argparse
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg

from adrs_models import ADRSParsedCampaign, ADRSArtifact

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/adrs"
)


class CampaignInserter:
    """Insert parsed campaign data into the database."""

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or DATABASE_URL
        self.system_id: uuid.UUID | None = None
        self.campaign_id: uuid.UUID | None = None
        self.candidate_ids: dict[str, uuid.UUID] = {}  # external_id -> UUID

    def test_connection(self):
        """Test database connection.

        Raises:
            Exception if connection fails
        """
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

    def get_campaign_count(self) -> int:
        """Get the count of campaigns in the database."""
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM campaigns;")
                return cur.fetchone()[0]

    def insert(self, parsed: ADRSParsedCampaign) -> bool:
        """Insert all campaign data. Returns True if successful."""
        with psycopg.connect(self.database_url) as conn:
            try:
                with conn.cursor() as cur:
                    self._ensure_system(cur, parsed.campaign.system)
                    self._insert_campaign(cur, parsed.campaign)
                    self._insert_candidates_bulk(cur, parsed.candidates)
                    self._insert_measurements_bulk(cur, parsed.measurements)
                    self._insert_artifacts_bulk(cur, parsed.artifacts)
                    self._insert_candidate_edges_bulk(cur, parsed.candidate_edges)

                conn.commit()
                return True

            except Exception as e:
                conn.rollback()
                print(f"Error inserting campaign: {e}")
                import traceback
                traceback.print_exc()
                raise

    def _ensure_system(self, cur: psycopg.Cursor, system):
        """Ensure the system exists and get its ID."""
        cur.execute(
            "SELECT id FROM systems WHERE name = %s AND version IS NOT DISTINCT FROM %s",
            (system.name, system.version),
        )
        row = cur.fetchone()

        if row:
            self.system_id = row[0]
        else:
            cur.execute(
                "INSERT INTO systems (name, version) VALUES (%s, %s) RETURNING id",
                (system.name, system.version),
            )
            self.system_id = cur.fetchone()[0]
            print(f"Created new system: {system.name} v{system.version}")

    def _insert_campaign(self, cur: psycopg.Cursor, campaign):
        """Insert the campaign record."""
        evaluator_setup_json = None
        if campaign.evaluator_setup:
            evaluator_setup_json = campaign.evaluator_setup.model_dump_json()

        cur.execute(
            """
            INSERT INTO campaigns (
                system_id, author, name, research_question,
                started_at, ended_at, config_used, algorithm_used, models_used,
                total_cost_usd, total_tokens, final_summary, final_metrics,
                evaluator_setup
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                self.system_id,
                campaign.author,
                campaign.name,
                campaign.research_question,
                campaign.started_at,
                campaign.ended_at,
                json.dumps(campaign.config_used) if campaign.config_used else None,
                campaign.algorithm_used,
                campaign.models_used,
                campaign.total_cost_usd,
                campaign.total_tokens,
                campaign.final_summary,
                json.dumps(campaign.final_metrics) if campaign.final_metrics else None,
                evaluator_setup_json,
            ),
        )
        self.campaign_id = cur.fetchone()[0]

    def _insert_candidates_bulk(self, cur: psycopg.Cursor, candidates: list):
        """Insert all candidate records sequentially to preserve ID mapping."""
        for candidate in candidates:
            embedding_str = None
            if candidate.solution_summary_embedding:
                embedding_str = "[" + ",".join(str(x) for x in candidate.solution_summary_embedding) + "]"

            cur.execute(
                """
                INSERT INTO candidates (
                    campaign_id, iteration_index, external_id, candidate_type, created_at,
                    solution_summary, solution_summary_embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    self.campaign_id,
                    candidate.iteration_index,
                    candidate.external_id,
                    candidate.candidate_type,
                    candidate.created_at,
                    candidate.solution_summary,
                    embedding_str,
                ),
            )
            candidate_db_id = cur.fetchone()[0]

            if candidate.external_id:
                self.candidate_ids[candidate.external_id] = candidate_db_id

    def _insert_measurements_bulk(self, cur: psycopg.Cursor, measurements: dict):
        """Insert all measurement records using bulk insert."""
        if not measurements:
            return

        rows = []
        for candidate_external_id, measurement_list in measurements.items():
            candidate_id = self.candidate_ids.get(candidate_external_id)
            if not candidate_id:
                print(f"Warning: Could not find candidate {candidate_external_id} for measurements")
                continue

            for measurement in measurement_list:
                rows.append((candidate_id, measurement.name, measurement.value))

        if rows:
            cur.executemany(
                "INSERT INTO measurements (candidate_id, name, value) VALUES (%s, %s, %s)",
                rows,
            )

    def _insert_artifacts_bulk(self, cur: psycopg.Cursor, artifacts: list):
        """Insert all artifact records using bulk insert."""
        if not artifacts:
            return

        rows = []
        for a in artifacts:
            candidate_id = None
            if a.external_id:
                candidate_id = self.candidate_ids.get(a.external_id)
            rows.append((
                self.campaign_id,
                candidate_id,
                a.iteration_index,
                a.uri,
                a.content_hash,
                a.size_bytes,
                a.mime_type,
            ))

        cur.executemany(
            """
            INSERT INTO artifacts (campaign_id, candidate_id, iteration_index, uri, content_hash, size_bytes, mime_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )

    def _insert_candidate_edges_bulk(self, cur: psycopg.Cursor, edges: list):
        """Insert candidate edges using bulk insert."""
        if not edges:
            return

        rows = []
        for edge in edges:
            source_id = self.candidate_ids.get(edge.source_external_id)
            target_id = self.candidate_ids.get(edge.target_external_id)

            if not source_id or not target_id:
                print(
                    f"Warning: Could not find candidate IDs for edge "
                    f"{edge.source_external_id} -> {edge.target_external_id}"
                )
                continue

            rows.append((source_id, target_id, edge.edge_type))

        if rows:
            cur.executemany(
                "INSERT INTO candidate_edges (source_candidate_id, target_candidate_id, edge_type) VALUES (%s, %s, %s)",
                rows,
            )


def detect_system_type(folder_path: Path) -> str | None:
    """Auto-detect the system type from folder structure."""
    # NOUS indicators
    if (folder_path / "state.json").exists() and (folder_path / "ledger.json").exists():
        return "nous"

    # GEPA indicators: summary.json with "framework": "gepa" and iterations.jsonl
    if (folder_path / "iterations.jsonl").exists() and (folder_path / "summary.json").exists():
        try:
            with open(folder_path / "summary.json") as f:
                summary = json.load(f)
                if summary.get("framework") == "gepa":
                    return "gepa"
                if summary.get("framework") == "coding_agent":
                    return "coding_agent"
        except Exception:
            pass

    # OpenEvolve indicators: evolution_trace.jsonl and checkpoints/
    if (folder_path / "evolution_trace.jsonl").exists() and (folder_path / "checkpoints").exists():
        return "openevolve"

    # SkyDiscover indicators - two layouts:
    # Layout 1: config.yaml + output/ (gamble-data-transformed style)
    if (folder_path / "config.yaml").exists() and (folder_path / "output").exists():
        config_yaml = folder_path / "config.yaml"
        try:
            import yaml
            with open(config_yaml) as f:
                config = yaml.safe_load(f)
                if config and ("llm" in config or "evaluator" in config):
                    return "skydiscover"
        except Exception:
            pass

    # Layout 2: checkpoints/ directly in root (odellia style, no output/ wrapper)
    # best/ may not exist for incomplete runs
    if (folder_path / "checkpoints").exists():
        # Verify it has the expected checkpoint structure (programs/ subdir)
        checkpoints_dir = folder_path / "checkpoints"
        for checkpoint in checkpoints_dir.iterdir():
            if checkpoint.is_dir() and (checkpoint / "programs").exists():
                return "skydiscover"

    return None


def insert_campaign(folder_path: Path, system_type: str | None = None) -> bool:
    """Insert a single campaign into the database.

    Args:
        folder_path: Path to campaign folder
        system_type: System type ('nous', 'skydiscover') or None for auto-detect

    Returns:
        True if successful, False otherwise
    """
    print(f"\nProcessing campaign: {folder_path}")

    # Auto-detect system type if not specified
    if system_type is None:
        system_type = detect_system_type(folder_path)
        if system_type:
            print(f"  Detected system type: {system_type}")
        else:
            print("  Error: Could not detect system type")
            return False

    # Parse based on system type
    parsed: ADRSParsedCampaign | None = None

    if system_type == "nous":
        from parse_nous import parse_nous_campaign

        parsed = parse_nous_campaign(folder_path)
    elif system_type == "skydiscover":
        from parse_skydiscover import parse_skydiscover_campaign

        parsed = parse_skydiscover_campaign(folder_path)
    elif system_type == "gepa":
        from parse_gepa import parse_gepa_campaign

        parsed = parse_gepa_campaign(folder_path)
    elif system_type == "openevolve":
        from parse_openevolve import parse_openevolve_campaign

        parsed = parse_openevolve_campaign(folder_path)
    elif system_type == "coding_agent":
        from parse_coding_agent import parse_coding_agent_campaign

        parsed = parse_coding_agent_campaign(folder_path)
    else:
        print(f"  Error: Unknown system type: {system_type}")
        return False

    if not parsed:
        print(f"  Failed to parse campaign: {folder_path}")
        return False

    total_measurements = sum(len(m) for m in parsed.measurements.values())
    num_iterations = len(set(c.iteration_index for c in parsed.candidates))
    print(
        f"  Parsed: {num_iterations} iterations, "
        f"{len(parsed.candidates)} candidates, "
        f"{total_measurements} measurements, "
        f"{len(parsed.artifacts)} artifacts"
    )

    # Insert into database
    inserter = CampaignInserter()
    if inserter.insert(parsed):
        print(f"  Successfully inserted campaign: {parsed.campaign.name}")
        return True
    else:
        print(f"  Failed to insert campaign: {folder_path}")
        return False


def _parse_single_campaign(
    folder_path: Path, system_type: str | None
) -> tuple[Path, ADRSParsedCampaign | None, str | None]:
    """Parse a single campaign. Returns (path, parsed_result, error_message)."""
    try:
        actual_type = system_type or detect_system_type(folder_path)
        if actual_type is None:
            return folder_path, None, "Could not detect system type"

        parsed: ADRSParsedCampaign | None = None
        if actual_type == "nous":
            from parse_nous import parse_nous_campaign
            parsed = parse_nous_campaign(folder_path)
        elif actual_type == "skydiscover":
            from parse_skydiscover import parse_skydiscover_campaign
            parsed = parse_skydiscover_campaign(folder_path)
        elif actual_type == "gepa":
            from parse_gepa import parse_gepa_campaign
            parsed = parse_gepa_campaign(folder_path)
        elif actual_type == "openevolve":
            from parse_openevolve import parse_openevolve_campaign
            parsed = parse_openevolve_campaign(folder_path)
        elif actual_type == "coding_agent":
            from parse_coding_agent import parse_coding_agent_campaign
            parsed = parse_coding_agent_campaign(folder_path)
        else:
            return folder_path, None, f"Unknown system type: {actual_type}"

        if not parsed:
            return folder_path, None, "Parser returned None"

        return folder_path, parsed, None

    except Exception as e:
        return folder_path, None, str(e)


def insert_all_campaigns(
    campaigns_dir: Path,
    system_type: str | None = None,
    max_workers: int | None = None,
) -> tuple[int, int]:
    """Insert all campaigns in a directory using parallel parsing and bulk inserts.

    Args:
        campaigns_dir: Directory containing campaign folders
        system_type: System type or None for auto-detect
        max_workers: Max parallel workers for parsing (default: CPU count)

    Returns:
        Tuple of (success_count, failure_count)
    """
    if not campaigns_dir.is_dir():
        print(f"Error: {campaigns_dir} is not a directory")
        return 0, 0

    # Collect campaign folders
    campaign_folders = []
    for item in sorted(campaigns_dir.iterdir()):
        if item.is_dir() and detect_system_type(item) is not None:
            campaign_folders.append(item)

    if not campaign_folders:
        print("No campaigns found")
        return 0, 0

    print(f"Found {len(campaign_folders)} campaigns to process")

    # Phase 1: Parse all campaigns in parallel
    print("Phase 1: Parsing campaigns...")
    parsed_campaigns: list[tuple[Path, ADRSParsedCampaign]] = []
    parse_errors: list[tuple[Path, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_parse_single_campaign, folder, system_type): folder
            for folder in campaign_folders
        }

        for future in as_completed(futures):
            path, parsed, error = future.result()
            if parsed:
                parsed_campaigns.append((path, parsed))
                print(f"  Parsed: {path.name}")
            else:
                parse_errors.append((path, error or "Unknown error"))
                print(f"  Failed to parse: {path.name} - {error}")

    if not parsed_campaigns:
        print("No campaigns were successfully parsed")
        return 0, len(parse_errors)

    # Phase 2: Insert all campaigns
    print("Phase 2: Inserting campaigns into database...")
    success_count = 0
    failure_count = len(parse_errors)

    inserter = CampaignInserter()
    for _i, (path, parsed) in enumerate(parsed_campaigns):
        try:
            if inserter.insert(parsed):
                print(f"  Inserted: {parsed.campaign.name or path.name}")
                success_count += 1
            else:
                print(f"  Failed to insert: {path.name}")
                failure_count += 1
        except Exception as e:
            print(f"  Error inserting {path.name}: {e}")
            failure_count += 1
        finally:
            # Reset state for next campaign
            inserter.system_id = None
            inserter.campaign_id = None
            inserter.candidate_ids = {}

    return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(
        description="Insert ADRS campaign data into the database."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to campaign folder or campaigns directory (with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Insert all campaigns in the given directory",
    )
    parser.add_argument(
        "--system",
        type=str,
        choices=["nous", "skydiscover", "gepa", "openevolve"],
        help="System type (auto-detect if not specified)",
    )

    args = parser.parse_args()

    if args.all:
        success, failure = insert_all_campaigns(
            args.path,
            args.system,
        )
        print(f"\nSummary: {success} campaigns inserted, {failure} failed")
    else:
        success = insert_campaign(args.path, args.system)
        exit(0 if success else 1)


if __name__ == "__main__":
    main()
