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
    python insert_adrs_campaign.py --system skydiscover /path/to/campaign_folder
"""

import argparse
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg

from adrs_models import ADRSParsedCampaign, ADRSArtifact, ADRSAssessor, ADRSSource
from provenance import ProvenanceError, label, normalize_submitter
from system_detection import detect_system_type

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/adrs"
)

PROVENANCE_KEYS = ("author", "version", "assessor", "source")
NAMED_REQUIRED_FIELDS = ("slug", "title", "version", "url", "license_spdx", "redistribution", "authors")
SOURCE_ROW_FIELDS = ("slug", "title", "version", "year", "url", "license_spdx", "redistribution",
                     "attribution", "authors", "bibtex", "owner")


def resolve_provenance_input(campaign_path: Path, root: Path) -> dict:
    """Merge every author_input.json from root down to campaign_path (both inclusive).

    Per top-level key, the file nearest the campaign wins; nested dicts (assessor,
    source) are replaced whole. Only author, version, assessor and source are returned.
    Never reads above root.
    """
    root = Path(root).resolve()
    campaign_path = Path(campaign_path).resolve()
    rel = campaign_path.relative_to(root)  # ValueError if campaign_path is not under root
    chain = [root]
    for part in rel.parts:
        chain.append(chain[-1] / part)

    merged: dict = {}
    for folder in chain:
        f = folder / "author_input.json"
        if not f.is_file():
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict):
            merged.update({k: data[k] for k in PROVENANCE_KEYS if k in data})
    return merged


def allocate_anon_slug(conn) -> tuple[str, str]:
    """Take nextval('anon_contributor_seq') once -> (slug, title)."""
    with conn.cursor() as cur:
        cur.execute("SELECT nextval('anon_contributor_seq')")
        n = cur.fetchone()[0]
    return f"contributor-{label(n)}@v1", f"Independent contributor {label(n)}"


def allocate_job_anon_slug(database_url: str | None = None) -> str:
    """Allocate (and commit) one anonymous slug for an upload job."""
    with psycopg.connect(database_url or DATABASE_URL) as conn:
        slug, _title = allocate_anon_slug(conn)
        conn.commit()
    return slug


def is_anonymous_first_upload(source) -> bool:
    """An anonymous source block without a slug (the server assigns one)."""
    if source is None:
        return False
    if isinstance(source, ADRSSource):
        source = source.model_dump()
    return isinstance(source, dict) and source.get("attribution") == "anonymous" and not source.get("slug")


def check_anonymous_blocks(sources) -> bool:
    """True if any source block is an anonymous first upload. Raises ProvenanceError
    unless all such blocks are identical (one anonymous contributor per upload)."""
    blocks = [ADRSSource.model_validate(s).model_dump() if isinstance(s, dict) else s.model_dump()
              for s in sources if is_anonymous_first_upload(s)]
    if any(b != blocks[0] for b in blocks):
        raise ProvenanceError(
            "an upload may contain only one anonymous contributor: all anonymous source blocks must be identical"
        )
    return bool(blocks)


def attach_provenance(parsed: ADRSParsedCampaign, prov: dict) -> ADRSParsedCampaign:
    """Set parsed.assessor / parsed.source from resolved provenance input."""
    if "assessor" in prov:
        parsed.assessor = ADRSAssessor.model_validate(prov["assessor"])
    if "source" in prov:
        parsed.source = ADRSSource.model_validate(prov["source"])
    return parsed


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

    def insert(
        self,
        parsed: ADRSParsedCampaign,
        assigned_anon_slug: str | None = None,
        submitted_by: str | None = None,
    ) -> bool:
        """Insert all campaign data with its source and generation-time evaluation, in
        one transaction. Returns True if successful; raises ProvenanceError (after
        rolling back) if a provenance rule is violated."""
        with psycopg.connect(self.database_url) as conn:
            try:
                with conn.cursor() as cur:
                    slug, anon_title = self._resolve_source(cur, parsed, assigned_anon_slug, submitted_by)
                    campaign = parsed.campaign
                    if anon_title is not None:
                        # Anonymous: the label replaces the author; submitted_by keeps the submitter.
                        campaign = campaign.model_copy(update={"author": anon_title})
                    self._ensure_system(cur, parsed.campaign.system)
                    self._insert_campaign(cur, campaign)
                    self._insert_candidates_bulk(cur, parsed.candidates)
                    evaluation_id = self._insert_evaluation(cur, parsed.assessor)
                    self._insert_measurements_bulk(cur, parsed.measurements, evaluation_id)
                    self._insert_artifacts_bulk(cur, parsed.artifacts)
                    self._insert_candidate_edges_bulk(cur, parsed.candidate_edges)
                    cur.execute(
                        "INSERT INTO campaign_sources (campaign_id, source_slug, submitted_by) VALUES (%s, %s, %s)",
                        (self.campaign_id, slug, normalize_submitter(submitted_by)),
                    )

                conn.commit()
                return True

            except Exception as e:
                conn.rollback()
                print(f"Error inserting campaign: {e}")
                import traceback
                traceback.print_exc()
                raise

    def _resolve_source(
        self, cur: psycopg.Cursor, parsed: ADRSParsedCampaign,
        assigned_anon_slug: str | None, submitted_by: str | None,
    ) -> tuple[str, str | None]:
        """Validate parsed.assessor/source and create the source row if needed.

        Returns (source_slug, anonymous title or None). Raises ProvenanceError.
        """
        if parsed.assessor is None:
            raise ProvenanceError("an assessor block is required")
        src = parsed.source
        if src is None:
            raise ProvenanceError("a source block is required")
        if src.consent is not True:
            raise ProvenanceError("source.consent must be true")
        if src.attribution not in (None, "named", "anonymous"):
            raise ProvenanceError("source.attribution must be 'named' or 'anonymous'")
        if src.redistribution not in (None, "full", "pointer_only"):
            raise ProvenanceError("source.redistribution must be 'full' or 'pointer_only'")
        submitter = normalize_submitter(submitted_by)
        supplied = {k: v for k, v in src.model_dump().items() if v is not None and k != "consent"}

        if src.attribution == "anonymous" and not src.slug:
            # Anonymous first upload: the server-assigned slug identifies the contributor.
            if any(v is not None for v in (src.title, src.authors, src.bibtex)):
                raise ProvenanceError("an anonymous source must not include slug, title, authors or bibtex")
            if assigned_anon_slug is None:
                raise ProvenanceError("an anonymous first upload needs a server-assigned slug")
            if submitter is None:
                raise ProvenanceError("an anonymous first upload needs a submitter")
            self._check_new_source(src)
            title = "Independent contributor " + assigned_anon_slug.removeprefix("contributor-").rsplit("@", 1)[0]
            row = dict(supplied, slug=assigned_anon_slug, title=title, owner=submitter)
            if self._create_source(cur, row):
                return assigned_anon_slug, title
            supplied = dict(supplied, slug=assigned_anon_slug)  # batch: the row already exists
        elif src.attribution == "named" and (not src.slug or self._fetch_source(cur, src.slug) is None):
            # New named source.
            missing = [f for f in NAMED_REQUIRED_FIELDS if getattr(src, f) is None]
            if missing:
                raise ProvenanceError(f"a named source requires {', '.join(missing)}")
            if src.title.lower().startswith("independent contributor"):
                raise ProvenanceError("a named source title must not start with 'Independent contributor'")
            self._check_new_source(src)
            if self._create_source(cur, dict(supplied, owner=submitter)):
                return src.slug, None
        elif not src.slug:
            raise ProvenanceError("a source block needs a slug, or attribution 'named' or 'anonymous'")

        # Existing-source reference.
        slug = supplied["slug"]
        row = self._fetch_source(cur, slug)
        if row is None:
            raise ProvenanceError(f"unknown source slug {slug!r}")
        if row["redistribution"] == "pointer_only":
            raise ProvenanceError(f"source {slug!r} is pointer_only and cannot receive uploads")
        if row["attribution"] == "anonymous" and (row["owner"] is None or row["owner"] != submitter):
            raise ProvenanceError(f"only the original submitter may upload to anonymous source {slug!r}")
        differing = [k for k, v in supplied.items() if row[k] != v]
        if differing:
            raise ProvenanceError(f"source {slug!r} exists with different {', '.join(differing)}")
        return slug, row["title"] if row["attribution"] == "anonymous" else None

    @staticmethod
    def _check_new_source(src):
        if src.redistribution is None:
            raise ProvenanceError("source.redistribution is required")
        if src.redistribution == "pointer_only":
            raise ProvenanceError("pointer_only sources cannot be uploaded")

    @staticmethod
    def _fetch_source(cur: psycopg.Cursor, slug: str) -> dict | None:
        cur.execute(f"SELECT {', '.join(SOURCE_ROW_FIELDS)} FROM source_datasets WHERE slug = %s", (slug,))
        row = cur.fetchone()
        return dict(zip(SOURCE_ROW_FIELDS, row)) if row else None

    @staticmethod
    def _create_source(cur: psycopg.Cursor, row: dict) -> bool:
        """Insert a source row (consent recorded now). False if the slug already exists."""
        cols = list(row)
        cur.execute(
            f"INSERT INTO source_datasets ({', '.join(cols)}, consent_recorded_at) "
            f"VALUES ({', '.join(['%s'] * len(cols))}, now()) "
            "ON CONFLICT (slug) DO NOTHING RETURNING slug",
            [row[c] for c in cols],
        )
        return cur.fetchone() is not None

    def _insert_evaluation(self, cur: psycopg.Cursor, assessor: ADRSAssessor):
        """Insert the generation-time evaluation for this campaign (kind is never taken from input)."""
        cur.execute(
            "INSERT INTO evaluations (assessor_name, assessor_version, platform, code_ref, kind) "
            "VALUES (%s, %s, %s, %s, 'generation_time') RETURNING id",
            (assessor.name, assessor.version, assessor.platform, assessor.code_ref),
        )
        return cur.fetchone()[0]

    def _ensure_system(self, cur: psycopg.Cursor, system):
        """Ensure the system exists and get its ID (atomic get-or-create)."""
        version = system.version or ""  # normalize missing version to '' (matches NOT NULL column)
        cur.execute(
            "INSERT INTO systems (name, version) VALUES (%s, %s) "
            "ON CONFLICT (name, version) DO NOTHING RETURNING id",
            (system.name, version),
        )
        row = cur.fetchone()

        if row:
            self.system_id = row[0]
            print(f"Created new system: {system.name} v{version!r}")
        else:
            # Row already existed (or a concurrent txn just created it) -> reuse it.
            cur.execute(
                "SELECT id FROM systems WHERE name = %s AND version = %s",
                (system.name, version),
            )
            self.system_id = cur.fetchone()[0]

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

    def _insert_measurements_bulk(self, cur: psycopg.Cursor, measurements: dict, evaluation_id=None):
        """Insert all measurement records using bulk insert, stamped with evaluation_id."""
        if not measurements:
            return

        rows = []
        for candidate_external_id, measurement_list in measurements.items():
            candidate_id = self.candidate_ids.get(candidate_external_id)
            if not candidate_id:
                print(f"Warning: Could not find candidate {candidate_external_id} for measurements")
                continue

            for measurement in measurement_list:
                rows.append((candidate_id, measurement.name, measurement.value, evaluation_id))

        if rows:
            cur.executemany(
                "INSERT INTO measurements (candidate_id, name, value, evaluation_id) VALUES (%s, %s, %s, %s)",
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


def insert_campaign(
    folder_path: Path, system_type: str | None = None, provenance_root: Path | None = None
) -> bool:
    """Insert a single campaign into the database.

    Args:
        folder_path: Path to campaign folder
        system_type: System type ('skydiscover', 'openevolve', 'gepa', 'shinka',
            'coding_agent') or None for auto-detect
        provenance_root: Topmost folder whose author_input.json applies (default:
            folder_path). An anonymous upload gets its own slug.

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

    if system_type == "skydiscover":
        from parse_skydiscover import parse_skydiscover_campaign

        parsed = parse_skydiscover_campaign(folder_path)
    elif system_type == "gepa":
        from parse_gepa import parse_gepa_campaign

        parsed = parse_gepa_campaign(folder_path)
    elif system_type == "openevolve":
        from parse_openevolve import parse_openevolve_campaign

        parsed = parse_openevolve_campaign(folder_path)
    elif system_type == "shinka":
        from parse_shinka import parse_shinka_campaign

        parsed = parse_shinka_campaign(folder_path)
    elif system_type == "coding_agent":
        from parse_coding_agent import parse_coding_agent_campaign

        parsed = parse_coding_agent_campaign(folder_path)
    elif system_type == "gepa_curated":
        from oneoff_parsers.parse_gepa_curated import parse_gepa_curated_campaign

        parsed = parse_gepa_curated_campaign(folder_path)
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

    prov = resolve_provenance_input(folder_path, provenance_root or folder_path)
    attach_provenance(parsed, prov)
    anon_slug = allocate_job_anon_slug() if is_anonymous_first_upload(parsed.source) else None

    # Insert into database
    inserter = CampaignInserter()
    if inserter.insert(parsed, assigned_anon_slug=anon_slug, submitted_by=prov.get("author")):
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
        if actual_type == "skydiscover":
            from parse_skydiscover import parse_skydiscover_campaign
            parsed = parse_skydiscover_campaign(folder_path)
        elif actual_type == "gepa":
            from parse_gepa import parse_gepa_campaign
            parsed = parse_gepa_campaign(folder_path)
        elif actual_type == "openevolve":
            from parse_openevolve import parse_openevolve_campaign
            parsed = parse_openevolve_campaign(folder_path)
        elif actual_type == "shinka":
            from parse_shinka import parse_shinka_campaign
            parsed = parse_shinka_campaign(folder_path)
        elif actual_type == "coding_agent":
            from parse_coding_agent import parse_coding_agent_campaign
            parsed = parse_coding_agent_campaign(folder_path)
        elif actual_type == "gepa_curated":
            from oneoff_parsers.parse_gepa_curated import parse_gepa_curated_campaign
            parsed = parse_gepa_curated_campaign(folder_path)
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

    # Resolve provenance per campaign (root .. campaign author_input.json files). All
    # anonymous first-upload blocks must be identical; they share one allocated slug.
    provenance = {path: resolve_provenance_input(path, campaigns_dir) for path, _ in parsed_campaigns}
    anon_slug = None
    if check_anonymous_blocks([p.get("source") for p in provenance.values()]):
        anon_slug = allocate_job_anon_slug()

    # Phase 2: Insert all campaigns
    print("Phase 2: Inserting campaigns into database...")
    success_count = 0
    failure_count = len(parse_errors)

    inserter = CampaignInserter()
    for _i, (path, parsed) in enumerate(parsed_campaigns):
        try:
            prov = provenance[path]
            attach_provenance(parsed, prov)
            if inserter.insert(parsed, assigned_anon_slug=anon_slug, submitted_by=prov.get("author")):
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
        choices=["skydiscover", "gepa", "openevolve", "shinka", "coding_agent"],
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
