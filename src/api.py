#!/usr/bin/env python3
"""FastAPI service for uploading and ingesting ADRS campaign data.

This service provides an HTTP API for uploading campaign data as zip files,
parsing them, and inserting them into the ADRS database.

Usage:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import shutil
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from insert_adrs_campaign import CampaignInserter, detect_system_type
# from parse_auto import parse_auto_campaign  # Disabled: auto-parsing fallback removed
from parse_skydiscover import parse_skydiscover_campaign
from parse_openevolve import parse_openevolve_campaign
from parse_gepa import parse_gepa_campaign
from parse_coding_agent import parse_coding_agent_campaign
from embeddings import get_embedding_model
from analytics import CampaignAnalytics


# In-memory job tracking (use Redis in production for persistence across restarts)
upload_jobs: dict[str, dict] = {}

# Configuration
DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# FastAPI app
app = FastAPI(
    title="ADRS Campaign Upload API",
    description="API for uploading and ingesting agentic research campaign data",
    version="1.0.0",
)


@app.on_event("startup")
async def preload_embedding_model():
    """Pre-load the embedding model at startup to avoid delay on first upload."""
    print("[STARTUP] Pre-loading embedding model...", flush=True)
    get_embedding_model()
    print("[STARTUP] Embedding model loaded and ready.", flush=True)


class SystemType(str, Enum):
    """Supported campaign system types."""

    SKYDISCOVER = "skydiscover"
    OPENEVOLVE = "openevolve"
    GEPA = "gepa"
    CODING_AGENT = "coding_agent"


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""

    success: bool
    message: str
    campaign_count: int = 0
    campaigns: list[dict] | None = None
    system_type: str | None = None
    total_stats: dict | None = None
    errors: list[str] | None = None


class JobStatus(BaseModel):
    """Status model for upload jobs."""

    job_id: str
    status: str  # "uploading", "processing", "complete", "error"
    progress: int = 0
    total: int = 0
    current: str | None = None
    message: str | None = None
    result: UploadResponse | None = None


def validate_tar_file(tar_path: Path) -> tuple[bool, str]:
    """Validate that the uploaded file is a valid tar.gz archive.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not tarfile.is_tarfile(tar_path):
        return False, "Uploaded file is not a valid tar archive. Expected .tar.gz format."

    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            # Check that it's not empty by reading first member
            first = tf.next()
            if first is None:
                return False, "Archive is empty"

        return True, ""
    except tarfile.TarError as e:
        return False, f"Invalid tar.gz archive: {e}"
    except Exception as e:
        return False, f"Error validating tar.gz file: {str(e)}"


def extract_tar(tar_path: Path, extract_to: Path) -> tuple[bool, str, list[Path]]:
    """Extract tar.gz file to directory and find campaign folders.

    Uses system tar for memory efficiency with large archives.

    Returns:
        Tuple of (success, error_message, list_of_campaign_paths)
    """
    try:
        print(f"[EXTRACT] Extracting to {extract_to}...", flush=True)

        # Use system tar with checkpoint for progress reporting
        # --checkpoint=N reports every N records (512-byte blocks)
        process = subprocess.Popen(
            [
                "tar", "-xzf", str(tar_path), "-C", str(extract_to),
                "--checkpoint=5000",
                "--checkpoint-action=echo=#%u",
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
        )

        # Stream progress from stderr
        last_records = 0
        for line in process.stderr:
            line = line.strip()
            if line.startswith("#"):
                try:
                    records = int(line[1:])
                    if records - last_records >= 5000:
                        print(f"[EXTRACT] {records:,} records processed", flush=True)
                        last_records = records
                except ValueError:
                    pass
            elif line:
                # Skip macOS xattr warnings (harmless, just noisy)
                if "Ignoring unknown extended header keyword" in line:
                    continue
                print(f"[EXTRACT] {line}", flush=True)

        return_code = process.wait()
        if return_code != 0:
            return False, f"tar extraction failed (code {return_code})", []

        # Remove macOS resource fork files (AppleDouble format)
        macos_files = list(extract_to.rglob("._*"))
        if macos_files:
            print(f"[EXTRACT] Removing {len(macos_files)} macOS resource fork files...", flush=True)
            for f in macos_files:
                try:
                    f.unlink()
                except OSError:
                    pass

        print(f"[EXTRACT] Extraction complete, finding campaigns...", flush=True)

        # Find campaign directories
        campaign_paths = []

        def find_campaigns(root: Path, depth: int = 0):
            """Recursively find campaign directories."""
            for item in root.iterdir():
                if item.is_dir():
                    system_type = detect_system_type(item)
                    if system_type:
                        campaign_paths.append(item)
                    elif depth < 3:
                        find_campaigns(item, depth + 1)

        find_campaigns(extract_to)

        if not campaign_paths:
            return False, (
                "No recognized campaign format found in archive. "
                "Supported formats: skydiscover, openevolve, gepa, coding_agent."
            ), []
        else:
            print(f"[EXTRACT] Found {len(campaign_paths)} recognized campaign folder(s)", flush=True)

        return True, "", campaign_paths

    except FileNotFoundError:
        return False, "tar command not found", []
    except PermissionError:
        return False, "Permission denied while extracting archive", []
    except OSError as e:
        if "No space" in str(e) or e.errno == 28:
            return False, f"Disk full during extraction: {e}", []
        return False, f"OS error extracting archive: {e}", []
    except Exception as e:
        return False, f"Error extracting archive: {str(e)}", []


def validate_campaign_format(
    campaign_path: Path, expected_type: SystemType
) -> tuple[bool, str, str | None]:
    """Validate that the extracted campaign matches the expected format.

    Returns:
        Tuple of (is_valid, error_message, detected_type)
    """
    detected_type = detect_system_type(campaign_path)

    if detected_type is None:
        files = [f.name for f in campaign_path.iterdir() if f.is_file()][:5]
        dirs = [d.name for d in campaign_path.iterdir() if d.is_dir()][:5]
        error_msg = (
            f"Unrecognized campaign format in '{campaign_path.name}' "
            f"(files: {files}, dirs: {dirs}). "
            f"Supported formats: skydiscover, openevolve, gepa, coding_agent."
        )
        return False, error_msg, None

    if detected_type != expected_type.value:
        error_msg = (
            f"Format mismatch: You specified '{expected_type.value}' "
            f"but the data appears to be '{detected_type}' format."
        )
        return False, error_msg, detected_type

    return True, "", detected_type


from adrs_models import ADRSParsedCampaign


async def parse_campaign(campaign_path: Path, system_type: str | None, author: str) -> list[ADRSParsedCampaign]:
    """Parse a campaign using the appropriate parser.

    Returns:
        List of ADRSParsedCampaign (may be empty or single item)
    """
    parsed = None

    if system_type:
        try:
            if system_type == "skydiscover":
                parsed = await asyncio.to_thread(parse_skydiscover_campaign, campaign_path)
            elif system_type == "openevolve":
                parsed = await asyncio.to_thread(parse_openevolve_campaign, campaign_path)
            elif system_type == "gepa":
                parsed = await asyncio.to_thread(parse_gepa_campaign, campaign_path)
            elif system_type == "coding_agent":
                parsed = await asyncio.to_thread(parse_coding_agent_campaign, campaign_path)
            else:
                print(f"Unknown system type: {system_type}")
        except Exception as e:
            print(f"Parser failed: {e}")
            parsed = None

    # # Fallback to agent-based parsing (disabled)
    # if parsed is None:
    #     import os
    #     if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
    #         print(f"Using agent-based parsing for {campaign_path}")
    #         try:
    #             parsed = await parse_auto_campaign(campaign_path)
    #         except Exception as e:
    #             print(f"Agent-based parsing failed: {e}")
    #             parsed = None
    #     else:
    #         print("ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN not set, cannot use agent-based parsing")

    # Normalize to list
    if parsed is None:
        return []
    elif isinstance(parsed, list):
        results = parsed
    else:
        results = [parsed]

    # Set author from form field only if not already set by parser (from author_input.json)
    for p in results:
        if not p.campaign.author:
            p.campaign.author = author

    return results


def inject_author_input(campaign_path: Path, author: str, version: str = "") -> None:
    """Create author_input.json in campaign folder if not already present."""
    author_input_path = campaign_path / "author_input.json"
    if not author_input_path.exists():
        data = {"author": author}
        if version:
            data["version"] = version
        with open(author_input_path, "w") as f:
            json.dump(data, f, indent=2)


async def process_upload_job(
    job_id: str,
    tar_path: Path,
    author: str,
    system_type: SystemType,
    version: str = "",
) -> None:
    """Background task to process an uploaded tar.gz file.

    Updates upload_jobs[job_id] with progress as processing happens.
    """
    extract_path = None
    start_time = time.time()

    try:
        # Validate tar.gz file (run in thread to avoid blocking)
        print(f"[JOB {job_id}] Validating tar.gz file...", flush=True)
        upload_jobs[job_id]["message"] = "Validating archive..."
        is_valid, error_msg = await asyncio.to_thread(validate_tar_file, tar_path)
        if not is_valid:
            upload_jobs[job_id].update({
                "status": "error",
                "message": error_msg,
            })
            return

        # Generate timestamp-based folder name for this upload
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        extract_path = DATA_DIR / f"upload_{timestamp}"
        extract_path.mkdir(parents=True, exist_ok=True)

        # Extract tar.gz file (run in thread to avoid blocking event loop)
        print(f"[JOB {job_id}] Extracting archive to {extract_path}...", flush=True)
        upload_jobs[job_id]["message"] = "Extracting archive..."
        success, error_msg, campaign_paths = await asyncio.to_thread(
            extract_tar, tar_path, extract_path
        )
        if not success:
            shutil.rmtree(extract_path, ignore_errors=True)
            upload_jobs[job_id].update({
                "status": "error",
                "message": error_msg,
            })
            return

        # Delete temp archive immediately after extraction to free disk space
        tar_path.unlink(missing_ok=True)
        print(f"[JOB {job_id}] Extraction complete, found {len(campaign_paths)} campaigns", flush=True)

        total_campaigns = len(campaign_paths)
        upload_jobs[job_id].update({
            "progress": 0,
            "total": total_campaigns,
            "message": "Processing campaigns...",
        })

        # Process each campaign
        campaign_results = []
        errors = []
        detected_type = None
        total_iterations = 0
        total_candidates = 0
        total_measurements = 0
        total_artifacts = 0
        total_edges = 0

        for idx, campaign_path in enumerate(campaign_paths):
            # Update progress
            upload_jobs[job_id].update({
                "progress": idx,
                "current": campaign_path.name,
            })
            print(f"[JOB {job_id}] Processing {idx + 1}/{total_campaigns}: {campaign_path.name}", flush=True)

            try:
                # Validate campaign format (run in thread)
                is_valid, error_msg, curr_type = await asyncio.to_thread(
                    validate_campaign_format, campaign_path, system_type
                )
                if not is_valid:
                    errors.append(f"{campaign_path.name}: {error_msg}")
                    continue

                # Track detected format
                if curr_type is not None:
                    if detected_type is None:
                        detected_type = curr_type
                    elif detected_type != curr_type:
                        errors.append(
                            f"{campaign_path.name}: Format mismatch - expected {detected_type}, got {curr_type}"
                        )
                        continue

                # Inject author_input.json if not already present (web UI upload path)
                await asyncio.to_thread(inject_author_input, campaign_path, author, version)

                # Parse the campaign(s)
                parse_start = time.time()
                parsed_list = await parse_campaign(campaign_path, curr_type, author)
                parse_elapsed = time.time() - parse_start
                print(f"[JOB {job_id}] Parsing {campaign_path.name}: {parse_elapsed:.2f}s ({len(parsed_list)} campaign(s))", flush=True)

                if not parsed_list:
                    format_desc = curr_type or "unrecognized"
                    errors.append(f"{campaign_path.name}: Failed to parse {format_desc} campaign")
                    continue

                # Insert each parsed campaign into database
                for parsed in parsed_list:
                    insert_start = time.time()
                    inserter = CampaignInserter()

                    # Insert into database (run in thread)
                    insert_success = await asyncio.to_thread(
                        inserter.insert, parsed
                    )
                    insert_elapsed = time.time() - insert_start
                    print(f"[JOB {job_id}] DB insert {parsed.campaign.name or campaign_path.name}: {insert_elapsed:.2f}s", flush=True)

                    if not insert_success:
                        errors.append(f"{parsed.campaign.name or campaign_path.name}: Database insertion failed")
                        continue

                    # Collect stats
                    campaign_measurements = sum(len(m) for m in parsed.measurements.values())
                    # Count iterations from unique iteration_index values
                    iteration_indices = set(c.iteration_index for c in parsed.candidates)
                    num_iterations = len(iteration_indices)
                    total_iterations += num_iterations
                    total_candidates += len(parsed.candidates)
                    total_measurements += campaign_measurements
                    total_artifacts += len(parsed.artifacts)
                    total_edges += len(parsed.candidate_edges)

                    campaign_results.append({
                        "campaign_id": parsed.campaign.name or campaign_path.name,
                        "name": parsed.campaign.name or campaign_path.name,
                        "iterations": num_iterations,
                        "candidates": len(parsed.candidates),
                    })

            except Exception as e:
                errors.append(f"{campaign_path.name}: {str(e)}")
                continue

            # Yield to event loop periodically to keep server responsive
            await asyncio.sleep(0)

        # Final progress update
        upload_jobs[job_id].update({
            "progress": total_campaigns,
            "current": None,
        })

        # Check if any campaigns succeeded
        if not campaign_results:
            shutil.rmtree(extract_path, ignore_errors=True)
            upload_jobs[job_id].update({
                "status": "error",
                "message": f"Failed to process any campaigns. Errors: {'; '.join(errors)}",
            })
            return

        # NOTE: We no longer rename the directory after insertion.
        # The artifact URIs stored in the database reference paths under extract_path,
        # so renaming would break those references. Keep the upload_{timestamp} name.

        # Build final response
        total_stats = {
            "campaigns": len(campaign_results),
            "iterations": total_iterations,
            "candidates": total_candidates,
            "measurements": total_measurements,
            "artifacts": total_artifacts,
            "edges": total_edges,
        }

        elapsed = time.time() - start_time
        print(f"[JOB {job_id}] Completed in {elapsed:.2f}s - {len(campaign_results)} campaign(s), {total_iterations} iterations", flush=True)

        upload_jobs[job_id].update({
            "status": "complete",
            "message": f"Successfully uploaded {len(campaign_results)} {detected_type} campaign(s)",
            "result": UploadResponse(
                success=True,
                message=f"Successfully uploaded {len(campaign_results)} {detected_type} campaign(s)",
                campaign_count=len(campaign_results),
                campaigns=campaign_results,
                system_type=detected_type,
                total_stats=total_stats,
                errors=errors if errors else None,
            ),
        })

    except Exception as e:
        import traceback
        elapsed = time.time() - start_time
        print(f"[JOB {job_id}] Failed after {elapsed:.2f}s - {str(e)}", flush=True)
        upload_jobs[job_id].update({
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
        })

    finally:
        # Clean up temporary archive file
        if tar_path.exists():
            try:
                tar_path.unlink()
            except Exception:
                pass
        # Clean up extract path if still exists
        if extract_path and extract_path.exists():
            shutil.rmtree(extract_path, ignore_errors=True)


@app.post("/upload")
async def upload_campaign(
    file: Annotated[UploadFile, File(description="tar.gz archive containing campaign data")],
    author: Annotated[str, Form(description="Email address of the campaign author")],
    system_type: Annotated[
        SystemType,
        Form(description="Campaign system type (skydiscover, openevolve, gepa, coding_agent)"),
    ],
    version: Annotated[str, Form(description="GitHub commit link (optional)")] = "",
) -> dict:
    """Upload a campaign data archive for background processing.

    This endpoint accepts a tar.gz file containing campaign data, saves it,
    and starts processing in the background. Use GET /upload/status/{job_id}
    to poll for progress.

    Args:
        file: tar.gz archive containing the campaign data
        author: Email address of the campaign author
        system_type: The format of the data (skydiscover, openevolve, gepa, coding_agent)
        version: GitHub commit link of the system that produced this campaign

    Returns:
        job_id: Unique identifier to poll for status
        status_url: URL to check processing status
    """
    job_id = str(uuid4())

    # Save file to disk
    tar_path = DATA_DIR / f"pending_{job_id}.tar.gz"
    file_size = 0

    try:
        print(f"[UPLOAD] Receiving {file.filename} (job {job_id})...", flush=True)
        with open(tar_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)
                file_size += len(chunk)
        print(f"[UPLOAD] Received {file_size} bytes for job {job_id}", flush=True)
    except Exception as e:
        tar_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to receive upload: {e}")

    # Initialize job status
    upload_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": 0,
        "total": 0,
        "current": None,
        "message": "Starting processing...",
        "result": None,
    }

    # Start background processing
    asyncio.create_task(process_upload_job(job_id, tar_path, author, system_type, version))

    return {
        "job_id": job_id,
        "status_url": f"/upload/status/{job_id}",
        "message": "Upload received, processing started. Poll status_url for progress.",
    }


async def process_parsed_upload_job(
    job_id: str,
    tar_path: Path,
    author: str,
) -> None:
    """Background task to process an uploaded tar.gz with _parsed.jsonl.

    Updates upload_jobs[job_id] with progress as processing happens.
    """
    extract_path = None
    start_time = time.time()

    try:
        # Validate tar.gz file
        print(f"[JOB {job_id}] Validating tar.gz file...", flush=True)
        upload_jobs[job_id]["message"] = "Validating archive..."
        is_valid, error_msg = await asyncio.to_thread(validate_tar_file, tar_path)
        if not is_valid:
            upload_jobs[job_id].update({
                "status": "error",
                "message": error_msg,
            })
            return

        # Generate timestamp-based folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        extract_path = DATA_DIR / f"upload_{timestamp}"
        extract_path.mkdir(parents=True, exist_ok=True)

        # Extract tar.gz file
        print(f"[JOB {job_id}] Extracting archive to {extract_path}...", flush=True)
        upload_jobs[job_id]["message"] = "Extracting archive..."

        # Simple extraction without campaign detection
        process = subprocess.Popen(
            ["tar", "-xzf", str(tar_path), "-C", str(extract_path)],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
        )
        _, stderr = process.communicate()
        if process.returncode != 0:
            upload_jobs[job_id].update({
                "status": "error",
                "message": f"tar extraction failed: {stderr}",
            })
            return

        # Delete temp archive
        tar_path.unlink(missing_ok=True)

        # Find _parsed.jsonl at root
        parsed_jsonl_path = extract_path / "_parsed.jsonl"
        if not parsed_jsonl_path.exists():
            # Check one level down (in case archive had a wrapper folder)
            for item in extract_path.iterdir():
                if item.is_dir():
                    candidate = item / "_parsed.jsonl"
                    if candidate.exists():
                        parsed_jsonl_path = candidate
                        break

        if not parsed_jsonl_path.exists():
            shutil.rmtree(extract_path, ignore_errors=True)
            upload_jobs[job_id].update({
                "status": "error",
                "message": "_parsed.jsonl not found at archive root. Create this file with one ADRSParsedCampaign JSON object per line.",
            })
            return

        # Read and validate each line
        print(f"[JOB {job_id}] Reading _parsed.jsonl...", flush=True)
        upload_jobs[job_id]["message"] = "Validating parsed campaigns..."

        lines = []
        try:
            with open(parsed_jsonl_path) as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            shutil.rmtree(extract_path, ignore_errors=True)
            upload_jobs[job_id].update({
                "status": "error",
                "message": f"Failed to read _parsed.jsonl: {e}",
            })
            return

        if not lines:
            shutil.rmtree(extract_path, ignore_errors=True)
            upload_jobs[job_id].update({
                "status": "error",
                "message": "_parsed.jsonl is empty",
            })
            return

        total_campaigns = len(lines)
        upload_jobs[job_id].update({
            "progress": 0,
            "total": total_campaigns,
            "message": "Processing campaigns...",
        })

        # Process each line
        campaign_results = []
        errors = []
        total_candidates = 0
        total_measurements = 0
        total_artifacts = 0
        total_edges = 0

        for idx, line in enumerate(lines):
            upload_jobs[job_id].update({
                "progress": idx,
                "current": f"Campaign {idx + 1}/{total_campaigns}",
            })

            try:
                # Parse JSON
                data = json.loads(line)

                # Validate against Pydantic model
                parsed = ADRSParsedCampaign.model_validate(data)

                # Set author from form field only if not already set (e.g. from author_input.json)
                if author and not parsed.campaign.author:
                    parsed.campaign.author = author

                # Insert into database
                insert_start = time.time()
                inserter = CampaignInserter()
                insert_success = await asyncio.to_thread(inserter.insert, parsed)
                insert_elapsed = time.time() - insert_start

                if not insert_success:
                    errors.append(f"Line {idx + 1}: Database insertion failed")
                    continue

                print(f"[JOB {job_id}] Inserted campaign {idx + 1}/{total_campaigns} in {insert_elapsed:.2f}s", flush=True)

                # Collect stats
                campaign_measurements = sum(len(m) for m in parsed.measurements.values())
                total_candidates += len(parsed.candidates)
                total_measurements += campaign_measurements
                total_artifacts += len(parsed.artifacts)
                total_edges += len(parsed.candidate_edges)

                campaign_results.append({
                    "campaign_id": parsed.campaign.name or f"campaign_{idx + 1}",
                    "name": parsed.campaign.name or f"campaign_{idx + 1}",
                    "candidates": len(parsed.candidates),
                })

            except json.JSONDecodeError as e:
                errors.append(f"Line {idx + 1}: Invalid JSON - {e}")
            except Exception as e:
                errors.append(f"Line {idx + 1}: Validation failed - {e}")

            await asyncio.sleep(0)

        # Final progress update
        upload_jobs[job_id].update({
            "progress": total_campaigns,
            "current": None,
        })

        # Check if any campaigns succeeded
        if not campaign_results:
            shutil.rmtree(extract_path, ignore_errors=True)
            upload_jobs[job_id].update({
                "status": "error",
                "message": f"Failed to process any campaigns. Errors: {'; '.join(errors)}",
            })
            return

        # NOTE: We no longer rename the directory after insertion.
        # The artifact URIs stored in the database reference paths under extract_path,
        # so renaming would break those references. Keep the upload_{timestamp} name.

        # Build final response
        total_stats = {
            "campaigns": len(campaign_results),
            "candidates": total_candidates,
            "measurements": total_measurements,
            "artifacts": total_artifacts,
            "edges": total_edges,
        }

        elapsed = time.time() - start_time
        print(f"[JOB {job_id}] Completed in {elapsed:.2f}s - {len(campaign_results)} campaign(s)", flush=True)

        upload_jobs[job_id].update({
            "status": "complete",
            "message": f"Successfully uploaded {len(campaign_results)} campaign(s) from _parsed.jsonl",
            "result": UploadResponse(
                success=True,
                message=f"Successfully uploaded {len(campaign_results)} campaign(s) from _parsed.jsonl",
                campaign_count=len(campaign_results),
                campaigns=campaign_results,
                system_type="parsed",
                total_stats=total_stats,
                errors=errors if errors else None,
            ),
        })

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[JOB {job_id}] Failed after {elapsed:.2f}s - {str(e)}", flush=True)
        upload_jobs[job_id].update({
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
        })

    finally:
        if tar_path.exists():
            try:
                tar_path.unlink()
            except Exception:
                pass
        if extract_path and extract_path.exists():
            shutil.rmtree(extract_path, ignore_errors=True)


@app.post("/upload/parsed")
async def upload_parsed_campaign(
    file: Annotated[UploadFile, File(description="tar.gz archive containing campaign data and _parsed.jsonl")],
    author: Annotated[str, Form(description="Email address of the campaign author (optional if in author_input.json)")] = "",
) -> dict:
    """Upload campaign data with agent-generated _parsed.jsonl.

    Use this endpoint when no built-in parser exists for your data format.
    The archive must contain:
    - Raw campaign folder(s) (for artifact storage)
    - _parsed.jsonl at the archive root (one ADRSParsedCampaign JSON per line)

    See GET /upload/formats for the schema reference.
    See GET /upload/instructions for the full workflow.

    Args:
        file: tar.gz archive containing campaign data and _parsed.jsonl
        author: Email address of the campaign author

    Returns:
        job_id: Unique identifier to poll for status
        status_url: URL to check processing status
    """
    job_id = str(uuid4())

    # Save file to disk
    tar_path = DATA_DIR / f"pending_{job_id}.tar.gz"
    file_size = 0

    try:
        print(f"[UPLOAD/PARSED] Receiving {file.filename} (job {job_id})...", flush=True)
        with open(tar_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
                file_size += len(chunk)
        print(f"[UPLOAD/PARSED] Received {file_size} bytes for job {job_id}", flush=True)
    except Exception as e:
        tar_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to receive upload: {e}")

    # Initialize job status
    upload_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": 0,
        "total": 0,
        "current": None,
        "message": "Starting processing...",
        "result": None,
    }

    # Start background processing
    asyncio.create_task(process_parsed_upload_job(job_id, tar_path, author))

    return {
        "job_id": job_id,
        "status_url": f"/upload/status/{job_id}",
        "message": "Upload received, processing started. Poll status_url for progress.",
    }


@app.get("/upload/status/{job_id}")
async def get_upload_status(job_id: str) -> JobStatus:
    """Get the status of an upload job.

    Poll this endpoint to track progress of a background upload.

    Args:
        job_id: The job ID returned from POST /upload

    Returns:
        JobStatus with current progress, status, and result when complete.
    """
    if job_id not in upload_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = upload_jobs[job_id]
    return JobStatus(**job)


@app.get("/upload/jobs")
async def list_upload_jobs() -> list[JobStatus]:
    """List all upload jobs (for debugging/monitoring).

    Returns:
        List of all jobs with their current status.
    """
    return [JobStatus(**job) for job in upload_jobs.values()]


@app.get("/upload/instructions")
async def get_upload_instructions():
    """Get structured workflow instructions for uploading campaign data.

    This is the primary reference for agents uploading campaign data.
    Follow these steps in order. Use /upload/formats for schema details.
    """
    return {
        "overview": "Try /upload first with detected system_type. Fall back to /upload/parsed if no parser matches OR if parsing fails.",
        "step_0_create_author_input": {
            "description": "ALWAYS create author_input.json before uploading (applies to both /upload and /upload/parsed)",
            "location": "Place at root of each campaign folder",
            "action": "Ask the user for their email and the GitHub repository link + commit hash of the system that produced this campaign",
            "schema": {
                "author": "Email address of the campaign author (required)",
                "version": "GitHub link + commit hash (e.g., https://github.com/org/repo/commit/abc123)",
            },
            "example": {
                "author": "researcher@example.com",
                "version": "https://github.com/acme/skydiscover/commit/a1b2c3d4e5f6",
            },
            "why": "Centralizes upload metadata in one place. Author identifies who ran the campaign; version links to the exact code that produced it.",
        },
        "step_1_detect_system_type": {
            "description": "Check folder structure to identify format",
            "decision_tree": [
                {"check": "evolution_trace.jsonl exists", "system_type": "openevolve"},
                {"check": "summary.json with framework='gepa' AND iterations.jsonl", "system_type": "gepa"},
                {"check": "summary.json with framework='coding_agent' AND iterations.jsonl", "system_type": "coding_agent"},
                {"check": "config.yaml + output/ OR checkpoints/*/programs/", "system_type": "skydiscover"},
                {"check": "none match", "action": "use /upload/parsed with a custom parser script"},
            ],
        },
        "step_2_try_upload": {
            "endpoint": "POST /upload",
            "when": "System type detected from step_1",
            "content_type": "multipart/form-data",
            "fields": {
                "file": "tar.gz archive containing campaign folder(s) with author_input.json",
                "author": "Email address of the uploader (required)",
                "system_type": "Detected value from step_1 (skydiscover, openevolve, gepa, coding_agent) — required, no auto-detect",
                "version": "GitHub commit link (optional, overridden by author_input.json if present)",
            },
            "create_archive": "tar -czf upload.tar.gz campaign_folder/",
            "poll": "GET /upload/status/{job_id} every 2-5 seconds",
            "success_condition": "status='complete' AND result.success=true",
            "fallback_triggers": [
                "status='error'",
                "status='complete' but result.success=false",
                "result.errors contains parse failures",
            ],
            "on_fallback": "proceed to step_3",
        },
        "step_3_fallback_parsed": {
            "endpoint": "POST /upload/parsed",
            "when": [
                "No system_type matched in step_1",
                "OR /upload failed/errored in step_2",
            ],
            "workflow": [
                "1. GET /upload/formats — read the schema to understand target structure",
                "2. Read raw campaign files to understand the source data",
                "3. Read author_input.json — include author in ADRSCampaign and version in ADRSSystem",
                "4. Write a one-off Python script that maps source → ADRSParsedCampaign",
                "5. Run script to generate _parsed.jsonl (one JSON object per line, one campaign per line)",
                "6. Place _parsed.jsonl at archive root alongside raw campaign folders",
                "7. tar -czf upload.tar.gz campaign_folder/ _parsed.jsonl",
                "8. POST /upload/parsed with file (author form field optional if set in _parsed.jsonl)",
                "9. Poll GET /upload/status/{job_id} until complete",
            ],
            "content_type": "multipart/form-data",
            "fields": {
                "file": "tar.gz archive containing campaign folder(s) AND _parsed.jsonl at root",
                "author": "Email address of the uploader (optional if author is set in _parsed.jsonl data)",
            },
            "schema_reference": "GET /upload/formats",
        },
        "endpoints_summary": {
            "GET /upload/instructions": "This endpoint - workflow guide",
            "GET /upload/formats": "Schema reference for building custom parsers",
            "POST /upload": "Upload with built-in parser (step 2)",
            "POST /upload/parsed": "Upload with agent-generated _parsed.jsonl (step 3)",
            "GET /upload/status/{job_id}": "Poll for progress",
        },
    }


@app.get("/stats")
async def get_stats():
    """Get campaign statistics.

    Returns:
        Count of campaigns in the database
    """
    try:
        inserter = CampaignInserter()
        campaign_count = inserter.get_campaign_count()
        return {"campaign_count": campaign_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get summary analytics for all campaigns.

    Returns:
        Summary statistics including counts and lists of models/algorithms.
    """
    try:
        analytics = CampaignAnalytics()
        return analytics.get_summary_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/analytics/clusters")
async def get_analytics_clusters(distance_threshold: float = 0.1):
    """Cluster campaigns by research question embedding similarity.

    Uses agglomerative clustering with cosine distance.

    Args:
        distance_threshold: Maximum distance for clusters (0-2 for cosine).
            Lower values create more, tighter clusters. Default is 0.1.

    Returns:
        cluster_count: Number of clusters found.
        clusters: List of clusters, each with cluster_id, campaign_count, and campaigns.
        campaigns_without_embeddings: Count of campaigns that couldn't be clustered.
    """
    try:
        analytics = CampaignAnalytics()
        return analytics.cluster_by_research_question(distance_threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")


@app.get("/analytics/variance")
async def get_analytics_variance(
    min_runs: int = 2,
    group_by: str = "model_algorithm",
):
    """Get bimodality statistics for replicated runs (Q1: Basin Structure / Multimodality).

    Groups campaigns by research question and configuration, calculates bimodality coefficient
    to reveal which configurations have multimodal score distributions (distinct attractors).

    Args:
        min_runs: Minimum number of runs required to include a cell. Default 2.
        group_by: Grouping mode - "model", "algorithm", or "model_algorithm". Default "model_algorithm".

    Returns:
        cells: List of cells with bimodality statistics:
            - research_question, problem_label: Problem identification
            - models: Model(s) used (or algorithm if group_by="algorithm")
            - algorithm: Algorithm used (if group_by="model_algorithm")
            - n: Number of runs
            - mean, stddev: Central tendency and spread
            - bc, skewness, kurtosis: Bimodality coefficient and components
            - min, max: Score range
            - scores: Raw scores for distribution visualization
        summary: Aggregate statistics (total_cells, total_campaigns, high_bc_cells)
    """
    from analytics.q1_variance import get_variance_analysis

    if group_by not in ("model", "algorithm", "model_algorithm"):
        raise HTTPException(status_code=400, detail="group_by must be 'model', 'algorithm', or 'model_algorithm'")

    try:
        return get_variance_analysis(None, min_runs, group_by)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Variance analysis error: {str(e)}")


@app.get("/analytics/variance/problems")
async def get_variance_problems(
    min_runs: int = 2,
):
    """Get list of problems available for variance analysis.

    Returns:
        List of problem labels.
    """
    from analytics.q1_variance import get_problem_list

    try:
        problems = await asyncio.to_thread(get_problem_list, None, min_runs)
        return {"problems": problems}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting problems: {str(e)}")


@app.get("/analytics/variance/figure/{problem}")
async def get_variance_figure_for_problem(
    problem: str,
    group_by: str = "model_algorithm",
    min_runs: int = 2,
    models: str | None = None,
):
    """Get variance figure for a specific problem.

    Args:
        problem: URL-encoded problem label.
        group_by: "aggregate", "model", "algorithm", or "model_algorithm"
        min_runs: Minimum number of runs required. Default 2.
        models: Comma-separated list of model names to include. If not provided, all models shown.

    Returns:
        PNG image.
    """
    from analytics.q1_variance import get_variance_figure_for_problem as _get_figure
    from fastapi.responses import Response
    from urllib.parse import unquote

    if group_by not in ("aggregate", "model", "algorithm", "model_algorithm"):
        raise HTTPException(status_code=400, detail="group_by must be 'aggregate', 'model', 'algorithm', or 'model_algorithm'")

    models_filter = [m.strip() for m in models.split(",")] if models else None

    try:
        problem_decoded = unquote(problem)
        figure_bytes = await asyncio.to_thread(
            _get_figure, problem_decoded, None, min_runs, group_by, models_filter
        )
        return Response(content=figure_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Figure generation error: {str(e)}")


@app.get("/analytics/calibration")
async def get_analytics_calibration():
    """Get rule-of-3 calibration analysis for BoN runs (Q2: Stagnation Heuristic Calibration).

    Tests whether the rule-of-3 heuristic (k failures → p < 3/k at 95% confidence)
    is calibrated for BoN runs. After k non-improving iterations, the heuristic
    predicts escape within k/3 more iterations with ~63% probability.

    Returns:
        by_threshold: Calibration stats for each stagnation threshold (5, 10, ..., 50)
            - escape_rate: Fraction that escaped in the prediction window
            - miscalibration: Deviation from 63% reference (negative = conservative)
        by_score_bucket: Same stats split by score level
        overall: Aggregate statistics
        metadata: Analysis parameters
    """
    from analytics.q2_calibration import get_calibration_analysis

    try:
        return get_calibration_analysis()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibration analysis error: {str(e)}")


@app.get("/analytics/calibration/problems")
async def get_calibration_problems():
    """Get list of problems available for calibration analysis."""
    from analytics.q2_calibration import get_problems

    try:
        problems = await asyncio.to_thread(get_problems)
        return {"problems": problems}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching problems: {str(e)}")


@app.get("/analytics/calibration/trend")
async def get_calibration_trend():
    """Mixed-effects logistic regression testing escape ~ k trend.

    Returns odds ratio per 10-unit increase in stagnation threshold,
    with 95% credible interval, testing whether escape probability
    declines with stagnation length.
    """
    from analytics.q2_calibration import get_calibration_analysis

    try:
        analysis = await asyncio.to_thread(get_calibration_analysis)
        return analysis.get("escape_trend_model", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend model error: {str(e)}")


@app.get("/analytics/calibration/aggregate/figure")
async def get_calibration_aggregate_figure(problems: str | None = None):
    """Get aggregate calibration bar chart for selected problems.

    Args:
        problems: Comma-separated list of problems (e.g., "Knapsack,Palindrome").
            If not provided, shows empty state.
    """
    from analytics.q2_calibration import get_aggregate_figure
    from fastapi.responses import Response

    problems_list = None
    if problems:
        problems_list = [p.strip() for p in problems.split(",")]

    try:
        figure_bytes = await asyncio.to_thread(get_aggregate_figure, None, problems_list)
        return Response(content=figure_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Figure generation error: {str(e)}")


@app.get("/analytics/calibration/figure")
async def get_calibration_grouped_figure(problems: str, group_by: str = "model"):
    """Get calibration figure for selected problems, grouped by model/algorithm.

    Args:
        problems: Comma-separated list of problem labels (URL-encoded)
        group_by: How to group lines - "model", "algorithm", or "model_algorithm"

    Returns:
        PNG image showing escape rate vs stagnation length, aggregated across selected problems.
    """
    from analytics.q2_calibration import get_grouped_figure
    from fastapi.responses import Response
    from urllib.parse import unquote

    if group_by not in ("model", "algorithm", "model_algorithm"):
        raise HTTPException(status_code=400, detail="group_by must be 'model', 'algorithm', or 'model_algorithm'")

    try:
        problems_list = [unquote(p.strip()) for p in problems.split(",") if p.strip()]
        if not problems_list:
            raise HTTPException(status_code=400, detail="At least one problem must be specified")
        figure_bytes = await asyncio.to_thread(get_grouped_figure, problems_list, group_by)
        return Response(content=figure_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Figure generation error: {str(e)}")


# ===========================================================================
# Q3: Diversity Analysis
# ===========================================================================


@app.get("/analytics/code-diversity/problems")
async def get_code_diversity_problems():
    """Get list of problems available for diversity analysis."""
    from analytics.q3_diversity import get_problems

    try:
        problems = await asyncio.to_thread(get_problems)
        return {"problems": problems}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching problems: {str(e)}")


@app.get("/analytics/code-diversity/early/figure")
async def get_code_diversity_early_figure(
    early_fraction: float = 0.25,
    group_by: str | None = None,
    problems: str | None = None,
):
    """Get figure for early code diversity vs final outcome.

    Args:
        early_fraction: Fraction of iterations to consider "early" (default 0.25)
        group_by: None (no grouping), "model", "algorithm", or "model_algorithm"
        problems: Comma-separated list of problems to filter
    """
    from analytics.q3_diversity import get_early_diversity_scatter_figure
    from fastapi.responses import Response

    if not 0 < early_fraction < 1:
        raise HTTPException(status_code=400, detail="early_fraction must be between 0 and 1")

    if group_by is not None and group_by not in ("model", "algorithm", "model_algorithm"):
        raise HTTPException(status_code=400, detail="group_by must be 'model', 'algorithm', or 'model_algorithm'")

    problems_list = [p.strip() for p in problems.split(",")] if problems else None

    try:
        figure_bytes = await asyncio.to_thread(
            get_early_diversity_scatter_figure, None, early_fraction, group_by, problems_list
        )
        return Response(content=figure_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Figure generation error: {str(e)}")


@app.get("/analytics/code-diversity/decile/figure")
async def get_code_diversity_decile_figure(
    problems: str | None = None,
    group_by: str | None = None,
):
    """Get decile diversity line plot as PNG.

    Args:
        problems: Comma-separated list of problems to include
        group_by: None (by problem), "model", or "algorithm"
    """
    from analytics.q3_diversity import get_decile_diversity_figure, GROUP_BYS
    from fastapi.responses import Response

    if group_by is not None and group_by not in GROUP_BYS:
        raise HTTPException(status_code=400, detail=f"group_by must be one of: {GROUP_BYS}")

    problems_list = [p.strip() for p in problems.split(",")] if problems else None

    try:
        figure_bytes = await asyncio.to_thread(get_decile_diversity_figure, None, problems_list, group_by)
        return Response(content=figure_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Figure generation error: {str(e)}")


@app.get("/analytics/code-factors/problems")
async def get_code_factors_problems():
    """Get list of problems available for factor importance analysis."""
    from analytics.q3_diversity import get_mutation_factors_problems

    try:
        problems = await asyncio.to_thread(get_mutation_factors_problems)
        return {"problems": problems}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching problems: {str(e)}")


@app.get("/analytics/code-factors/importance/figure")
async def get_code_factors_importance_figure(
    group_by: str | None = None,
    problems: str | None = None,
):
    """Get factor importance bar chart.

    Args:
        group_by: None (aggregate), "model", "algorithm", or "model_algorithm"
        problems: Comma-separated list of problems to include
    """
    from analytics.q3_diversity import get_factor_importance_figure
    from fastapi.responses import Response

    if group_by is not None and group_by not in ("model", "algorithm", "model_algorithm"):
        raise HTTPException(status_code=400, detail="group_by must be 'model', 'algorithm', or 'model_algorithm'")

    problems_list = None
    if problems:
        problems_list = [p.strip() for p in problems.split(",")]

    try:
        figure_bytes = await asyncio.to_thread(get_factor_importance_figure, None, group_by, problems_list)
        return Response(content=figure_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Figure generation error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        Simple health status
    """
    try:
        # Test database connection via CampaignInserter
        inserter = CampaignInserter()
        inserter.test_connection()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "data_dir": str(DATA_DIR.absolute()),
    }


def _get_pydantic_schema_summary() -> dict:
    """Generate a summary of ADRS Pydantic models for agent consumption."""
    from adrs_models import (
        ADRSParsedCampaign,
        ADRSCampaign,
        ADRSCandidate,
        ADRSMeasurement,
        ADRSArtifact,
        ADRSCandidateEdge,
        ADRSSystem,
    )

    def summarize_model(model_class) -> dict:
        """Extract field info from a Pydantic model."""
        fields = {}
        for name, field_info in model_class.model_fields.items():
            field_type = str(field_info.annotation).replace("typing.", "").replace("<class '", "").replace("'>", "")
            fields[name] = {
                "type": field_type,
                "required": field_info.is_required(),
                "description": field_info.description or "",
            }
            if field_info.default is not None and field_info.default is not ...:
                fields[name]["default"] = str(field_info.default)
        return {
            "description": model_class.__doc__.strip().split("\n")[0] if model_class.__doc__ else "",
            "fields": fields,
        }

    return {
        "ADRSParsedCampaign": summarize_model(ADRSParsedCampaign),
        "ADRSCampaign": summarize_model(ADRSCampaign),
        "ADRSSystem": summarize_model(ADRSSystem),
        "ADRSCandidate": summarize_model(ADRSCandidate),
        "ADRSMeasurement": summarize_model(ADRSMeasurement),
        "ADRSArtifact": summarize_model(ADRSArtifact),
        "ADRSCandidateEdge": summarize_model(ADRSCandidateEdge),
    }


@app.get("/upload/formats")
async def get_upload_formats():
    """Get the ADRS schema for building custom parsers.

    Returns Pydantic model schemas, relationships, and author_input.json spec.
    Use this when writing a one-off parser for /upload/parsed.
    """
    return {
        "description": "ADRS schema reference for building custom parsers",
        "pydantic_models": _get_pydantic_schema_summary(),
        "relationships": {
            "campaign": "Exactly one ADRSCampaign per ADRSParsedCampaign",
            "candidates": "List of ADRSCandidate, each with unique external_id",
            "measurements": "Dict keyed by candidate.external_id → list of ADRSMeasurement",
            "artifacts": "List of ADRSArtifact, optionally linked to candidates via external_id",
            "candidate_edges": "List of ADRSCandidateEdge; source_external_id and target_external_id must match candidate external_ids",
        },
        "example_mapping": {
            "description": "How SkyDiscover/OpenEvolve raw files map to ADRS. Use as a reference for building parsers.",
            "source_files": {
                "programs": "checkpoints/checkpoint_N/programs/*.json — one JSON per program with id, parent_id, other_context_ids, metrics",
                "config": "config.yaml — campaign configuration (optional)",
                "summary": "output/summary.json — final metrics",
                "logs": "output/logs/*.log — timestamps for started_at/ended_at",
            },
            "system": {
                "name": "Hardcoded string identifying the framework (e.g., 'skydiscover', 'openevolve', 'gepa', 'coding_agent')",
                "version": "author_input.json → version (GitHub commit link)",
            },
            "campaign": {
                "name": "Folder name or summary.json → output_dir",
                "research_question": "config.yaml → research_question",
                "started_at": "First timestamp in log files",
                "ended_at": "Last timestamp in log files",
                "algorithm_used": "config.yaml → search.type or detected from logs",
                "models_used": "config.yaml → llm.model (extracted recursively)",
                "final_metrics": "summary.json (best_score, improvement_percent, etc.)",
            },
            "candidates": {
                "description": "One candidate per unique program. iteration_index = when first discovered.",
                "iteration_index": "program.json → iteration_found (0 = seed programs)",
                "external_id": "program.json → id (UUID, must be unique within campaign)",
                "candidate_type": "'program' (hardcoded)",
                "created_at": "program.json → timestamp",
            },
            "candidate_edges": {
                "description": "Two edge types track evolution lineage",
                "parent_edge": {
                    "source_external_id": "program.json → parent_id",
                    "target_external_id": "program.json → id",
                    "edge_type": "'parent'",
                    "meaning": "This program was mutated from the parent",
                },
                "context_edge": {
                    "source_external_id": "Each ID in program.json → other_context_ids[]",
                    "target_external_id": "program.json → id",
                    "edge_type": "'context'",
                    "meaning": "These programs were shown as history in Best-of-N sampling",
                },
            },
            "measurements": {
                "description": "Keyed by candidate external_id. All metrics stored as name-value pairs.",
                "common_measurements": [
                    {"name": "status", "source": "'VALID' if metrics.validity == 1.0 else 'INVALID'"},
                    {"name": "combined_score", "source": "program.json → metrics.combined_score"},
                    {"name": "validity", "source": "program.json → metrics.validity"},
                    {"name": "result_summary", "source": "Formatted string of all metrics"},
                ],
            },
            "artifacts": {
                "description": "References to files on disk. URI is relative path within campaign.",
                "examples": [
                    {"uri": "config.yaml", "iteration_index": None},
                    {"uri": "output/summary.json", "iteration_index": None},
                    {"uri": "checkpoints/checkpoint_5/programs/abc123.json", "iteration_index": 5, "external_id": "abc123"},
                ],
            },
        },
        "author_input": {
            "description": "Metadata file the uploading agent MUST create before uploading",
            "location": "author_input.json at each campaign folder root",
            "when_to_create": "ALWAYS create this file before uploading, for both /upload and /upload/parsed paths",
            "action": "Ask the user for their email and the GitHub repository link + commit hash of the system that produced this campaign",
            "fields": {
                "author": {
                    "type": "string",
                    "description": "Email address of the person who ran/owns this campaign. Maps to ADRSCampaign.author.",
                    "required": True,
                },
                "version": {
                    "type": "string",
                    "description": "GitHub link + commit hash of the system that produced this campaign. Maps to ADRSSystem.version.",
                    "format": "https://github.com/{org}/{repo}/commit/{hash}",
                    "required": False,
                },
            },
            "example": {
                "author": "researcher@example.com",
                "version": "https://github.com/acme/skydiscover/commit/a1b2c3d4e5f6",
            },
            "note": "All parsers (built-in and custom) should read this file and map author to ADRSCampaign.author and version to ADRSSystem.version",
        },
        "output_format": {
            "description": "For /upload/parsed, create _parsed.jsonl at archive root",
            "format": "JSON Lines - one ADRSParsedCampaign JSON object per line",
            "example_line": {
                "campaign": {
                    "system": {"name": "my_system", "version": "https://github.com/org/repo/commit/abc123"},
                    "name": "experiment_001",
                    "research_question": "Optimize matrix multiplication",
                },
                "candidates": [
                    {"iteration_index": 0, "external_id": "seed-1", "candidate_type": "program"},
                    {"iteration_index": 1, "external_id": "child-1", "candidate_type": "program"},
                ],
                "measurements": {
                    "seed-1": [{"name": "status", "value": "VALID"}, {"name": "combined_score", "value": "100.0"}],
                    "child-1": [{"name": "status", "value": "VALID"}, {"name": "combined_score", "value": "150.0"}],
                },
                "artifacts": [{"uri": "config.yaml", "iteration_index": None}],
                "candidate_edges": [{"source_external_id": "seed-1", "target_external_id": "child-1", "edge_type": "parent"}],
            },
        },
    }


# Mount static files for CSS and JS
app.mount("/css", StaticFiles(directory=Path(__file__).parent / "static" / "css"), name="css")
app.mount("/js", StaticFiles(directory=Path(__file__).parent / "static" / "js"), name="js")


@app.get("/")
async def root():
    """Serve the frontend HTML page."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
