#!/usr/bin/env python3
"""Import embeddings from JSONL into database.

Two modes:
  --mode summarize  Import summary text + embeddings into solution_summary and
                    solution_summary_embedding columns. Requires both --summaries
                    (from export) and --embeddings (from embed_batch).

  --mode direct     Import code embeddings into direct_code_embedding column.
                    Only requires --embeddings file.

Usage:
    python import_embeddings.py --mode summarize --summaries summaries.jsonl --embeddings summary_embeddings.jsonl
    python import_embeddings.py --mode direct --embeddings code_embeddings.jsonl
    python import_embeddings.py --mode direct --embeddings code_embeddings.jsonl --dry-run
"""

import argparse
import json
import os
import sys
import uuid
from collections import defaultdict

import numpy as np
import psycopg

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/adrs"
)

BATCH_SIZE = 500


def load_jsonl(path: str) -> dict[str, dict]:
    """Load JSONL file, return dict keyed by candidate_id."""
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["candidate_id"]] = record
    return records


def import_summarize(
    summaries_path: str,
    embeddings_path: str,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Import summaries and their embeddings into database."""
    print(f"Loading summaries from {summaries_path}...")
    summaries = load_jsonl(summaries_path)
    print(f"  Loaded {len(summaries)} summaries.")

    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = load_jsonl(embeddings_path)
    print(f"  Loaded {len(embeddings)} embeddings.")

    # Find candidates that have both summary and embedding
    candidate_ids = set(summaries.keys()) & set(embeddings.keys())
    print(f"  {len(candidate_ids)} candidates have both summary and embedding.")

    if not candidate_ids:
        print("No candidates to import.")
        return 0, 0

    if dry_run:
        sample = list(candidate_ids)[:5]
        for cid in sample:
            summary_preview = summaries[cid]["text"][:80] + "..."
            dim = len(embeddings[cid]["embedding"])
            print(f"  {cid}: {dim}-dim embedding, summary: {summary_preview}")
        if len(candidate_ids) > 5:
            print(f"  ... and {len(candidate_ids) - 5} more")
        return 0, 0

    print("Importing to database...")
    success_count = 0
    failure_count = 0

    candidate_list = list(candidate_ids)
    total = len(candidate_list)

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for batch_start in range(0, total, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total)
                batch = candidate_list[batch_start:batch_end]

                for cid in batch:
                    summary = summaries[cid]["text"]
                    embedding = embeddings[cid]["embedding"]
                    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

                    try:
                        cur.execute(
                            """
                            UPDATE candidates
                            SET solution_summary = %s,
                                solution_summary_embedding = %s
                            WHERE id = %s
                            """,
                            (summary, embedding_str, uuid.UUID(cid)),
                        )
                        if cur.rowcount > 0:
                            success_count += 1
                        else:
                            failure_count += 1
                    except Exception as e:
                        print(f"  Error updating {cid}: {e}")
                        failure_count += 1

                conn.commit()
                print(f"  Progress: {batch_end}/{total}")

    print(f"\nCompleted: {success_count} updated, {failure_count} failures")
    return success_count, failure_count


def import_direct(
    embeddings_path: str,
    dry_run: bool = False,
    resume: bool = False,
) -> tuple[int, int]:
    """Import direct code embeddings into database.

    Streams records to avoid loading entire file into memory.
    If resume=True, skips candidates that already have embeddings.
    """
    # Count lines first
    print(f"Counting records in {embeddings_path}...")
    total = 0
    with open(embeddings_path) as f:
        for line in f:
            if line.strip():
                total += 1
    print(f"  Found {total} embeddings.")

    if total == 0:
        print("No embeddings to import.")
        return 0, 0

    if dry_run:
        print("Sample records:")
        with open(embeddings_path) as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                record = json.loads(line)
                dim = len(record["embedding"])
                print(f"  {record['candidate_id']}: {dim}-dim embedding")
        if total > 5:
            print(f"  ... and {total - 5} more")
        return 0, 0

    # Get already imported IDs if resuming
    skip_ids: set[str] = set()
    if resume:
        print("Checking for already imported embeddings...")
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM candidates WHERE direct_code_embedding IS NOT NULL")
                skip_ids = {str(row[0]) for row in cur.fetchall()}
        print(f"  Found {len(skip_ids)} already imported, will skip.")

    print("Importing to database (streaming)...")
    success_count = 0
    failure_count = 0
    skipped_count = 0
    processed = 0

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            batch_records: list[tuple[str, str]] = []

            with open(embeddings_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    record = json.loads(line)
                    cid = record["candidate_id"]

                    if cid in skip_ids:
                        skipped_count += 1
                        processed += 1
                        continue

                    embedding = record["embedding"]
                    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                    batch_records.append((embedding_str, cid))

                    if len(batch_records) >= BATCH_SIZE:
                        for emb_str, candidate_id in batch_records:
                            try:
                                cur.execute(
                                    """
                                    UPDATE candidates
                                    SET direct_code_embedding = %s
                                    WHERE id = %s
                                    """,
                                    (emb_str, uuid.UUID(candidate_id)),
                                )
                                if cur.rowcount > 0:
                                    success_count += 1
                                else:
                                    failure_count += 1
                            except Exception as e:
                                print(f"  Error updating {candidate_id}: {e}")
                                failure_count += 1

                        conn.commit()
                        processed += len(batch_records)
                        print(f"  Progress: {processed}/{total} ({skipped_count} skipped)")
                        batch_records = []

            # Final batch
            if batch_records:
                for emb_str, candidate_id in batch_records:
                    try:
                        cur.execute(
                            """
                            UPDATE candidates
                            SET direct_code_embedding = %s
                            WHERE id = %s
                            """,
                            (emb_str, uuid.UUID(candidate_id)),
                        )
                        if cur.rowcount > 0:
                            success_count += 1
                        else:
                            failure_count += 1
                    except Exception as e:
                        print(f"  Error updating {candidate_id}: {e}")
                        failure_count += 1

                conn.commit()
                processed += len(batch_records)
                print(f"  Progress: {processed}/{total} ({skipped_count} skipped)")

    print(f"\nCompleted: {success_count} updated, {skipped_count} skipped, {failure_count} failures")
    return success_count, failure_count


def compute_context_code_diversity(
    candidate_ids: list[str] | None = None,
    dry_run: bool = False,
    batch_size: int = 1000,
) -> tuple[int, int]:
    """Compute and store context_code_diversity for candidates with direct code embeddings.

    For each candidate, computes the average pairwise cosine distance among its
    context candidates' embeddings. This is O(k²) per candidate where k is context size,
    but runs once at import time rather than on every analytics query.

    Args:
        candidate_ids: Optional list of candidate IDs to process. If None, processes
                       all candidates that have direct_code_embedding but no
                       context_code_diversity yet.
        dry_run: If True, show what would be computed without updating database.
        batch_size: Number of candidates to process per batch (default 1000).

    Returns:
        Tuple of (success_count, failure_count)
    """
    print("Computing context code diversity...")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Find candidates that need diversity computed
            if candidate_ids:
                placeholders = ",".join(["%s"] * len(candidate_ids))
                cur.execute(f"""
                    SELECT DISTINCT c.id
                    FROM candidates c
                    JOIN candidate_edges ce ON ce.target_candidate_id = c.id AND ce.edge_type = 'context'
                    WHERE c.direct_code_embedding IS NOT NULL
                      AND c.id IN ({placeholders})
                """, [uuid.UUID(cid) for cid in candidate_ids])
            else:
                cur.execute("""
                    SELECT DISTINCT c.id
                    FROM candidates c
                    JOIN candidate_edges ce ON ce.target_candidate_id = c.id AND ce.edge_type = 'context'
                    WHERE c.direct_code_embedding IS NOT NULL
                      AND c.context_code_diversity IS NULL
                """)

            candidates_to_process = [row[0] for row in cur.fetchall()]
            total_candidates = len(candidates_to_process)
            print(f"  Found {total_candidates} candidates to process.")

            if not candidates_to_process:
                return 0, 0

            if dry_run:
                print("  Dry run - would compute diversity for these candidates:")
                for cid in candidates_to_process[:5]:
                    print(f"    {cid}")
                if len(candidates_to_process) > 5:
                    print(f"    ... and {len(candidates_to_process) - 5} more")
                return 0, 0

            # Process in batches to avoid memory issues
            success_count = 0
            failure_count = 0

            for batch_start in range(0, total_candidates, batch_size):
                batch_end = min(batch_start + batch_size, total_candidates)
                batch_candidates = candidates_to_process[batch_start:batch_end]

                # Get context edges for this batch
                placeholders = ",".join(["%s"] * len(batch_candidates))
                cur.execute(f"""
                    SELECT ce.target_candidate_id, ce.source_candidate_id, c.direct_code_embedding
                    FROM candidate_edges ce
                    JOIN candidates c ON c.id = ce.source_candidate_id
                    WHERE ce.edge_type = 'context'
                      AND ce.target_candidate_id IN ({placeholders})
                      AND c.direct_code_embedding IS NOT NULL
                """, batch_candidates)

                # Group context embeddings by child candidate
                context_by_child: dict[uuid.UUID, list[list[float]]] = defaultdict(list)
                for row in cur.fetchall():
                    child_id, context_id, embedding = row
                    if embedding is not None:
                        # pgvector returns embeddings as strings like "[0.1,0.2,...]"
                        if isinstance(embedding, str):
                            embedding = json.loads(embedding)
                        context_by_child[child_id].append(embedding)

                # Compute diversity for each candidate in batch
                batch_updates: list[tuple[float, uuid.UUID]] = []

                for child_id, embeddings in context_by_child.items():
                    if len(embeddings) < 2:
                        # Need at least 2 context candidates for pairwise diversity
                        continue

                    try:
                        # Compute average pairwise cosine distance
                        emb_array = np.array(embeddings)
                        # Normalize for cosine distance
                        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
                        normalized = emb_array / norms

                        # Pairwise cosine distances: 1 - cosine_similarity
                        n = len(embeddings)
                        total_distance = 0.0
                        pair_count = 0
                        for i in range(n):
                            for j in range(i + 1, n):
                                cosine_sim = np.dot(normalized[i], normalized[j])
                                total_distance += 1 - cosine_sim
                                pair_count += 1

                        avg_diversity = total_distance / pair_count if pair_count > 0 else 0.0
                        batch_updates.append((float(avg_diversity), child_id))

                    except Exception as e:
                        print(f"  Error computing diversity for {child_id}: {e}")
                        failure_count += 1

                # Update database for this batch
                for diversity, child_id in batch_updates:
                    try:
                        cur.execute(
                            "UPDATE candidates SET context_code_diversity = %s WHERE id = %s",
                            (diversity, child_id),
                        )
                        if cur.rowcount > 0:
                            success_count += 1
                        else:
                            failure_count += 1
                    except Exception as e:
                        print(f"  Error updating {child_id}: {e}")
                        failure_count += 1

                conn.commit()
                print(f"  Progress: {batch_end}/{total_candidates} ({success_count} updated)")

    print(f"  Completed: {success_count} updated, {failure_count} failures")
    return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(
        description="Import embeddings from JSONL into database."
    )
    parser.add_argument(
        "--mode",
        choices=["summarize", "direct"],
        required=True,
        help="summarize: import summary + embedding. direct: import code embedding only.",
    )
    parser.add_argument(
        "--summaries",
        help="Summaries JSONL file (required for summarize mode)",
    )
    parser.add_argument(
        "--embeddings",
        "-e",
        required=True,
        help="Embeddings JSONL file from embed_batch.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without doing it",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip candidates that already have embeddings",
    )

    args = parser.parse_args()

    if args.mode == "summarize":
        if not args.summaries:
            print("Error: --summaries is required for summarize mode", file=sys.stderr)
            sys.exit(1)
        success, failures = import_summarize(
            summaries_path=args.summaries,
            embeddings_path=args.embeddings,
            dry_run=args.dry_run,
        )
    else:
        success, failures = import_direct(
            embeddings_path=args.embeddings,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        # Compute context diversity for newly imported embeddings
        if success > 0 and not args.dry_run:
            print("\nComputing context code diversity for imported candidates...")
            compute_context_code_diversity(dry_run=args.dry_run)

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
