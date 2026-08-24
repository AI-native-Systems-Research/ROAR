#!/usr/bin/env python3
"""Export candidate data for GPU embedding.

Two modes:
  --mode summarize  Query candidates needing summary embeddings, call LLM to summarize
                    code, output JSONL with {candidate_id, text} where text is the summary.
                    LLM calls are cached to disk.

  --mode direct     Query candidates needing direct code embeddings, load code from
                    artifacts, output JSONL with {candidate_id, text} where text is raw code.

Output is designed for embed_batch.py which runs on GPU cluster.

Usage:
    python export_for_embedding.py --mode summarize --output summaries.jsonl
    python export_for_embedding.py --mode direct --output code_to_embed.jsonl
    python export_for_embedding.py --mode summarize --output summaries.jsonl --limit 1000
"""

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Callable

import psycopg

from litellm_client import llm_batch_call_sync

# ---------------------------------------------------------------------------
# LLM Summarization with caching
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / ".cache"
SUMMARY_CACHE_FILE = CACHE_DIR / "code_summaries.json"

SUMMARIZE_SYSTEM_PROMPT = """You are an expert algorithm analyst. Your task is to summarize code solutions by identifying their core algorithmic approach.

Focus on:
1. **Algorithm type**: What class of algorithm is this? (e.g., dynamic programming, greedy, branch-and-bound, genetic algorithm, simulated annealing, constraint propagation, graph search, divide-and-conquer, linear programming relaxation)

2. **Key data structures**: What structures enable the solution? (e.g., priority queue, union-find, segment tree, hash map, adjacency list, memoization table)

3. **Optimization strategy**: How does it find good solutions? (e.g., local search with restarts, beam search, iterative refinement, pruning heuristics, relaxation and rounding)

4. **Scientific/mathematical concepts**: What principles from CS, math, or other sciences inform the approach? (e.g., graph theory, combinatorial optimization, probabilistic methods, physics-inspired annealing, biological evolution)

Output a 1-2 sentence summary that captures the essence of HOW the code solves the problem, not WHAT problem it solves. Be specific about the algorithm family and techniques used.

Examples of good summaries:
- "Dynamic programming with bitmask state compression to track visited subsets, using memoization to avoid recomputation."
- "Greedy bin-packing heuristic using first-fit-decreasing strategy, with items sorted by height for shelf-based placement."
- "Genetic algorithm with tournament selection and two-point crossover, using a penalty function for constraint violations."
- "Branch-and-bound search with linear programming relaxation for bounds, pruning branches that cannot improve the incumbent."
"""

SUMMARIZE_USER_TEMPLATE = """Summarize the algorithmic approach of this code:

```
{code}
```

Provide a 1-2 sentence summary focusing on algorithm type, key data structures, and optimization strategy."""


def _content_hash(content: str) -> str:
    """Hash content for caching."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _load_summary_cache() -> dict[str, str]:
    """Load cached summaries from disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if SUMMARY_CACHE_FILE.exists():
        try:
            with open(SUMMARY_CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_summary_cache(cache: dict[str, str]) -> None:
    """Save summaries cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def summarize_code_batch(
    code_items: list[tuple[str, str]],
    max_concurrency: int = 10,
    progress_callback: Callable[[int, int], None] | None = None,
    save_interval: int = 500,
) -> dict[str, str]:
    """Summarize a batch of code solutions via LLM with caching.

    Args:
        code_items: List of (identifier, code_string) tuples
        max_concurrency: Maximum concurrent LLM calls
        progress_callback: Optional callback(completed, total) for progress
        save_interval: Save cache to disk every N completions (default 500)

    Returns:
        Dict mapping identifier to summary string
    """
    summary_cache = _load_summary_cache()
    results: dict[str, str] = {}
    to_summarize: list[tuple[str, str, str]] = []  # (identifier, content_hash, code)

    for identifier, code in code_items:
        content_hash = _content_hash(code)
        if content_hash in summary_cache:
            results[identifier] = summary_cache[content_hash]
        else:
            to_summarize.append((identifier, content_hash, code))

    if not to_summarize:
        if progress_callback:
            progress_callback(len(results), len(results))
        return results

    prompts = [
        SUMMARIZE_USER_TEMPLATE.format(code=code[:12000])
        for _, _, code in to_summarize
    ]

    new_completions = [0]
    last_save_count = [0]

    def on_result(index: int, result: str | None):
        identifier, content_hash, _ = to_summarize[index]
        if result:
            summary_cache[content_hash] = result
            results[identifier] = result
        new_completions[0] += 1
        if new_completions[0] - last_save_count[0] >= save_interval:
            _save_summary_cache(summary_cache)
            last_save_count[0] = new_completions[0]

    def batch_progress(completed: int, total: int):
        cached_count = len(code_items) - len(to_summarize)
        if progress_callback:
            progress_callback(cached_count + completed, cached_count + total)

    llm_batch_call_sync(
        prompts=prompts,
        system_prompt=SUMMARIZE_SYSTEM_PROMPT,
        max_concurrency=max_concurrency,
        max_tokens=256,
        progress_callback=batch_progress,
        result_callback=on_result,
    )

    _save_summary_cache(summary_cache)
    return results


DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/adrs"
)


def load_code_from_uri(uri: str) -> str | None:
    """Load solution code from artifact URI.

    Handles multiple formats:
    1. JSON files with 'solution' key (skydiscover format)
    2. JSON files with 'code' key (openevolve format)
    3. Raw code files (.rs, .py, .cpp, etc.)
    """
    if not uri:
        return None
    try:
        with open(uri) as f:
            content = f.read()

        # If it's a JSON file, try to extract code from known keys
        if uri.endswith(".json"):
            try:
                data = json.loads(content)
                # Try 'solution' first (skydiscover), then 'code' (openevolve)
                return data.get("solution") or data.get("code")
            except json.JSONDecodeError:
                return None

        # Otherwise treat as raw code
        return content
    except OSError:
        return None


def get_candidates_needing_summary_embedding(limit: int | None = None) -> list[dict]:
    """Query candidates where solution_summary_embedding IS NULL.

    Returns candidates with code artifacts (JSON program files or raw source files).
    """
    query = """
        SELECT DISTINCT ON (c.id)
            c.id as candidate_id,
            a.uri as artifact_uri
        FROM candidates c
        JOIN artifacts a ON a.candidate_id = c.id
        WHERE c.solution_summary_embedding IS NULL
          AND (
            a.uri LIKE '%/programs/%'
            OR a.uri LIKE '%.rs'
            OR a.uri LIKE '%.py'
            OR a.uri LIKE '%.cpp'
            OR a.uri LIKE '%.c'
            OR a.uri LIKE '%.java'
            OR a.uri LIKE '%.js'
            OR a.uri LIKE '%.ts'
            OR a.uri LIKE '%.go'
          )
        ORDER BY c.id,
            CASE WHEN a.uri LIKE '%/programs/%' THEN 0 ELSE 1 END,
            COALESCE(a.size_bytes, 0) DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    results = []
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                results.append({
                    "candidate_id": row[0],
                    "artifact_uri": row[1],
                })
    return results


def get_candidates_needing_direct_embedding(limit: int | None = None) -> list[dict]:
    """Query candidates where direct_code_embedding IS NULL.

    Returns candidates with code artifacts (JSON program files or raw source files).
    For candidates with multiple artifacts, prefers .json files, then picks the largest
    code file to get the most representative code.
    """
    query = """
        SELECT DISTINCT ON (c.id)
            c.id as candidate_id,
            a.uri as artifact_uri
        FROM candidates c
        JOIN artifacts a ON a.candidate_id = c.id
        WHERE c.direct_code_embedding IS NULL
          AND (
            a.uri LIKE '%/programs/%'
            OR a.uri LIKE '%.rs'
            OR a.uri LIKE '%.py'
            OR a.uri LIKE '%.cpp'
            OR a.uri LIKE '%.c'
            OR a.uri LIKE '%.java'
            OR a.uri LIKE '%.js'
            OR a.uri LIKE '%.ts'
            OR a.uri LIKE '%.go'
          )
        ORDER BY c.id,
            CASE WHEN a.uri LIKE '%/programs/%' THEN 0 ELSE 1 END,
            COALESCE(a.size_bytes, 0) DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    results = []
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                results.append({
                    "candidate_id": row[0],
                    "artifact_uri": row[1],
                })
    return results


def export_summarize(
    output_path: str,
    limit: int | None = None,
    max_concurrency: int = 10,
    dry_run: bool = False,
) -> int:
    """Export summaries for embedding (calls LLM, uses cache)."""
    print("Querying candidates needing summary embeddings...")
    candidates = get_candidates_needing_summary_embedding(limit=limit)

    if not candidates:
        print("No candidates need summary embedding.")
        return 0

    print(f"Found {len(candidates)} candidates.")

    if dry_run:
        for c in candidates[:10]:
            print(f"  {c['candidate_id']}: {c['artifact_uri']}")
        if len(candidates) > 10:
            print(f"  ... and {len(candidates) - 10} more")
        return 0

    print("Loading code from artifacts...")
    code_items: list[tuple[str, str]] = []
    candidate_ids: list[uuid.UUID] = []
    load_failures = 0

    for c in candidates:
        code = load_code_from_uri(c["artifact_uri"])
        if code:
            code_items.append((str(c["candidate_id"]), code))
            candidate_ids.append(c["candidate_id"])
        else:
            load_failures += 1

    if load_failures > 0:
        print(f"  Warning: Failed to load code for {load_failures} candidates")

    if not code_items:
        print("No code loaded - nothing to process.")
        return 0

    print(f"Loaded code for {len(code_items)} candidates.")

    def progress(completed: int, total: int):
        if completed % 50 == 0 or completed == total:
            print(f"  Summarization: {completed}/{total}")

    print("Summarizing code (LLM calls, cached)...")
    summaries = summarize_code_batch(
        code_items,
        max_concurrency=max_concurrency,
        progress_callback=progress,
    )

    print(f"Writing to {output_path}...")
    count = 0
    with open(output_path, "w") as f:
        for candidate_id_str, _ in code_items:
            summary = summaries.get(candidate_id_str)
            if summary:
                record = {
                    "candidate_id": candidate_id_str,
                    "text": summary,
                }
                f.write(json.dumps(record) + "\n")
                count += 1

    print(f"Exported {count} summaries to {output_path}")
    return count


def export_direct(
    output_path: str,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Export code directly for embedding (no LLM calls)."""
    print("Querying candidates needing direct code embedding...")
    candidates = get_candidates_needing_direct_embedding(limit=limit)

    if not candidates:
        print("No candidates need direct embedding.")
        return 0

    print(f"Found {len(candidates)} candidates.")

    if dry_run:
        for c in candidates[:10]:
            print(f"  {c['candidate_id']}: {c['artifact_uri']}")
        if len(candidates) > 10:
            print(f"  ... and {len(candidates) - 10} more")
        return 0

    print("Loading code from artifacts...")
    count = 0
    load_failures = 0

    with open(output_path, "w") as f:
        for c in candidates:
            code = load_code_from_uri(c["artifact_uri"])
            if code:
                record = {
                    "candidate_id": str(c["candidate_id"]),
                    "text": code,
                }
                f.write(json.dumps(record) + "\n")
                count += 1
            else:
                load_failures += 1

    if load_failures > 0:
        print(f"  Warning: Failed to load code for {load_failures} candidates")

    print(f"Exported {count} code samples to {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Export candidate data for GPU embedding."
    )
    parser.add_argument(
        "--mode",
        choices=["summarize", "direct"],
        required=True,
        help="summarize: LLM summary (cached). direct: raw code.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max candidates to export (default: all pending)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Max concurrent LLM calls for summarize mode (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without doing it",
    )

    args = parser.parse_args()

    if args.mode == "summarize":
        export_summarize(
            output_path=args.output,
            limit=args.limit,
            max_concurrency=args.max_concurrency,
            dry_run=args.dry_run,
        )
    else:
        export_direct(
            output_path=args.output,
            limit=args.limit,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
