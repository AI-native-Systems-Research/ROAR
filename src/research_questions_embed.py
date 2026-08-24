#!/usr/bin/env python3
"""Backfill research_question_embedding for campaigns missing them.

This script queries campaigns where research_question_embedding IS NULL
but research_question IS NOT NULL, computes embeddings, and updates the database.

Usage:
    # Process all pending campaigns
    python research_questions_embed.py

    # Process in batches
    python research_questions_embed.py --batch-size 100

    # Dry run (show what would be processed)
    python research_questions_embed.py --dry-run
"""

import argparse
import os
import uuid

import psycopg

from embeddings import embed_texts

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/adrs"
)


def get_campaigns_needing_embeddings(
    limit: int | None = None,
) -> list[dict]:
    """Query campaigns where research_question_embedding IS NULL.

    Returns:
        List of dicts with keys: campaign_id, research_question
    """
    query = """
        SELECT id, research_question
        FROM campaigns
        WHERE research_question_embedding IS NULL
          AND research_question IS NOT NULL
        ORDER BY id
    """
    if limit:
        query += f" LIMIT {limit}"

    results = []
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                results.append({
                    "campaign_id": row[0],
                    "research_question": row[1],
                })
    return results


def update_campaign_embedding(
    campaign_id: uuid.UUID,
    embedding: list[float],
) -> bool:
    """Update a campaign's research_question_embedding."""
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE campaigns
                SET research_question_embedding = %s
                WHERE id = %s
                """,
                (embedding_str, campaign_id),
            )
        conn.commit()
        return cur.rowcount > 0


def process_pending(
    batch_size: int | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Process campaigns that need research question embeddings.

    Args:
        batch_size: Max campaigns to process (None = all)
        dry_run: If True, only report what would be processed

    Returns:
        Tuple of (success_count, failure_count)
    """
    print("Querying campaigns needing research question embeddings...")
    campaigns = get_campaigns_needing_embeddings(limit=batch_size)

    if not campaigns:
        print("No campaigns need processing.")
        return 0, 0

    print(f"Found {len(campaigns)} campaigns to process.")

    if dry_run:
        print("Dry run - would process:")
        for c in campaigns[:10]:
            q = c["research_question"][:80] + "..." if len(c["research_question"]) > 80 else c["research_question"]
            print(f"  {c['campaign_id']}: {q}")
        if len(campaigns) > 10:
            print(f"  ... and {len(campaigns) - 10} more")
        return 0, 0

    # Batch compute embeddings
    print("Computing embeddings...")
    questions = [c["research_question"] for c in campaigns]
    embeddings = embed_texts(questions)

    # Update database
    print("Updating database...")
    success_count = 0
    failure_count = 0

    for campaign, embedding in zip(campaigns, embeddings):
        try:
            if update_campaign_embedding(campaign["campaign_id"], embedding):
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            print(f"  Error updating {campaign['campaign_id']}: {e}")
            failure_count += 1

    print(f"\nCompleted: {success_count} updated, {failure_count} failures")
    return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(
        description="Backfill research question embeddings for campaigns."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Max campaigns to process (default: all pending)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes",
    )

    args = parser.parse_args()

    success, failures = process_pending(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    if failures > 0:
        exit(1)


if __name__ == "__main__":
    main()
