#!/usr/bin/env python3
"""Check database status, list tables, show row counts, and inspect sample data."""

import json
import os
import psycopg

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/adrs"
)


def inspect_sample_campaign(cur):
    """Inspect a random campaign with detailed data."""

    # Get a random campaign
    cur.execute("""
        SELECT
            c.id,
            c.name,
            s.name as system_name,
            s.version as system_version,
            c.research_question,
            c.algorithm_used,
            c.started_at,
            c.ended_at,
            c.total_cost_usd,
            c.total_tokens
        FROM campaigns c
        JOIN systems s ON c.system_id = s.id
        ORDER BY RANDOM()
        LIMIT 1;
    """)

    campaign = cur.fetchone()
    if not campaign:
        print("\n✗ No campaigns found")
        return

    campaign_id, name, system_name, system_version, research_q, algorithm, started, ended, cost, tokens = campaign

    print(f"\n📊 Randomly Selected Campaign")
    print(f"   Name: {name}")
    print(f"   System: {system_name} ({system_version})")
    print(f"   Algorithm: {algorithm}")
    if research_q:
        print(f"   Research Question: {research_q[:80]}...")
    if started:
        print(f"   Started: {started}")
    if ended:
        print(f"   Ended: {ended}")
    if cost:
        print(f"   Total Cost: ${cost:.2f}")
    if tokens:
        print(f"   Total Tokens: {tokens:,}")

    # Count candidates for this campaign (iterations are implicit via iteration_index)
    cur.execute("""
        SELECT COUNT(*), COUNT(DISTINCT iteration_index) FROM candidates WHERE campaign_id = %s;
    """, (campaign_id,))
    cand_count, iter_count = cur.fetchone()
    print(f"\n   Total Iterations: {iter_count}")
    print(f"   Total Candidates: {cand_count}")

    if cand_count == 0:
        print("\n   ✗ No candidates found for this campaign")
        return

    # Get a random candidate
    cur.execute("""
        SELECT
            id,
            iteration_index,
            external_id,
            candidate_type,
            created_at
        FROM candidates
        WHERE campaign_id = %s
        ORDER BY RANDOM()
        LIMIT 1;
    """, (campaign_id,))

    candidate = cur.fetchone()
    cand_id, iter_idx, external_id, cand_type, created_at = candidate

    print(f"\n💡 Randomly Selected Candidate")
    print(f"   Iteration Index: {iter_idx}")
    print(f"   External ID: {external_id}")
    print(f"   Type: {cand_type}")
    if created_at:
        print(f"   Created: {created_at}")

    # Get measurements for this candidate
    cur.execute("""
        SELECT
            name,
            value
        FROM measurements
        WHERE candidate_id = %s
        ORDER BY name
        LIMIT 10;
    """, (cand_id,))

    measurements = cur.fetchall()
    if measurements:
        print(f"\n   📊 Measurements ({len(measurements)} shown, max 10):")
        for meas in measurements:
            meas_name, meas_value = meas
            value_str = meas_value[:60] + "..." if len(meas_value) > 60 else meas_value
            print(f"      - {meas_name}: {value_str}")

    # Show artifact count
    cur.execute("""
        SELECT COUNT(*) FROM artifacts WHERE campaign_id = %s;
    """, (campaign_id,))
    artifact_count = cur.fetchone()[0]

    print(f"\n📁 Campaign Artifacts: {artifact_count} files")

    # Show sample artifacts
    cur.execute("""
        SELECT
            uri,
            size_bytes,
            mime_type
        FROM artifacts
        WHERE campaign_id = %s
        ORDER BY RANDOM()
        LIMIT 3;
    """, (campaign_id,))

    sample_artifacts = cur.fetchall()
    if sample_artifacts:
        print(f"\n   Sample artifacts:")
        for art in sample_artifacts:
            uri, size, mime = art
            size_kb = size / 1024 if size else 0
            print(f"      - {uri}")
            print(f"        Size: {size_kb:.2f} KB, Type: {mime}")

    # Show candidate edges
    cur.execute("""
        SELECT COUNT(*)
        FROM candidate_edges ce
        JOIN candidates c ON ce.source_candidate_id = c.id
        WHERE c.campaign_id = %s;
    """, (campaign_id,))
    edge_count = cur.fetchone()[0]

    if edge_count > 0:
        print(f"\n🔗 Candidate Edges: {edge_count} relationships")

        # Show sample edges (with candidate types and edge type)
        cur.execute("""
            SELECT
                c1.candidate_type as source_type,
                c2.candidate_type as target_type,
                ce.edge_type
            FROM candidate_edges ce
            JOIN candidates c1 ON ce.source_candidate_id = c1.id
            JOIN candidates c2 ON ce.target_candidate_id = c2.id
            WHERE c1.campaign_id = %s
            ORDER BY RANDOM()
            LIMIT 3;
        """, (campaign_id,))

        sample_edges = cur.fetchall()
        if sample_edges:
            print(f"\n   Sample relationships:")
            for edge in sample_edges:
                source_type, target_type, edge_type = edge
                print(f"      - {source_type} --[{edge_type}]--> {target_type}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Check database status and inspect sample data"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check: show only table counts, skip detailed inspection",
    )
    args = parser.parse_args()

    print(f"Connecting to: {DATABASE_URL}\n")

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Check database version
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"✓ Database connected")
                print(f"  {version.split(',')[0]}\n")

                # List all tables in public schema
                cur.execute("""
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY tablename;
                """)
                tables = [row[0] for row in cur.fetchall()]

                if not tables:
                    print("✗ No tables found. Run 'python create_tables.py' first.")
                    return

                print(f"✓ Tables found: {len(tables)}\n")

                # Get row count for each table
                print("Table                    | Row Count")
                print("-" * 45)
                for table in tables:
                    cur.execute(f"SELECT COUNT(*) FROM {table};")
                    count = cur.fetchone()[0]
                    print(f"{table:24} | {count:>9}")

                # Skip detailed inspection if --quick flag is set
                if args.quick:
                    print("\n✓ Quick check complete (use without --quick for detailed inspection)")
                    return

                # Show sample data inspection
                print("\n" + "=" * 80)
                print("SAMPLE DATA INSPECTION")
                print("=" * 80)

                cur.execute("SELECT COUNT(*) FROM campaigns;")
                campaigns_count = cur.fetchone()[0]

                if campaigns_count == 0:
                    print("\n✗ No campaigns found. Insert data first.")
                else:
                    inspect_sample_campaign(cur)

    except psycopg.OperationalError as e:
        print(f"✗ Connection failed: {e}")
        print("\nMake sure PostgreSQL is running and DATABASE_URL is correct.")
        print(f"Current DATABASE_URL: {DATABASE_URL}")


if __name__ == "__main__":
    main()
