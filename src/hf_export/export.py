#!/usr/bin/env python3
"""Export ADRS database tables to Parquet for HuggingFace dataset upload."""

import json
import os
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/adrs",
)

OUTPUT_DIR = Path(__file__).parent / "data"


def parse_vector(text):
    """Parse pgvector text format '[0.1,0.2,...]' into a list of floats."""
    if text is None:
        return None
    return [float(x) for x in text.strip("[]").split(",")]


def export_table(cur, table_name: str, schema: pa.Schema, query: str):
    """Export a single table to Parquet."""
    cur.execute(query)
    columns = [desc.name for desc in cur.description]
    rows = cur.fetchall()

    if not rows:
        print(f"  {table_name}: 0 rows (skipping)")
        return

    arrays = {}
    for col_idx, col_name in enumerate(columns):
        field = schema.field(col_name)
        values = [row[col_idx] for row in rows]
        if field.type == pa.list_(pa.float32()):
            values = [parse_vector(v) for v in values]
        arrays[col_name] = pa.array(values, type=field.type)

    table = pa.table(arrays, schema=schema)
    output_path = OUTPUT_DIR / f"{table_name}.parquet"
    pq.write_table(table, output_path)
    print(f"  {table_name}: {len(rows)} rows -> {output_path.name}")


def export_systems(cur):
    schema = pa.schema([
        ("id", pa.string()),
        ("name", pa.string()),
        ("version", pa.string()),
    ])
    export_table(cur, "systems", schema, """
        SELECT id::text, name, version FROM systems
    """)


def export_campaigns(cur):
    schema = pa.schema([
        ("id", pa.string()),
        ("system_id", pa.string()),
        ("author", pa.string()),
        ("name", pa.string()),
        ("research_question", pa.string()),
        ("research_question_embedding", pa.list_(pa.float32())),
        ("started_at", pa.string()),
        ("ended_at", pa.string()),
        ("config_used", pa.string()),
        ("algorithm_used", pa.string()),
        ("models_used", pa.string()),
        ("total_cost_usd", pa.float64()),
        ("total_tokens", pa.int64()),
        ("final_summary", pa.string()),
        ("final_metrics", pa.string()),
        ("evaluator_setup", pa.string()),
    ])
    export_table(cur, "campaigns", schema, """
        SELECT
            id::text,
            system_id::text,
            author,
            name,
            research_question,
            research_question_embedding::text,
            started_at::text,
            ended_at::text,
            config_used::text,
            algorithm_used,
            models_used::text,
            total_cost_usd,
            total_tokens,
            final_summary,
            final_metrics::text,
            evaluator_setup::text
        FROM campaigns
    """)


def export_candidates(cur):
    schema = pa.schema([
        ("id", pa.string()),
        ("campaign_id", pa.string()),
        ("iteration_index", pa.int32()),
        ("external_id", pa.string()),
        ("candidate_type", pa.string()),
        ("created_at", pa.string()),
        ("solution_summary", pa.string()),
        ("solution_summary_embedding", pa.list_(pa.float32())),
        ("direct_code_embedding", pa.list_(pa.float32())),
        ("context_code_diversity", pa.float64()),
    ])
    export_table(cur, "candidates", schema, """
        SELECT
            id::text,
            campaign_id::text,
            iteration_index,
            external_id,
            candidate_type,
            created_at::text,
            solution_summary,
            solution_summary_embedding::text,
            direct_code_embedding::text,
            context_code_diversity
        FROM candidates
    """)


def export_measurements(cur):
    schema = pa.schema([
        ("id", pa.string()),
        ("candidate_id", pa.string()),
        ("name", pa.string()),
        ("value", pa.string()),
    ])
    export_table(cur, "measurements", schema, """
        SELECT id::text, candidate_id::text, name, value FROM measurements
    """)


def export_candidate_edges(cur):
    schema = pa.schema([
        ("id", pa.string()),
        ("source_candidate_id", pa.string()),
        ("target_candidate_id", pa.string()),
        ("edge_type", pa.string()),
    ])
    export_table(cur, "candidate_edges", schema, """
        SELECT
            id::text,
            source_candidate_id::text,
            target_candidate_id::text,
            edge_type
        FROM candidate_edges
    """)


def export_artifacts(cur):
    schema = pa.schema([
        ("id", pa.string()),
        ("campaign_id", pa.string()),
        ("candidate_id", pa.string()),
        ("iteration_index", pa.int32()),
        ("uri", pa.string()),
        ("content_hash", pa.string()),
        ("size_bytes", pa.int64()),
        ("mime_type", pa.string()),
    ])
    export_table(cur, "artifacts", schema, """
        SELECT
            id::text,
            campaign_id::text,
            candidate_id::text,
            iteration_index,
            uri,
            content_hash,
            size_bytes,
            mime_type
        FROM artifacts
    """)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ADRS database to {OUTPUT_DIR}/\n")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            export_systems(cur)
            export_campaigns(cur)
            export_candidates(cur)
            export_measurements(cur)
            export_candidate_edges(cur)
            export_artifacts(cur)

    print("\nDone.")


if __name__ == "__main__":
    main()
