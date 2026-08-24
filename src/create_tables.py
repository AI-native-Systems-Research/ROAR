#!/usr/bin/env python3
"""Create ADRS database and all tables."""

import os
import psycopg
from psycopg import sql

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/adrs"
)

# Connection URL to postgres database (for creating the adrs database)
POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)

SCHEMA = """
-- Enable pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Systems: ADRS framework or implementation that produced the data
CREATE TABLE IF NOT EXISTS systems (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    version TEXT
);

-- Campaigns: one top-level investigation or discovery run
CREATE TABLE IF NOT EXISTS campaigns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id UUID REFERENCES systems(id) ON DELETE CASCADE,
    author TEXT NOT NULL,
    name TEXT,
    research_question TEXT,
    research_question_embedding vector(384),
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    config_used JSONB,
    algorithm_used TEXT,
    models_used TEXT[],
    total_cost_usd DOUBLE PRECISION,
    total_tokens BIGINT,
    final_summary TEXT,
    final_metrics JSONB,
    evaluator_setup JSONB
);

-- Candidates: the central object - what the ADRS system proposed
-- Each candidate belongs to one iteration (by index) within a campaign.
-- For SkyDiscover: 1 candidate per iteration (the program)
-- For NOUS: 1 candidate per iteration (the 4-arm bundle, with arm statuses as measurements)
-- Note: Actual content (code, hypothesis text) is stored in artifacts, not here.
CREATE TABLE IF NOT EXISTS candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    iteration_index INTEGER NOT NULL,
    external_id TEXT,
    candidate_type TEXT,
    created_at TIMESTAMP,
    solution_summary TEXT,
    solution_summary_embedding vector(384),
    direct_code_embedding vector(768),
    context_code_diversity DOUBLE PRECISION,
    UNIQUE (campaign_id, external_id)
);

-- Candidate edges: relationships between candidates for graph visualization
-- Captures evolution history: parent_id (who we derived from), context_ids (what history we saw)
CREATE TABLE IF NOT EXISTS candidate_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    target_candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    edge_type TEXT DEFAULT 'parent'
);

-- Measurements: named values (metrics, results, outcomes) for candidates
-- For SkyDiscover: combined_score, validity, etc.
-- For NOUS: h-main_status, h-control_status, h-ablation_status, h-robustness_status,
--           prediction_accuracy, arms_correct, arms_total
CREATE TABLE IF NOT EXISTS measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value TEXT NOT NULL
);

-- Artifacts: references to preserved files (stored on disk, not in database)
CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    iteration_index INTEGER,
    uri TEXT NOT NULL,
    content_hash TEXT,
    size_bytes BIGINT,
    mime_type TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_candidates_campaign_id ON candidates(campaign_id);
CREATE INDEX IF NOT EXISTS idx_candidates_iteration_index ON candidates(campaign_id, iteration_index);
CREATE INDEX IF NOT EXISTS idx_candidate_edges_source ON candidate_edges(source_candidate_id);
CREATE INDEX IF NOT EXISTS idx_candidate_edges_target ON candidate_edges(target_candidate_id);
CREATE INDEX IF NOT EXISTS idx_candidate_edges_type ON candidate_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_measurements_candidate_id ON measurements(candidate_id);
CREATE INDEX IF NOT EXISTS idx_measurements_name ON measurements(name);
CREATE INDEX IF NOT EXISTS idx_artifacts_campaign_id ON artifacts(campaign_id);

-- Vector similarity index for research question embeddings (HNSW for better out-of-box performance)
CREATE INDEX IF NOT EXISTS idx_campaigns_research_question_embedding ON campaigns
    USING hnsw (research_question_embedding vector_cosine_ops);

-- GIN index for efficient array queries on models_used (e.g., WHERE 'claude-3-opus' = ANY(models_used))
CREATE INDEX IF NOT EXISTS idx_campaigns_models_used ON campaigns USING gin (models_used);

-- Vector similarity index for solution summary embeddings
CREATE INDEX IF NOT EXISTS idx_candidates_solution_summary_embedding ON candidates
    USING hnsw (solution_summary_embedding vector_cosine_ops);

-- Vector similarity index for direct code embeddings
CREATE INDEX IF NOT EXISTS idx_candidates_direct_code_embedding ON candidates
    USING hnsw (direct_code_embedding vector_cosine_ops);
"""


def create_database_if_not_exists(db_name):
    """Create the database if it doesn't exist."""
    print(f"Checking if database '{db_name}' exists...")

    with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (db_name,)
            )
            exists = cur.fetchone()

            if not exists:
                print(f"Creating database '{db_name}'...")
                cur.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(db_name)
                ))
                print(f"✓ Database '{db_name}' created")
            else:
                print(f"✓ Database '{db_name}' already exists")


def main():
    # Extract database name from DATABASE_URL
    db_name = DATABASE_URL.rsplit('/', 1)[-1]

    # Create database if it doesn't exist
    create_database_if_not_exists(db_name)

    # Connect to the database and create tables
    print(f"\nConnecting to: {DATABASE_URL}")
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
        conn.commit()
    print("✓ All tables created successfully")


if __name__ == "__main__":
    main()
