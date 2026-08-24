#!/usr/bin/env python3
"""Common intermediate data models for ADRS database insertion.

This module defines Pydantic models that serve as an intermediate representation
for agentic research system outputs. Parsers for specific systems (NOUS, SkyDiscover,
etc.) should convert their output into these models before database insertion.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EvaluatorMetric(BaseModel):
    """A metric computed/measured by an evaluator."""

    name: str = Field(
        ...,
        description="Metric name as used in the evaluator (e.g., 'makespan', 'throughput')."
    )
    description: str = Field(
        ...,
        description="Brief description of what this metric measures (e.g., 'total time to complete all scheduled tasks')."
    )
    direction: str = Field(
        ...,
        description="What direction is better: 'higher_is_better', 'lower_is_better', or 'binary_pass_fail'."
    )


class EvaluatorSetup(BaseModel):
    """Structured representation of an evaluation setup.

    This captures what metrics are measured and how success is defined,
    enabling semantic comparison of evaluators across different campaigns
    and systems. Two campaigns with similar EvaluatorSetup embeddings
    can be meaningfully compared.
    """

    metrics: list[EvaluatorMetric] = Field(
        ...,
        description="List of metrics computed/measured by the evaluator."
    )
    success_criteria: str = Field(
        ...,
        description="What constitutes success/failure (e.g., 'program must complete without crash')."
    )


class ADRSSystem(BaseModel):
    """The research system/framework that produced the campaign data."""

    name: str = Field(
        ...,
        description="Name of the research system (e.g., 'nous', 'skydiscover', 'openevolve'). Use lowercase."
    )
    version: str | None = Field(
        None,
        description="Version identifier of the system. Can be semver, git hash, or descriptive string."
    )


class ADRSCampaign(BaseModel):
    """Top-level metadata for a research campaign/experiment run."""

    system: ADRSSystem = Field(
        ...,
        description="The research system that produced this campaign."
    )
    author: str = Field(
        default="",
        description="Email address of the person who ran/owns this campaign."
    )
    name: str | None = Field(
        None,
        description="Human-readable name for this campaign. Often derived from folder name or run ID."
    )
    research_question: str | None = Field(
        None,
        description="The hypothesis or question this campaign investigated. Free-form text."
    )
    started_at: datetime | None = Field(
        None,
        description="When the campaign started. ISO 8601 format (e.g., '2024-03-15T10:30:00')."
    )
    ended_at: datetime | None = Field(
        None,
        description="When the campaign ended. ISO 8601 format."
    )
    config_used: dict[str, Any] | None = Field(
        None,
        description="Initial configuration used for the campaign (from config.yaml or equivalent)."
    )
    algorithm_used: str | None = Field(
        None,
        description="Search/optimization algorithm used (e.g., 'nous', 'openevolve', 'best_of_n', 'evox')."
    )
    models_used: list[str] | None = Field(
        None,
        description="List of LLM model names/IDs used in this campaign. Normalized for easy querying (e.g., ['claude-3-opus', 'gpt-4']). Extracted from config_used by parsers."
    )
    total_cost_usd: float | None = Field(
        None,
        description="Total API cost in US dollars, if available."
    )
    total_tokens: int | None = Field(
        None,
        description="Total tokens consumed (input + output combined), if available."
    )
    final_summary: str | None = Field(
        None,
        description="Qualitative summary of what was discovered or achieved. Free-form text."
    )
    final_metrics: dict[str, Any] | None = Field(
        None,
        description="Final quantitative metrics. Example: {'best_score': 95.2, 'improvement_percent': 15.3}."
    )
    evaluator_setup: EvaluatorSetup | None = Field(
        None,
        description="Structured representation of the evaluation setup (metrics, directions, success criteria). Extracted by LLM from evaluator code (SkyDiscover) or campaign config (NOUS). Used for semantic similarity matching of evaluators."
    )


class ADRSCandidate(BaseModel):
    """A candidate solution, hypothesis, or program generated during the campaign.

    This is the central entity — what the research system proposed and evaluated.
    For SkyDiscover: one candidate per iteration (the program).
    For NOUS: one candidate per iteration (the 4-arm bundle, with arm statuses as measurements).

    Note: The actual content (code, hypothesis text) is stored in artifacts, not in this table.
    This keeps the database lean while artifacts preserve full fidelity.
    """

    iteration_index: int = Field(
        ...,
        description="Zero-based iteration number. Each iteration produces exactly one candidate."
    )
    external_id: str | None = Field(
        None,
        description="Unique ID for this candidate within the campaign. Required for measurements and edges. Use the source system's ID if available."
    )
    candidate_type: str | None = Field(
        None,
        description="Category of candidate. SkyDiscover: 'program'. NOUS: 'hypothesis_bundle'."
    )
    created_at: datetime | None = Field(
        None,
        description="When this candidate was created. ISO 8601 format."
    )
    solution_summary: str | None = Field(
        None,
        description="LLM-generated summary of the algorithmic approach. Describes the algorithm type, data structures, optimization strategy, and scientific concepts used."
    )
    solution_summary_embedding: list[float] | None = Field(
        None,
        description="384-dimensional embedding of the solution_summary for diversity analysis."
    )


class ADRSMeasurement(BaseModel):
    """A named metric, result, or outcome associated with a candidate.

    All values are stored as strings. Convert numbers to strings (e.g., "327.0", "0.85").

    Common measurement names:
    - 'status': evaluation result ('VALID', 'INVALID', 'SUCCEED', 'FAILED')
    - 'combined_score': overall score from the evaluator
    - 'conclusion': summary of evaluation outcome
    - 'error_type': type of error if evaluation failed
    - System-specific metrics: 'makespan', 'accuracy', 'validity', etc.
    """

    name: str = Field(
        ...,
        description="Measurement name. Use lowercase_with_underscores. Examples: 'status', 'combined_score', 'makespan'."
    )
    value: str = Field(
        ...,
        description="Measurement value as a string. Numbers should be converted to strings (e.g., '95.5', '1.0')."
    )


class ADRSArtifact(BaseModel):
    """A source file that was read during mapping.

    IMPORTANT FOR AGENTS: You must include an artifact entry for EVERY file you read
    while mapping source data to this schema. This enables verification that the mapping
    is correct by allowing reviewers to trace values back to their source files.

    Examples of files to include:
    - Config files (config.yaml, run_metadata.json)
    - Result files (best_program_info.json, summary.json)
    - Program/code files (best_program.cpp, solution.py)
    - Checkpoint data (checkpoint_*/metadata.json)
    - Log files (*.log)

    If you read it to extract data, include it as an artifact.
    """

    iteration_index: int | None = Field(
        None,
        description="Which iteration this artifact belongs to, or null for campaign-level artifacts."
    )
    external_id: str | None = Field(
        None,
        description="External ID of the candidate this artifact belongs to (e.g., program ID from checkpoint)."
    )
    uri: str = Field(
        ...,
        description="Path to the source file, relative to campaign root."
    )
    content_hash: str | None = Field(
        None,
        description="SHA256 hash of file content."
    )
    size_bytes: int | None = Field(
        None,
        description="File size in bytes."
    )
    mime_type: str | None = Field(
        None,
        description="MIME type."
    )


class ADRSCandidateEdge(BaseModel):
    """A directed relationship between two candidates (parent → child).

    Used to track lineage: which candidate was derived from which.
    Both IDs must match external_id values in the candidates list.

    Edge types:
    - 'parent': Direct derivation (SkyDiscover parent_id, NOUS iter(N-1) → iter(N))
    - 'context': Historical context shown during generation (SkyDiscover other_context_ids)
    """

    source_external_id: str = Field(
        ...,
        description="external_id of the parent/source candidate."
    )
    target_external_id: str = Field(
        ...,
        description="external_id of the child/derived candidate."
    )
    edge_type: str = Field(
        default="parent",
        description="Type of relationship: 'parent' (derived from) or 'context' (shown as history)."
    )


class ADRSParsedCampaign(BaseModel):
    """Complete campaign data ready for database insertion.

    This is the top-level model to submit to POST /upload/json.

    Relationships:
    - campaign: exactly one, required
    - candidates: one per iteration, each has iteration_index
    - measurements: keyed by candidate external_id
    - artifacts: zero or more, optionally linked to iterations by index
    - candidate_edges: parent/context relationships between candidates
    """

    campaign: ADRSCampaign = Field(
        ...,
        description="Campaign metadata. Required."
    )
    candidates: list[ADRSCandidate] = Field(
        default_factory=list,
        description="List of candidates, one per iteration. Each has a unique iteration_index."
    )
    measurements: dict[str, list[ADRSMeasurement]] = Field(
        default_factory=dict,
        description="Measurements keyed by candidate external_id."
    )
    artifacts: list[ADRSArtifact] = Field(
        default_factory=list,
        description="Source files read during mapping."
    )
    candidate_edges: list[ADRSCandidateEdge] = Field(
        default_factory=list,
        description="Parent/context relationships between candidates."
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
