#!/usr/bin/env python3
"""Fallback parser using Claude Agent SDK for unknown campaign formats.

This parser activates when deterministic parsers (NOUS, SkyDiscover) fail.
It uses Claude to explore the campaign directory and map files to the
ADRSParsedCampaign schema.

Requires ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN environment variable to be set.
"""

import asyncio
import json
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, query

from adrs_models import (
    ADRSArtifact,
    ADRSCampaign,
    ADRSCandidate,
    ADRSCandidateEdge,
    ADRSMeasurement,
    ADRSParsedCampaign,
    ADRSSystem,
)

# The schema as a string for the prompt
SCHEMA_DESCRIPTION = """
You must map the campaign data to this exact JSON structure (ADRSParsedCampaign):

{
  "campaign": {
    "system": {"name": "string (lowercase)", "version": "string|null"},
    "author": "",  // leave empty, will be filled by caller
    "name": "string|null",
    "research_question": "string|null",
    "started_at": "ISO8601 datetime|null",
    "ended_at": "ISO8601 datetime|null",
    "config_used": {...}|null,  // initial configuration used for the campaign
    "algorithm_used": "string|null",
    "models_used": ["string", ...]|null,  // LLM model names/IDs used (e.g., ["claude-3-opus", "gpt-4"])
    "total_cost_usd": number|null,
    "total_tokens": integer|null,
    "final_summary": "string|null",
    "final_metrics": {"metric": value, ...}|null
  },
  "candidates": [
    {
      "iteration_index": integer (0-based),
      "external_id": "string|null (required if has measurements or edges)",
      "candidate_type": "string|null",
      "created_at": "ISO8601 datetime|null"
    }
  ],
  "measurements": {
    "candidate_external_id": [
      {"name": "string (lowercase_with_underscores)", "value": "string (numbers as strings)"}
    ]
  },
  "artifacts": [
    {
      "iteration_index": integer|null (null for campaign-level files),
      "uri": "relative/path/from/campaign/root",
      "content_hash": "string|null",
      "size_bytes": integer|null,
      "mime_type": "string|null"
    }
  ],
  "candidate_edges": [
    {"source_external_id": "parent_id", "target_external_id": "child_id", "edge_type": "parent|context"}
  ]
}

IMPORTANT:
- Include an artifact entry for EVERY file you read during mapping
- All measurement values must be strings (convert numbers: 95.5 -> "95.5")
- iteration_index on candidates is 0-based
- external_id is required for candidates that have measurements or appear in edges
- edge_type defaults to "parent" if not specified
"""


async def parse_campaign_with_agent(campaign_path: Path) -> dict | None:
    """Use Claude Agent SDK to parse an unknown campaign format.

    Args:
        campaign_path: Path to the extracted campaign directory

    Returns:
        Dictionary matching ADRSParsedCampaign structure, or None on failure
    """
    prompt = f"""You are a data mapping agent. Your task is to explore a directory
and map its contents to a standardized schema.

Directory: {campaign_path}

TARGET SCHEMA (for each campaign):
{SCHEMA_DESCRIPTION}

INSTRUCTIONS:
1. First, study the schema above - understand what each field represents
2. Use Glob with "**/*" to get a full view of the directory structure
3. Identify how many separate campaigns exist - the directory may contain:
   - A single campaign at the root
   - A single campaign nested in subdirectories
   - Multiple campaigns in separate subdirectories
4. For each campaign found, read its files and map to the schema:
   - campaign: look for config files, metadata, summaries
   - candidates: look for solutions, hypotheses, programs, individuals proposed (one per iteration/round/generation)
   - measurements: look for scores, metrics, fitness values, evaluations
   - artifacts: track all files used for mapping
   - candidate_edges: look for relationships or sequences between candidates

Be thorough - the data may be organized differently than the schema. Your job is to
find the best mapping from whatever structure exists to the schema fields.

OUTPUT FORMAT:
- If you find ONE campaign: output a single JSON object matching the schema
- If you find MULTIPLE campaigns: output a JSON array of objects, each matching the schema

No explanation, no markdown code blocks - just the raw JSON.
"""

    result_text = None

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Glob", "Grep"],
            cwd=str(campaign_path),
            model="claude-sonnet-4-6",
        ),
    ):
        if hasattr(message, "result"):
            result_text = message.result

    if not result_text:
        return None

    # Extract JSON from the response (handle potential markdown wrapping)
    json_text = result_text.strip()
    if json_text.startswith("```"):
        # Remove markdown code block
        lines = json_text.split("\n")
        json_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse agent response as JSON: {e}")
        print(f"Response was: {result_text[:500]}...")
        return None


def dict_to_parsed_campaign(data: dict) -> ADRSParsedCampaign | None:
    """Convert a dictionary to ADRSParsedCampaign with validation."""
    try:
        # Build campaign
        system_data = data.get("campaign", {}).get("system", {})
        system = ADRSSystem(
            name=system_data.get("name", "unknown"),
            version=system_data.get("version"),
        )

        campaign_data = data.get("campaign", {})
        campaign = ADRSCampaign(
            system=system,
            author=campaign_data.get("author", ""),
            name=campaign_data.get("name"),
            research_question=campaign_data.get("research_question"),
            started_at=campaign_data.get("started_at"),
            ended_at=campaign_data.get("ended_at"),
            config_used=campaign_data.get("config_used"),
            algorithm_used=campaign_data.get("algorithm_used"),
            models_used=campaign_data.get("models_used"),
            total_cost_usd=campaign_data.get("total_cost_usd"),
            total_tokens=campaign_data.get("total_tokens"),
            final_summary=campaign_data.get("final_summary"),
            final_metrics=campaign_data.get("final_metrics"),
        )

        # Build candidates
        candidates = [
            ADRSCandidate(
                iteration_index=c.get("iteration_index", 0),
                external_id=c.get("external_id"),
                candidate_type=c.get("candidate_type"),
                created_at=c.get("created_at"),
            )
            for c in data.get("candidates", [])
        ]

        # Build measurements
        measurements: dict[str, list[ADRSMeasurement]] = {}
        for ext_id, measurement_list in data.get("measurements", {}).items():
            measurements[ext_id] = [
                ADRSMeasurement(name=m.get("name", ""), value=str(m.get("value", "")))
                for m in measurement_list
            ]

        # Build artifacts
        artifacts = [
            ADRSArtifact(
                iteration_index=a.get("iteration_index"),
                uri=a.get("uri", ""),
                content_hash=a.get("content_hash"),
                size_bytes=a.get("size_bytes"),
                mime_type=a.get("mime_type"),
            )
            for a in data.get("artifacts", [])
        ]

        # Build edges
        candidate_edges = [
            ADRSCandidateEdge(
                source_external_id=e.get("source_external_id", ""),
                target_external_id=e.get("target_external_id", ""),
                edge_type=e.get("edge_type", "parent"),
            )
            for e in data.get("candidate_edges", [])
        ]

        return ADRSParsedCampaign(
            campaign=campaign,
            candidates=candidates,
            measurements=measurements,
            artifacts=artifacts,
            candidate_edges=candidate_edges,
        )

    except Exception as e:
        print(f"Failed to convert dict to ADRSParsedCampaign: {e}")
        return None


async def parse_auto_campaign(campaign_path: Path) -> list[ADRSParsedCampaign] | ADRSParsedCampaign | None:
    """Parse campaign(s) using the Claude Agent SDK fallback.

    This is an async entry point for use in the FastAPI async context.

    Args:
        campaign_path: Path to the campaign directory (may contain multiple campaigns)

    Returns:
        Single ADRSParsedCampaign, list of them, or None if parsing fails
    """
    try:
        # Run the async agent
        print(f"[AUTO-PARSE] Starting agent for {campaign_path.name}...", flush=True)
        data = await parse_campaign_with_agent(campaign_path)

        if not data:
            print(f"[AUTO-PARSE] Agent returned no data for {campaign_path.name}", flush=True)
            return None

        # Handle multiple campaigns (array) or single campaign (object)
        if isinstance(data, list):
            total = len(data)
            results = []
            for i, campaign_data in enumerate(data):
                parsed = dict_to_parsed_campaign(campaign_data)
                if parsed:
                    name = parsed.campaign.name or f"campaign-{i+1}"
                    print(f"[AUTO-PARSE] [{i+1}/{total}] {name}: "
                          f"{len(parsed.iterations)} iterations, "
                          f"{len(parsed.candidates)} candidates, "
                          f"{len(parsed.artifacts)} artifacts", flush=True)
                    results.append(parsed)
                else:
                    print(f"[AUTO-PARSE] [{i+1}/{total}] Failed to convert campaign data", flush=True)
            print(f"[AUTO-PARSE] Completed: {len(results)}/{total} campaigns parsed successfully", flush=True)
            return results if results else None
        else:
            parsed = dict_to_parsed_campaign(data)
            if parsed:
                name = parsed.campaign.name or campaign_path.name
                print(f"[AUTO-PARSE] {name}: "
                      f"{len(parsed.iterations)} iterations, "
                      f"{len(parsed.candidates)} candidates, "
                      f"{len(parsed.artifacts)} artifacts", flush=True)
            else:
                print(f"[AUTO-PARSE] Failed to convert campaign data", flush=True)
            return parsed

    except Exception as e:
        print(f"Auto-parse failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_auto.py <campaign_path>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Path does not exist: {path}")
        sys.exit(1)

    result = asyncio.run(parse_auto_campaign(path))
    if result:
        print("\n--- Parsed Campaign ---")
        print(f"System: {result.campaign.system.name}")
        print(f"Name: {result.campaign.name}")
        print(f"Candidates: {len(result.candidates)}")
        print(f"Artifacts: {len(result.artifacts)}")
    else:
        print("Failed to parse campaign")
        sys.exit(1)
