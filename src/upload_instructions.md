# ADRS Upload

Use these API endpoints:

1. `GET /upload/instructions` — Start here. Returns workflow guide with step-by-step process.
2. `GET /upload/formats` — Schema reference and mapping examples for custom parsers.
3. `POST /upload` — Upload with built-in parser (skydiscover, openevolve, gepa).
4. `POST /upload/parsed` — Upload with agent-generated `_parsed.jsonl`.
5. `GET /upload/status/{job_id}` — Poll until complete.

Always create `author_input.json` first — see `/upload/instructions` for details.
