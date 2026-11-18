# ADRS Upload

Use these API endpoints:

1. `GET /upload/instructions` — Start here. Returns workflow guide with step-by-step process.
2. `GET /upload/formats` — Schema reference and mapping examples for custom parsers.
3. `POST /upload` — Upload with built-in parser (skydiscover, openevolve, gepa).
4. `POST /upload/parsed` — Upload with agent-generated `_parsed.jsonl`.
5. `GET /upload/status/{job_id}` — Poll until complete.

Always create `author_input.json` first — see `/upload/instructions` for details.

## Provenance: `assessor` and `source` blocks (required)

Every campaign needs an `assessor` block and a `source` block, in `author_input.json`
(a root file applies to every campaign below it; a nearer file overrides it per key) or,
for `/upload/parsed`, on each `_parsed.jsonl` line. Uploads without them are rejected.

- `assessor`: the evaluator that produced the scores: `name`, `version`, `platform`
  (non-empty; use `"unknown"` if not known) and optional `code_ref` (e.g. a commit URL).
- `source`: the dataset the campaign belongs to, used for citation and licensing.
  `consent` must be `true`. `redistribution` must be `full` (`pointer_only` is not accepted).
  - New named source: `attribution: "named"` with `slug` (e.g. `mydataset@v1`), `title`,
    `version`, `url`, `license_spdx` (SPDX id, e.g. `CC-BY-4.0`), `redistribution`, `authors`;
    optional `year` and `bibtex`.
  - Anonymous contributor: `attribution: "anonymous"` with `redistribution`, `license_spdx`
    and `consent`, and no `slug`, `title` or `authors`. The server assigns a slug such as
    `contributor-A@v1` and returns it as `anon_slug` in the job status.
  - Existing source: `{"slug": "<slug>", "consent": true}`. Anonymous sources accept
    re-uploads only from their original submitter.

Example `author_input.json`:

```json
{
  "author": "researcher@example.com",
  "version": "https://github.com/acme/skydiscover/commit/a1b2c3d4e5f6",
  "assessor": {"name": "my-evaluator", "version": "1.2.0", "platform": "linux-x86_64",
               "code_ref": "https://github.com/acme/evaluator/commit/abc123"},
  "source": {"attribution": "anonymous", "redistribution": "full",
             "license_spdx": "CC-BY-4.0", "consent": true}
}
```

Rules:

- One anonymous contributor per upload; keep the returned `anon_slug`.
- Don't put personal names in folder names or file paths; they become campaign names and artifact URIs.
