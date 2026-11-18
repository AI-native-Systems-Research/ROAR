# ROAR provenance and assessor versioning: interface spec

This is the contract shared by the tests (`src/tests/`) and the implementation. Names, signatures and behaviours here are normative. Paths are relative to `src/` unless stated otherwise.

## Goals
1. Every score ingested **from now on** records the assessor that produced it: name, version, platform and code reference. The schema can represent a future regrade, meaning a new score for an existing candidate from a different assessor, and it distinguishes generation-time scores from regrades.
2. ROAR can cite every dataset it incorporates, now and in future:
   - per-source provenance and license;
   - a tool that emits exactly the citations a subset of campaigns needs;
   - support for anonymous contributors.

## Hard constraints
- **v1 data is immutable.** No existing row in `systems`, `campaigns`, `candidates`, `measurements`, `artifacts` or `candidate_edges` is updated or deleted by the migration or backfill. Existing analytics (`analytics/*`) are **not modified** and must return identical results after the migration and backfill.
- **Provenance DDL is a separate constant.** It is `create_tables.PROVENANCE_SCHEMA`; the existing `create_tables.SCHEMA` string is left byte-identical. This matters because external loaders apply `SCHEMA` verbatim to frozen databases. `create_tables.main()` applies `SCHEMA` and then `PROVENANCE_SCHEMA`.
- **Never write to frozen databases.** `backfill_provenance.py` and any test must refuse to write to a database named `roar_v1` or `roar_full`. They raise `provenance.ProvenanceError` before doing anything.
- **No personal names** of contributors may appear in committed code, seed files, tests, fixtures or any output written by this feature. Tests detect them at runtime only (see `provenance.contributor_prefix`).

## Schema (`create_tables.PROVENANCE_SCHEMA`)
The schema is additive and idempotent: `IF NOT EXISTS` everywhere.
```
CREATE TABLE IF NOT EXISTS source_datasets (
  slug TEXT PRIMARY KEY,            -- versioned, e.g. 'cheng2025barbarians@v1'
  title TEXT, version TEXT, year INTEGER, url TEXT,
  license_spdx TEXT,                -- NULL = none declared/recorded
  redistribution TEXT NOT NULL CHECK (redistribution IN ('full','pointer_only')),
  attribution TEXT NOT NULL CHECK (attribution IN ('named','anonymous')),
  authors TEXT, bibtex TEXT,        -- bibtex only if supplied; never generated into the DB
  consent_recorded_at TIMESTAMPTZ,  -- NULL = not recorded; set on first insert only
  owner TEXT,                       -- normalized submitter; NEVER exported
  CHECK (attribution <> 'anonymous' OR (authors IS NULL AND bibtex IS NULL)),
  CHECK (attribution <> 'named' OR (authors IS NOT NULL AND title IS NOT NULL
         AND title NOT ILIKE 'independent contributor%'))
);
CREATE SEQUENCE IF NOT EXISTS anon_contributor_seq START 1;
CREATE TABLE IF NOT EXISTS campaign_sources (
  campaign_id UUID PRIMARY KEY REFERENCES campaigns(id) ON DELETE CASCADE,
  source_slug TEXT NOT NULL REFERENCES source_datasets(slug),
  submitted_by TEXT                 -- normalized; NEVER exported; NULL for v1
);
CREATE TABLE IF NOT EXISTS evaluations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  assessor_name TEXT NOT NULL, assessor_version TEXT NOT NULL,
  platform TEXT NOT NULL, code_ref TEXT,
  kind TEXT NOT NULL CHECK (kind IN ('generation_time','regrade')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE measurements ADD COLUMN IF NOT EXISTS evaluation_id UUID REFERENCES evaluations(id);
```
- **Generation-time predicate.** A measurement is generation-time iff `evaluation_id IS NULL OR evaluations.kind = 'generation_time'`.
- **Drop order.** `drop_tables.py` drops `campaign_sources`, `source_datasets`, `evaluations` and the sequence (CASCADE), in addition to the existing tables.
- **Owner semantics.** `owner` is checked only for anonymous sources and is informational for named ones.

## Models (`adrs_models.py`)
- `ADRSAssessor(name: str, version: str, platform: str, code_ref: str | None = None)`.
  - `platform` must be a non-empty string; `"unknown"` is allowed.
- `ADRSSource(slug: str | None = None, title: str | None = None, version: str | None = None, year: int | None = None, url: str | None = None, license_spdx: str | None = None, redistribution: str | None = None, attribution: str | None = None, authors: str | None = None, bibtex: str | None = None, consent: bool | None = None)`.
- `ADRSParsedCampaign` gains `assessor: ADRSAssessor | None = None` and `source: ADRSSource | None = None`.
  - It must **not** have any field that carries a server-assigned anonymous slug.
  - Extra fields in uploaded JSON, such as `assigned_anon_slug`, are ignored or rejected and never honoured.

## Provenance input (`insert_adrs_campaign.resolve_provenance_input(campaign_path: Path, root: Path) -> dict`)
- **Merge order.** It merges every `author_input.json` found from `root` down to `campaign_path`, inclusive of both. The merge is per top-level key, and the file nearest the campaign wins. Nested dicts such as `assessor` and `source` are replaced whole, not deep-merged.
- **Return value.** A dict with any of the keys `author`, `version`, `assessor`, `source`.
- **Boundary.** It never reads above `root`.

## Submitter (`provenance.normalize_submitter(s) -> str | None`)
It strips and lowercases the value. `None` or an empty or whitespace-only string returns `None`.

## Anonymous labels
- **`provenance.label(n: int) -> str`.** Spreadsheet-style: 1→`A`, 26→`Z`, 27→`AA`, 52→`AZ`, 53→`BA`. `n < 1` raises `ValueError`.
- **`insert_adrs_campaign.allocate_anon_slug(conn) -> tuple[str, str]`.** Takes `nextval('anon_contributor_seq')` once and returns `(f"contributor-{label(n)}@v1", f"Independent contributor {label(n)}")`.

## Inserter (`insert_adrs_campaign.CampaignInserter.insert(parsed, assigned_anon_slug: str | None = None, submitted_by: str | None = None) -> bool`)
The existing behaviour is unchanged otherwise. All work happens in one transaction; any `ProvenanceError` rolls back everything for that campaign.

**Raises `provenance.ProvenanceError` if:**
- `parsed.assessor` or `parsed.source` is missing;
- `source.consent` is not `True`;
- the source would be, or is, `redistribution='pointer_only'`;
- **named source** (`attribution='named'`):
  - required fields are missing (`slug`, `title`, `version`, `url`, `license_spdx`, `redistribution`, `authors`);
  - or the slug exists and differs on any field the upload **supplied** (omitted `bibtex` or `year` is not compared);
- **anonymous first upload** (`attribution='anonymous'` without a slug):
  - the block includes `slug`, `title` or `authors`;
  - or `assigned_anon_slug` is None;
  - or the normalized `submitted_by` is empty;
- **existing-source reference** (a block whose only meaningful field is `slug`, or any block with a `slug` that is not a new named source):
  - the slug doesn't exist;
  - or the row is anonymous and the normalized `submitted_by` ≠ `owner`, or `owner` is NULL.

**Otherwise it:**
- creates the source row if needed, setting `consent_recorded_at=now()` and `owner=normalize_submitter(submitted_by)` on first insert only. An anonymous row gets `slug=assigned_anon_slug`, the title from `allocate_anon_slug`, and `authors` and `bibtex` NULL.
- inserts `campaign_sources(campaign_id, source_slug, submitted_by)`;
- inserts one `evaluations` row (`kind='generation_time'`, from `parsed.assessor`) and sets `measurements.evaluation_id` on **every** measurement inserted for that campaign;
- if the source is anonymous, writes the source title as `campaigns.author`. `submitted_by` and `owner` keep the original submitter.

`kind` is never taken from input; the inserter only creates `generation_time` evaluations.

## Callers
- **CLI `insert_all_campaigns(root, ...)`.**
  - It resolves provenance per campaign via `resolve_provenance_input(campaign, root)` and attaches `assessor` and `source` to each parsed campaign.
  - If any campaign has an anonymous first-upload block, it checks that all such blocks are identical, else raises `ProvenanceError` before inserting anything. It then calls `allocate_anon_slug` **once** and passes the slug to every insert.
  - `submitted_by` comes from the resolved `author`, read before any substitution.
- **CLI `insert_campaign(folder_path, system_type=None, provenance_root=None)`.** This is a one-campaign job: root is `provenance_root or folder_path`, and it allocates its own slug if the upload is anonymous.
- **API `/upload` and `/upload/parsed`.** Same rules, one slug per job.
  - `submitted_by` is the form `author`.
  - `/upload/parsed` takes a per-line `assessor`/`source` block over a root `author_input.json` found next to `_parsed.jsonl`.
  - The job result (`JobStatus`) exposes `anon_slug: str | None`.
- **`benchmark_direct_insert.py`.** Passes resolved provenance through; it does not allocate anonymous slugs, so anonymous first uploads fail there.

## Backfill (`backfill_provenance.py`)
- **Entry point.** `backfill(database_url: str) -> dict`, with CLI `python backfill_provenance.py <DATABASE_URL>`.
- **Refusals.** It raises `ProvenanceError` for databases named `roar_v1` or `roar_full`.
- **Steps.**
  - Apply `PROVENANCE_SCHEMA`.
  - Upsert the four seed sources from `sources_seed.yaml` (in `src/`). This is idempotent, and an existing identical row is a no-op.
  - Insert `campaign_sources` rows for every campaign that has none:
    - `ibm-ai-native-systems@v1`: `author == 'Roar Authors'` and `provenance.contributor_prefix(name) is not None`;
    - `ellis-castro-2026@v1`: every other `author == 'Roar Authors'` campaign;
    - `cheng2025barbarians@v1`: `author == 'Roar Authors 2'`.
  - Any campaign that is not mapped raises.
- **Return value.** Counts per slug. On a copy of `roar_v1` this is exactly `{ellis-castro-2026@v1: 914, ibm-ai-native-systems@v1: 14}`. On a copy of `roar_full` it adds `cheng2025barbarians@v1: 144`. Gupta has 0.
- **Re-running** returns the same counts and changes nothing.
- **Never** updates existing v1 rows or sets `evaluation_id` on v1 measurements.
- **`provenance.contributor_prefix(name) -> str | None`.** This is the i06 rule.
  - Match `^([^_/][^/]*?)(__|/)` and take group 1.
  - Return None unless that prefix is outside `{"cbl","cbl-multi","cloudcast","llm-sql","mas","other_cases","ns3"}` and the name does not match `^(eb1[a-z-]*|kimi-k2\.5|gpt-5\.4|gpt-5-mini|gpt-oss[a-z0-9-]*|opus|gemini-2\.5-flash|qwen3\.5[a-z0-9:]*)_(adaevolve|evox|best_of_n)`.

## Seed (`sources_seed.yaml`)
All four seed sources are `attribution: named`, `redistribution: full` and `license_spdx: CC-BY-4.0`, with consent recorded (permission granted 2026-11-17).

| slug | title | year | url | authors | bibtex |
|---|---|---|---|---|---|
| `ellis-castro-2026@v1` | Don't gamble, GAMBLe: An analytical framework for AI-driven research systems | 2026 | https://arxiv.org/abs/2606.02863 | Marquita Ellis and Paul Castro | arXiv entry for 2606.02863 |
| `cheng2025barbarians@v1` | Barbarians at the Gate: How AI is Upending Systems Research | 2025 | https://arxiv.org/abs/2510.06189 | the 17 authors of arXiv 2510.06189 | the arXiv entry |
| `gupta2026harness@v1` | Automated Discovery Has No Universally Superior Harness | 2026 | https://huggingface.co/datasets/harness-generalization/dataset | Akshat Gupta and Jermaine Lei and Alexander Lu and Gopala Anumanchipalli and Leshem Choshen | the given entry, key `gupta2026automateddiscoveryuniversallysuperior`, eprint 2607.18235 |
| `ibm-ai-native-systems@v1` | IBM Research AI-Native Systems contributed runs | none | none | IBM Research AI-Native Systems | none (generated) |

## Citation tooling (`provenance.py`)
- **`ProvenanceError(ValueError)`.**
- **`ROAR_SOURCE`.** A dict for ROAR itself: slug `roar`, plus the bibtex `@misc{lin2026roar, ...}` from the README.
- **`load_provenance(conn) -> tuple[DataFrame, DataFrame]`.** Returns `(campaign_sources[campaign_id, source_slug], sources[...])`. It never selects `owner` or `submitted_by`.
- **`required_citations(campaign_ids, campaign_sources, sources, extra_sources=()) -> list[dict]`.**
  - Returns ROAR first, then each distinct source used by `campaign_ids`, then each slug in `extra_sources`, deduplicated and sorted by slug after ROAR.
  - Raises `ProvenanceError` for any campaign id that has no `campaign_sources` row, or for an unknown extra slug.
  - Accepts ids as `str` or `UUID`.
- **`bibtex_key(slug) -> str`.** Replaces each run of characters outside `[A-Za-z0-9_-]` with `_`. For example, `cheng2025barbarians@v1` → `cheng2025barbarians_v1`.
- **`format_citations(sources, fmt="bibtex"|"text") -> str`.**
  - **bibtex:** uses the stored bibtex verbatim if present. Otherwise it generates `@misc{<bibtex_key(slug)>, author = {...}, title = {...}, year = {...}, howpublished = {\url{<url>}}, note = {version <version>}}`, omitting absent fields.
  - **anonymous sources:** render as their title only, with no authors.
  - **pointer_only sources:** the output includes the phrase `no license declared` when `license_spdx` is NULL.
  - **text:** one line per source.
  - Any other `fmt` raises `ValueError`.

## Export (`hf_export/export.py`)
- **`export_provenance(cur, output_dir: Path) -> list[Path]`.**
  - Writes `sources.parquet` (every `source_datasets` column except `owner`) and `campaign_sources.parquet` (`campaign_id`, `source_slug` only).
  - Does **not** rewrite the six v1 parquet files.
- **`evaluations.parquet`** is deferred and not part of this change.

## Docs
`upload_instructions.md`, `/upload/instructions` and `/upload/formats` describe:
- the `assessor` and `source` blocks;
- the rule "one anonymous contributor per upload; keep the returned `anon_slug`";
- "don't put personal names in folder names or file paths; they become campaign names and artifact URIs".

## Out of scope (do not build)
- A regrade ingestion path.
- Changing any analytics reader.
- Exporting `evaluations.parquet`.
- Scrubbing v1 identifiers.

## Appendix: exact Gupta et al. bibtex (normative; seed must equal this after whitespace normalization)
```
@misc{gupta2026automateddiscoveryuniversallysuperior,
      title={Automated Discovery Has No Universally Superior Harness}, 
      author={Akshat Gupta and Jermaine Lei and Alexander Lu and Gopala Anumanchipalli and Leshem Choshen},
      year={2026},
      eprint={2607.18235},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.18235}, 
}
```
