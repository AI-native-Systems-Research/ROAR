# Provenance test suite

Tests for `docs/PROVENANCE_SPEC.md`. Run from `src/`:

```
.venv/bin/python -m pytest tests            # default: everything except golden
.venv/bin/python -m pytest tests -m golden  # golden only (needs the env vars below)
.venv/bin/python -m pytest tests -m ""      # everything
```

DB-backed tests need Postgres with pgvector. They create and drop throwaway databases
named `roar_provtest_*` (golden copies: `<src>_provcopy_*`), and sweep stale ones at
session start. Databases named `roar_v1` or `roar_full` are never written.

| Variable | Used by | Meaning |
|---|---|---|
| `ROAR_PG_ADMIN_URL` | all DB tests | Admin URL (default `postgresql://postgres:postgres@localhost:55432/postgres`) |
| `ROAR_GOLDEN_DBS` | golden, name-leak | Comma-separated source DB names to copy and compare (read-only) |
| `ROAR_TEST_PG_CONTAINER` | golden | Docker container running the server; copies use `pg_dump \| psql` there |
| `ROAR_PARQUET_DIR` | golden | Directory of previously exported v1 parquet files |
| `ROAR_PARQUET_SHA256_FILE` | golden | `sha256sum`-format manifest for the files in `ROAR_PARQUET_DIR` |
| `ROAR_BASE_REV` | name-leak, analytics-unchanged | Pre-feature git revision to diff against |

Tests whose variables are unset are skipped.
