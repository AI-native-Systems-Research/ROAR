"""Spec "Hard constraints": no personal names of contributors in committed code, seed
files, tests, fixtures or any output written by this feature.

Contributor prefixes are computed at runtime (read-only SELECT on roar_v1) with the
spec's i06 rule. They are NEVER printed or written: every assertion is on a count."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import psycopg
import pytest

from conftest import REPO_ROOT, SRC_DIR, db_url, require_pg

EXCLUDED_PREFIXES = {"cbl", "cbl-multi", "cloudcast", "llm-sql", "mas", "other_cases", "ns3"}
_PREFIX_RE = re.compile(r"^([^_/][^/]*?)(__|/)")
_MODEL_RE = re.compile(
    r"^(eb1[a-z-]*|kimi-k2\.5|gpt-5\.4|gpt-5-mini|gpt-oss[a-z0-9-]*|opus|gemini-2\.5-flash|qwen3\.5[a-z0-9:]*)"
    r"_(adaevolve|evox|best_of_n)"
)


def local_contributor_prefix(name: str | None) -> str | None:
    """Independent re-statement of the spec rule (so this test does not trust the code under test)."""
    if not name:
        return None
    m = _PREFIX_RE.match(name)
    if not m:
        return None
    p = m.group(1)
    if p in EXCLUDED_PREFIXES or _MODEL_RE.match(name):
        return None
    return p


class _Prefixes:
    """Opaque holder so pytest never renders the values in a failure report."""

    def __init__(self, values: frozenset[str]):
        self._values = values
        self._patterns = [
            re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", re.IGNORECASE) for v in values
        ]

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"<{len(self._values)} contributor prefixes>"

    def hits(self, text: str) -> int:
        """Number of distinct prefixes occurring in text (case-insensitive, token-bounded)."""
        return sum(1 for p in self._patterns if p.search(text))

    def values(self):
        return self._values


@pytest.fixture(scope="session")
def v1_prefixes():
    require_pg()
    try:
        with psycopg.connect(db_url("roar_v1"), connect_timeout=5) as conn:
            conn.read_only = True
            names = [r[0] for r in conn.execute("SELECT name FROM campaigns").fetchall()]
    except psycopg.OperationalError:
        pytest.skip("roar_v1 not reachable")
    prefixes = frozenset(p for p in (local_contributor_prefix(n) for n in names) if p)
    assert len(prefixes) > 0, "rule found no contributor prefixes in roar_v1; the leak test would be vacuous"
    return _Prefixes(prefixes)


def test_local_rule_matches_provenance_contributor_prefix(v1_prefixes):
    import provenance

    with psycopg.connect(db_url("roar_v1"), connect_timeout=5) as conn:
        conn.read_only = True
        names = [r[0] for r in conn.execute("SELECT name FROM campaigns").fetchall()]
    mismatches = sum(1 for n in names if provenance.contributor_prefix(n) != local_contributor_prefix(n))
    n_prefixed = sum(1 for n in names if provenance.contributor_prefix(n) is not None)
    assert mismatches == 0
    assert n_prefixed == 14


def _repo_change_texts() -> list[tuple[str, str]]:
    diff = subprocess.run(["git", "diff", "e594ccc", "--", "."], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout
    out = [("git diff e594ccc", diff)]
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.split("\0")
    for rel in filter(None, untracked):
        p = REPO_ROOT / rel
        try:
            out.append((rel, p.read_text(errors="ignore")))
        except (OSError, UnicodeDecodeError):
            continue
        out.append((rel + " (path)", rel))
    return out


def test_no_contributor_names_in_repo_changes(v1_prefixes):
    texts = _repo_change_texts()
    assert texts and texts[0][1], "empty diff vs e594ccc: nothing to check (spec commit should differ)"
    leaked_files = sum(1 for _, t in texts if v1_prefixes.hits(t))
    assert leaked_files == 0, f"{leaked_files} changed/untracked file(s) contain a contributor prefix"


def test_no_contributor_names_in_seed_file(v1_prefixes):
    p = SRC_DIR / "sources_seed.yaml"
    assert p.exists(), "src/sources_seed.yaml missing"
    assert v1_prefixes.hits(p.read_text()) == 0


@pytest.mark.golden
def test_no_contributor_names_in_export_outputs(golden_copies, v1_prefixes, tmp_path):
    import pyarrow.parquet as pq

    gc = golden_copies("roar_v1")
    gc.require_backfill()
    from hf_export.export import export_provenance

    with psycopg.connect(gc.url) as conn:
        with conn.cursor() as cur:
            paths = export_provenance(cur, tmp_path)
    assert len(paths) == 2
    n = 0
    for path in paths:
        path = Path(path)
        n += v1_prefixes.hits(path.read_bytes().decode("latin-1"))
        n += v1_prefixes.hits(str(pq.read_table(path).to_pylist()))
    assert n == 0


@pytest.mark.golden
def test_no_contributor_names_in_v1_citations(golden_copies, v1_prefixes):
    gc = golden_copies("roar_v1")
    gc.require_backfill()
    import provenance

    with psycopg.connect(gc.url) as conn:
        cs, srcs = provenance.load_provenance(conn)
        ids = [r[0] for r in conn.execute("SELECT id FROM campaigns").fetchall()]
    cites = provenance.required_citations(ids, cs, srcs)
    for fmt in ("bibtex", "text"):
        assert v1_prefixes.hits(provenance.format_citations(cites, fmt=fmt)) == 0, fmt
