"""Spec "Hard constraints": no personal names of contributors in committed code, seed
files, tests, fixtures or any output written by this feature.

Contributor prefixes are computed at runtime (read-only SELECT on the ROAR_GOLDEN_DBS
sources) with the spec's i06 rule; the diff base is ROAR_BASE_REV. Tests skip when these
are unset. Prefixes are NEVER printed or written: every assertion is on a count."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest

from conftest import REPO_ROOT, SRC_DIR, db_url, env_list, require_pg

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


def _source_names() -> dict[str, list]:
    """campaigns.name per ROAR_GOLDEN_DBS source (read-only SELECT)."""
    dbs = env_list("ROAR_GOLDEN_DBS")
    if not dbs:
        pytest.skip("set ROAR_GOLDEN_DBS (source DBs whose campaign names define contributor prefixes)")
    require_pg()
    out = {}
    for d in dbs:
        try:
            with psycopg.connect(db_url(d), connect_timeout=5) as conn:
                conn.read_only = True
                out[d] = [r[0] for r in conn.execute("SELECT name FROM campaigns").fetchall()]
        except psycopg.OperationalError:
            pytest.skip("a ROAR_GOLDEN_DBS source is not reachable")
    return out


@pytest.fixture(scope="session")
def v1_prefixes():
    names = [n for ns in _source_names().values() for n in ns]
    prefixes = frozenset(p for p in (local_contributor_prefix(n) for n in names) if p)
    found = len(prefixes)
    assert found > 0, "rule found no contributor prefixes; the leak tests would be vacuous"
    return _Prefixes(prefixes)


def _base_rev() -> str:
    rev = os.environ.get("ROAR_BASE_REV")
    if not rev:
        pytest.skip("set ROAR_BASE_REV to the pre-feature commit")
    ok = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], cwd=REPO_ROOT,
                        capture_output=True, text=True)
    if ok.returncode != 0:
        pytest.skip("ROAR_BASE_REV does not resolve in this checkout")
    return rev


def test_local_rule_matches_provenance_contributor_prefix(v1_prefixes):
    import provenance

    for _db, names in _source_names().items():
        mismatches = sum(1 for n in names if provenance.contributor_prefix(n) != local_contributor_prefix(n))
        n_prefixed = sum(1 for n in names if provenance.contributor_prefix(n) is not None)
        assert mismatches == 0
        if len(names) in (928, 1072):  # the spec's v1 / full corpora
            assert n_prefixed == 14


def _repo_change_texts(rev: str) -> list[str]:
    diff = subprocess.run(["git", "diff", rev, "--", "."], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout
    out = [diff]
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.split("\0")
    for rel in filter(None, untracked):
        try:
            out.append((REPO_ROOT / rel).read_text(errors="ignore"))
        except (OSError, UnicodeDecodeError):
            continue
        out.append(rel)
    return out


# Every assertion below compares a count: no failure message can contain a prefix or
# the scanned text.

def test_no_contributor_names_in_repo_changes(v1_prefixes):
    texts = _repo_change_texts(_base_rev())
    has_diff = bool(texts[0])
    assert has_diff, "empty diff vs ROAR_BASE_REV: nothing to check"
    leaked = sum(1 for t in texts if v1_prefixes.hits(t))
    assert leaked == 0


def test_no_contributor_names_in_seed_file(v1_prefixes):
    p = SRC_DIR / "sources_seed.yaml"
    exists = p.exists()
    assert exists, "src/sources_seed.yaml missing"
    n = v1_prefixes.hits(p.read_text())
    assert n == 0


def _first_golden(golden_copies):
    from conftest import golden_dbs

    gc = golden_copies(golden_dbs()[0])
    gc.require_backfill()
    return gc


@pytest.mark.golden
def test_no_contributor_names_in_export_outputs(golden_copies, v1_prefixes, tmp_path):
    import pyarrow.parquet as pq

    gc = _first_golden(golden_copies)
    from hf_export.export import export_provenance

    with psycopg.connect(gc.url) as conn:
        with conn.cursor() as cur:
            paths = export_provenance(cur, tmp_path)
    n_paths = len(paths)
    assert n_paths == 2
    n = 0
    for path in paths:
        path = Path(path)
        n += v1_prefixes.hits(path.read_bytes().decode("latin-1"))
        n += v1_prefixes.hits(str(pq.read_table(path).to_pylist()))
    assert n == 0


@pytest.mark.golden
def test_no_contributor_names_in_citations(golden_copies, v1_prefixes):
    gc = _first_golden(golden_copies)
    import provenance

    with psycopg.connect(gc.url) as conn:
        cs, srcs = provenance.load_provenance(conn)
        ids = [r[0] for r in conn.execute("SELECT id FROM campaigns").fetchall()]
    cites = provenance.required_citations(ids, cs, srcs)
    for fmt in ("bibtex", "text"):
        n = v1_prefixes.hits(provenance.format_citations(cites, fmt=fmt))
        assert n == 0, fmt


@pytest.mark.golden
def test_no_contributor_names_in_backfill_cli_output(golden_copies, v1_prefixes):
    gc = _first_golden(golden_copies)
    p = subprocess.run([sys.executable, str(SRC_DIR / "backfill_provenance.py"), gc.url],
                       cwd=SRC_DIR, capture_output=True, text=True, timeout=600)
    rc = p.returncode
    n = v1_prefixes.hits(p.stdout) + v1_prefixes.hits(p.stderr)
    assert n == 0
    assert rc == 0


def test_analytics_code_unchanged():
    """Spec: existing analytics are not modified (optional; needs ROAR_BASE_REV)."""
    rev = _base_rev()
    diff = subprocess.run(["git", "diff", "--stat", rev, "--", "src/analytics"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout
    changed_lines = len(diff.splitlines())
    assert changed_lines == 0
