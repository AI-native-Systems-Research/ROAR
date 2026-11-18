"""Spec "Seed (sources_seed.yaml)": the four seed sources and their exact fields."""

from __future__ import annotations

import re

import pytest
import yaml

from conftest import SEED_SLUGS, SRC_DIR

SEED_PATH = SRC_DIR / "sources_seed.yaml"

ALLOWED_KEYS = {"slug", "title", "version", "year", "url", "license_spdx", "redistribution", "attribution",
                "authors", "bibtex", "consent", "consent_recorded_at"}

EXPECTED = {
    "ellis-castro-2026@v1": {
        "title": "Don't gamble, GAMBLe: An analytical framework for AI-driven research systems",
        "year": 2026,
        "url": "https://arxiv.org/abs/2606.02863",
        "authors": "Marquita Ellis and Paul Castro",
    },
    "cheng2025barbarians@v1": {
        "title": "Barbarians at the Gate: How AI is Upending Systems Research",
        "year": 2025,
        "url": "https://arxiv.org/abs/2510.06189",
    },
    "gupta2026harness@v1": {
        "title": "Automated Discovery Has No Universally Superior Harness",
        "year": 2026,
        "url": "https://huggingface.co/datasets/harness-generalization/dataset",
        "authors": "Akshat Gupta and Jermaine Lei and Alexander Lu and Gopala Anumanchipalli and Leshem Choshen",
    },
    "ibm-ai-native-systems@v1": {
        "title": "IBM Research AI-Native Systems contributed runs",
        "year": None,
        "url": None,
        "authors": "IBM Research AI-Native Systems",
        "bibtex": None,
    },
}


def _load() -> dict[str, dict]:
    if not SEED_PATH.exists():
        pytest.fail("src/sources_seed.yaml is missing (spec: Seed)")
    data = yaml.safe_load(SEED_PATH.read_text())
    if isinstance(data, dict) and "sources" in data:
        data = data["sources"]
    if isinstance(data, dict):  # slug -> fields
        entries = [dict(v, slug=v.get("slug", k)) for k, v in data.items()]
    else:
        entries = list(data)
    out = {}
    for e in entries:
        assert isinstance(e, dict) and "slug" in e
        assert e["slug"] not in out, f"duplicate seed slug {e['slug']}"
        out[e["slug"]] = e
    return out


def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


def _field(bib: str, name: str) -> str | None:
    """Value of a bibtex field (brace-matched, quoted or bare)."""
    m = re.search(rf"(?<![A-Za-z]){name}\s*=\s*", bib, re.I)
    if not m:
        return None
    i = m.end()
    if i < len(bib) and bib[i] == "{":
        depth, j = 0, i
        while j < len(bib):
            depth += {"{": 1, "}": -1}.get(bib[j], 0)
            if depth == 0:
                return bib[i + 1:j]
            j += 1
        return None
    if i < len(bib) and bib[i] == '"':
        k = bib.find('"', i + 1)
        return bib[i + 1:k] if k != -1 else None
    m2 = re.match(r"[^,}]*", bib[i:])
    return m2.group(0).strip()


def test_seed_has_exactly_four_slugs():
    assert set(_load()) == set(SEED_SLUGS)


@pytest.mark.parametrize("slug", SEED_SLUGS)
def test_seed_common_fields(slug):
    e = _load()[slug]
    assert set(e) <= ALLOWED_KEYS, f"unexpected keys {set(e) - ALLOWED_KEYS}"
    assert "owner" not in e and "submitted_by" not in e
    assert e["attribution"] == "named"
    assert e["redistribution"] == "full"
    assert e["license_spdx"] == "CC-BY-4.0"
    consent = e.get("consent")
    recorded = e.get("consent_recorded_at")
    assert consent is True or str(recorded or "").startswith("2026-11-17"), "consent must be recorded"
    if recorded is not None:
        assert str(recorded).startswith("2026-11-17")


@pytest.mark.parametrize("slug", SEED_SLUGS)
def test_seed_table_fields(slug):
    e = _load()[slug]
    for k, v in EXPECTED[slug].items():
        assert e.get(k) == v, f"{slug}.{k}"
    assert e.get("title") and not e["title"].lower().startswith("independent contributor")
    assert e.get("authors")


def test_seed_cheng_has_17_authors_and_arxiv_bibtex():
    e = _load()["cheng2025barbarians@v1"]
    authors = [a.strip() for a in e["authors"].split(" and ")]
    assert len(authors) == 17 and all(authors)
    bib = e["bibtex"]
    assert isinstance(bib, str) and bib.lstrip().startswith("@") and _balanced(bib)
    assert "2510.06189" in bib
    assert "Barbarians at the Gate" in bib


def test_seed_ellis_bibtex_is_arxiv_entry():
    bib = _load()["ellis-castro-2026@v1"]["bibtex"]
    assert isinstance(bib, str) and bib.lstrip().startswith("@") and _balanced(bib)
    assert "2606.02863" in bib
    assert "GAMBLe" in bib
    assert "Ellis" in bib and "Castro" in bib


def test_seed_gupta_bibtex_entry():
    bib = _load()["gupta2026harness@v1"]["bibtex"]
    assert isinstance(bib, str) and _balanced(bib)
    m = re.match(r"\s*@(\w+)\{([^,\s]+)\s*,", bib)
    assert m, "bibtex must start with @type{key,"
    assert m.group(2) == "gupta2026automateddiscoveryuniversallysuperior"
    assert _field(bib, "eprint") == "2607.18235"
    title = _field(bib, "title") or ""
    assert "Automated Discovery Has No Universally Superior Harness" in title.replace("{", "").replace("}", "")
    assert (_field(bib, "year") or "").strip() == "2026"
    authors = _field(bib, "author") or ""
    for surname in ("Gupta", "Lei", "Lu", "Anumanchipalli", "Choshen"):
        assert surname in authors, surname
    assert bib.count("@") == 1


def test_seed_ibm_has_no_bibtex():
    e = _load()["ibm-ai-native-systems@v1"]
    assert e.get("bibtex") is None
    assert e.get("year") is None and e.get("url") is None
