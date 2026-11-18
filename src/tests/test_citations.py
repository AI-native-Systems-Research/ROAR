"""Spec "Citation tooling (provenance.py)": ROAR_SOURCE, load_provenance,
required_citations, format_citations (bibtex_key is in test_provenance_units.py)."""

from __future__ import annotations

import re
import uuid

import psycopg
import pytest

from conftest import REPO_ROOT, insert_sql_campaign, insert_sql_source, map_campaign, unique

ALICE = "alice-test@example.com"


def _readme_roar_bibtex() -> str:
    text = (REPO_ROOT / "README.md").read_text()
    m = re.search(r"@misc\{lin2026roar,.*?\n\}", text, re.S)
    assert m, "README has no @misc{lin2026roar,...} block"
    return m.group(0)


def _norm(s: str) -> str:
    return " ".join(s.split())


def _entries(bib: str) -> list[str]:
    """Split a bibtex string into top-level @entries (brace-matched)."""
    out, i = [], 0
    while (j := bib.find("@", i)) != -1:
        k = bib.find("{", j)
        if k == -1:
            break
        depth, e = 0, k
        while e < len(bib):
            depth += {"{": 1, "}": -1}.get(bib[e], 0)
            if depth == 0:
                break
            e += 1
        out.append(bib[j:e + 1])
        i = e + 1
    return out


def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


# --- ROAR_SOURCE ---------------------------------------------------------------

def test_roar_source(prov):
    rs = prov.ROAR_SOURCE
    assert isinstance(rs, dict)
    assert rs["slug"] == "roar"
    assert _norm(rs["bibtex"]) == _norm(_readme_roar_bibtex())


# --- fixtures in the session DB ---------------------------------------------------

@pytest.fixture
def cited(db, prov):
    tag = uuid.uuid4().hex[:6]
    a, b = f"zz-a-{tag}@v1", f"aa-b-{tag}@v2"
    anon = f"contributor-T{tag.upper()}@v1"
    insert_sql_source(db, a, title=f"Dataset A {tag}", owner=ALICE)
    insert_sql_source(db, b, title=f"Dataset B {tag}", version="v2", bibtex="@misc{bkey, title = {B}}", owner=ALICE)
    insert_sql_source(db, anon, attribution="anonymous", title="Independent contributor T", authors=None,
                      bibtex=None, owner=ALICE)
    c = {k: insert_sql_campaign(db, unique(f"cite-{k}-"), n_candidates=1) for k in ("a1", "a2", "b1", "anon1", "unmapped")}
    map_campaign(db, c["a1"], a, ALICE)
    map_campaign(db, c["a2"], a, ALICE)
    map_campaign(db, c["b1"], b, ALICE)
    map_campaign(db, c["anon1"], anon, ALICE)
    with psycopg.connect(db) as conn:
        cs, srcs = prov.load_provenance(conn)
    return {"a": a, "b": b, "anon": anon, "c": c, "cs": cs, "srcs": srcs, "tag": tag}


# --- load_provenance ---------------------------------------------------------------

def test_load_provenance_shapes_and_no_private_columns(cited):
    cs, srcs = cited["cs"], cited["srcs"]
    assert set(cs.columns) == {"campaign_id", "source_slug"}
    assert "slug" in srcs.columns
    for df in (cs, srcs):
        assert "owner" not in df.columns and "submitted_by" not in df.columns
        assert ALICE not in df.astype(str).to_string()
    assert {cited["a"], cited["b"], cited["anon"]} <= set(srcs["slug"])
    assert {str(x) for x in cs["campaign_id"]} >= {cited["c"][k] for k in ("a1", "a2", "b1", "anon1")}


# --- required_citations ---------------------------------------------------------------

def test_required_citations_roar_first_then_sorted_dedup(prov, cited):
    c = cited["c"]
    out = prov.required_citations([c["a1"], c["b1"], c["a2"], c["anon1"], c["a1"]], cited["cs"], cited["srcs"])
    slugs = [d["slug"] for d in out]
    assert slugs == ["roar"] + sorted({cited["a"], cited["b"], cited["anon"]})
    assert out[0]["slug"] == prov.ROAR_SOURCE["slug"]
    by = {d["slug"]: d for d in out}
    assert by[cited["a"]]["title"] == f"Dataset A {cited['tag']}"
    for d in out:
        assert "owner" not in d and "submitted_by" not in d


def test_required_citations_accepts_str_and_uuid(prov, cited):
    c = cited["c"]
    as_str = prov.required_citations([c["a1"], c["b1"]], cited["cs"], cited["srcs"])
    as_uuid = prov.required_citations([uuid.UUID(c["a1"]), uuid.UUID(c["b1"])], cited["cs"], cited["srcs"])
    mixed = prov.required_citations([c["a1"], uuid.UUID(c["b1"])], cited["cs"], cited["srcs"])
    assert [d["slug"] for d in as_str] == [d["slug"] for d in as_uuid] == [d["slug"] for d in mixed]
    assert [d["slug"] for d in as_str] == ["roar"] + sorted([cited["a"], cited["b"]])


def test_required_citations_empty_is_roar_only(prov, cited):
    out = prov.required_citations([], cited["cs"], cited["srcs"])
    assert [d["slug"] for d in out] == ["roar"]


def test_required_citations_extra_sources(prov, cited):
    c = cited["c"]
    out = prov.required_citations([c["a1"]], cited["cs"], cited["srcs"], extra_sources=(cited["b"], cited["a"]))
    assert [d["slug"] for d in out] == ["roar"] + sorted([cited["a"], cited["b"]])
    out2 = prov.required_citations([], cited["cs"], cited["srcs"], extra_sources=[cited["anon"]])
    assert [d["slug"] for d in out2] == ["roar", cited["anon"]]


def test_required_citations_unmapped_campaign_raises(prov, cited, ProvenanceError):
    with pytest.raises(ProvenanceError):
        prov.required_citations([cited["c"]["a1"], cited["c"]["unmapped"]], cited["cs"], cited["srcs"])
    with pytest.raises(ProvenanceError):
        prov.required_citations([uuid.uuid4()], cited["cs"], cited["srcs"])


def test_required_citations_unknown_extra_slug_raises(prov, cited, ProvenanceError):
    with pytest.raises(ProvenanceError):
        prov.required_citations([cited["c"]["a1"]], cited["cs"], cited["srcs"], extra_sources=("jdoe@v1",))


# --- format_citations ---------------------------------------------------------------

def _src(**kw):
    base = {"slug": "synthetic-src@v2", "title": "A Synthetic Dataset", "version": "v2", "year": 2026,
            "url": "https://example.com/synthetic", "license_spdx": "CC-BY-4.0", "redistribution": "full",
            "attribution": "named", "authors": "Alice Test and Bob Test", "bibtex": None,
            "consent_recorded_at": None}
    base.update(kw)
    return base


def test_format_stored_bibtex_verbatim(prov):
    stored = "@article{stored2025,\n  title  = {Stored {Entry}},\n  author = {Test, Alice},\n  year = {2025}\n}"
    out = prov.format_citations([_src(bibtex=stored)], fmt="bibtex")
    assert stored in out
    assert "@misc{synthetic-src_v2" not in out


def test_format_generated_misc(prov):
    out = prov.format_citations([_src()], fmt="bibtex")
    entries = _entries(out)
    assert len(entries) == 1
    e = entries[0]
    assert _balanced(out) and _balanced(e)
    m = re.match(r"@misc\{([^,\s]+)\s*,", e)
    assert m and m.group(1) == "synthetic-src_v2"
    assert re.search(r"author\s*=\s*\{Alice Test and Bob Test\}", e)
    assert re.search(r"title\s*=\s*\{A Synthetic Dataset\}", e)
    assert re.search(r"year\s*=\s*\{2026\}", e)
    assert re.search(r"howpublished\s*=\s*\{\\url\{https://example\.com/synthetic\}\}", e)
    assert re.search(r"note\s*=\s*\{version v2\}", e)


def test_format_generated_misc_sanitizes_key_and_omits_absent_fields(prov):
    out = prov.format_citations([_src(slug="we!rd slug@v1", year=None, url=None, version=None)], fmt="bibtex")
    e = _entries(out)[0]
    assert _balanced(e)
    m = re.match(r"@misc\{([^,\s]+)\s*,", e)
    assert m and m.group(1) == "we_rd_slug_v1"
    assert re.fullmatch(r"[A-Za-z0-9_-]+", m.group(1))
    assert "year" not in e and "howpublished" not in e and "note" not in e
    assert re.search(r"title\s*=\s*\{A Synthetic Dataset\}", e)


def test_format_roar_source_uses_stored_bibtex(prov):
    out = prov.format_citations([prov.ROAR_SOURCE], fmt="bibtex")
    assert prov.ROAR_SOURCE["bibtex"] in out


def test_format_anonymous_title_only_no_authors(prov):
    anon = _src(slug="contributor-Q@v1", title="Independent contributor Q", attribution="anonymous",
                authors=None, bibtex=None, year=None, url=None)
    bib = prov.format_citations([anon], fmt="bibtex")
    assert "Independent contributor Q" in bib
    assert not re.search(r"\bauthor\s*=", bib)
    assert _balanced(bib)
    txt = prov.format_citations([anon], fmt="text")
    assert "Independent contributor Q" in txt


def test_format_anonymous_never_shows_authors_even_if_present(prov):
    anon = _src(slug="contributor-R@v1", title="Independent contributor R", attribution="anonymous",
                authors="Alice Test")
    for fmt in ("bibtex", "text"):
        out = prov.format_citations([anon], fmt=fmt)
        assert "Independent contributor R" in out and "Alice Test" not in out, fmt


@pytest.mark.parametrize("fmt", ["bibtex", "text"])
def test_format_pointer_only_null_license(prov, fmt):
    s = _src(slug="pointer-src@v1", redistribution="pointer_only", license_spdx=None)
    out = prov.format_citations([s], fmt=fmt)
    assert "no license declared" in out
    if fmt == "bibtex":
        assert _balanced(out)


@pytest.mark.parametrize("fmt", ["bibtex", "text"])
def test_format_pointer_only_with_license_has_no_missing_license_phrase(prov, fmt):
    s = _src(slug="pointer-lic@v1", redistribution="pointer_only", license_spdx="MIT")
    assert "no license declared" not in prov.format_citations([s], fmt=fmt)


def test_format_text_one_line_per_source(prov):
    srcs = [prov.ROAR_SOURCE, _src(), _src(slug="other@v1", title="Other Dataset", bibtex="@misc{o, title={O}}"),
            _src(slug="contributor-S@v1", title="Independent contributor S", attribution="anonymous", authors=None)]
    out = prov.format_citations(srcs, fmt="text")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == len(srcs)
    assert "A Synthetic Dataset" in lines[1]
    assert "Other Dataset" in lines[2]
    assert "Independent contributor S" in lines[3]


def test_format_bibtex_one_entry_per_source(prov):
    srcs = [prov.ROAR_SOURCE, _src(), _src(slug="other@v1", title="Other Dataset")]
    out = prov.format_citations(srcs, fmt="bibtex")
    assert len(_entries(out)) == 3 and _balanced(out)


@pytest.mark.parametrize("fmt", ["BIBTEX", "json", "", "markdown"])
def test_format_bad_fmt_raises_value_error(prov, fmt):
    with pytest.raises(ValueError):
        prov.format_citations([_src()], fmt=fmt)
