#!/usr/bin/env python3
"""Provenance helpers: anonymous labels, submitter normalization, the i06 contributor
rule, and citation tooling (docs/PROVENANCE_SPEC.md).

Usage:
    with psycopg.connect(url) as conn:
        cs, sources = load_provenance(conn)
    cites = required_citations(campaign_ids, cs, sources)
    print(format_citations(cites, fmt="bibtex"))
"""

import math
import re
from uuid import UUID

import pandas as pd


class ProvenanceError(ValueError):
    """A provenance rule was violated (missing/invalid source or assessor, frozen DB, ...)."""


ROAR_SOURCE = {
    "slug": "roar",
    "title": "ROAR: Unifying Runs across Heterogeneous AI-Driven Research Systems",
    "version": None,
    "year": 2026,
    "url": None,
    "license_spdx": "CC-BY-4.0",
    "redistribution": "full",
    "attribution": "named",
    "authors": "Lin, Leo Y. and Ramani, Vishakha and Celik, Z. Berkay and Castro, Paul and Ellis, Marquita",
    "bibtex": """@misc{lin2026roar,
  title        = {{ROAR}: Unifying Runs across Heterogeneous {AI}-Driven Research Systems},
  author       = {Lin, Leo Y. and Ramani, Vishakha and Celik, Z. Berkay and Castro, Paul and Ellis, Marquita},
  year         = {2026},
  note         = {Preprint},
}""",
}

# Columns of source_datasets that may leave the database (never owner).
SOURCE_COLUMNS = ["slug", "title", "version", "year", "url", "license_spdx", "redistribution",
                  "attribution", "authors", "bibtex", "consent_recorded_at"]

FROZEN_DATABASES = frozenset({"roar_v1", "roar_full"})


def label(n: int) -> str:
    """Spreadsheet-style label: 1 -> A, 26 -> Z, 27 -> AA, 53 -> BA."""
    if n < 1:
        raise ValueError(f"label() needs n >= 1, got {n}")
    out = ""
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(ord("A") + r) + out
    return out


def normalize_submitter(s: str | None) -> str | None:
    """Strip and lowercase; None/empty/whitespace-only -> None."""
    if s is None:
        return None
    s = s.strip().lower()
    return s or None


_PREFIX_RE = re.compile(r"^([^_/][^/]*?)(__|/)")
_EXCLUDED_PREFIXES = {"cbl", "cbl-multi", "cloudcast", "llm-sql", "mas", "other_cases", "ns3"}
_MODEL_MECHANISM_RE = re.compile(
    r"^(eb1[a-z-]*|kimi-k2\.5|gpt-5\.4|gpt-5-mini|gpt-oss[a-z0-9-]*|opus|gemini-2\.5-flash|qwen3\.5[a-z0-9:]*)"
    r"_(adaevolve|evox|best_of_n)"
)


def contributor_prefix(name: str | None) -> str | None:
    """The i06 rule: the contributor prefix of a v1 campaign name, or None."""
    if not name:
        return None
    m = _PREFIX_RE.match(name)
    if not m:
        return None
    prefix = m.group(1)
    if prefix in _EXCLUDED_PREFIXES or _MODEL_MECHANISM_RE.match(name):
        return None
    return prefix


def bibtex_key(slug: str) -> str:
    """BibTeX-safe key: each run of characters outside [A-Za-z0-9_-] becomes '_'."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", slug)


def load_provenance(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(campaign_sources[campaign_id, source_slug], sources[SOURCE_COLUMNS]).

    Never selects owner or submitted_by.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT campaign_id::text, source_slug FROM campaign_sources")
        cs = pd.DataFrame(cur.fetchall(), columns=["campaign_id", "source_slug"])
        cur.execute(f"SELECT {', '.join(SOURCE_COLUMNS)} FROM source_datasets")
        sources = pd.DataFrame(cur.fetchall(), columns=SOURCE_COLUMNS)
    return cs, sources


def _clean(v):
    """pandas missing values (NaN/NaT/None) -> None."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _source_record(row: dict) -> dict:
    rec = {k: _clean(v) for k, v in row.items()}
    if rec.get("year") is not None:
        rec["year"] = int(rec["year"])
    return rec


def required_citations(campaign_ids, campaign_sources: pd.DataFrame, sources: pd.DataFrame,
                       extra_sources=()) -> list[dict]:
    """ROAR first, then every distinct source used by campaign_ids plus extra_sources,
    deduplicated and sorted by slug. ROAR appears exactly once, even when campaigns map
    to the `roar` source or extra_sources contains it.

    Raises ProvenanceError for a campaign without a campaign_sources row or an unknown
    extra slug. Accepts campaign ids as str or UUID.
    """
    mapping = {str(c): s for c, s in zip(campaign_sources["campaign_id"], campaign_sources["source_slug"])}
    by_slug = {r["slug"]: _source_record(r) for r in sources.to_dict("records")}

    slugs = set()
    for cid in campaign_ids:
        try:
            key = str(cid if isinstance(cid, UUID) else UUID(str(cid)))
        except ValueError:
            raise ProvenanceError(f"invalid campaign id {cid!r}") from None
        if key not in mapping:
            raise ProvenanceError(f"campaign {key} has no campaign_sources row")
        slugs.add(mapping[key])
    for slug in extra_sources:
        if slug not in by_slug and slug != ROAR_SOURCE["slug"]:
            raise ProvenanceError(f"unknown source slug {slug!r}")
        slugs.add(slug)
    for slug in slugs:
        if slug not in by_slug:
            raise ProvenanceError(f"campaign_sources references unknown source {slug!r}")

    slugs.discard(ROAR_SOURCE["slug"])
    return [dict(ROAR_SOURCE)] + [by_slug[s] for s in sorted(slugs)]


def _missing_license(src: dict) -> bool:
    return src.get("redistribution") == "pointer_only" and not src.get("license_spdx")


def _bibtex_entry(src: dict) -> str:
    anonymous = src.get("attribution") == "anonymous"
    if src.get("bibtex") and not anonymous:
        entry = src["bibtex"].strip()
        if _missing_license(src):
            entry += f"\n% {bibtex_key(src['slug'])}: no license declared"
        return entry
    # Anonymous sources render as their title only, never with authors.
    fields = []
    if src.get("authors") and not anonymous:
        fields.append(("author", src["authors"]))
    if src.get("title"):
        fields.append(("title", src["title"]))
    if src.get("year") is not None and not anonymous:
        fields.append(("year", str(src["year"])))
    if src.get("url") and not anonymous:
        fields.append(("howpublished", f"\\url{{{src['url']}}}"))
    notes = []
    if src.get("version") and not anonymous:
        notes.append(f"version {src['version']}")
    if _missing_license(src):
        notes.append("no license declared")
    if notes:
        fields.append(("note", "; ".join(notes)))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
    return f"@misc{{{bibtex_key(src['slug'])},\n{body}\n}}"


def _text_line(src: dict) -> str:
    if src.get("attribution") == "anonymous":
        parts = [src.get("title") or src["slug"]]
    else:
        parts = [src.get("authors"), src.get("title"),
                 str(src["year"]) if src.get("year") is not None else None,
                 src.get("url"),
                 f"version {src['version']}" if src.get("version") else None]
    if _missing_license(src):
        parts.append("no license declared")
    line = ". ".join(p for p in parts if p) + "."
    return " ".join(line.split())


def format_citations(sources, fmt: str = "bibtex") -> str:
    """Render sources as BibTeX entries (fmt='bibtex') or one line each (fmt='text')."""
    if fmt == "bibtex":
        return "\n\n".join(_bibtex_entry(s) for s in sources) + "\n"
    if fmt == "text":
        return "\n".join(_text_line(s) for s in sources) + "\n"
    raise ValueError(f"fmt must be 'bibtex' or 'text', got {fmt!r}")
