"""Spec "Docs": upload_instructions.md, GET /upload/instructions and GET /upload/formats
describe the assessor and source blocks and state the two rules verbatim (modulo case,
whitespace, backticks and typographic apostrophes)."""

from __future__ import annotations

import json

import pytest

from conftest import SRC_DIR

RULE_ANON = "one anonymous contributor per upload; keep the returned anon_slug"
RULE_NAMES = "don't put personal names in folder names or file paths; they become campaign names and artifact URIs"

# Terms a description of the two blocks must mention.
ASSESSOR_TERMS = ("assessor", "platform", "code_ref")
SOURCE_TERMS = ("source", "attribution", "redistribution", "license_spdx", "consent", "anonymous")


def _norm(s: str) -> str:
    s = s.replace("`", "").replace("’", "'").replace("‘", "'")
    return " ".join(s.lower().split())


def _docs() -> dict[str, str]:
    from fastapi.testclient import TestClient

    import api

    client = TestClient(api.app)  # no lifespan: instruction endpoints need no DB/model
    out = {"upload_instructions.md": (SRC_DIR / "upload_instructions.md").read_text()}
    for ep in ("/upload/instructions", "/upload/formats"):
        r = client.get(ep)
        assert r.status_code == 200, ep
        out[ep] = json.dumps(r.json(), ensure_ascii=False)
    return out


DOCS = ["upload_instructions.md", "/upload/instructions", "/upload/formats"]


@pytest.mark.parametrize("doc", DOCS)
def test_anonymous_rule_present(doc):
    assert _norm(RULE_ANON) in _norm(_docs()[doc])


@pytest.mark.parametrize("doc", DOCS)
def test_personal_names_rule_present(doc):
    assert _norm(RULE_NAMES) in _norm(_docs()[doc])


@pytest.mark.parametrize("doc", DOCS)
def test_assessor_and_source_blocks_described(doc):
    text = _norm(_docs()[doc])
    for term in ASSESSOR_TERMS + SOURCE_TERMS:
        assert term in text, f"{doc} does not mention {term!r}"
