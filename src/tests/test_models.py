"""Spec "Models (adrs_models.py)": ADRSAssessor, ADRSSource, and ADRSParsedCampaign's
new optional fields; no field may carry a server-assigned anonymous slug."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from conftest import DEFAULT_ASSESSOR, anon_source_block, build_parsed_dict


def _models():
    import adrs_models

    for n in ("ADRSAssessor", "ADRSSource"):
        if not hasattr(adrs_models, n):
            pytest.fail(f"adrs_models.{n} is not defined (spec: Models)")
    return adrs_models


def test_assessor_fields_and_defaults():
    m = _models()
    a = m.ADRSAssessor(name="ev", version="2", platform="unknown")
    assert (a.name, a.version, a.platform, a.code_ref) == ("ev", "2", "unknown", None)
    b = m.ADRSAssessor(name="ev", version="2", platform="linux", code_ref="git:abc")
    assert b.code_ref == "git:abc"


@pytest.mark.parametrize("missing", ["name", "version", "platform"])
def test_assessor_required_fields(missing):
    m = _models()
    data = dict(DEFAULT_ASSESSOR)
    data.pop(missing)
    with pytest.raises(ValidationError):
        m.ADRSAssessor.model_validate(data)


@pytest.mark.parametrize("platform", ["", None])
def test_assessor_platform_must_be_non_empty_string(platform):
    m = _models()
    with pytest.raises(ValidationError):
        m.ADRSAssessor(name="ev", version="1", platform=platform)


def test_source_all_fields_optional_with_none_defaults():
    m = _models()
    s = m.ADRSSource()
    for f in ("slug", "title", "version", "year", "url", "license_spdx", "redistribution",
              "attribution", "authors", "bibtex", "consent"):
        assert getattr(s, f) is None, f
    s2 = m.ADRSSource(slug="x@v1", year=2026, consent=True)
    assert (s2.slug, s2.year, s2.consent) == ("x@v1", 2026, True)


def test_parsed_campaign_new_fields_default_none():
    m = _models()
    p = m.ADRSParsedCampaign.model_validate(build_parsed_dict(assessor=None))
    assert "assessor" in m.ADRSParsedCampaign.model_fields
    assert "source" in m.ADRSParsedCampaign.model_fields
    assert p.assessor is None and p.source is None


def test_parsed_campaign_accepts_blocks():
    m = _models()
    p = m.ADRSParsedCampaign.model_validate(build_parsed_dict(source=anon_source_block()))
    assert isinstance(p.assessor, m.ADRSAssessor)
    assert isinstance(p.source, m.ADRSSource)
    assert p.assessor.platform == DEFAULT_ASSESSOR["platform"]
    assert p.source.attribution == "anonymous" and p.source.consent is True


def _all_field_names(model_cls, seen=None) -> set[str]:
    seen = seen or set()
    names: set[str] = set()
    if model_cls in seen:
        return names
    seen.add(model_cls)
    for fname, finfo in model_cls.model_fields.items():
        names.add(fname)
        ann = finfo.annotation
        for sub in getattr(ann, "__args__", ()) or (ann,):
            if isinstance(sub, type) and hasattr(sub, "model_fields"):
                names |= _all_field_names(sub, seen)
    return names


def test_no_field_carries_anonymous_slug():
    m = _models()
    names = {n.lower() for n in _all_field_names(m.ADRSParsedCampaign)}
    assert not any("anon" in n for n in names), "no model field may carry an anonymous slug"
    assert "assigned_anon_slug" not in m.ADRSParsedCampaign.model_fields
    assert "assigned_anon_slug" not in m.ADRSSource.model_fields


def test_assigned_anon_slug_in_json_is_never_honoured():
    m = _models()
    forged = "contributor-ZZZ@v1"
    data = build_parsed_dict(source=anon_source_block())
    data["assigned_anon_slug"] = forged
    data["source"]["assigned_anon_slug"] = forged
    try:
        p = m.ADRSParsedCampaign.model_validate(data)
    except ValidationError:
        return  # rejected: acceptable per spec
    assert getattr(p, "assigned_anon_slug", None) is None
    assert forged not in json.dumps(p.model_dump(mode="json"))
    assert p.source.slug is None
