"""Spec "Provenance input": insert_adrs_campaign.resolve_provenance_input(campaign_path, root)."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import DEFAULT_ASSESSOR, anon_source_block, named_source_block, write_author_input


def _resolve():
    import insert_adrs_campaign

    fn = getattr(insert_adrs_campaign, "resolve_provenance_input", None)
    if fn is None:
        pytest.fail("insert_adrs_campaign.resolve_provenance_input is not defined")
    return fn


ROOT_SOURCE = named_source_block("root-src@v1")
ROOT_ASSESSOR = dict(DEFAULT_ASSESSOR, name="root-assessor")


def _tree(tmp_path: Path):
    root = tmp_path / "root"
    mid = root / "group"
    camp = mid / "campaign1"
    camp.mkdir(parents=True)
    return root, mid, camp


def test_root_only(tmp_path):
    root, _, camp = _tree(tmp_path)
    write_author_input(root, {"author": "alice-test@example.com", "version": "v9",
                              "assessor": ROOT_ASSESSOR, "source": ROOT_SOURCE})
    assert _resolve()(camp, root) == {"author": "alice-test@example.com", "version": "v9",
                                      "assessor": ROOT_ASSESSOR, "source": ROOT_SOURCE}


def test_campaign_file_with_only_author_keeps_root_assessor_and_source(tmp_path):
    root, _, camp = _tree(tmp_path)
    write_author_input(root, {"author": "alice-test@example.com", "assessor": ROOT_ASSESSOR, "source": ROOT_SOURCE})
    write_author_input(camp, {"author": "bob-test@example.com"})
    out = _resolve()(camp, root)
    assert out["author"] == "bob-test@example.com"
    assert out["assessor"] == ROOT_ASSESSOR
    assert out["source"] == ROOT_SOURCE


def test_nearest_file_wins_per_top_level_key(tmp_path):
    root, mid, camp = _tree(tmp_path)
    mid_assessor = dict(DEFAULT_ASSESSOR, name="mid-assessor")
    write_author_input(root, {"author": "alice-test@example.com", "version": "root-v",
                              "assessor": ROOT_ASSESSOR, "source": ROOT_SOURCE})
    write_author_input(mid, {"assessor": mid_assessor, "version": "mid-v"})
    write_author_input(camp, {"version": "camp-v"})
    out = _resolve()(camp, root)
    assert out == {"author": "alice-test@example.com", "version": "camp-v",
                   "assessor": mid_assessor, "source": ROOT_SOURCE}


def test_nested_dicts_replaced_not_merged(tmp_path):
    root, _, camp = _tree(tmp_path)
    root_assessor = dict(DEFAULT_ASSESSOR, code_ref="git:root")
    camp_assessor = {"name": "camp-assessor", "version": "2", "platform": "unknown"}
    camp_source = anon_source_block()
    write_author_input(root, {"assessor": root_assessor, "source": ROOT_SOURCE})
    write_author_input(camp, {"assessor": camp_assessor, "source": camp_source})
    out = _resolve()(camp, root)
    assert out["assessor"] == camp_assessor  # no code_ref leaked in from root
    assert out["source"] == camp_source  # no slug/title/authors leaked in from root
    assert "code_ref" not in out["assessor"]
    assert "slug" not in out["source"] and "title" not in out["source"]


def test_never_reads_above_root(tmp_path):
    root, _, camp = _tree(tmp_path)
    write_author_input(tmp_path, {"author": "outside@example.com", "version": "outside",
                                  "assessor": ROOT_ASSESSOR, "source": ROOT_SOURCE})
    write_author_input(camp, {"author": "alice-test@example.com"})
    assert _resolve()(camp, root) == {"author": "alice-test@example.com"}


def test_nothing_found_returns_empty(tmp_path):
    root, _, camp = _tree(tmp_path)
    write_author_input(tmp_path, {"author": "outside@example.com"})
    assert _resolve()(camp, root) == {}


def test_campaign_equals_root(tmp_path):
    root, _, _ = _tree(tmp_path)
    write_author_input(root, {"author": "alice-test@example.com", "source": ROOT_SOURCE})
    assert _resolve()(root, root) == {"author": "alice-test@example.com", "source": ROOT_SOURCE}


def test_only_known_keys_returned(tmp_path):
    root, _, camp = _tree(tmp_path)
    write_author_input(root, {"author": "alice-test@example.com", "unrelated": 1,
                              "assigned_anon_slug": "contributor-A@v1"})
    out = _resolve()(camp, root)
    assert set(out) <= {"author", "version", "assessor", "source"}
    assert out["author"] == "alice-test@example.com"
