"""Spec "Callers": CLI insert_all_campaigns(root, ...), CLI insert_campaign(folder_path,
system_type=None, provenance_root=None) and benchmark_direct_insert.benchmark_insert.

Campaign folders are real (minimal) SkyDiscover runs parsed by the real parser; the
campaign name recorded in the DB is the folder name."""

from __future__ import annotations

import re

import pytest

from conftest import (
    DEFAULT_ASSESSOR,
    anon_source_block,
    campaign_row,
    fetchall,
    fetchone,
    named_source_block,
    table_counts,
    unique,
    write_author_input,
    write_skydiscover_campaign,
)

ALICE = "alice-test@example.com"
BOB = "bob-test@example.com"


def _mapping_by_name(url, name):
    return fetchone(
        url,
        "SELECT cs.source_slug, cs.submitted_by, c.author FROM campaigns c "
        "JOIN campaign_sources cs ON cs.campaign_id=c.id WHERE c.name=%s",
        (name,),
    )


def _evals_by_name(url, name):
    return fetchall(
        url,
        "SELECT DISTINCT m.evaluation_id, e.assessor_name, e.kind FROM campaigns c "
        "JOIN candidates k ON k.campaign_id=c.id JOIN measurements m ON m.candidate_id=k.id "
        "LEFT JOIN evaluations e ON e.id=m.evaluation_id WHERE c.name=%s",
        (name,),
    )


def _batch(tmp_path, n, root_ai, per_campaign=None):
    root = tmp_path / unique("batch-")
    names = [unique("camp-") for _ in range(n)]
    for i, nm in enumerate(names):
        write_skydiscover_campaign(root / nm, (per_campaign or {}).get(i))
    write_author_input(root, root_ai)
    return root, names


# --- insert_all_campaigns ----------------------------------------------------

def test_insert_all_anonymous_batch_one_source_three_mappings(db, tmp_path):
    from insert_adrs_campaign import insert_all_campaigns

    root, names = _batch(tmp_path, 3, {"author": ALICE, "assessor": DEFAULT_ASSESSOR, "source": anon_source_block()})
    before = table_counts(db)
    ok, failed = insert_all_campaigns(root)
    assert (ok, failed) == (3, 0)
    after = table_counts(db)
    assert after["source_datasets"] == before["source_datasets"] + 1
    assert after["campaign_sources"] == before["campaign_sources"] + 3
    assert after["evaluations"] == before["evaluations"] + 3
    maps = [_mapping_by_name(db, n) for n in names]
    slugs = {m[0] for m in maps}
    assert len(slugs) == 1
    slug = slugs.pop()
    m = re.match(r"^contributor-([A-Z]+)@v1$", slug)
    assert m
    title = f"Independent contributor {m.group(1)}"
    assert all(mp[1] == ALICE for mp in maps)  # submitted_by = resolved author before substitution
    assert all(mp[2] == title for mp in maps)  # campaigns.author = anonymous label
    owner, stored_title = fetchone(db, "SELECT owner, title FROM source_datasets WHERE slug=%s", (slug,))
    assert (owner, stored_title) == (ALICE, title)
    for n in names:
        ev = _evals_by_name(db, n)
        assert len(ev) == 1 and ev[0][0] is not None and ev[0][2] == "generation_time"


def test_insert_all_differing_anonymous_blocks_raise_before_any_insert(db, tmp_path, ProvenanceError):
    from insert_adrs_campaign import insert_all_campaigns

    root, names = _batch(
        tmp_path, 3, {"author": ALICE, "assessor": DEFAULT_ASSESSOR, "source": anon_source_block()},
        per_campaign={1: {"source": anon_source_block(license_spdx="MIT")}},
    )
    before = table_counts(db)
    seq_before = fetchone(db, "SELECT last_value, is_called FROM anon_contributor_seq")
    with pytest.raises(ProvenanceError):
        insert_all_campaigns(root)
    assert table_counts(db) == before
    for n in names:
        assert campaign_row(db, n) is None
    # Checked before allocating a slug.
    assert fetchone(db, "SELECT last_value, is_called FROM anon_contributor_seq") == seq_before


def test_insert_all_named_with_per_campaign_author(db, tmp_path):
    from insert_adrs_campaign import insert_all_campaigns

    src = named_source_block()
    root, names = _batch(
        tmp_path, 2, {"author": ALICE, "assessor": DEFAULT_ASSESSOR, "source": src},
        per_campaign={1: {"author": BOB}},
    )
    assert insert_all_campaigns(root) == (2, 0)
    m0, m1 = (_mapping_by_name(db, n) for n in names)
    assert m0[:2] == (src["slug"], ALICE)
    assert m1[:2] == (src["slug"], BOB)
    assert m1[2] == BOB  # named: campaigns.author is not substituted
    # Insert order is not specified (parsing completes in any order): first inserter owns it.
    assert fetchone(db, "SELECT owner FROM source_datasets WHERE slug=%s", (src["slug"],))[0] in {ALICE, BOB}


def test_insert_all_mixed_named_and_anonymous_batch(db, tmp_path):
    """Anonymous first-upload campaigns share one allocated slug; a campaign with its own
    named block maps to the named source."""
    from insert_adrs_campaign import insert_all_campaigns

    src = named_source_block()
    root, names = _batch(
        tmp_path, 3, {"author": ALICE, "assessor": DEFAULT_ASSESSOR, "source": anon_source_block()},
        per_campaign={1: {"source": src}},
    )
    before = table_counts(db)
    assert insert_all_campaigns(root) == (3, 0)
    after = table_counts(db)
    assert after["source_datasets"] == before["source_datasets"] + 2
    assert after["campaign_sources"] == before["campaign_sources"] + 3
    m0, m1, m2 = (_mapping_by_name(db, n) for n in names)
    assert m0[0] == m2[0] and re.match(r"^contributor-[A-Z]+@v1$", m0[0])
    assert m1[0] == src["slug"]
    assert {m0[1], m1[1], m2[1]} == {ALICE}
    title = "Independent contributor " + m0[0][len("contributor-"):-len("@v1")]
    assert m0[2] == m2[2] == title
    assert m1[2] != title
    assert fetchone(db, "SELECT owner FROM source_datasets WHERE slug=%s", (m0[0],))[0] == ALICE


def test_insert_all_campaign_level_assessor_override(db, tmp_path):
    from insert_adrs_campaign import insert_all_campaigns

    root, names = _batch(
        tmp_path, 2, {"author": ALICE, "assessor": DEFAULT_ASSESSOR, "source": named_source_block()},
        per_campaign={0: {"assessor": dict(DEFAULT_ASSESSOR, name="override-assessor")}},
    )
    assert insert_all_campaigns(root) == (2, 0)
    assert _evals_by_name(db, names[0])[0][1] == "override-assessor"
    assert _evals_by_name(db, names[1])[0][1] == DEFAULT_ASSESSOR["name"]


def test_insert_all_without_provenance_inserts_nothing(db, tmp_path):
    from insert_adrs_campaign import insert_all_campaigns

    root, names = _batch(tmp_path, 2, {"author": ALICE})
    before = table_counts(db)
    try:
        ok, _failed = insert_all_campaigns(root)
        assert ok == 0
    except Exception as e:  # raising ProvenanceError is also compliant
        from provenance import ProvenanceError

        assert isinstance(e, ProvenanceError)
    assert table_counts(db) == before


# --- insert_campaign ---------------------------------------------------------

def test_insert_campaign_with_provenance_root(db, tmp_path):
    from insert_adrs_campaign import insert_campaign

    src = named_source_block()
    root = tmp_path / unique("root-")
    name = unique("camp-")
    folder = write_skydiscover_campaign(root / "group" / name)
    write_author_input(root, {"author": ALICE, "assessor": DEFAULT_ASSESSOR, "source": src})
    assert insert_campaign(folder, "skydiscover", provenance_root=root) is True
    assert _mapping_by_name(db, name)[:2] == (src["slug"], ALICE)
    assert _evals_by_name(db, name)[0][2] == "generation_time"


def test_insert_campaign_default_root_is_folder(db, tmp_path, ProvenanceError):
    """Without provenance_root, root == folder_path, so a parent's blocks are not seen."""
    from insert_adrs_campaign import insert_campaign

    root = tmp_path / unique("root-")
    name = unique("camp-")
    folder = write_skydiscover_campaign(root / name, {"author": ALICE})
    write_author_input(root, {"assessor": DEFAULT_ASSESSOR, "source": named_source_block()})
    before = table_counts(db)
    try:
        assert insert_campaign(folder, "skydiscover") is False
    except ProvenanceError:
        pass
    assert campaign_row(db, name) is None
    assert table_counts(db) == before


def test_insert_campaign_own_folder_blocks(db, tmp_path):
    from insert_adrs_campaign import insert_campaign

    src = named_source_block()
    name = unique("camp-")
    folder = write_skydiscover_campaign(
        tmp_path / name, {"author": ALICE, "assessor": DEFAULT_ASSESSOR, "source": src}
    )
    assert insert_campaign(folder) is True  # auto-detect + root defaults to folder
    assert _mapping_by_name(db, name)[:2] == (src["slug"], ALICE)


def test_insert_campaign_anonymous_allocates_own_slug_per_call(db, tmp_path):
    from insert_adrs_campaign import insert_campaign

    root = tmp_path / unique("root-")
    write_author_input(root, {"author": ALICE, "assessor": DEFAULT_ASSESSOR, "source": anon_source_block()})
    names = [unique("camp-") for _ in range(2)]
    for n in names:
        write_skydiscover_campaign(root / n)
        assert insert_campaign(root / n, "skydiscover", provenance_root=root) is True
    slugs = [_mapping_by_name(db, n)[0] for n in names]
    assert slugs[0] != slugs[1]
    assert all(re.match(r"^contributor-[A-Z]+@v1$", s) for s in slugs)
    assert all(_mapping_by_name(db, n)[1] == ALICE for n in names)


# --- benchmark_direct_insert -------------------------------------------------

def test_benchmark_passes_resolved_provenance(db, tmp_path):
    from benchmark_direct_insert import benchmark_insert

    src = named_source_block()
    root, names = _batch(tmp_path, 2, {"author": ALICE, "assessor": DEFAULT_ASSESSOR, "source": src})
    res = benchmark_insert(root, max_workers=2)
    assert res["success_count"] == 2
    for n in names:
        assert _mapping_by_name(db, n)[:2] == (src["slug"], ALICE)
        assert _evals_by_name(db, n)[0][0] is not None


def test_benchmark_rejects_anonymous_first_upload(db, tmp_path):
    from benchmark_direct_insert import benchmark_insert

    root, names = _batch(tmp_path, 2, {"author": ALICE, "assessor": DEFAULT_ASSESSOR, "source": anon_source_block()})
    before = table_counts(db)
    seq_before = fetchone(db, "SELECT last_value, is_called FROM anon_contributor_seq")
    try:
        res = benchmark_insert(root, max_workers=2)
        assert res["success_count"] == 0
    except Exception as e:  # raising ProvenanceError is also compliant
        from provenance import ProvenanceError

        assert isinstance(e, ProvenanceError)
    assert table_counts(db) == before
    assert fetchone(db, "SELECT last_value, is_called FROM anon_contributor_seq") == seq_before
