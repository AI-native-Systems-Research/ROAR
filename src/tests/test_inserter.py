"""Spec "Inserter": CampaignInserter.insert(parsed, assigned_anon_slug=None, submitted_by=None),
every ProvenanceError case, every success effect, one-transaction rollback; and spec
"Anonymous labels": allocate_anon_slug(conn)."""

from __future__ import annotations

import re
import threading
import uuid

import psycopg
import pytest

from conftest import (
    DEFAULT_ASSESSOR,
    allocate,
    anon_source_block,
    build_parsed,
    campaign_row,
    fetchall,
    fetchone,
    insert,
    insert_sql_campaign,
    insert_sql_source,
    label_to_int,
    named_source_block,
    table_counts,
    unique,
)

ALICE = "alice-test@example.com"
BOB = "bob-test@example.com"
SLUG_RE = re.compile(r"^contributor-([A-Z]+)@v1$")


def _source_row(url, slug):
    row = fetchone(
        url,
        "SELECT slug, title, version, year, url, license_spdx, redistribution, attribution, authors, "
        "bibtex, consent_recorded_at, owner FROM source_datasets WHERE slug=%s",
        (slug,),
    )
    if row is None:
        return None
    keys = ["slug", "title", "version", "year", "url", "license_spdx", "redistribution", "attribution",
            "authors", "bibtex", "consent_recorded_at", "owner"]
    return dict(zip(keys, row))


def _mapping(url, campaign_id):
    return fetchone(url, "SELECT source_slug, submitted_by FROM campaign_sources WHERE campaign_id=%s", (campaign_id,))


def _expect_reject(url, parsed, ProvenanceError, exc=None, **kw):
    """insert must raise and leave no trace (one transaction, full rollback)."""
    before = table_counts(url)
    with pytest.raises(exc or ProvenanceError):
        insert(url, parsed, **kw)
    assert campaign_row(url, parsed.campaign.name) is None
    assert table_counts(url) == before


def _new_named(url, submitter=ALICE, **overrides):
    block = named_source_block(**overrides)
    p = build_parsed(source=block)
    assert insert(url, p, submitted_by=submitter) is True
    return block, p


# =============================================================================
# Raise cases
# =============================================================================

def test_missing_assessor_rejected(db, ProvenanceError):
    _expect_reject(db, build_parsed(assessor=None, source=named_source_block()), ProvenanceError, submitted_by=ALICE)


def test_missing_source_rejected(db, ProvenanceError):
    _expect_reject(db, build_parsed(source=None), ProvenanceError, submitted_by=ALICE)


@pytest.mark.parametrize("consent", [False, None])
def test_consent_must_be_true_named(db, ProvenanceError, consent):
    block = named_source_block(consent=consent)
    if consent is None:
        block.pop("consent")
    _expect_reject(db, build_parsed(source=block), ProvenanceError, submitted_by=ALICE)
    assert _source_row(db, block["slug"]) is None


@pytest.mark.parametrize("consent", [False, None])
def test_consent_must_be_true_anonymous(db, ProvenanceError, consent):
    block = anon_source_block(consent=consent)
    if consent is None:
        block.pop("consent")
    slug, _ = allocate(db)
    _expect_reject(db, build_parsed(source=block), ProvenanceError, assigned_anon_slug=slug, submitted_by=ALICE)
    assert _source_row(db, slug) is None


def test_new_pointer_only_named_rejected(db, ProvenanceError):
    block = named_source_block(redistribution="pointer_only")
    _expect_reject(db, build_parsed(source=block), ProvenanceError, submitted_by=ALICE)
    assert _source_row(db, block["slug"]) is None


def test_new_pointer_only_anonymous_rejected(db, ProvenanceError):
    slug, _ = allocate(db)
    block = anon_source_block(redistribution="pointer_only")
    _expect_reject(db, build_parsed(source=block), ProvenanceError, assigned_anon_slug=slug, submitted_by=ALICE)
    assert _source_row(db, slug) is None


def test_reference_to_existing_pointer_only_source_rejected(db, ProvenanceError):
    slug = f"ptr-{uuid.uuid4().hex[:8]}@v1"
    insert_sql_source(db, slug, redistribution="pointer_only", license_spdx=None, owner=ALICE)
    _expect_reject(db, build_parsed(source={"slug": slug, "consent": True}), ProvenanceError, submitted_by=ALICE)


@pytest.mark.parametrize("field", ["slug", "title", "version", "url", "license_spdx", "redistribution", "authors"])
def test_named_missing_required_field_rejected(db, ProvenanceError, field):
    block = named_source_block()
    slug = block["slug"]
    block.pop(field)
    _expect_reject(db, build_parsed(source=block), ProvenanceError, submitted_by=ALICE)
    assert _source_row(db, slug) is None


@pytest.mark.parametrize(
    "field,new",
    [
        ("title", "A different title"),
        ("version", "v1-other"),
        ("url", "https://example.com/other"),
        ("license_spdx", "MIT"),
        ("authors", "Carol Test"),
        ("year", 2020),
        ("bibtex", "@misc{other, title={Other}}"),
        ("attribution", "anonymous"),
        ("redistribution", "pointer_only"),
    ],
)
def test_named_existing_slug_with_differing_supplied_field_rejected(db, ProvenanceError, field, new):
    block, _ = _new_named(db, bibtex="@misc{orig, title={Orig}}")
    before = _source_row(db, block["slug"])
    changed = dict(block, **{field: new})
    _expect_reject(db, build_parsed(source=changed), ProvenanceError, submitted_by=ALICE)
    assert _source_row(db, block["slug"]) == before


def test_named_title_independent_contributor_rejected_and_rolled_back(db):
    """Rejected either by the inserter or by the DB CHECK; either way nothing persists."""
    block = named_source_block(title="Independent contributor Z")
    p = build_parsed(source=block)
    before = table_counts(db)
    with pytest.raises(Exception) as ei:
        insert(db, p, submitted_by=ALICE)
    assert not isinstance(ei.value, (TypeError, AttributeError, ImportError)), repr(ei.value)
    assert campaign_row(db, p.campaign.name) is None
    assert table_counts(db) == before
    assert _source_row(db, block["slug"]) is None


@pytest.mark.parametrize(
    "extra",
    [{"title": "Independent contributor Q"}, {"authors": "Alice Test"}, {"title": "x", "authors": "y"}],
)
def test_anonymous_first_upload_with_identity_fields_rejected(db, ProvenanceError, extra):
    slug, _ = allocate(db)
    _expect_reject(db, build_parsed(source=anon_source_block(**extra)), ProvenanceError,
                   assigned_anon_slug=slug, submitted_by=ALICE)
    assert _source_row(db, slug) is None


def test_anonymous_first_upload_without_assigned_slug_rejected(db, ProvenanceError):
    _expect_reject(db, build_parsed(source=anon_source_block()), ProvenanceError, submitted_by=ALICE)
    _expect_reject(db, build_parsed(source=anon_source_block()), ProvenanceError,
                   assigned_anon_slug=None, submitted_by=ALICE)


@pytest.mark.parametrize("submitter", [None, "", "   "])
def test_anonymous_first_upload_without_submitter_rejected(db, ProvenanceError, submitter):
    slug, _ = allocate(db)
    _expect_reject(db, build_parsed(source=anon_source_block()), ProvenanceError,
                   assigned_anon_slug=slug, submitted_by=submitter)
    assert _source_row(db, slug) is None


@pytest.mark.parametrize("slug", ["jdoe@v1", "no-such-source@v1"])
def test_reference_to_unknown_slug_rejected(db, ProvenanceError, slug):
    assert _source_row(db, slug) is None
    _expect_reject(db, build_parsed(source={"slug": slug, "consent": True}), ProvenanceError, submitted_by=ALICE)
    assert _source_row(db, slug) is None


def test_reference_to_unallocated_contributor_slug_rejected(fresh_db, ProvenanceError):
    url = fresh_db(provenance=True)
    for block in ({"slug": "contributor-E@v1", "consent": True},
                  {"slug": "contributor-E@v1", "attribution": "anonymous", "consent": True}):
        _expect_reject(url, build_parsed(source=block), ProvenanceError, submitted_by=ALICE)
        _expect_reject(url, build_parsed(source=block), ProvenanceError,
                       assigned_anon_slug="contributor-E@v1", submitted_by=ALICE)
    assert _source_row(url, "contributor-E@v1") is None


def _anon_first(url, submitter=ALICE):
    slug, title = allocate(url)
    p = build_parsed(author=submitter, source=anon_source_block())
    assert insert(url, p, assigned_anon_slug=slug, submitted_by=submitter) is True
    return slug, title, p


@pytest.mark.parametrize("other", [BOB, "", None, "   "])
def test_anonymous_reupload_by_non_owner_rejected(db, ProvenanceError, other):
    slug, _, _ = _anon_first(db, ALICE)
    for block in ({"slug": slug, "consent": True}, {"slug": slug, "attribution": "anonymous", "consent": True}):
        _expect_reject(db, build_parsed(source=block), ProvenanceError, submitted_by=other)
    assert fetchone(db, "SELECT count(*) FROM campaign_sources WHERE source_slug=%s", (slug,))[0] == 1


def test_anonymous_reference_with_null_owner_rejected(db, ProvenanceError):
    slug = f"contributor-NULL{uuid.uuid4().hex[:4].upper()}@v1"
    insert_sql_source(db, slug, attribution="anonymous", title="Independent contributor X",
                      authors=None, bibtex=None, owner=None)
    _expect_reject(db, build_parsed(source={"slug": slug, "consent": True}), ProvenanceError, submitted_by=ALICE)


@pytest.mark.parametrize("consent", [False, None])
def test_slug_only_reference_requires_consent(db, ProvenanceError, consent):
    block, _ = _new_named(db, ALICE)
    ref = {"slug": block["slug"]} if consent is None else {"slug": block["slug"], "consent": consent}
    _expect_reject(db, build_parsed(source=ref), ProvenanceError, submitted_by=ALICE)
    slug, _, _ = _anon_first(db, ALICE)
    ref = {"slug": slug} if consent is None else {"slug": slug, "consent": consent}
    _expect_reject(db, build_parsed(source=ref), ProvenanceError, submitted_by=ALICE)


# =============================================================================
# Success effects
# =============================================================================

def _consent_age_seconds(url, slug) -> float:
    return fetchone(url, "SELECT abs(extract(epoch FROM now() - consent_recorded_at)) "
                         "FROM source_datasets WHERE slug=%s", (slug,))[0]


def test_consent_recorded_now_on_first_insert_named_and_anonymous(db):
    block, _ = _new_named(db, ALICE)
    assert _consent_age_seconds(db, block["slug"]) < 30
    slug, _, _ = _anon_first(db, ALICE)
    assert _consent_age_seconds(db, slug) < 30


def test_named_insert_omitting_bibtex_stores_null(db):
    block = named_source_block()
    assert "bibtex" not in block
    assert insert(db, build_parsed(source=block), submitted_by=ALICE) is True
    assert fetchone(db, "SELECT bibtex IS NULL FROM source_datasets WHERE slug=%s", (block["slug"],))[0] is True


def test_named_first_insert_effects(db):
    block = named_source_block(bibtex="@misc{synthkey, title={Synthetic}}")
    p = build_parsed(author=ALICE, source=block)
    assert insert(db, p, submitted_by="  Alice-Test@Example.com ") is True
    cid, author = campaign_row(db, p.campaign.name)
    assert author == ALICE  # named: campaigns.author untouched
    row = _source_row(db, block["slug"])
    for k in ("title", "version", "year", "url", "license_spdx", "redistribution", "attribution", "authors", "bibtex"):
        assert row[k] == block[k], k
    assert row["consent_recorded_at"] is not None
    assert row["owner"] == ALICE
    assert _mapping(db, cid) == (block["slug"], ALICE)


def test_exactly_one_generation_time_evaluation_and_every_measurement_stamped(db):
    block = named_source_block()
    assessor = {"name": "ev-x", "version": "3.1", "platform": "darwin-arm64", "code_ref": "git:deadbeef"}
    p = build_parsed(source=block, assessor=assessor, n_candidates=4)
    before_evals = table_counts(db)["evaluations"]
    assert insert(db, p, submitted_by=ALICE) is True
    cid, _ = campaign_row(db, p.campaign.name)
    assert table_counts(db)["evaluations"] == before_evals + 1
    rows = fetchall(
        db,
        "SELECT m.evaluation_id FROM measurements m JOIN candidates c ON c.id=m.candidate_id WHERE c.campaign_id=%s",
        (cid,),
    )
    assert len(rows) == 8
    eval_ids = {r[0] for r in rows}
    assert len(eval_ids) == 1 and None not in eval_ids
    ev = fetchone(db, "SELECT assessor_name, assessor_version, platform, code_ref, kind, created_at "
                      "FROM evaluations WHERE id=%s", (eval_ids.pop(),))
    assert ev[:5] == ("ev-x", "3.1", "darwin-arm64", "git:deadbeef", "generation_time")
    assert ev[5] is not None


def test_assessor_without_code_ref_and_unknown_platform(db):
    p = build_parsed(source=named_source_block(), assessor={"name": "ev", "version": "1", "platform": "unknown"})
    assert insert(db, p, submitted_by=ALICE) is True
    cid, _ = campaign_row(db, p.campaign.name)
    ev = fetchall(db, "SELECT DISTINCT e.platform, e.code_ref, e.kind FROM evaluations e JOIN measurements m "
                      "ON m.evaluation_id=e.id JOIN candidates c ON c.id=m.candidate_id WHERE c.campaign_id=%s", (cid,))
    assert ev == [("unknown", None, "generation_time")]


def test_kind_is_never_taken_from_input(db):
    from pydantic import ValidationError

    from adrs_models import ADRSParsedCampaign
    from conftest import build_parsed_dict

    d = build_parsed_dict(source=named_source_block(), assessor=dict(DEFAULT_ASSESSOR, kind="regrade"))
    try:
        p = ADRSParsedCampaign.model_validate(d)
    except ValidationError:
        return  # rejecting the extra key is also compliant
    assert insert(db, p, submitted_by=ALICE) is True
    cid, _ = campaign_row(db, p.campaign.name)
    kinds = fetchall(db, "SELECT DISTINCT e.kind FROM evaluations e JOIN measurements m ON m.evaluation_id=e.id "
                         "JOIN candidates c ON c.id=m.candidate_id WHERE c.campaign_id=%s", (cid,))
    assert kinds == [("generation_time",)]


def test_other_campaigns_measurements_not_stamped(db):
    v1_id = insert_sql_campaign(db, unique("v1like-"), n_candidates=3)
    assert insert(db, build_parsed(source=named_source_block()), submitted_by=ALICE) is True
    n = fetchone(db, "SELECT count(*) FROM measurements m JOIN candidates c ON c.id=m.candidate_id "
                     "WHERE c.campaign_id=%s AND m.evaluation_id IS NOT NULL", (v1_id,))[0]
    assert n == 0


def test_each_insert_gets_its_own_evaluation(db):
    block = named_source_block()
    p1, p2 = build_parsed(source=block), build_parsed(source=block)
    assert insert(db, p1, submitted_by=ALICE) is True
    assert insert(db, p2, submitted_by=ALICE) is True

    def evals(name):
        cid, _ = campaign_row(db, name)
        return {r[0] for r in fetchall(db, "SELECT m.evaluation_id FROM measurements m JOIN candidates c "
                                           "ON c.id=m.candidate_id WHERE c.campaign_id=%s", (cid,))}

    e1, e2 = evals(p1.campaign.name), evals(p2.campaign.name)
    assert len(e1) == len(e2) == 1 and e1 != e2


def test_named_reupload_identical_ok_and_owner_consent_set_on_first_insert_only(db):
    block, _ = _new_named(db, ALICE, bibtex="@misc{k, title={T}}")
    first = _source_row(db, block["slug"])
    p2 = build_parsed(source=dict(block))
    assert insert(db, p2, submitted_by=BOB) is True
    after = _source_row(db, block["slug"])
    assert after == first  # owner and consent_recorded_at untouched
    assert after["owner"] == ALICE
    cid, _ = campaign_row(db, p2.campaign.name)
    assert _mapping(db, cid) == (block["slug"], BOB)


@pytest.mark.parametrize("omit", [("bibtex",), ("year",), ("bibtex", "year")])
def test_named_reupload_omitting_bibtex_or_year_ok(db, omit):
    block, _ = _new_named(db, ALICE, bibtex="@misc{k2, title={T2}}")
    first = _source_row(db, block["slug"])
    reup = {k: v for k, v in block.items() if k not in omit}
    p2 = build_parsed(source=reup)
    assert insert(db, p2, submitted_by=ALICE) is True
    assert _source_row(db, block["slug"]) == first


def test_named_slug_only_reference_ok_for_anyone(db):
    block, _ = _new_named(db, ALICE)
    first = _source_row(db, block["slug"])
    p2 = build_parsed(source={"slug": block["slug"], "consent": True})
    assert insert(db, p2, submitted_by=BOB) is True
    cid, author = campaign_row(db, p2.campaign.name)
    assert _mapping(db, cid) == (block["slug"], BOB)
    assert author == p2.campaign.author
    assert _source_row(db, block["slug"]) == first


def test_named_same_title_new_version_slug_ok(db):
    base = f"versioned-{uuid.uuid4().hex[:6]}"
    b1, _ = _new_named(db, ALICE, slug=f"{base}@v1", title=f"Shared title {base}", version="v1")
    b2 = named_source_block(slug=f"{base}@v2", title=f"Shared title {base}", version="v2")
    p2 = build_parsed(source=b2)
    assert insert(db, p2, submitted_by=ALICE) is True
    assert _source_row(db, b1["slug"])["version"] == "v1"
    assert _source_row(db, b2["slug"])["version"] == "v2"


def test_anonymous_first_upload_effects(db):
    slug, title = allocate(db)
    p = build_parsed(author="  Alice-Test@Example.com", source=anon_source_block())
    assert insert(db, p, assigned_anon_slug=slug, submitted_by="  Alice-Test@Example.com") is True
    cid, author = campaign_row(db, p.campaign.name)
    assert author == title  # campaigns.author is the anonymous label
    row = _source_row(db, slug)
    assert row["title"] == title
    assert row["attribution"] == "anonymous"
    assert row["authors"] is None and row["bibtex"] is None
    assert row["owner"] == ALICE
    assert row["consent_recorded_at"] is not None
    assert row["redistribution"] == "full"
    assert _mapping(db, cid) == (slug, ALICE)  # submitted_by keeps the submitter


def test_anonymous_title_derived_from_assigned_slug(db):
    slug, title = allocate(db)
    m = SLUG_RE.match(slug)
    assert m and title == f"Independent contributor {m.group(1)}"
    p = build_parsed(source=anon_source_block())
    assert insert(db, p, assigned_anon_slug=slug, submitted_by=ALICE) is True
    assert _source_row(db, slug)["title"] == title


@pytest.mark.parametrize("submitter", [ALICE, "ALICE-TEST@example.com  "])
def test_anonymous_reupload_by_owner_ok(db, submitter):
    slug, title, _ = _anon_first(db, ALICE)
    first = _source_row(db, slug)
    for block in ({"slug": slug, "consent": True}, {"slug": slug, "attribution": "anonymous", "consent": True}):
        p = build_parsed(author=submitter, source=block)
        assert insert(db, p, submitted_by=submitter) is True
        cid, author = campaign_row(db, p.campaign.name)
        assert author == title
        assert _mapping(db, cid) == (slug, ALICE)
    assert _source_row(db, slug) == first


def test_anonymous_batch_shares_one_slug(db):
    slug, title = allocate(db)
    names = []
    for _ in range(3):
        p = build_parsed(source=anon_source_block())
        assert insert(db, p, assigned_anon_slug=slug, submitted_by=ALICE) is True
        names.append(p.campaign.name)
    assert fetchone(db, "SELECT count(*) FROM source_datasets WHERE slug=%s", (slug,))[0] == 1
    assert fetchone(db, "SELECT count(*) FROM campaign_sources WHERE source_slug=%s", (slug,))[0] == 3
    for n in names:
        assert campaign_row(db, n)[1] == title


def test_rejected_insert_leaves_no_rows(db, ProvenanceError):
    """A rejected anonymous re-upload by a non-owner must not leave a campaign, its
    candidates, measurements, artifacts, edges, system or evaluation behind."""
    slug, _, _ = _anon_first(db, ALICE)
    p = build_parsed(source={"slug": slug, "consent": True}, system_name=unique("sys-"), n_candidates=5)
    _expect_reject(db, p, ProvenanceError, submitted_by=BOB)
    assert fetchone(db, "SELECT count(*) FROM systems WHERE name=%s", (p.campaign.system.name,))[0] == 0


@pytest.mark.parametrize("kind", ["named", "anonymous"])
def test_late_db_failure_rolls_back_source_evaluation_and_campaign(db, kind):
    """A DB error during candidate insertion (duplicate external_id violates
    UNIQUE(campaign_id, external_id)) must roll back everything for the campaign,
    including any source row and evaluation created in the same insert."""
    from adrs_models import ADRSParsedCampaign
    from conftest import build_parsed_dict

    if kind == "named":
        block, kw = named_source_block(), {}
        slug = block["slug"]
    else:
        slug, _ = allocate(db)
        block, kw = anon_source_block(), {"assigned_anon_slug": slug}
    d = build_parsed_dict(source=block, system_name=unique("sys-"), n_candidates=2)
    d["candidates"][1]["external_id"] = d["candidates"][0]["external_id"]
    p = ADRSParsedCampaign.model_validate(d)
    before = table_counts(db)
    with pytest.raises(Exception) as ei:
        insert(db, p, submitted_by=ALICE, **kw)
    assert not isinstance(ei.value, (TypeError, AttributeError, ImportError)), repr(ei.value)
    assert campaign_row(db, p.campaign.name) is None
    assert table_counts(db) == before
    assert _source_row(db, slug) is None


def test_existing_behaviour_preserved(db):
    p = build_parsed(source=named_source_block(), n_candidates=3)
    assert insert(db, p, submitted_by=ALICE) is True
    cid, _ = campaign_row(db, p.campaign.name)
    assert fetchone(db, "SELECT count(*) FROM candidates WHERE campaign_id=%s", (cid,))[0] == 3
    assert fetchone(db, "SELECT count(*) FROM artifacts WHERE campaign_id=%s", (cid,))[0] == 1
    assert fetchone(db, "SELECT count(*) FROM candidate_edges e JOIN candidates c ON c.id=e.target_candidate_id "
                        "WHERE c.campaign_id=%s", (cid,))[0] == 2


# =============================================================================
# allocate_anon_slug
# =============================================================================

def test_allocate_starts_at_A_and_is_sequential(fresh_db):
    url = fresh_db(provenance=True)
    assert allocate(url) == ("contributor-A@v1", "Independent contributor A")
    assert allocate(url) == ("contributor-B@v1", "Independent contributor B")
    assert fetchone(url, "SELECT last_value FROM anon_contributor_seq")[0] == 2


def test_allocate_takes_nextval_once(db):
    s1, t1 = allocate(db)
    s2, t2 = allocate(db)
    m1, m2 = SLUG_RE.match(s1), SLUG_RE.match(s2)
    assert m1 and m2
    assert label_to_int(m2.group(1)) == label_to_int(m1.group(1)) + 1
    assert t1 == f"Independent contributor {m1.group(1)}"
    assert fetchone(db, "SELECT last_value FROM anon_contributor_seq")[0] == label_to_int(m2.group(1))


def test_allocate_does_not_create_source_rows(db):
    before = table_counts(db)
    slug, _ = allocate(db)
    assert table_counts(db) == before
    assert _source_row(db, slug) is None


def test_allocate_concurrent_connections_get_distinct_slugs(db):
    import insert_adrs_campaign

    fn = getattr(insert_adrs_campaign, "allocate_anon_slug", None)
    assert fn is not None, "insert_adrs_campaign.allocate_anon_slug is not defined"
    results: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)
    lock = threading.Lock()

    def worker():
        try:
            with psycopg.connect(db) as conn:  # each thread: its own connection, open txn
                barrier.wait()
                got = [fn(conn)[0] for _ in range(5)]
                barrier.wait()  # all transactions still open here
                conn.commit()
            with lock:
                results.extend(got)
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors
    assert len(results) == 20 and len(set(results)) == 20
    assert all(SLUG_RE.match(s) for s in results)
