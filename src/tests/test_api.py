"""Spec "Callers" -> API /upload and /upload/parsed (one anonymous slug per job, returned
as JobStatus.anon_slug; submitted_by = form author; per-line blocks beat a root
author_input.json in /upload/parsed) and "Models" (assigned_anon_slug in uploaded JSON
is never honoured). Uses FastAPI's TestClient against the session throwaway DB."""

from __future__ import annotations

import json
import os
import re
import time

import pytest

from conftest import (
    DEFAULT_ASSESSOR,
    anon_source_block,
    build_parsed_dict,
    fetchall,
    fetchone,
    make_targz,
    named_source_block,
    table_counts,
    unique,
    write_author_input,
    write_skydiscover_campaign,
)

ALICE = "alice-test@example.com"
BOB = "bob-test@example.com"
SLUG_RE = re.compile(r"^contributor-[A-Z]+@v1$")


def _shim_bsdtar(tmp_path, monkeypatch):
    """api.extract_tar passes GNU-only --checkpoint flags. On hosts whose `tar` is bsdtar
    (macOS), put a PATH shim that drops them; the extraction itself is unchanged."""
    import shutil
    import subprocess

    real = shutil.which("tar")
    if not real:
        return
    ver = subprocess.run([real, "--version"], capture_output=True, text=True).stdout
    if "bsdtar" not in ver:
        return
    shim_dir = tmp_path / "tar_shim"
    shim_dir.mkdir()
    shim = shim_dir / "tar"
    shim.write_text(
        "#!/bin/bash\nargs=()\nfor a in \"$@\"; do case \"$a\" in --checkpoint*) ;; *) args+=(\"$a\");; esac; done\n"
        f"exec {real} \"${{args[@]}}\"\n"
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim_dir}:{os.environ.get('PATH', '')}")


@pytest.fixture
def client(db, tmp_path, monkeypatch):
    import api

    _shim_bsdtar(tmp_path, monkeypatch)

    monkeypatch.setattr(api, "DATA_DIR", tmp_path / "api_data", raising=True)
    (tmp_path / "api_data").mkdir()
    monkeypatch.setattr(api, "get_embedding_model", lambda *a, **k: None, raising=False)
    from fastapi.testclient import TestClient

    with TestClient(api.app) as c:  # context manager keeps the loop alive for background jobs
        yield c


def _run_job(client, endpoint, archive: bytes, data: dict, timeout=90) -> dict:
    r = client.post(endpoint, files={"file": ("upload.tar.gz", archive, "application/gzip")}, data=data)
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/upload/status/{job_id}").json()
        if st["status"] in ("complete", "error"):
            return st
        time.sleep(0.2)
    pytest.fail(f"job {job_id} did not finish in {timeout}s")


def _by_name(url, name):
    return fetchone(
        url,
        "SELECT c.author, cs.source_slug, cs.submitted_by FROM campaigns c "
        "LEFT JOIN campaign_sources cs ON cs.campaign_id=c.id WHERE c.name=%s",
        (name,),
    )


def _assessor_of(url, name):
    return fetchall(
        url,
        "SELECT DISTINCT e.assessor_name, e.kind FROM campaigns c JOIN candidates k ON k.campaign_id=c.id "
        "JOIN measurements m ON m.candidate_id=k.id JOIN evaluations e ON e.id=m.evaluation_id WHERE c.name=%s",
        (name,),
    )


def test_job_status_model_exposes_anon_slug():
    import api

    assert "anon_slug" in api.JobStatus.model_fields
    js = api.JobStatus(job_id="x", status="complete")
    assert js.anon_slug is None


# --- /upload -------------------------------------------------------------------

def _upload_tree(tmp_path, n, root_ai):
    top = tmp_path / unique("up-")
    root = top / "batch"
    names = [unique("camp-") for _ in range(n)]
    for nm in names:
        write_skydiscover_campaign(root / nm)
    write_author_input(root, root_ai)
    return top, names


def test_upload_anonymous_one_slug_per_job(client, db, tmp_path):
    top, names = _upload_tree(tmp_path, 2, {"assessor": DEFAULT_ASSESSOR, "source": anon_source_block()})
    st = _run_job(client, "/upload", make_targz(top), {"author": "  Alice-Test@Example.com ", "system_type": "skydiscover"})
    assert st["status"] == "complete", st
    slug = st["anon_slug"]
    assert isinstance(slug, str) and SLUG_RE.match(slug)
    rows = [_by_name(db, n) for n in names]
    title = "Independent contributor " + slug[len("contributor-"):-len("@v1")]
    assert all(r == (title, slug, ALICE) for r in rows), rows
    assert fetchone(db, "SELECT owner, title, authors FROM source_datasets WHERE slug=%s", (slug,)) == (ALICE, title, None)
    for n in names:
        assert _assessor_of(db, n) == [(DEFAULT_ASSESSOR["name"], "generation_time")]


def test_upload_two_jobs_get_distinct_slugs(client, db, tmp_path):
    slugs = []
    for _ in range(2):
        top, _ = _upload_tree(tmp_path, 1, {"assessor": DEFAULT_ASSESSOR, "source": anon_source_block()})
        st = _run_job(client, "/upload", make_targz(top), {"author": ALICE, "system_type": "skydiscover"})
        assert st["status"] == "complete", st
        slugs.append(st["anon_slug"])
    assert slugs[0] != slugs[1] and all(SLUG_RE.match(s) for s in slugs)


def test_upload_named_has_no_anon_slug(client, db, tmp_path):
    src = named_source_block()
    top, names = _upload_tree(tmp_path, 1, {"assessor": DEFAULT_ASSESSOR, "source": src})
    st = _run_job(client, "/upload", make_targz(top), {"author": BOB, "system_type": "skydiscover"})
    assert st["status"] == "complete", st
    assert st["anon_slug"] is None
    assert _by_name(db, names[0])[1:] == (src["slug"], BOB)


def test_upload_anonymous_reupload_by_other_submitter_rejected(client, db, tmp_path):
    top, _ = _upload_tree(tmp_path, 1, {"assessor": DEFAULT_ASSESSOR, "source": anon_source_block()})
    st = _run_job(client, "/upload", make_targz(top), {"author": ALICE, "system_type": "skydiscover"})
    slug = st["anon_slug"]
    assert slug
    top2, names2 = _upload_tree(tmp_path, 1, {"assessor": DEFAULT_ASSESSOR, "source": {"slug": slug, "consent": True}})
    st2 = _run_job(client, "/upload", make_targz(top2), {"author": BOB, "system_type": "skydiscover"})
    assert st2["status"] == "error" or (st2.get("result") or {}).get("campaign_count", 0) == 0
    assert _by_name(db, names2[0]) is None
    # The owner can re-upload.
    top3, names3 = _upload_tree(tmp_path, 1, {"assessor": DEFAULT_ASSESSOR, "source": {"slug": slug, "consent": True}})
    st3 = _run_job(client, "/upload", make_targz(top3), {"author": ALICE, "system_type": "skydiscover"})
    assert st3["status"] == "complete", st3
    assert _by_name(db, names3[0])[1:] == (slug, ALICE)


def test_upload_without_provenance_inserts_nothing(client, db, tmp_path):
    top, names = _upload_tree(tmp_path, 1, {"version": "v1"})
    before = table_counts(db)
    st = _run_job(client, "/upload", make_targz(top), {"author": ALICE, "system_type": "skydiscover"})
    assert st["status"] == "error" or (st.get("result") or {}).get("campaign_count", 0) == 0
    assert _by_name(db, names[0]) is None
    assert table_counts(db) == before


# --- /upload/parsed --------------------------------------------------------------

def _parsed_archive(tmp_path, lines: list[dict], root_ai: dict | None, wrapper: bool = False):
    top = tmp_path / unique("parsed-")
    base = top / "wrap" if wrapper else top
    base.mkdir(parents=True)
    (base / "_parsed.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    if root_ai is not None:
        write_author_input(base, root_ai)
    (base / "raw").mkdir()
    (base / "raw" / "readme.txt").write_text("raw campaign files")
    return make_targz(top)


def test_parsed_per_line_block_beats_root_author_input(client, db, tmp_path):
    root_src = named_source_block()
    line_src = named_source_block()
    root_assessor = dict(DEFAULT_ASSESSOR, name="root-assessor")
    line_assessor = dict(DEFAULT_ASSESSOR, name="line-assessor")
    l1 = build_parsed_dict(author="", assessor=line_assessor, source=line_src)
    l2 = build_parsed_dict(author="", assessor=None, source=None)
    l3 = build_parsed_dict(author="", assessor=None, source=line_src)
    archive = _parsed_archive(tmp_path, [l1, l2, l3], {"author": ALICE, "assessor": root_assessor, "source": root_src})
    st = _run_job(client, "/upload/parsed", archive, {"author": ALICE})
    assert st["status"] == "complete", st
    n1, n2, n3 = (l["campaign"]["name"] for l in (l1, l2, l3))
    assert _by_name(db, n1)[1:] == (line_src["slug"], ALICE)
    assert _assessor_of(db, n1) == [("line-assessor", "generation_time")]
    assert _by_name(db, n2)[1:] == (root_src["slug"], ALICE)
    assert _assessor_of(db, n2) == [("root-assessor", "generation_time")]
    assert _by_name(db, n3)[1:] == (line_src["slug"], ALICE)
    assert _assessor_of(db, n3) == [("root-assessor", "generation_time")]
    assert st["anon_slug"] is None


def test_parsed_root_author_input_in_wrapper_folder(client, db, tmp_path):
    root_src = named_source_block()
    l1 = build_parsed_dict(author="", assessor=None, source=None)
    archive = _parsed_archive(tmp_path, [l1], {"assessor": DEFAULT_ASSESSOR, "source": root_src}, wrapper=True)
    st = _run_job(client, "/upload/parsed", archive, {"author": BOB})
    assert st["status"] == "complete", st
    assert _by_name(db, l1["campaign"]["name"])[1:] == (root_src["slug"], BOB)


def test_parsed_anonymous_one_slug_per_job(client, db, tmp_path):
    lines = [build_parsed_dict(author=ALICE, source=anon_source_block()) for _ in range(3)]
    st = _run_job(client, "/upload/parsed", _parsed_archive(tmp_path, lines, None), {"author": ALICE})
    assert st["status"] == "complete", st
    slug = st["anon_slug"]
    assert slug and SLUG_RE.match(slug)
    rows = [_by_name(db, l["campaign"]["name"]) for l in lines]
    assert {r[1] for r in rows} == {slug}
    assert {r[2] for r in rows} == {ALICE}
    assert fetchone(db, "SELECT count(*) FROM campaign_sources WHERE source_slug=%s", (slug,))[0] == 3


def test_parsed_assigned_anon_slug_in_json_is_ignored(client, db, tmp_path):
    forged = "contributor-ZZZZ@v1"
    line = build_parsed_dict(author=ALICE, source=anon_source_block())
    line["assigned_anon_slug"] = forged
    line["source"]["assigned_anon_slug"] = forged
    st = _run_job(client, "/upload/parsed", _parsed_archive(tmp_path, [line], None), {"author": ALICE})
    assert fetchone(db, "SELECT count(*) FROM source_datasets WHERE slug=%s", (forged,))[0] == 0
    row = _by_name(db, line["campaign"]["name"])
    if st["status"] == "complete" and row is not None:
        assert row[1] == st["anon_slug"] and row[1] != forged
    assert st.get("anon_slug") != forged


def test_parsed_cannot_hijack_existing_anonymous_slug(client, db, tmp_path):
    first = [build_parsed_dict(author=ALICE, source=anon_source_block())]
    st = _run_job(client, "/upload/parsed", _parsed_archive(tmp_path, first, None), {"author": ALICE})
    slug = st["anon_slug"]
    assert slug
    line = build_parsed_dict(author=BOB, source={"slug": slug, "consent": True})
    st2 = _run_job(client, "/upload/parsed", _parsed_archive(tmp_path, [line], None), {"author": BOB})
    assert st2["status"] == "error" or (st2.get("result") or {}).get("campaign_count", 0) == 0
    assert _by_name(db, line["campaign"]["name"]) is None
