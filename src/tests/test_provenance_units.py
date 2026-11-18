"""Pure-function units in ``provenance``: ProvenanceError, label(), normalize_submitter(),
contributor_prefix() (i06 rule) and bibtex_key(). Spec sections "Submitter",
"Anonymous labels", "Backfill" (contributor_prefix) and "Citation tooling"."""

from __future__ import annotations

import pytest


def test_provenance_error_is_value_error(prov):
    assert issubclass(prov.ProvenanceError, ValueError)


# --- label -----------------------------------------------------------------

@pytest.mark.parametrize(
    "n,expected",
    [(1, "A"), (2, "B"), (5, "E"), (26, "Z"), (27, "AA"), (28, "AB"), (52, "AZ"), (53, "BA"),
     (78, "BZ"), (702, "ZZ"), (703, "AAA")],
)
def test_label(prov, n, expected):
    assert prov.label(n) == expected


@pytest.mark.parametrize("n", [0, -1, -27])
def test_label_rejects_n_below_one(prov, n):
    with pytest.raises(ValueError):
        prov.label(n)


def test_label_is_bijective_on_prefix(prov):
    seen = [prov.label(n) for n in range(1, 2000)]
    assert len(set(seen)) == len(seen)
    assert all(s.isalpha() and s.isupper() for s in seen)


# --- normalize_submitter -----------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("alice-test@example.com", "alice-test@example.com"),
        ("  Alice-Test@Example.COM \n", "alice-test@example.com"),
        ("BOB-TEST", "bob-test"),
        (None, None),
        ("", None),
        ("   ", None),
        ("\t\n ", None),
    ],
)
def test_normalize_submitter(prov, raw, expected):
    assert prov.normalize_submitter(raw) == expected


# --- contributor_prefix (synthetic names only) -------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("someone__run1", "someone"),
        ("alice-test/run_7", "alice-test"),
        ("bob-test__p11__seed3", "bob-test"),
        ("carol_test__x", "carol_test"),
        ("dave-test/sub__y", "dave-test"),
        ("someone__a/b", "someone"),
        # Excluded group-1 prefixes.
        ("cbl/x", None),
        ("cbl__x", None),
        ("cbl-multi/x", None),
        ("cloudcast__run", None),
        ("llm-sql/run", None),
        ("mas__run", None),
        ("other_cases/run", None),
        ("ns3/run", None),
        # Excluded model_mechanism names.
        ("eb1-preview_evox_p11_run1_abc", None),
        ("eb1_adaevolve__x", None),
        ("kimi-k2.5_best_of_n__x", None),
        ("gpt-5.4_evox/x", None),
        ("gpt-5-mini_adaevolve__x", None),
        ("gpt-oss-20b_evox__x", None),
        ("opus_best_of_n/x", None),
        ("gemini-2.5-flash_evox__x", None),
        ("qwen3.5:27b_adaevolve__x", None),
        # No separator / leading separator chars.
        ("plainname", None),
        ("single_underscore_only", None),
        ("_hidden__x", None),
        ("/abs/path", None),
        ("", None),
    ],
)
def test_contributor_prefix(prov, name, expected):
    assert prov.contributor_prefix(name) == expected


def test_contributor_prefix_model_rule_needs_mechanism_suffix(prov):
    # A model-like prefix not followed by an ADRS mechanism is not excluded.
    assert prov.contributor_prefix("opus-test__x") == "opus-test"
    assert prov.contributor_prefix("kimi-k2.5_other__x") == "kimi-k2.5_other"


# --- bibtex_key --------------------------------------------------------------

@pytest.mark.parametrize(
    "slug,expected",
    [
        ("cheng2025barbarians@v1", "cheng2025barbarians_v1"),
        ("contributor-A@v1", "contributor-A_v1"),
        ("ibm-ai-native-systems@v1", "ibm-ai-native-systems_v1"),
        ("a@@b", "a_b"),
        ("x y.z", "x_y_z"),
        ("we!rd slug@v2", "we_rd_slug_v2"),
        ("ok_key-1", "ok_key-1"),
        ("roar", "roar"),
    ],
)
def test_bibtex_key(prov, slug, expected):
    assert prov.bibtex_key(slug) == expected
