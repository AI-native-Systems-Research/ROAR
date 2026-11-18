"""Check the parquet mirrors against ROAR's own SQL on a Postgres load of the same mirror.

Needs DATABASE_URL pointing at a database holding the submitted corpus (928 campaigns),
e.g. one built with src/create_tables.py from the same parquet files. Skipped otherwise.
usage: python -I tests/test_backend_vs_postgres.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from iclr2027 import paths, parquet_backend as pb  # noqa: E402

if not os.environ.get("DATABASE_URL"):
    print("SKIP: DATABASE_URL not set")
    sys.exit(0)

import analytics.q1_variance as q1  # noqa: E402
import analytics.q2_calibration as q2  # noqa: E402
import analytics.q3_diversity.early_diversity as ed  # noqa: E402
import analytics.q3_diversity.factor_importance as fi  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (" " + detail if detail else ""))
    fails += 0 if ok else 1


# q2
a = {c["campaign_id"]: c for c in q2.query_iteration_scores()}
b = {c["campaign_id"]: c for c in pb.query_iteration_scores()}
check("q2 campaigns", set(a) == set(b), f"{len(a)} vs {len(b)}")
check("q2 order", [c for c in a] == [c for c in b])
bad = [k for k in a if a[k]["iterations"] != b[k]["iterations"] or list(a[k]["models_used"] or []) != list(b[k]["models_used"] or [])
       or a[k]["algorithm_used"] != b[k]["algorithm_used"]]
check("q2 iterations/models identical", not bad, f"{len(bad)} differ")

# q1
a = {c["campaign_id"]: c for c in q1.query_skydiscover_scores()}
b = {c["campaign_id"]: c for c in pb.query_skydiscover_scores()}
check("q1 campaigns", set(a) == set(b), f"{len(a)} vs {len(b)}")
bad = [k for k in a if abs(a[k]["primary_score"] - b[k]["primary_score"]) > 1e-12]
check("q1 best scores identical", not bad, f"{len(bad)} differ")

# q3 early diversity
a = {c["campaign_id"]: c for c in ed._compute_early_diversity_campaign_data()}
b = {c["campaign_id"]: c for c in pb._compute_early_diversity_campaign_data()}
check("early campaigns", set(a) == set(b), f"{len(a)} vs {len(b)}")
dd = max(abs(a[k]["early_diversity"] - b[k]["early_diversity"]) for k in a if k in b)
nn = sum(a[k]["n_early_candidates"] != b[k]["n_early_candidates"] for k in a if k in b)
check("early diversity within 1e-4 (4-dp rounding)", dd <= 1.0001e-4 and nn == 0, f"max diff {dd}, n_early differ {nn}")

# q3 factors
a = {r["candidate_id"]: r for r in fi.query_mutation_factors()}
b = {r["candidate_id"]: r for r in pb.query_mutation_factors()}
check("factor rows same set", set(a) == set(b), f"{len(a)} vs {len(b)}")
keys = ["score_diversity", "context_diversity", "better_ratio", "worse_ratio", "context_size", "child_score", "parent_score"]
# VAR_POP is computed by Postgres in extended precision; numpy's float64 variance can
# differ in the last of the 4 rounded decimals.
bad = [k for k in a if k in b and any(abs(a[k][f] - b[k][f]) > 1.0001e-4 + 1e-12 * abs(a[k][f]) for f in keys)]
check("factor values equal to 4-dp rounding", not bad, f"{len(bad)} differ")
flip = sum((a[k]["score_delta_global"] > 0) != (b[k]["score_delta_global"] > 0) for k in a if k in b)
check("improvement flags differ only through tied iteration_index (<= 5 rows)", flip <= 5, f"{flip} rows")
sys.exit(1 if fails else 0)
