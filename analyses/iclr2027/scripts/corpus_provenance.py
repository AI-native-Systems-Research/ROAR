"""Table 4 (provenance of the campaigns in ROAR) and the corpus facts of Section 4.1 / Table 3.

Sources:
  * submitted corpus = mirror campaigns with author "Roar Authors" (928). Independent
    researchers' campaigns are those whose name carries a contributor prefix
    ("<prefix>__..." or "<prefix>/..."), other than the task-family prefixes of the
    post-submission corpus and the generator_mechanism naming of Ellis & Castro (2026).
    The rest are attributed to Ellis & Castro (2026). Prefixes are never written out:
    contributors are reported as anonymous labels.
  * post-submission campaigns of Cheng et al. (2025b) = author "Roar Authors 2" (144).
  * Gupta et al. (2026) = runs listed in HARNESS_GEN_DATA/metadata/runs.parquet (3,269).
"""
import collections
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from iclr2027 import paths, parquet_backend as pb, results_io  # noqa: E402

from analytics.utils import abbreviate_problem  # noqa: E402

STD = re.compile(r"^(eb1[a-z-]*|kimi-k2\.5|gpt-5\.4|gpt-5-mini|gpt-oss[a-z0-9-]*|opus|gemini-2\.5-flash|qwen3\.5[a-z0-9:]*)_"
                 r"(adaevolve|evox|best_of_n)")
FAMILIES = {"cbl", "cbl-multi", "cloudcast", "llm-sql", "mas", "other_cases", "ns3"}


def prefix(name):
    m = re.match(r"^([^_/][^/]*?)(__|/)", name)
    return m.group(1) if m else None


def main():
    pb.set_corpus("full")
    t = pb.tables()
    camp = t["campaigns"]
    sub = camp[camp.author == paths.SUBMITTED_AUTHOR]
    post = camp[camp.author == paths.POST_SUBMISSION_AUTHOR]
    pref = collections.Counter(p for n in sub.name for p in [prefix(n)] if p and p not in FAMILIES and not STD.match(n))
    labels = {p: "contributor " + chr(65 + i) for i, p in enumerate(sorted(pref))}
    contributed = sub[sub.name.map(lambda n: prefix(n) in labels)]
    by_label = collections.Counter(labels[prefix(n)] for n in contributed.name)
    runs = os.path.join(paths.HARNESS_GEN_DATA, "metadata", "runs.parquet")
    if not os.path.exists(runs):
        sys.exit(f"missing {runs}; set HARNESS_GEN_DATA to the Gupta et al. (2026) corpus")
    meta = pd.read_parquet(runs, columns=["task"])
    n_sub, n_ind = len(sub), len(contributed)
    rows_at = {"Ellis & Castro (2026)": n_sub - n_ind, "Independent researchers": n_ind}
    rows_after = dict(rows_at)
    rows_after.update({"Cheng et al. (2025b)": len(post), "Gupta et al. (2026)": int(len(meta))})

    def share(rows):
        tot = sum(rows.values())
        return {k: {"campaigns": v, "share_pct": round(100 * v / tot, 1)} for k, v in rows.items()} | {"Total": tot}

    # Section 4.1 / Table 3 corpus facts, submitted corpus
    cand = t["candidates"][t["candidates"].campaign_id.isin(set(sub.id))]
    sc = pb.numeric_scores()
    best = cand[["id", "campaign_id"]].merge(sc, left_on="id", right_on="candidate_id").groupby("campaign_id").score.max()
    sub = sub.assign(problem=sub.research_question.map(abbreviate_problem), best=sub.id.map(best))
    per_problem = {}
    for p, s in sub.groupby("research_question"):
        per_problem[abbreviate_problem(p)] = {"campaigns": int(len(s)),
                                              "candidates": int(cand.campaign_id.isin(set(s.id)).sum()),
                                              "campaigns_with_final_score_zero": int((s.best == 0).sum())}
    res = {"table4_at_submission": share(rows_at), "table4_after_submission": share(rows_after),
           "independent_researchers_by_anonymous_label": dict(sorted(by_label.items())),
           "submitted_campaigns": n_sub, "submitted_candidates": int(len(cand)),
           "post_submission_campaigns": int(len(post)), "mirror_campaigns": int(len(camp)),
           "gupta_runs": int(len(meta)), "per_problem_submitted": per_problem}
    results_io.write_result("corpus_provenance.json", res)
    print(res["table4_after_submission"], per_problem.get("Palindrome", per_problem.keys()))


if __name__ == "__main__":
    main()
