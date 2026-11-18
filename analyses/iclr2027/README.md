# Analyses of the ICLR 2027 revision

This folder regenerates the numbers and figures that are new or changed in the revised paper,
relative to the reviewed version. Everything else in the paper comes from the unchanged code in
`src/analytics/` (for example `src/analytics/q5/results/` for the leave-one-out RQ4 of the
reviewed version).

ROAR's own analytics functions do the computation. Nothing in `src/` is modified. ROAR's four
database query functions are replaced at run time by pandas mirrors over the Hugging Face parquet
tables (`iclr2027/parquet_backend.py`), so no Postgres is needed. Set `ROAR_BACKEND=postgres`
(with `DATABASE_URL`) to use ROAR's SQL instead.

## Reproduce

```
export ROAR_DATA=<directory with the mirror's six *.parquet files>     # SHA-256 pinned in iclr2027/paths.py
export HARNESS_GEN_DATA=<Gupta et al. (2026) harness-generalization runs: metadata/runs.parquet, data/...>
bash run_all.sh                       # about 20 min on 8 workers; RQ4 on the external corpus dominates
bash tests/quick_test.sh              # about 1 min: quick re-run compared with the committed results
```

Optional inputs, not part of the public mirror:
- `ROAR_RAW_DIR` holds the raw SkyDiscover run directories. With it, `run_all.sh` also
  re-extracts `data/code_tokens.csv.gz` and runs the App. D validation.
- `GAMBLE_RUNS_DB` is the run database used to select App. D's 911 campaigns.
- `TIKTOKEN_CACHE_DIR` must point at a cached `o200k_base` file for an offline token extraction.

Environment: the ROAR `src/pyproject.toml` / `uv.lock` environment, plus `tiktoken`, which is
needed only for the token extraction. Python 3.12 was used.

| Script | Output | Paper | Runtime (16-core laptop) |
|---|---|---|---|
| `scripts/corpus_provenance.py` | `results/corpus_provenance.json` | Table 4, §4.1, Tables 2–3 counts | 5 s |
| `scripts/rq1_failed_campaigns.py` | `results/rq1_failed_campaigns.json`, `figures/fig4_*`, `fig5_*` | §4.2, App. G | 8 s |
| `scripts/rq2_escape.py` | `results/rq2_escape.json`, `figures/fig2_*` | §4.3, Fig. 2 | 30 s |
| `scripts/rq3_extract_code_tokens.py` (needs raw files) | `data/code_tokens.csv.gz` | input to RQ3 | 25 s |
| `scripts/rq3_length_control.py` | `results/rq3_length_control.json`, `results/rq3_early_campaigns.csv`, `figures/fig3_*` | §4.4, Fig. 3, App. I.2 | 35 s |
| `scripts/rq4_budget_allocation.py --corpus roar` | `results/rq4_roar.json`, `results/rq4_roar_splits.json.gz` | Table 1 (left), §4.5, Conclusion, App. K | 2.7 min (14 workers) |
| `scripts/rq4_budget_allocation.py --corpus external` | `results/rq4_external.json`, `..._splits.json.gz` | Table 1 (right) | 12 min (14 workers) |
| `scripts/rq4_figures.py` | `figures/fig16_*`, `fig17_*`, `fig18_*` | Figs. 16–18 | 5 s |
| `scripts/parser_validation.py` (needs raw files) | `results/parser_validation.json` | App. D | 40 s |

Seeds are fixed. RQ4 split r uses ROAR `split_calibration_evaluation(seed=r)` for r = 0..199,
`estimate_policy(seed=42)`, and replay seed 42 + the campaign's index in its configuration.
ROAR's own bootstraps use seed 42 internally. Committed outputs are byte-reproducible, with one
exception: the VB GLMM p-values carry floating-point noise of about 1e-6.

## Mapping: paper value → output

"Exact" means the committed output rounds to the printed value. Paths are in `results/`.

| Paper location | Printed | Ours | Output (key) |
|---|---|---|---|
| Table 4, at submission | 914 (98.5%) / 14 (1.5%) / 928 | same | `corpus_provenance.json` `table4_at_submission` |
| Table 4, after submission | 914 / 14 / 144 / 3,269 / 4,341; 21.1 / 0.3 / 3.3 / 75.3% | same | `table4_after_submission` |
| §4.1 | Palindrome: all 290 campaigns end at 0 | 290 / 290 | `per_problem_submitted.Palindrome` |
| App. G | 1 failed campaign, Knapsack, 30 candidates, all judge-server failures; configuration < 10 runs | 1; 30; 30; n = 7 | `rq1_failed_campaigns.json` |
| §4.2 / App. G | exclusion does not alter Figs. 1, 4, 5: 14 (7/6/1), 10 (5/3/2), 12 of 24 | same; cells identical before and after | `rq1_counts_excluding_failed`, `rq1_cells_unchanged_by_exclusion` |
| §4.3 | OR 1.00 per 10 iterations [0.95, 1.06] | 1.004 [0.949, 1.063] (three problems) | `rq2_escape.json` `trend.paper_three` |
| §4.3 | Knapsack 4–9%; CUDA Attention 40% at k ≥ 30 | 3.8–8.6%; 2 of 5 | `per_problem` |
| §4.3 | Polynomino 13–14% | 12.3 / 14.2 / 12.9% at k = 10/20/30; 0 of 87 at k = 40 | `per_problem` (see Differences) |
| §4.3 | "fewer than one episode in six" | maximum pooled rate 15.4% (k = 15) | `pooled.paper_three` |
| §4.4, App. I.2 | early diversity ρ = 0.16, p = 0.004 (Polynomino); 0.04, p = 0.51 (Knapsack); −0.50, p = 0.667 (CUDA) | 0.165, 0.004; 0.041, 0.51; −0.50, 0.667 | `rq3_length_control.json` `early_diversity.*.problem` |
| App. I.2 | Polynomino Best-of-N 0.41 (p < 0.001), eb1-preview 0.39 (0.001), Kimi-K2.5 0.36 (0.031), EvoX −0.12 (0.31), GPT-5.4 −0.39 (0.072) | same | `early_diversity.Polynomino.by_mechanism / by_model` |
| §4.4, Fig. 3 | Polynomino: worse +0.17, better +0.10, score div. +0.10 | +0.164, +0.102, +0.097 (parquet); +0.166, +0.102, +0.099 (Postgres) | `glmm.Polynomino.with_context_length` |
| §4.4, Fig. 3 | Knapsack: score div. −0.13, worse +0.13, code div. +0.14, better +0.02 (p = 0.72) | −0.118, +0.126, +0.129, +0.011 (p = 0.80) parquet; −0.117, +0.129, +0.133, +0.016 (p = 0.715) Postgres | `glmm.Knapsack` (see Differences) |
| §4.4, Fig. 3 | CUDA Attention: better −0.48, code div. +0.35 | −0.529, +0.389 | `glmm.CUDA Attention` (see Differences) |
| §4.5 | 17 configurations with ≥ 8 campaigns, 12 tested | 17, 12 | `rq4_roar.json` |
| Table 1, our corpus | 12/2/+9.2; 11/2/+12.4; 10/5/+12.2; 11/1/+11.7; 12/1/+9.5; 10/1/+8.2 | identical | `rq4_roar.json` `table1` |
| §4.5 | restart after 5 negative in 2 (ours) and 6 (external); after 10 negative in 1 and none | ours: 2 and 1; external: 7 and 2 | `table1.*.negative` |
| Conclusion | restart after 10 beats one campaign in 11 of 12 (ours) | 11 of 12 | `table1.Restart after 10` |
| Conclusion, Table 1 | restart after 10 positive in all 27 external configurations | 25 of 27 (k_stop = 11); 26 of 27 with k_stop = 10 | `rq4_external.json` (see Differences) |
| Table 1, external corpus | 24/3/+2.8; 19/2/+1.4; 21/12/+2.0; 27/6/+3.9; 26/2/+3.1; 21/2/+2.8 | 24/3/+2.6; 18/2/+1.3; 20/9/+1.9; 25/6/+3.7; 25/5/+3.0; 19/2/+2.8 | `rq4_external.json` `table1` (see Differences) |
| §4.5, App. K, Figs. 17–18 | 27 external configurations; their n and fit/evaluation sizes | the same 27, same n and sizes | `rq4_external.json` `per_configuration` |
| App. K.2 | excluded σs ≤ 1.3, four within a point of 100, tested σs ≥ 8; fits on 4–16 campaigns | 1.28; 4; 8.07; 4–16 | `rq4_roar.json` `excluded_configurations`, `per_configuration` |
| App. K.2 | median SD across splits: fitted 8.8, fixed rules 2.5–6.7 | 8.85; 2.55–6.72 | `median_sd_across_splits` |
| Fig. 16 | 12 rows, labels (n: fit/eval), band end −61 | same values; row order differs | `figures/fig16_*` |
| App. D | 911 campaigns, 47,168 candidates; identical scores and iteration indices; identical best score and iteration count per campaign | 911; 47,168; 47,168 / 47,168; 911 / 911 | `parser_validation.json` |

## Differences between the paper and the code

The paper is not edited here. Each item lists what the code gives.

1. **Table 1, external corpus, and the Conclusion's "all 27".** No local output produced these
   numbers. `rq4_budget_allocation.py --corpus external` reconstructs them from the protocol in
   §4.5 / App. K.1.
   - What matches: the set of 27 configurations, defined as (task, search method, model) with
     ≥ 8 runs of ≥ 50 steps and final-best SD ≥ 2 on the 0–100 scale, matches the paper, as do
     every row's n and half sizes.
   - What does not: the counts and means are close but not identical. Restart after 10 is
     positive in 25 of 27 configurations. The two negative ones are Heilbronn UCT/GPT-OSS-20B
     (−0.41) and Sequential/GPT-OSS-120B (−0.09), both 10–15-run configurations.
   - Unknown protocol details that could explain the gap:
     - which 60 evaluation campaigns are scored (we take the first 60 of the evaluation half in
       split order);
     - the seeds;
     - the step translation (we use step score = max over children, an unscored step = 0, ×100);
     - the stagnation-counter convention (`k_stop` 11 vs 10).
   - The same protocol reproduces the left half of Table 1 exactly.
2. **RQ3 coefficients.**
   - Polynomino matches. Knapsack score diversity and code diversity are −0.12 and +0.13, not
     the printed −0.13 and +0.14. CUDA Attention gives −0.53 and +0.39, not −0.48 and +0.35.
   - The CUDA code text is not in the mirror. Its control uses artifact bytes, which include the
     stored prompt.
   - The paper's numbers may come from a different length measure or fit. No local output
     reproduces them.
3. **Parquet vs Postgres.** Two places depend on row orders that ROAR's SQL leaves unspecified.
   - **RQ3 rows.** The running best in `query_mutation_factors` orders tied `iteration_index`
     values arbitrarily, and 2 of 45,947 improvement flags differ between backends. As a result,
     RQ3 coefficients move by up to 0.005.
   - **RQ1 bootstrap.** ROAR's BC bootstrap resamples in the order the campaigns arrive, and q1
     has no ORDER BY. With campaigns sorted by id, the confirmed/inconclusive counts are unchanged
     (7/6/1 and 5/3/2), but two Polynomino cells swap classes (Kimi-K2.5 / Best-of-N and
     eb1-delta-preview / Best-of-N), and some CIs move by up to 0.03. Polynomino GPT-5.4 /
     AdaEvolve's lower bound changes from 0.55 to 0.00.
   - With `ROAR_BACKEND=postgres` the scripts reproduce the original outputs exactly
     (`tests/test_results.py`).
4. **Polynomino 13–14% (§4.3).** This holds at k = 20/30 only. k = 10 gives 12.3%, and at k = 40
   no escape is observed in 87 episodes.

## Tests

- `tests/test_results.py` compares the committed results with every printed value above and with
  the outputs of the original, independently verified analyses (`tests/reference_values.json`).
  - Known differences are listed as KNOWN-DIFF, not failures.
  - The file comparison runs in seconds and needs no data.
- `tests/quick_test.sh` re-runs everything into a temporary directory. RQ4 uses 4 splits; the
  test then checks:
  - the JSON outputs equal the committed ones;
  - the 4 RQ4 splits equal splits 0–3 of the committed 200;
  - the independent re-implementation (`tests/indep_replay.py`) reproduces those splits exactly.
    That code has its own loader, simulator and grid fit, and imports no ROAR function.
  - `FULL=1 python -I tests/test_rq4_independent.py` checks all 200 splits, in about 25 min.
- `tests/test_backend_vs_postgres.py` compares each parquet mirror with ROAR's SQL on a Postgres
  load of the same mirror. It is skipped without `DATABASE_URL`.

## Scope

**Included:** only analyses whose numbers or figures appear in the revision.

**Not included:**
- the earlier leave-one-out baselines and refit bootstrap;
- cross-fitting, learning curves and temporal splits on the external runs;
- the hazard / Kaplan-Meier / GEE analyses of RQ2;
- cross-problem heterogeneity tests;
- dip / mixture checks of RQ1;
- the GAMBLe-database overlap counts;
- exploratory restart-ranking analyses.

None of them is printed in the revision.
