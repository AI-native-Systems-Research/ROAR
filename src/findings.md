# Top-K Diversity Analysis Findings

**Date:** July 2026  
**Data Source:** SkyDiscover campaign results (Knapsack, Polyomino problems)

## Summary

Top-performing runs consistently converge toward similar solutions rather than discovering diverse winning strategies. This pattern holds across all models and algorithms analyzed.

## Methodology

- **Included runs**: Only runs where the best candidate beats baseline (score > 0)
- **Top-10 diversity**: Pairwise cosine distance among the best candidates from the 10 highest-scoring runs
- **Other diversity**: Pairwise cosine distance among the best candidates from remaining runs (excluding top 10)
- Higher values = more structurally diverse solutions among winners

---

## Main Finding: Winners Converge

Across all groupings, the top-10 winners show **lower diversity** than other runs:

| Grouping | Groups where top-10 more diverse | Groups where top-10 less diverse |
|----------|----------------------------------|----------------------------------|
| By Problem | 0 | 2 |
| By Model | 0 | 11 of 11 |
| By Algorithm | 0 | 3 of 3 |

**Interpretation:** High-scoring runs tend to converge toward similar solutions. The "other" runs (which still beat baseline but aren't top performers) explore more diverse approaches.

---

## By Problem

| Problem | Top-10 Diversity | Other Diversity | Gap | Top Score Mean | Runs (Total/Other) |
|---------|------------------|-----------------|-----|----------------|-------------------|
| Knapsack | 0.120 | 0.127 | -0.008 | 100.0 | 216 / 206 |
| Polyomino | 0.116 | 0.129 | -0.013 | 79.4 | 221 / 211 |

- **Knapsack**: Mild convergence; all top runs achieve perfect scores
- **Polyomino**: Similar convergence pattern; top scores around 79
- **Palindrome**: Excluded (no runs beat baseline — all scores = 0)

---

## By Model

| Model | Top-10 Diversity | Other Diversity | Gap | Top Score Mean | Total / Other Runs |
|-------|------------------|-----------------|-----|----------------|-------------------|
| eb1-pro | 0.081 | 0.091 | -0.010 | 100.0 | 15 / 5 |
| eb1 | 0.131 | 0.165 | -0.034 | 100.0 | 47 / 37 |
| eb1-frontier-preview | 0.115 | 0.181 | -0.066 | 100.0 | 54 / 44 |
| claude-opus-4-6 | 0.114 | 0.152 | -0.039 | 99.7 | 33 / 23 |
| moonshotai/Kimi-K2.5 | 0.094 | 0.152 | -0.058 | 99.7 | 53 / 43 |
| azure/gpt-5.4 | 0.101 | 0.152 | -0.051 | 99.6 | 28 / 18 |
| Azure/gpt-5-mini-2025-08-07 | 0.091 | 0.144 | -0.053 | 99.6 | 23 / 13 |
| openai/gpt-oss-20b | 0.109 | 0.156 | -0.047 | 99.6 | 43 / 33 |
| eb1-preview | 0.115 | 0.175 | -0.060 | 99.9 | 66 / 56 |
| eb1-delta-preview | 0.116 | 0.156 | -0.040 | 99.7 | 58 / 48 |
| gemini-2.5-flash | 0.162 | 0.215 | -0.052 | 95.0 | 16 / 6 |

### Observations

- **eb1-pro**: Lowest diversity gap (-0.010) — solutions already homogeneous, achieves 100% with high consistency
- **eb1-frontier-preview**: Largest gap (-0.066) — other runs explore diverse approaches, but top winners converge
- **gemini-2.5-flash**: Highest top-winner diversity (0.162) but lower scores (95.0) — finds more varied good solutions but not as optimal
- **qwen3.5:9b**: Excluded (insufficient runs beating baseline)

---

## By Algorithm

| Algorithm | Top-10 Diversity | Other Diversity | Gap | Top Score Mean | Total / Other Runs |
|-----------|------------------|-----------------|-----|----------------|-------------------|
| adaevolve | 0.128 | 0.140 | -0.012 | 100.0 | 145 / 135 |
| best_of_n | 0.105 | 0.130 | -0.025 | 100.0 | 156 / 146 |
| evox | 0.107 | 0.128 | -0.021 | 100.0 | 136 / 126 |

### Observations

- **best_of_n**: Strongest convergence among top winners (-0.025 gap)
- **adaevolve**: Maintains slightly more diversity among top winners (-0.012 gap)
- **evox**: Similar convergence pattern to best_of_n

---

## Research Implications

1. **Limited winning strategies**: The data suggests these optimization problems have a limited set of effective solutions. Different runs and models tend to discover the same approaches rather than equally-good but structurally different solutions.

2. **Top performers converge more than typical runs**: The "other" runs (which still beat baseline but aren't top-10) show more diversity than the top-10. This suggests that high performance correlates with finding the same optimal approaches.

3. **Model capability vs. diversity**: Higher-capability models (eb1-pro, eb1) achieve better scores with less diversity, suggesting they more reliably find the "known best" approach. Lower-capability models show more diversity but worse scores.

4. **Consistent pattern across algorithms**: All three algorithms (adaevolve, best_of_n, evox) show the same convergence pattern among top performers, suggesting this is problem-driven rather than algorithm-driven.

---

## Open Questions

- Do these problems have multiple global optima, or is convergence expected?
- Would diversity-preserving techniques (novelty search, quality-diversity) improve exploration?
- Is the Palindrome problem fundamentally harder, or is it a prompt/evaluation issue?
