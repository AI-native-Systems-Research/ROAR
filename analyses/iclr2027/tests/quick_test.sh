#!/bin/bash
# Quick regression test (about 2-4 minutes): re-runs every analysis into a temporary directory
# with 4 RQ4 splits and compares with the committed results:
#   * provenance, RQ1, RQ2, RQ3: the re-run JSON must equal the committed JSON (runtime fields ignored);
#   * RQ4 (our corpus): the 4 re-run splits must equal splits 0-3 of the committed 200-split records;
#   * RQ4 (our corpus): the independent re-implementation must reproduce those splits exactly;
#   * tests/test_results.py on the committed results (paper values and original outputs).
# usage: ROAR_DATA=<dir> HARNESS_GEN_DATA=<dir> [PYTHON=...] bash tests/quick_test.sh
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
TMP="${TMPDIR:-/tmp}/iclr2027_quick_$$"
mkdir -p "$TMP/results" "$TMP/figures"
export ICLR_RESULTS_DIR="$TMP/results" ICLR_FIGURES_DIR="$TMP/figures" QUICK=1
T0=$(date +%s)
for s in corpus_provenance rq1_failed_campaigns rq2_escape rq3_length_control; do
  echo "== $s"; "$PY" -I "scripts/$s.py" > "$TMP/$s.log" 2>&1 || { cat "$TMP/$s.log"; exit 1; }
done
echo "== rq4 (4 splits)"; "$PY" -I scripts/rq4_budget_allocation.py --corpus roar --splits 4 --counter-variants --workers "${WORKERS:-8}" > "$TMP/rq4.log" 2>&1
"$PY" -I tests/compare_quick.py "$TMP/results" results
echo "== independent RQ4 re-implementation on the quick splits"
SPLITS=4 "$PY" -I tests/test_rq4_independent.py
unset ICLR_RESULTS_DIR ICLR_FIGURES_DIR QUICK
echo "== committed results vs paper and original outputs"
"$PY" -I tests/test_results.py | tail -1
rm -rf "$TMP"
echo "quick test passed in $(( $(date +%s) - T0 )) s"
