#!/bin/bash
# Regenerate every result and figure of the ICLR 2027 revision that this folder covers.
#   ROAR_DATA=<dir with the mirror's *.parquet> HARNESS_GEN_DATA=<Gupta et al. runs> bash run_all.sh
# Optional: ROAR_RAW_DIR (+ RUNS_DB) re-reads raw run directories (code tokens, App. D);
#           PYTHON (default: python3), WORKERS (default 8), QUICK=1 (4 RQ4 splits; smoke test only).
# Seeds are fixed inside the scripts (see README). Writes results/ and figures/.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
W="${WORKERS:-8}"
T0=$(date +%s)
step() { local t; t=$(date +%s); echo "== $*"; "$PY" -I "$@"; echo "   $(( $(date +%s) - t )) s"; }
step scripts/corpus_provenance.py
step scripts/rq1_failed_campaigns.py
step scripts/rq2_escape.py
if [ -n "$ROAR_RAW_DIR" ]; then step scripts/rq3_extract_code_tokens.py; fi
step scripts/rq3_length_control.py
step scripts/rq4_budget_allocation.py --corpus roar --workers "$W" --counter-variants
if [ -f "$HARNESS_GEN_DATA/metadata/runs.parquet" ]; then
  step scripts/rq4_budget_allocation.py --corpus external --workers "$W" --counter-variants
else
  echo "== skip external corpus (HARNESS_GEN_DATA not set)"
fi
step scripts/rq4_figures.py
if [ -n "$ROAR_RAW_DIR" ]; then step scripts/parser_validation.py; fi
step tests/test_results.py
echo "run_all.sh finished in $(( $(date +%s) - T0 )) s"
