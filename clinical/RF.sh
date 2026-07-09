#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/RF.py"
TASK="${TASK:-Prognosis}"
RANDOM_STATE_LIST="${RANDOM_STATE_LIST:-0 10 20 30 40}"
GRIDSEARCH_RANGE="${GRIDSEARCH_RANGE:-train}"
SELECTOR_LIST="${SELECTOR_LIST:-none lasso rfecv}"

for RANDOM_STATE in ${RANDOM_STATE_LIST}; do
  for selector in ${SELECTOR_LIST}; do
    echo "========== Clinical RF | task=${TASK} | random_state=${RANDOM_STATE} | selector=${selector} | top_k=None =========="
    python3 "${PYTHON_SCRIPT}" \
      --classifier RF \
      --task "${TASK}" \
      --gridsearch_range "${GRIDSEARCH_RANGE}" \
      --random_state "${RANDOM_STATE}" \
      --rf_feature_selector "${selector}" \
      --top_k None
  done
done
