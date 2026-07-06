#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/SVM.py"
TASK="${TASK:-Prognosis}"
RANDOM_STATE_LIST="${RANDOM_STATE_LIST:-0 10 20 30 40}"
SELECTOR_LIST="${SELECTOR_LIST:-none lasso rfecv}"

for RANDOM_STATE in ${RANDOM_STATE_LIST}; do
  for selector in ${SELECTOR_LIST}; do
    echo "========== Clinical SVM | task=${TASK} | random_state=${RANDOM_STATE} | selector=${selector} | top_k=None =========="
    python3 "${PYTHON_SCRIPT}" \
      --classifier SVM \
      --task "${TASK}" \
      --random_state "${RANDOM_STATE}" \
      --svm_feature_selector "${selector}" \
      --top_k None
  done
done
