#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/RF.py"
TASK="${TASK:-Prognosis}"
RANDOM_STATE_LIST="${RANDOM_STATE_LIST:-0 15 30 45 60}"
TOP_K_LIST="${TOP_K_LIST:-20 25 30}"

for RANDOM_STATE in ${RANDOM_STATE_LIST}; do
for selector in rfe; do
  for top_k in ${TOP_K_LIST}; do
    echo "========== RF | task=${TASK} | random_state=${RANDOM_STATE} | selector=${selector} | top_k=${top_k} =========="
    python3 "${PYTHON_SCRIPT}" \
      --classifier RF \
      --task "${TASK}" \
      --random_state "${RANDOM_STATE}" \
      --rf_feature_selector "${selector}" \
      --top_k "${top_k}"
  done
done
echo "========== RF | task=${TASK} | random_state=${RANDOM_STATE} | selector=rfecv =========="
python3 "${PYTHON_SCRIPT}" \
  --classifier RF \
  --task "${TASK}" \
  --random_state "${RANDOM_STATE}" \
  --rf_feature_selector rfecv
done
