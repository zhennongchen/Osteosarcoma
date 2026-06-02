#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/RF.py"

for RANDOM_STATE in 0 30 60; do
  for selector in rfe sfs; do
    for top_k in 15 20 25; do
      echo "========== RF | random_state=${RANDOM_STATE} | selector=${selector} | top_k=${top_k} =========="
      python3 "${PYTHON_SCRIPT}" \
        --classifier RF \
        --random_state "${RANDOM_STATE}" \
        --rf_feature_selector "${selector}" \
        --top_k "${top_k}"
    done
  done

  echo "========== RF | random_state=${RANDOM_STATE} | selector=rfecv =========="
  python3 "${PYTHON_SCRIPT}" \
    --classifier RF \
    --random_state "${RANDOM_STATE}" \
    --rf_feature_selector rfecv
done
