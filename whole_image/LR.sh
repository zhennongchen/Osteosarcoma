#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/LR.py"

for RANDOM_STATE in 0 30 60; do
  for top_k in None 15 20 25; do
    echo "========== LR | random_state=${RANDOM_STATE} | selector=lasso | top_k=${top_k} =========="
    python3 "${PYTHON_SCRIPT}" \
      --classifier LR \
      --random_state "${RANDOM_STATE}" \
      --lr_feature_selector lasso \
      --top_k "${top_k}"
  done
done
