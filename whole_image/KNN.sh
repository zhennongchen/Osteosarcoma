#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/KNN.py"

for RANDOM_STATE in 0 30 60; do
  for top_k in 15 20 25; do
    echo "========== KNN | random_state=${RANDOM_STATE} | selector=sfs | top_k=${top_k} =========="
    python3 "${PYTHON_SCRIPT}" \
      --classifier KNN \
      --random_state "${RANDOM_STATE}" \
      --knn_feature_selector sfs \
      --top_k "${top_k}"
  done
done
