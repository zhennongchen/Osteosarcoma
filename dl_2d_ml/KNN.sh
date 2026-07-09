#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/KNN.py"
TASK="${TASK:-Prognosis}"
TRIAL_NAME="${TRIAL_NAME:-dl_2d_ml}"
RANDOM_STATE_LIST="${RANDOM_STATE_LIST:-0 10 20 30 40}"
GRIDSEARCH_RANGE="${GRIDSEARCH_RANGE:-train}"
TOP_K_LIST="${TOP_K_LIST:-20 25 30}"
LASSO_TOP_K_LIST="${LASSO_TOP_K_LIST:-None ${TOP_K_LIST}}"

for RANDOM_STATE in ${RANDOM_STATE_LIST}; do
  for selector in sfs; do
    for top_k in ${TOP_K_LIST}; do
      echo "========== KNN | task=${TASK} | random_state=${RANDOM_STATE} | selector=${selector} | top_k=${top_k} =========="
      python3 "${PYTHON_SCRIPT}" \
        --classifier KNN \
        --task "${TASK}" \
        --gridsearch_range "${GRIDSEARCH_RANGE}" \
    --trial_name "${TRIAL_NAME}" \
        --random_state "${RANDOM_STATE}" \
        --knn_feature_selector "${selector}" \
        --top_k "${top_k}"
    done
  done
  for top_k in ${LASSO_TOP_K_LIST}; do
    echo "========== KNN | task=${TASK} | random_state=${RANDOM_STATE} | selector=lasso | top_k=${top_k} =========="
    python3 "${PYTHON_SCRIPT}" \
      --classifier KNN \
      --task "${TASK}" \
      --gridsearch_range "${GRIDSEARCH_RANGE}" \
    --trial_name "${TRIAL_NAME}" \
      --random_state "${RANDOM_STATE}" \
      --knn_feature_selector lasso \
      --top_k "${top_k}"
  done
done
