#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/SVM.py"
TASK="${TASK:-Prognosis}"
RANDOM_STATE_LIST="${RANDOM_STATE_LIST:-0 10 20 30 40}"
GRIDSEARCH_RANGE="${GRIDSEARCH_RANGE:-train}"
TOP_K_LIST="${TOP_K_LIST:-20 25 30}"
LASSO_TOP_K_LIST="${LASSO_TOP_K_LIST:-None ${TOP_K_LIST}}"

for RANDOM_STATE in ${RANDOM_STATE_LIST}; do
  for selector in rfe; do
    for top_k in ${TOP_K_LIST}; do
      echo "========== SVM | task=${TASK} | random_state=${RANDOM_STATE} | selector=${selector} | top_k=${top_k} =========="
      python3 "${PYTHON_SCRIPT}" \
        --classifier SVM \
        --task "${TASK}" \
        --gridsearch_range "${GRIDSEARCH_RANGE}" \
        --random_state "${RANDOM_STATE}" \
        --svm_feature_selector "${selector}" \
        --top_k "${top_k}"
    done
  done
  echo "========== SVM | task=${TASK} | random_state=${RANDOM_STATE} | selector=rfecv =========="
  python3 "${PYTHON_SCRIPT}" \
    --classifier SVM \
    --task "${TASK}" \
    --gridsearch_range "${GRIDSEARCH_RANGE}" \
    --random_state "${RANDOM_STATE}" \
    --svm_feature_selector rfecv
  for top_k in ${LASSO_TOP_K_LIST}; do
    echo "========== SVM | task=${TASK} | random_state=${RANDOM_STATE} | selector=lasso | top_k=${top_k} =========="
    python3 "${PYTHON_SCRIPT}" \
      --classifier SVM \
      --task "${TASK}" \
      --gridsearch_range "${GRIDSEARCH_RANGE}" \
      --random_state "${RANDOM_STATE}" \
      --svm_feature_selector lasso \
      --top_k "${top_k}"
  done
done
