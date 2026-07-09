#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/RF.py"
TASK="${TASK:-Prognosis}"
RANDOM_STATE_LIST="${RANDOM_STATE_LIST:-0 10 20 30 40}"
GRIDSEARCH_RANGE="${GRIDSEARCH_RANGE:-train}"
TOP_K_LIST="${TOP_K_LIST:-20 25 30}"
LASSO_TOP_K_LIST="${LASSO_TOP_K_LIST:-None ${TOP_K_LIST}}"
HABITAT_MODE="${HABITAT_MODE:-individual}"
PCC_RADIOMICS_PATH="${PCC_RADIOMICS_PATH:-/host/d/projects/Habitats/radiomics/habitats_individual/habitat_radiomics_measurements_avg_PCC.xlsx}"
IMAGE_TYPE="${IMAGE_TYPE:-habitats_individual}"

for RANDOM_STATE in ${RANDOM_STATE_LIST}; do
  for selector in rfe; do
    for top_k in ${TOP_K_LIST}; do
      echo "========== RF | task=${TASK} | habitat_mode=${HABITAT_MODE} | random_state=${RANDOM_STATE} | selector=${selector} | top_k=${top_k} =========="
      python3 "${PYTHON_SCRIPT}" \
        --classifier RF \
        --task "${TASK}" \
        --gridsearch_range "${GRIDSEARCH_RANGE}" \
        --pcc_radiomics_path "${PCC_RADIOMICS_PATH}" \
        --image_type "${IMAGE_TYPE}" \
        --random_state "${RANDOM_STATE}" \
        --rf_feature_selector "${selector}" \
        --top_k "${top_k}"
    done
  done
  echo "========== RF | task=${TASK} | habitat_mode=${HABITAT_MODE} | random_state=${RANDOM_STATE} | selector=rfecv =========="
  python3 "${PYTHON_SCRIPT}" \
    --classifier RF \
    --task "${TASK}" \
    --gridsearch_range "${GRIDSEARCH_RANGE}" \
    --pcc_radiomics_path "${PCC_RADIOMICS_PATH}" \
    --image_type "${IMAGE_TYPE}" \
    --random_state "${RANDOM_STATE}" \
    --rf_feature_selector rfecv
  for top_k in ${LASSO_TOP_K_LIST}; do
    echo "========== RF | task=${TASK} | habitat_mode=${HABITAT_MODE} | random_state=${RANDOM_STATE} | selector=lasso | top_k=${top_k} =========="
    python3 "${PYTHON_SCRIPT}" \
      --classifier RF \
      --task "${TASK}" \
      --gridsearch_range "${GRIDSEARCH_RANGE}" \
      --pcc_radiomics_path "${PCC_RADIOMICS_PATH}" \
      --image_type "${IMAGE_TYPE}" \
      --random_state "${RANDOM_STATE}" \
      --rf_feature_selector lasso \
      --top_k "${top_k}"
  done
done
