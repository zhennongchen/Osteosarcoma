#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Edit these variables for a run.
TASK="${TASK:-Prognosis}"
METHOD_LIST="${METHOD_LIST:-SVM LR RF XGBoost}"
RANDOM_STATE_LIST="${RANDOM_STATE_LIST:-0 10 20 30 40}"
GRIDSEARCH_RANGE="${GRIDSEARCH_RANGE:-all}"  # choices: train, all
TOP_K_LIST="${TOP_K_LIST:-7 10 12 15 17 20 22 25 27 30}"
HABITAT_MODE="${HABITAT_MODE:-sum}" # choices: individual, avg, sum

if [[ "${HABITAT_MODE}" == "individual" ]]; then
  PCC_RADIOMICS_PATH="${PCC_RADIOMICS_PATH:-/host/d/projects/Habitats/radiomics/habitats_individual/habitat_radiomics_measurements_avg_PCC.xlsx}"
  IMAGE_TYPE="${IMAGE_TYPE:-habitats_individual}"
elif [[ "${HABITAT_MODE}" == "avg" ]]; then
  PCC_RADIOMICS_PATH="${PCC_RADIOMICS_PATH:-/host/d/projects/Habitats/radiomics/habitats/habitat_radiomics_measurements_avg_PCC.xlsx}"
  IMAGE_TYPE="${IMAGE_TYPE:-habitats_avg}"
elif [[ "${HABITAT_MODE}" == "sum" ]]; then
  PCC_RADIOMICS_PATH="${PCC_RADIOMICS_PATH:-/host/d/projects/Habitats/radiomics/habitats/habitat_radiomics_measurements_sum_PCC.xlsx}"
  IMAGE_TYPE="${IMAGE_TYPE:-habitats_sum}"
else
  echo "Unsupported HABITAT_MODE: ${HABITAT_MODE}. Use individual, avg, or sum." >&2
  exit 1
fi

# LASSO can use numeric top_k values or None. None keeps all non-zero LASSO features.
LASSO_TOP_K_LIST="${LASSO_TOP_K_LIST:-None ${TOP_K_LIST}}"
LR_TOP_K_LIST="${LR_TOP_K_LIST:-${LASSO_TOP_K_LIST}}"

export TASK GRIDSEARCH_RANGE RANDOM_STATE_LIST TOP_K_LIST LASSO_TOP_K_LIST LR_TOP_K_LIST HABITAT_MODE PCC_RADIOMICS_PATH IMAGE_TYPE

echo "========== Main settings =========="
echo "TASK=${TASK}"
echo "METHOD_LIST=${METHOD_LIST}"
echo "RANDOM_STATE_LIST=${RANDOM_STATE_LIST}"
echo "GRIDSEARCH_RANGE=${GRIDSEARCH_RANGE}"
echo "SPLIT_DESIGN=set123 train folds 0-4, internal fold 5, external fold 6"
echo "TOP_K_LIST=${TOP_K_LIST}"
echo "LASSO_TOP_K_LIST=${LASSO_TOP_K_LIST}"
echo "HABITAT_MODE=${HABITAT_MODE}"
echo "PCC_RADIOMICS_PATH=${PCC_RADIOMICS_PATH}"
echo "IMAGE_TYPE=${IMAGE_TYPE}"

for method in ${METHOD_LIST}; do
  method_script="${method}.sh"
  if [[ ! -f "${SCRIPT_DIR}/${method_script}" ]]; then
    echo "Missing method script: ${SCRIPT_DIR}/${method_script}" >&2
    exit 1
  fi
  echo "========== Running ${method_script} | task=${TASK} | habitat_mode=${HABITAT_MODE} | set123 5-fold CV + internal/external =========="
  bash "${SCRIPT_DIR}/${method_script}"
done

echo "========== Summarizing selected methods | task=${TASK} | image_type=${IMAGE_TYPE} =========="
python3 "${SCRIPT_DIR}/summarize.py" --task "${TASK}" --image_type "${IMAGE_TYPE}"
