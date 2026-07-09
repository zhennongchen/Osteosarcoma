#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Edit these variables for a run.
TASK="${TASK:-Prognosis}"
TRIAL_NAME="${TRIAL_NAME:-dl_2d_ml_cv}"
METHOD_LIST="${METHOD_LIST:-SVM LR RF KNN XGBoost}"
RANDOM_STATE_LIST="${RANDOM_STATE_LIST:-0 10 20 30 40}"
GRIDSEARCH_RANGE="${GRIDSEARCH_RANGE:-all}"  # choices: train, all
TOP_K_LIST="${TOP_K_LIST:-7 10 12 15 17 20 22 25 27 30}"

# LASSO can use numeric top_k values or None. None keeps all non-zero LASSO features.
LASSO_TOP_K_LIST="${LASSO_TOP_K_LIST:-None ${TOP_K_LIST}}"
LR_TOP_K_LIST="${LR_TOP_K_LIST:-${LASSO_TOP_K_LIST}}"

export TASK TRIAL_NAME GRIDSEARCH_RANGE RANDOM_STATE_LIST TOP_K_LIST LASSO_TOP_K_LIST LR_TOP_K_LIST

echo "========== Main settings =========="
echo "TASK=${TASK}"
echo "TRIAL_NAME=${TRIAL_NAME}"
echo "METHOD_LIST=${METHOD_LIST}"
echo "RANDOM_STATE_LIST=${RANDOM_STATE_LIST}"
echo "GRIDSEARCH_RANGE=${GRIDSEARCH_RANGE}"
echo "SPLIT_DESIGN=set123 train folds 0-4, internal fold 5, external fold 6"
echo "TOP_K_LIST=${TOP_K_LIST}"
echo "LASSO_TOP_K_LIST=${LASSO_TOP_K_LIST}"

for method in ${METHOD_LIST}; do
  method_script="${method}.sh"
  if [[ ! -f "${SCRIPT_DIR}/${method_script}" ]]; then
    echo "Missing method script: ${SCRIPT_DIR}/${method_script}" >&2
    exit 1
  fi
  echo "========== Running ${method_script} | task=${TASK} | set123 5-fold CV + internal/external =========="
  bash "${SCRIPT_DIR}/${method_script}"
done

echo "========== Summarizing selected methods | task=${TASK} =========="
python3 "${SCRIPT_DIR}/summarize.py" --task "${TASK}" --trial_name "${TRIAL_NAME}"
