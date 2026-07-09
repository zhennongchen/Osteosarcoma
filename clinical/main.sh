#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${TASK:-Prognosis}"
METHOD_LIST="${METHOD_LIST:-SVM RF LR}"
RANDOM_STATE_LIST="${RANDOM_STATE_LIST:-0 10 20 30 40}"
GRIDSEARCH_RANGE="${GRIDSEARCH_RANGE:-all}"  # choices: train, all
SELECTOR_LIST="${SELECTOR_LIST:-none lasso rfecv}"

export TASK GRIDSEARCH_RANGE RANDOM_STATE_LIST SELECTOR_LIST

echo "========== Clinical main settings =========="
echo "TASK=${TASK}"
echo "METHOD_LIST=${METHOD_LIST}"
echo "RANDOM_STATE_LIST=${RANDOM_STATE_LIST}"
echo "GRIDSEARCH_RANGE=${GRIDSEARCH_RANGE}"
echo "SPLIT_DESIGN=set123 train folds 0-4, internal fold 5, external fold 6"
echo "SELECTOR_LIST=${SELECTOR_LIST}"

for METHOD in ${METHOD_LIST}; do
  echo "========== Running ${METHOD}.sh | task=${TASK} | set123 5-fold CV + internal/external =========="
  "${SCRIPT_DIR}/${METHOD}.sh"
done

echo "========== Summarizing clinical models | task=${TASK} =========="
python3 "${SCRIPT_DIR}/summarize.py" --task "${TASK}"
