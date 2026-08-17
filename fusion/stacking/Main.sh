#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-Prognosis}"
METHOD_LIST="${METHOD_LIST:-SVM LR RF XGBoost}"
RANDOM_STATE_LIST="${RANDOM_STATE_LIST:-0 10 20 30 40}"
GRIDSEARCH_RANGE="${GRIDSEARCH_RANGE:-all}"

export TASK
export RANDOM_STATE_LIST
export GRIDSEARCH_RANGE

echo "========== Stacking Main settings =========="
echo "TASK=${TASK}"
echo "METHOD_LIST=${METHOD_LIST}"
echo "RANDOM_STATE_LIST=${RANDOM_STATE_LIST}"
echo "GRIDSEARCH_RANGE=${GRIDSEARCH_RANGE}"

for METHOD in ${METHOD_LIST}; do
    echo "========== Running ${METHOD}.sh =========="
    bash "${METHOD}.sh"
done

echo "========== Running summarize.py =========="
python3 summarize.py --task "${TASK}"
