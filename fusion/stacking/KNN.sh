#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-Prognosis}"
RANDOM_STATE_LIST="${RANDOM_STATE_LIST:-0 10 20 30 40}"
GRIDSEARCH_RANGE="${GRIDSEARCH_RANGE:-all}"

for RANDOM_STATE in ${RANDOM_STATE_LIST}; do
    echo "========== Stacking KNN | task=${TASK} | random_state=${RANDOM_STATE} | gridsearch_range=${GRIDSEARCH_RANGE} =========="
    python3 KNN.py \
        --task "${TASK}" \
        --random_state "${RANDOM_STATE}" \
        --gridsearch_range "${GRIDSEARCH_RANGE}"
done
