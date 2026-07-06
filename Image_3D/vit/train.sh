#!/bin/bash

LABEL="Prognosis"
TRIAL_NAME="vit_3D"
TRAINED_MODEL_PATH="/host/d/projects/Habitats/models/Prognosis/vit/vit_3D/random0_fold5/models/model-55.pt"
START_STEP=55

RANDOM_STATE_LIST="0"
VAL_FOLD_LIST="5"

TRAIN_BATCH_SIZE=30
TRAIN_NUM_STEPS=300
SAVE_MODELS_EVERY=5

for RANDOM_STATE in ${RANDOM_STATE_LIST}; do
    for VAL_FOLD in ${VAL_FOLD_LIST}; do
        echo "============================================================"
        echo "3D ViT | label=${LABEL} | trial=${TRIAL_NAME} | random_state=${RANDOM_STATE} | val_fold=${VAL_FOLD}"

        python3 train.py \
            --label "${LABEL}" \
            --trial_name "${TRIAL_NAME}" \
            --trained_model_path "${TRAINED_MODEL_PATH}" \
            --start_step "${START_STEP}" \
            --random_state "${RANDOM_STATE}" \
            --val_fold "${VAL_FOLD}" \
            --train_batch_size "${TRAIN_BATCH_SIZE}" \
            --train_num_steps "${TRAIN_NUM_STEPS}" \
            --save_models_every "${SAVE_MODELS_EVERY}"
    done
done
