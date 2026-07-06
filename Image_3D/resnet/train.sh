#!/bin/bash

LABEL="Prognosis"
MODEL_DEPTH=18
FINE_TUNE_STAGE="all"   # choices: all, fc, 1, 2
ONLY_TUMOR_PIXELS="seg"  # kept for compatibility; generator returns [full,bbox,tumor]
AUGMENT_CONTEXT="full"  # choices: simple, full
IN_CHANNELS=3
USE_MEDICALNET_PRETRAINED="no"  # choices: yes, no
TRIAL_NAME="resnet18_3D_FTall_AUGfull_96x96x64_nomed_adam"     # Exact output folder name; no automatic naming
TRAINED_MODEL_PATH="None"
START_STEP=0
OPTIMIZER="adam"  # choices: sgd, adam

RANDOM_STATE_LIST="0"
SPLIT_MODE="cv"  # choices: cv, all
FOLD_LIST="01234"  # examples: 01234 or 012345
VAL_FOLD_LIST=("4" "3" "2" "1" "0")  # list of validation folds to run; must be in FOLD_LIST

TRAIN_BATCH_SIZE=20
TRAIN_NUM_STEPS=100
SAVE_MODELS_EVERY=5

for RANDOM_STATE in ${RANDOM_STATE_LIST}; do
    for VAL_FOLD in "${VAL_FOLD_LIST[@]}"; do
        echo "============================================================"
        echo "3D ResNet${MODEL_DEPTH} | label=${LABEL} | split_mode=${SPLIT_MODE} | fold_list=${FOLD_LIST} | fine_tune_stage=${FINE_TUNE_STAGE} | only_tumor_pixels=${ONLY_TUMOR_PIXELS} | optimizer=${OPTIMIZER} | in_channels=${IN_CHANNELS} | pretrained=${USE_MEDICALNET_PRETRAINED} | random_state=${RANDOM_STATE} | val_fold=${VAL_FOLD}"

        python3 train.py \
            --label "${LABEL}" \
            --trial_name "${TRIAL_NAME}" \
            --model_depth "${MODEL_DEPTH}" \
            --fine_tune_stage "${FINE_TUNE_STAGE}" \
            --only_tumor_pixels "${ONLY_TUMOR_PIXELS}" \
            --augment_context "${AUGMENT_CONTEXT}" \
            --in_channels "${IN_CHANNELS}" \
            --use_medicalnet_pretrained "${USE_MEDICALNET_PRETRAINED}" \
            --trained_model_path "${TRAINED_MODEL_PATH}" \
            --start_step "${START_STEP}" \
            --optimizer "${OPTIMIZER}" \
            --random_state "${RANDOM_STATE}" \
            --split_mode "${SPLIT_MODE}" \
            --fold_list "${FOLD_LIST}" \
            --val_fold "${VAL_FOLD}" \
            --train_batch_size "${TRAIN_BATCH_SIZE}" \
            --train_num_steps "${TRAIN_NUM_STEPS}" \
            --save_models_every "${SAVE_MODELS_EVERY}"
    done
done
