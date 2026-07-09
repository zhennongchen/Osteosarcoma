#!/bin/bash

LABEL="Prognosis"
MODEL_FAMILY="resnet"  # choices: resnet, efficientnet, shallow_cnn
MODEL_DEPTH=18
EFFICIENTNET_VERSION="b0"  # choices: b0, b1, b2, b3, v2_s
INIT_CHANNEL=8  # shallow_cnn version; first conv channel count
FINE_TUNE_STAGE="all"   # choices: all, fc, 1, 2
ONLY_TUMOR_PIXELS="roi"  # kept for compatibility; generator returns [full,bbox,tumor]
AUGMENT_CONTEXT="full"  # choices: simple, full
TRIAL_NAME="resnet18_2.5D_FTall_AUGfull_sgd"     # Exact output folder name; no automatic naming
TRAINED_MODEL_PATH="None" #/host/d/projects/Habitats/models/Prognosis/resnet18_2.5D_FT1_AUGfull/random0_fold5/models/model-25.pt"
START_STEP=0
OPTIMIZER="sgd"  # choices: sgd, adam
INPUT_MODE="2.5d"  # choices: 2d, 2.5d

RANDOM_STATE_LIST="0" 
SPLIT_MODE="cv"  # choices: cv, all, all_data
FOLD_LIST="01234"  # examples: 01234 or 012345
VAL_FOLD_LIST=("4" "3" "2" "1" "0")  # for cross-validation, the fold to use as validation; must match FOLD_LIST

TRAIN_BATCH_SIZE=20
TRAIN_NUM_STEPS=100
SAVE_MODELS_EVERY=5

for RANDOM_STATE in ${RANDOM_STATE_LIST}; do
    for VAL_FOLD in "${VAL_FOLD_LIST[@]}"; do
        echo "============================================================"
        echo "2D CNN | family=${MODEL_FAMILY} | resnet_depth=${MODEL_DEPTH} | efficientnet=${EFFICIENTNET_VERSION} | init_channel=${INIT_CHANNEL} | label=${LABEL} | split_mode=${SPLIT_MODE} | fold_list=${FOLD_LIST} | fine_tune_stage=${FINE_TUNE_STAGE} | only_tumor_pixels=${ONLY_TUMOR_PIXELS} | augment_context=${AUGMENT_CONTEXT} | optimizer=${OPTIMIZER} | input_mode=${INPUT_MODE} | random_state=${RANDOM_STATE} | val_fold=${VAL_FOLD}"
    
        python3 train.py \
            --label "${LABEL}" \
            --trial_name "${TRIAL_NAME}" \
            --model_family "${MODEL_FAMILY}" \
            --model_depth "${MODEL_DEPTH}" \
            --efficientnet_version "${EFFICIENTNET_VERSION}" \
            --init_channel "${INIT_CHANNEL}" \
            --fine_tune_stage "${FINE_TUNE_STAGE}" \
            --only_tumor_pixels "${ONLY_TUMOR_PIXELS}" \
            --augment_context "${AUGMENT_CONTEXT}" \
            --trained_model_path "${TRAINED_MODEL_PATH}" \
            --start_step "${START_STEP}" \
            --optimizer "${OPTIMIZER}" \
            --input_mode "${INPUT_MODE}" \
            --random_state "${RANDOM_STATE}" \
            --split_mode "${SPLIT_MODE}" \
            --fold_list "${FOLD_LIST}" \
            --val_fold "${VAL_FOLD}" \
            --train_batch_size "${TRAIN_BATCH_SIZE}" \
            --train_num_steps "${TRAIN_NUM_STEPS}" \
            --save_models_every "${SAVE_MODELS_EVERY}"
    done
done