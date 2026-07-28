#!/bin/bash

LABEL="Prognosis"
TRIAL_NAME="resnet18_3D_FTall_AUGfull_96x96x64_nomed_adam"
MODEL_DEPTH=18
FINE_TUNE_STAGE="all"
ONLY_TUMOR_PIXELS="seg"
AUGMENT_CONTEXT="full"
IN_CHANNELS=3
RANDOM_STATE=0
SPLIT_MODE="all"
VAL_FOLD="56"
PRED_FOLDS="0123456"
EPOCH=12
BATCH_SIZE=4
NUM_WORKERS=0
DEVICE="auto"

echo "============================================================"
echo "3D ResNet prediction"
echo "label=${LABEL}"
echo "trial_name=${TRIAL_NAME}"
echo "random_state=${RANDOM_STATE}"
echo "split_mode=${SPLIT_MODE}"
echo "model val_fold=${VAL_FOLD}"
echo "prediction folds=${PRED_FOLDS}"
echo "epoch=${EPOCH}"
echo "============================================================"

python3 predict.py \
    --label "${LABEL}" \
    --trial_name "${TRIAL_NAME}" \
    --model_depth "${MODEL_DEPTH}" \
    --fine_tune_stage "${FINE_TUNE_STAGE}" \
    --only_tumor_pixels "${ONLY_TUMOR_PIXELS}" \
    --augment_context "${AUGMENT_CONTEXT}" \
    --in_channels "${IN_CHANNELS}" \
    --random_state "${RANDOM_STATE}" \
    --split_mode "${SPLIT_MODE}" \
    --val_fold "${VAL_FOLD}" \
    --pred_folds "${PRED_FOLDS}" \
    --epoch "${EPOCH}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}"
