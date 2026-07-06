---
name: habitat-deep-learning
description: Use when designing, explaining, or implementing deep learning components for the Osteosarcoma/Habitats project, especially 2D/3D ResNet feature extraction, Med3D/ImageNet pretrained models, ROI/bounding-box inputs, Vision Transformer plaque-style models, patch design, augmentation, and DL-to-ML feature extraction, final-selection probability fusion, and fusion with radiomics/habitat models.
---

# Habitat Deep Learning

Use this skill for DL-related work in the Osteosarcoma/Habitats project. Read `references/dl_paper_notes.md` when the user asks about article methods, model inputs, pretrained weights, data augmentation, 2D/3D ResNet, ViT, patching, or how to adapt these ideas to the project. Read `references/dl_current_workflow.md` when editing the current Image_3D/ViT code, preprocessing, generators, training scripts, or output folders.

Related skills:

- `habitat-papers` for local paper paths and literature retrieval.
- `habitat-ensemble-fusion` for probability-level voting/stacking fusion.
- `habitat-voxel-workflow` for voxel habitat generation and radiomics pipeline.

## Current Project Sizing Memory - 2026-06-18

For the current osteosarcoma DL preprocessing branch:

- preprocessing notebook: `/host/d/Github/Osteosarcoma/deep_learning/image_preprocessing.ipynb`
- current resampled spacing for DL prep: `[2, 2, 2] mm`
- outputs live under:
  `/host/e/D/Data/Habitats/Jishuitan/resampled_data/{Patient_set}/{Patient_index}/`
- summary table:
  `/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12_resampled_bbox_2x2x2.xlsx`

Buffered tumor bbox statistics at `[2,2,2]` mm (after resample and bbox extraction) were:

- `bbox_x_shape`: min 26, median 46, 95% 63, max 84
- `bbox_y_shape`: min 24, median 43, 95% 62, max 118
- `bbox_z_shape`: min 16, median 54, 95% 97.1, max 154

Physical bbox sizes (mm):

- `bbox_x_mm`: min 52, median 92, 95% 126, max 168
- `bbox_y_mm`: min 48, median 86, 95% 124, max 236
- `bbox_z_mm`: min 32, median 108, 95% 194.2, max 308

Agreed current first-pass ViT direction:

- use a fixed input size
- simple zero padding is acceptable for v1
- do not add extra handcrafted size/fraction features
- do not use CLS token in the first version
- do not specially mask/drop invalid all-zero tokens in v1; let zero patches flow through
- likely use simple token aggregation such as mean pooling rather than CLS

The sizing discussion has now been provisionally locked to the following first-pass ViT input design:

- resample spacing for DL preprocessing: `[2,2,2] mm`
- per-case input is the tumor bounding box from the resampled label, with the existing preprocessing buffer logic
- final network input shape: `128 x 128 x 160`
- first-pass patch size: `16 x 16 x 8`
- resulting token grid: `8 x 8 x 20 = 1280 tokens`

This is currently a practical project decision rather than a literature-matched requirement. The intent is:

- keep all cases in one fixed padded shape
- preserve enough z-direction detail for osteosarcoma volumes
- use a simple patching rule that evenly divides the input shape
- keep the first version straightforward before adding masking or more advanced token handling

So the current provisional pipeline is:

1. resample original image/label to `[2,2,2] mm`
2. compute tumor bbox from the resampled label
3. crop the image to the bbox region
4. pad/crop to uniform `128 x 128 x 160`
5. patchify with `16 x 16 x 8`
6. feed tokens into the first-pass ViT without CLS-token tricks or special invalid-token masking

## Current ViT Implementation Memory - 2026-06-18

The current runnable DL branch is under:

- shared/generator code: `/host/d/Github/Osteosarcoma/Image_3D/Generator.py`
- ViT model/trainer: `/host/d/Github/Osteosarcoma/Image_3D/vit/model.py`
- ViT training entrypoint: `/host/d/Github/Osteosarcoma/Image_3D/vit/train.py`
- ViT batch script: `/host/d/Github/Osteosarcoma/Image_3D/vit/train.sh`

Current first-pass model is a 3D ViT trained from scratch for binary classification:

- input tensor from generator: `[B, 1, 128, 128, 160]`
- patch size: `[16, 16, 8]`
- token grid: `8 x 8 x 20 = 1280`
- no CLS token in v1
- no invalid zero-token masking in v1
- aggregation: mean pooling over patch tokens
- transformer depth: `6` sequential encoder blocks, following the plaque ViT paper/supplement figure with `n=6`
- output logits: `[B, 2]`; use softmax column 1 as positive-class probability for AUC

Training and split conventions:

- `train.py` is CLI-based and accepts `--label`, `--trial_name`, `--trained_model_path`, `--start_step`, `--random_state`, `--val_fold`, `--train_batch_size`, `--train_num_steps`, and `--save_models_every`.
- `--val_fold` is required. Train folds are the other four folds among `0..4`.
- patient split path pattern:
  `/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12_5fold_{label.lower()}_random{random_state}.xlsx`
- default batch script loops random states `0 15 30 45 60` and validation folds `0 1 2 3 4`.

Output folder convention locked on 2026-06-18:

```text
/host/d/projects/Habitats/models/{label}/vit/{trial_name}/random{random_state}_fold{val_fold}/models/
/host/d/projects/Habitats/models/{label}/vit/{trial_name}/random{random_state}_fold{val_fold}/log/
```

The Trainer should receive `results_folder` as the setting folder:
`.../{trial_name}/randomX_foldY`.
It then saves weights under `models/` and `training_log.xlsx` under `log/`.

Trainer behavior:

- `CrossEntropyLoss` is used for `[B, 2]` logits and integer 0/1 labels.
- Gradient accumulation should backpropagate every mini-batch using `loss / accum_iter`, and step the optimizer every `accum_iter` mini-batches or at the final mini-batch of an epoch.
- Validation happens every `save_models_every`/`validation_every` step.
- At validation steps, run deterministic prediction over both train and val generators, compute loss and AUC, and append to `training_log.xlsx`.
- Temporarily disable train-generator augmentation during full train-set AUC evaluation, then restore it.

Generator/preprocessing notes:

- DL data live under `/host/e/D/Data/Habitats/Jishuitan/resampled_data/{Patient_set}/{Patient_index}/`.
- Current image input file is `img_n4.nii.gz` and bbox mask is `bbox_mask.nii.gz`.
- The generator crops the image using the bbox mask, normalizes each case individually, then crops/pads to `[128,128,160]`.
- A previous bug came from pairing train image paths with validation patient_set/patient_index lists; bbox masks must match the same case as the image path.
- If creating debug subsets, slice `x_file_list`, labels, `patient_set_list`, and `patient_index_list` consistently.

N4/resampling decision memory:

- N4 bias correction was discussed as an MRI intensity nonuniformity correction.
- Original-space N4 was too slow for this dataset, so practical current direction is N4 after resampling.
- Per-case intensity normalization is preferred because intensity ranges vary substantially across cases.

See `references/dl_current_workflow.md` for a fuller chronological summary and code-level details.

## Current ResNet / DL Classifier Memory - 2026-06-24

The active DL branch now includes 2D, 2.5D, and 3D ResNet-style classifiers. The latest design is documented in `references/dl_current_workflow.md` under "Current 2D / 2.5D / 3D ResNet Workflow".

Core current points:

- data root: `/host/e/D/Data/Habitats/Jishuitan/resampled_data_new`
- 2D uses `img_slices.nii.gz`, `label_slices.nii.gz`, and `bbox_mask_slices.nii.gz`; target size currently `144 x 144`
- 3D uses `img.nii.gz`, `label.nii.gz`, and `bbox_mask.nii.gz`; target size currently `96 x 96 x 64` in code unless the user changes it
- generator now creates semantic channels: full context, bbox-only, and tumor-only
- 2.5D feeds front/middle/rear slices through the same 2D ResNet and averages logits
- 3D MedicalNet ResNet can use 1 or 3 input channels; if loading a 1-channel MedicalNet checkpoint into a 3-channel model, the first conv weights are copied/averaged across channels
- `trial_name` is now explicit: no automatic trial-name construction
- 2D/3D trainers support `--optimizer sgd|adam`, `--fine_tune_stage all|fc|1|2`, and periodic train/val AUC logging
- repeated experiments showed strong overfitting for end-to-end DL; current recommended next direction is early-epoch/pretrained DL feature extraction followed by the existing ML pipeline, with optional PCA fitted consistently according to the user's current feature-selection convention

Read `references/dl_current_workflow.md` before editing `Image_2D`, `Image_3D/resnet`, generators, train scripts, prediction scripts, or DL feature-extraction code.

## Current DL-to-ML Decision - 2026-06-27

After extensive end-to-end DL experiments with ViT, 2D/2.5D/3D ResNet, MedicalNet, EfficientNet, and shallow CNN showing poor validation generalization/overfitting, the current practical decision is to pause end-to-end DL classifier optimization and use a DL-feature-then-ML route.

Current accepted shortcut for the 2.5D branch:

- Use a single 2.5D DL model trained on all 330 cases rather than strict fold-specific out-of-fold feature extraction.
- Use epoch 35 of that all-data 2.5D model as the feature extractor.
- Extract DL features for all 330 cases using this model.
- Feed the resulting DL feature table into the same ML framework used for radiomics/habitat features.
- This is intentionally not the strict leakage-free CV feature-extraction design; it is the user-approved current experimental route.


For the current DL-to-ML status, leakage caveats, and July 2026 redo plan, read `references/dl_current_workflow.md`.
