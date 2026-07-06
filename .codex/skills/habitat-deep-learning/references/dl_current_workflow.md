# DL Current Workflow Memory - Osteosarcoma/Habitats

Date recorded: 2026-06-18

This note summarizes the current deep-learning thread so future work can resume without re-deriving the project decisions.

## Literature-Derived Context

Two local references were used for DL planning:

1. `Integration of multi-scale radiomics and deep learning for Ki-67 prediction in clear cell renal carcinoma` plus supplement.
2. `MRI-based habitat radiomics combined with vision transformer for identifying vulnerable intracranial atherosclerotic plaques and predicting stroke events a multicenter, retrospective study` plus supplement/docx.

Important takeaways:

- The plaque ViT paper uses an ROI volume, patch embedding, position embedding, transformer encoder, and MLP classifier.
- Its supplement figure reports ROI size `48 x 64 x 64`, patch size `16`, frame patch size `2`, and transformer encoder `n=6`.
- The `n=6` encoder blocks should be understood as six sequential transformer blocks.
- Their patching gives `4 x 4 x 24 = 384` tokens.
- For our first pass, we copied the spirit of this design but changed sizing to fit osteosarcoma data.

## Current Preprocessing Decision

Current practical DL preprocessing direction:

1. Resample image/label to `[2,2,2] mm`.
2. Create tumor bbox from the resampled label.
3. Crop image by bbox.
4. Apply per-case intensity normalization.
5. Crop/pad to `[128,128,160]`.
6. Feed fixed-size volume to 3D ViT.

Why `[128,128,160]`:

- Earlier bbox stats at `[2,2,2]` mm showed x/y mostly below 128, z max around 154.
- Fixed input is simpler for batching and ViT positional embeddings.
- v1 intentionally uses simple padding rather than invalid-token masking.

Current files and data paths:

- preprocessing notebook context: `/host/d/Github/Osteosarcoma/deep_learning/image_preprocessing.ipynb` or current `Image_3D` preprocessing work.
- resampled data root: `/host/e/D/Data/Habitats/Jishuitan/resampled_data`
- per-case files expected by generator:
  - `img_n4.nii.gz`
  - `bbox_mask.nii.gz`

## Generator

Current generator file:

`/host/d/Github/Osteosarcoma/Image_3D/Generator.py`

Expected output from generator:

```text
image: [1, 128, 128, 160]
label: scalar LongTensor 0/1
```

Behavior:

- Loads image and bbox mask.
- Crops image by bbox coordinates.
- Normalizes per case.
- Pads/crops to target image size.
- Optional augmentation for train generator.

Important pitfall already encountered:

- If using a small debug subset, do not mix train image paths with validation patient_set/patient_index lists.
- The bbox mask path is inferred from patient_set/patient_index, so image path and bbox case identity must match.

## ViT Model

Current model/trainer file:

`/host/d/Github/Osteosarcoma/Image_3D/vit/model.py`

Current ViT design:

- input image size: `(128,128,160)`
- patch size: `(16,16,8)`
- in_channels: `1`
- num_classes: `2`
- embed_dim: `256`
- depth: `6`
- num_heads: `8`
- mlp_ratio: `4`
- dropout: `0.1`
- attention_dropout: `0.1`

Patch/token math:

```text
128 / 16 = 8
128 / 16 = 8
160 / 8  = 20
num_tokens = 8 * 8 * 20 = 1280
```

Architecture:

1. 3D Conv patch embedding with kernel=stride patch size.
2. Learned positional embedding.
3. Six explicit sequential transformer encoder blocks.
4. Mean pooling over tokens.
5. MLP head to two logits.

v1 deliberately does not use:

- CLS token
- all-zero/invalid token mask
- handcrafted tumor-size/fraction input

## Trainer

Trainer is in `model.py` and follows the user’s Example_UNet style where useful:

- uses `Accelerator`
- uses EMA
- writes Excel training logs
- saves periodic models
- supports gradient accumulation

Classification-specific behavior:

- model output is logits `[B,2]`
- loss is `CrossEntropyLoss`
- class-1 probability is `softmax(logits, dim=1)[:,1]`
- AUC is computed with `roc_auc_score`
- train and validation AUC are computed only at validation/save intervals because full-volume prediction is expensive

Gradient accumulation rule:

```python
loss_to_backward = loss / accum_iter
accelerator.backward(loss_to_backward)
# optimizer step every accum_iter mini-batches, or at final mini-batch
```

This is the standard accumulation pattern. The user’s older scaler code was conceptually similar but should ensure division by `accum_iter` and final-batch stepping.

## Training Entrypoint

Current training entrypoint:

`/host/d/Github/Osteosarcoma/Image_3D/vit/train.py`

CLI arguments:

- `--label`, default `Prognosis`
- `--trial_name`, default `vit_3D`
- `--trained_model_path`, default `None`
- `--start_step`, default `0`
- `--random_state`, default `0`
- `--val_fold`, required
- `--train_batch_size`, default `10`
- `--train_num_steps`, default `500`
- `--save_models_every`, default `1`

Patient split path pattern:

```text
/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12_5fold_{label.lower()}_random{random_state}.xlsx
```

`val_fold` is one fold among `0..4`; train folds are the other four.

## Output Directory Convention

Locked convention after the latest change:

```text
/host/d/projects/Habitats/models/{label}/vit/{trial_name}/random{random_state}_fold{val_fold}/models/
/host/d/projects/Habitats/models/{label}/vit/{trial_name}/random{random_state}_fold{val_fold}/log/
```

`train.py` builds the setting folder and passes it as `results_folder` to `Trainer`.
`Trainer` then creates and uses `models/` and `log/` inside it.

## Batch Script

Current batch script:

`/host/d/Github/Osteosarcoma/Image_3D/vit/train.sh`

Default settings:

```bash
LABEL="Prognosis"
TRIAL_NAME="vit_3D"
TRAINED_MODEL_PATH="None"
START_STEP=0
RANDOM_STATE_LIST="0 15 30 45 60"
VAL_FOLD_LIST="0 1 2 3 4"
TRAIN_BATCH_SIZE=10
TRAIN_NUM_STEPS=500
SAVE_MODELS_EVERY=1
```

Loop order:

```bash
for RANDOM_STATE in ${RANDOM_STATE_LIST}; do
    for VAL_FOLD in ${VAL_FOLD_LIST}; do
        python3 train.py ...
    done
done
```

## N4 Bias Correction / Intensity Memory

N4 bias correction was added to the preprocessing discussion as a correction for low-frequency MRI intensity inhomogeneity.

Project decision notes:

- Doing N4 in original space was too slow.
- Practical direction is to apply N4 after resampling.
- Intensity ranges differ widely across cases, so per-case normalization is preferred over one cohort-level min/max.

## Verification Already Done

After updating paths, these checks passed:

```bash
python3 -m py_compile /host/d/Github/Osteosarcoma/Image_3D/vit/train.py /host/d/Github/Osteosarcoma/Image_3D/vit/model.py
bash -n /host/d/Github/Osteosarcoma/Image_3D/vit/train.sh
```

Full training was not run by Codex after the latest path change; the user had already started running the deep learning model.

---

# ResNet 2D/3D Update - 2026-06-23

This section records the pivot away from first-pass ViT toward ResNet-based DL baselines.

## Why ViT Was Paused

The first 3D ViT branch was not selected as the immediate next direction because:

- bbox crop sizes vary widely across osteosarcoma cases
- fixed-size 3D ViT inputs created many empty/padded patches
- prediction probabilities were often compressed into a narrow range near the positive-class prior
- train AUC and internal-test/validation AUC differed substantially, suggesting weak or non-generalizable signal
- from-scratch ViT is likely too data-hungry for the current 330-case dataset

The user decided to pause ViT and pursue ResNet baselines.

## 2D ResNet Direction

Current 2D idea:

1. From each 3D MRI/label case, take the slice with the largest tumor area.
2. Use a 2D ResNet classifier, initially ResNet50.
3. Use ImageNet transfer learning because 2D pretrained weights are straightforward in torchvision.
4. Crop each 2D slice to a uniform shape by center crop:
   - if mask is available, center on the mask bbox center
   - otherwise, center on the image center
5. Check and print if crop removes mask pixels; do not raise an error for this check.
6. Normalize MRI per case to `[0,1]`, copy to 3 channels, then apply ImageNet mean/std normalization.

Current files:

- 2D generator: `/host/d/Github/Osteosarcoma/Image_2D/Generator.py`
- 2D ResNet50 train entrypoint: `/host/d/Github/Osteosarcoma/Image_2D/resnet50/train.py`

2D generator expected item output:

```text
image: [3, H, W]
label: scalar LongTensor 0/1
```

The DataLoader therefore produces:

```text
[B, 3, H, W]
```

ImageNet preprocessing memory:

- Do not use uint8 `0..255`; use `float32`.
- Preferred preprocessing for ImageNet-pretrained ResNet:
  1. per-case MRI min-max normalize to `[0,1]`
  2. repeat grayscale MRI to 3 channels
  3. apply ImageNet mean/std:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

2D ResNet50 training memory:

- Build model with torchvision:

```python
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, 2)
```

- If `--trained_model_path None`, initialize from ImageNet weights.
- If `--trained_model_path` is provided, load the saved project checkpoint and continue training.
- Trainer mirrors existing project style:
  - `Accelerator`
  - `EMA`
  - `CrossEntropyLoss`
  - periodic checkpoint saving
  - train/validation loss
  - train/validation AUC
  - `training_log.xlsx`

2D output convention currently implemented:

```text
/host/d/projects/Habitats/models/{label}/Image_2D/{trial_name}/random{random_state}_fold{val_fold}/models/
/host/d/projects/Habitats/models/{label}/Image_2D/{trial_name}/random{random_state}_fold{val_fold}/log/
```

## 3D ResNet / MedicalNet Direction

Current 3D ResNet idea:

- Use MedicalNet pretrained 3D ResNet weights instead of training a 3D CNN from scratch.
- Do not import the whole `/host/d/Github/MedicalNet` folder during Osteosarcoma training; extract the architecture into the Osteosarcoma repo.
- First model: 3D ResNet50 with MedicalNet `resnet_50_23dataset.pth`.
- Fixed input size for first pass: `[96,96,96]`.

Current files:

- MedicalNet-derived 3D ResNet architecture and Trainer:
  `/host/d/Github/Osteosarcoma/Image_3D/resnet/model.py`
- 3D ResNet train entrypoint:
  `/host/d/Github/Osteosarcoma/Image_3D/resnet/train.py`
- MedicalNet-style generator:
  `/host/d/Github/Osteosarcoma/Image_3D/Generator_ResNet.py`

MedicalNet pretrained weights are stored locally at:

```text
/host/e/D/Data/Habitats/MedicalNet_weights/pretrain/
```

The current hard-coded pretrained path in `Image_3D/resnet/train.py` is:

```python
MEDICALNET_PRETRAIN_PATH = '/host/e/D/Data/Habitats/MedicalNet_weights/pretrain/resnet_50_23dataset.pth'
```

Available weights observed in that folder include:

```text
resnet_10.pth
resnet_10_23dataset.pth
resnet_18.pth
resnet_18_23dataset.pth
resnet_34.pth
resnet_34_23dataset.pth
resnet_50.pth
resnet_50_23dataset.pth
resnet_101.pth
resnet_152.pth
resnet_200.pth
```

3D ResNet training logic:

- If `--trained_model_path None`, build ResNet50 and load the hard-coded MedicalNet pretrained backbone.
- If `--trained_model_path` is provided, build the same architecture and load the project checkpoint for continued training.
- MedicalNet checkpoint keys have `module.` prefixes and contain backbone weights. The extracted classifier strips `module.` and loads all shape-matched keys.
- Verified loading result for `resnet_50_23dataset.pth`:

```text
matched keys: 318
skipped keys: 0
forward output shape: (1, 2)
```

MedicalNet-derived model adaptation:

- Original MedicalNet model is segmentation-oriented and ends in `conv_seg`.
- Osteosarcoma version keeps compatible backbone names:
  `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, `layer4`.
- The segmentation head is replaced by:

```text
AdaptiveAvgPool3d((1,1,1)) -> Linear(..., 2)
```

3D ResNet generator preprocessing:

- data root: `/host/e/D/Data/Habitats/Jishuitan/resampled_data`
- expected per-case files:
  - `img_n4.nii.gz`
  - `bbox_mask.nii.gz`
- pipeline:
  1. load 3D image
  2. crop by bbox mask
  3. crop/pad to `[96,96,96]` with padding value `0`
  4. MedicalNet-style z-score normalization
  5. output single-channel tensor `[1,96,96,96]`

MedicalNet normalization reference:

- Original MedicalNet normalizes non-zero voxels using mean/std.
- Background voxels equal to `0` are filled with random standard-normal noise.
- The current `Generator_ResNet.py` follows this idea after crop/pad.

3D output convention currently implemented:

```text
/host/d/projects/Habitats/models/{label}/Image_3D/{trial_name}/random{random_state}_fold{val_fold}/models/
/host/d/projects/Habitats/models/{label}/Image_3D/{trial_name}/random{random_state}_fold{val_fold}/log/
```

Example command:

```bash
python3 /host/d/Github/Osteosarcoma/Image_3D/resnet/train.py \
  --label Prognosis \
  --trial_name resnet50_3D \
  --random_state 0 \
  --val_fold 5
```

## Dependency Note for nnUNet / NumPy

During the external dataset segmentation work, nnUNet v2 was present but required a newer NumPy stack:

- `nnUNetv2_predict` exists in the environment.
- `nnunetv2` package exists; old `nnunet` v1 package does not.
- To run nnUNet v2, NumPy was temporarily upgraded to `1.26.4`.
- After nnUNet work, NumPy was restored to `1.23.5` for the main radiomics/ML/DL project environment.
- With NumPy `1.23.5`, `pip check` again reports expected conflicts for `blosc2` and `nnunetv2`; this is intentional unless actively running nnUNet.

---

# Current 2D / 2.5D / 3D ResNet Workflow - 2026-06-24

This section records the current state after the ViT branch was paused and the project moved through 2D ResNet, 2.5D ResNet, and 3D MedicalNet ResNet experiments. Use this section when editing `Image_2D`, `Image_3D/resnet`, DL generators, prediction scripts, or when designing DL-feature-extraction plus ML experiments.

## Current Data Root and Preprocessing

The current DL data branch is:

```text
/host/e/D/Data/Habitats/Jishuitan/resampled_data_new/{Patient_set}/{Patient_index}/
```

The data were regenerated from MRI/label with spacing `[1,1,3] mm`. N4 bias correction is currently not used in this branch.

For each case, expected files are:

```text
img.nii.gz
label.nii.gz
bbox_mask.nii.gz
img_slices.nii.gz
label_slices.nii.gz
bbox_mask_slices.nii.gz
```

Meaning:

- `img.nii.gz`, `label.nii.gz`, `bbox_mask.nii.gz`: full 3D resampled data and 3D bbox mask.
- `img_slices.nii.gz`, `label_slices.nii.gz`, `bbox_mask_slices.nii.gz`: three consecutive slices around the largest tumor slice. These are used for 2D/2.5D.

Current sizing judgment from the regenerated tables:

- 2D target size: `144 x 144`
- 3D target size: currently `96 x 96 x 64` in code, though `128 x 128 x 64` was also discussed as a more inclusive option.

Do not silently change these sizes. The user may tune them in scripts.

## Current Crop Logic

The current preferred crop method is not "bbox crop then pad". It is center-crop/pad around the bbox center:

1. Given original image and bbox mask, find bbox center.
2. Crop/pad a fixed target-sized window around that center.
3. Preserve the original pixels in this window. If the crop exceeds image boundaries, pad with background value `0`.
4. Apply the same crop to image, bbox mask, and tumor label so they remain spatially aligned.
5. Construct three semantic image channels:
   - channel 0: full context crop
   - channel 1: bbox-only crop, with pixels outside bbox set to `0`
   - channel 2: tumor-only crop, with pixels outside tumor label set to `0`

For 2D, this creates slice-level arrays shaped conceptually as:

```text
[semantic_channel=3, target_x, target_y, slice_index=3]
```

For 3D, this creates:

```text
[semantic_channel=3, target_x, target_y, target_z]
```

## Percentile Cutoff and Normalization

MRI intensity ranges vary strongly across cases. Current generator logic clips extreme high values before normalization:

- Use channel 1, the bbox-only channel, to find valid positive-intensity voxels/pixels.
- Compute `percentile_cutoff`, default `95`.
- Clip all semantic channels to this cutoff.
- Then apply the existing per-case normalization.

For 2D ImageNet-pretrained ResNet:

1. normalize to `[0,1]`
2. apply ImageNet mean/std per semantic channel

For 3D MedicalNet-style ResNet:

- The previous MedicalNet normalization idea was non-zero z-score with random standard-normal background replacement.
- The current generator has evolved toward semantic 3-channel inputs; check the current `Generator_ResNet.py` before changing normalization.

## Augmentation

Generators support `augment_context`:

```text
simple: flip, rotate, translate
full: random_noise -> random_brightness -> random_contrast -> random_sharpness -> flip -> rotate -> translate
```

Each augmentation step is gated independently with:

```python
np.random.uniform(0, 1) < self.augment_frequency
```

Important rule: all semantic channels and all three 2.5D slices must receive the same spatial augmentation parameters so channel/slice alignment is not broken.

For brightness/contrast/sharpness, be careful with PIL-style functions that expect `uint8`/`0..255`. The current preference is to keep augmentation clinically conservative and avoid transformations that destroy MRI intensity meaning.

## 2D ResNet

Code locations:

```text
/host/d/Github/Osteosarcoma/Image_2D/Generator.py
/host/d/Github/Osteosarcoma/Image_2D/resnet50/model.py
/host/d/Github/Osteosarcoma/Image_2D/resnet50/train.py
/host/d/Github/Osteosarcoma/Image_2D/resnet50/train.sh
/host/d/Github/Osteosarcoma/Image_2D/resnet50/predict.py
```

Despite the folder name `resnet50`, the scripts can build ResNet18/34/50 through `--model_depth`.

Current `Dataset_2D` output:

```text
slice_front:  [3, target_x, target_y]
slice_middle: [3, target_x, target_y]
slice_rear:   [3, target_x, target_y]
y: scalar label
```

`input_mode` controls how the model uses these outputs:

- `2d`: use only `slice_middle`
- `2.5d`: pass front, middle, rear slices independently through the same 2D ResNet and average logits before loss/probability computation

The 2.5D logic is average-logits, not average-probabilities.

## 3D MedicalNet ResNet

Code locations:

```text
/host/d/Github/Osteosarcoma/Image_3D/Generator_ResNet.py
/host/d/Github/Osteosarcoma/Image_3D/resnet/model.py
/host/d/Github/Osteosarcoma/Image_3D/resnet/train.py
/host/d/Github/Osteosarcoma/Image_3D/resnet/train.sh
```

MedicalNet pretrained weights live in:

```text
/host/e/D/Data/Habitats/MedicalNet_weights/pretrain/
```

The default pretrained checkpoint used in code has been:

```text
resnet_50_23dataset.pth
```

The current 3D ResNet model supports `in_channels`:

- `in_channels=1`: load MedicalNet conv1 normally
- `in_channels=3`: if loading a 1-channel MedicalNet checkpoint, copy the conv1 weights across three channels and divide/average so the scale remains reasonable

This lets the 3D model consume the semantic channels `[full, bbox-only, tumor-only]` while still using MedicalNet weights.

`--use_medicalnet_pretrained yes|no` controls whether to load MedicalNet weights when `--trained_model_path None`.

## Fine-Tuning Controls

2D and 3D ResNet trainers use the same conceptual fine-tune options:

```text
all: train all layers
fc: train classifier head only
1: train the final block of layer4 plus classifier head
2: train the final two blocks of layer4 plus classifier head
```

Do not interpret `1` as `layer1`; it means "last 1 block in layer4". This was changed after the user noted that full layer4 fine-tuning was too many parameters.

For torchvision 2D ResNet:

- ResNet18 layer4 has 2 residual blocks.
- ResNet50 layer4 has 3 bottleneck blocks.

`avgpool` has no trainable parameters; it cannot be fine-tuned.

## Optimizer and Trial Naming

Trainers support:

```text
--optimizer sgd|adam
```

Current default requested behavior:

- support both Adam and SGD
- default optimizer in code may be SGD, but train.sh can explicitly set Adam
- default learning rate was set around `1e-3`, with Adam often empirically fitting train data faster than SGD

Trial naming decision:

- automatic trial-name construction is disabled
- `--trial_name` must be set explicitly by the user
- whatever the user passes is the output folder name

This avoids hidden naming changes when toggling fine-tune stage, optimizer, input mode, or ROI/seg mode.

## Output and Prediction

Model outputs are binary logits `[B,2]`. Use:

```python
prob = softmax(logits, dim=1)[:, 1]
```

Prediction files should report probabilities, not hard labels. Metrics are computed by:

- AUC from probabilities
- best threshold by maximizing sensitivity + specificity / Youden-style criterion
- then accuracy, sensitivity, and specificity at that threshold

For 2D/2.5D prediction summary, five fold-specific models may be applied to the same prediction cohort. The user explored:

- `fold_mean`: mean probability across five fold models
- `best_fold_p`: probability from the fold model with best AUC on that prediction cohort
- `fold_mean_indicated`: an intentionally cheating diagnostic using label to choose high/low probabilities; do not use for valid reporting

## Overfitting and Current Interpretation

Across prognosis and pathologic labels, the user observed persistent overfitting in:

- 2D ResNet50
- 2D ResNet18
- 2.5D ResNet
- 3D MedicalNet ResNet
- several ROI/seg/context-channel variants
- several fine-tuning settings, including `fc` and last-block fine-tuning

This does not necessarily prove the task is impossible for DL, but it strongly suggests end-to-end DL is unstable for the current sample size, acquisition heterogeneity, and tumor-size variability.

Current interpretation:

- train AUC can rise while validation/internal-test AUC stays poor: likely memorization rather than useful representation learning
- if the end-to-end classifier is badly overfit, features from late overfit epochs are unlikely to generalize magically after ML
- if using DL as feature extractor, prefer early/pre-overfit epochs or pretrained/weakly fine-tuned features
- choose epochs using validation behavior inside the training folds, not internal test

## DL Feature Extractor + ML Direction

The current preferred next experimental direction is:

1. Train or load a DL model.
2. Extract penultimate/global-average-pooling features, not final class probabilities.
3. Build a feature table with identifiers plus DL features.
4. Optionally standardize and apply PCA.
5. Feed the resulting DL feature table into the existing ML pipeline, like radiomics.

Feature vector size reminder:

- 2D torchvision ResNet18/34 avgpool feature: 512
- 2D torchvision ResNet50 avgpool feature: 2048
- 3D MedicalNet ResNet18-style avgpool feature: usually 512
- 3D MedicalNet ResNet50-style avgpool feature: usually 2048
- final avgpool vector size depends on architecture width/depth, not input image depth, because adaptive average pooling collapses spatial dimensions

PCA convention, as currently accepted by the user for consistency with the existing radiomics ML workflow:

- Because radiomics feature selection is currently done on all cases including internal test, PCA for DL features can also be fit on all cases for this exploratory pipeline.
- This gives all cases the same columns `PC1..PCN` and fits the user's existing fixed-feature-table workflow.
- If a stricter leakage-free experiment is later needed, fit PCA only on train data inside each evaluation split and transform held-out data.

## Important Caution for Future Edits

When changing generators, always preserve spatial alignment among:

```text
image crop
bbox crop
label crop
semantic channels
2.5D front/middle/rear slices
```

Most subtle DL bugs in this branch come from mismatched case identity or mismatched spatial crop/augmentation between image and mask-derived channels.

## Current DL-to-ML Decision - 2026-06-27

The end-to-end DL branch has been deprioritized after repeated overfitting/generalization failures across ViT, 2D/2.5D/3D ResNet, MedicalNet, EfficientNet, and shallow CNN experiments. The project now proceeds with DL as a feature extractor followed by the existing ML framework.

Current user-approved working decision:

1. Use the 2.5D DL model branch.
2. Do not enforce strict cross-validated feature extraction with fold-specific models for now.
3. Train/use one model on all 330 cases.
4. Use epoch 35 of that all-data model to extract features from all 330 cases.
5. Treat the extracted DL features like radiomics/habitat features and send them through the existing ML pipeline.

Important caveat: this route is not leakage-free in the formal CV sense because the feature extractor was trained on all 330 cases. This is deliberate and accepted for the current exploratory experiment.

## July 2026 DL-to-ML Status And Redo Plan

By 2026-07-01, DL is still not being treated as a reliable end-to-end classifier because repeated 2D, 2.5D, and 3D models overfit. The practical DL use is feature extraction followed by the same ML framework used for radiomics.

Current DL-feature ML branches:

```text
dl_2d_ml
dl_2d_ml_cv
dl_3d_ml
```

Interpretation:

- `dl_2d_ml`: exploratory/optimistic branch. The DL feature extractor was trained on all 330 cases and then used to extract features for all cases. This is leakage-prone and should be described as a performance-seeking or upper-bound style experiment, not a clean validation design.
- `dl_2d_ml_cv`: more scientifically defensible branch. Feature extraction follows the fold-specific/CV logic so validation cases are not represented by a DL extractor trained on themselves. The user expects worse performance because the DL models overfit.
- `dl_3d_ml`: 3D DL-feature branch sent into the same downstream ML scripts.

Downstream ML outputs follow the same structure as radiomics:

```text
/host/d/projects/Habitats/models/Prognosis/dl_2d_ml/
/host/d/projects/Habitats/models/Prognosis/dl_3d_ml/
```

Final-selection notebooks now exist for both DL branches:

```text
/host/d/Github/Osteosarcoma/dl_2d_ml/final_selection.ipynb
/host/d/Github/Osteosarcoma/dl_3d_ml/final_selection.ipynb
```

They use the same manual probability-level fusion pattern as whole-image and habitat models:

- validation/CV: selected experiments -> `prob_mean` and best fold-wise `prob_mix`;
- only the best mix is saved;
- mean and mix ROC plots are saved separately;
- internal test: selected experiments -> mean of each experiment's `prob_final` only.

Upcoming redo with new data:

- New 20+ cases will be added.
- Old-case radiomics feature extraction does not need to be repeated, but new cases need basic information and radiomics extraction.
- The full ML/DL-to-ML evaluation will be redone after a new three-way split is defined: train/CV, internal_test, external_test.
- The external_test set will include a portion of the previous train data plus the new cases; wait for exact user instructions before coding.
- Any future DL-to-ML scripts should support explicit external-test prediction/metrics rather than assuming only fold 5 internal test.

