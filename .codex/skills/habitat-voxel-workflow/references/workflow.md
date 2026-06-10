# Osteosarcoma Voxel-Based Habitat Workflow

Project: `/host/d/Github/Osteosarcoma`

Main notebook: `/host/d/Github/Osteosarcoma/habitats/step1_make_habitat.ipynb`

Detailed project memory: `/host/d/Github/Osteosarcoma/docs/habitat_voxel_workflow_memory.md`

## Core Decision

Keep PyRadiomics voxel feature extraction at `[1,1,1]` to match the whole-tumor radiomics setting. Do not change the PyRadiomics voxel extraction spacing just to reduce K-means cost.

Instead:

1. Extract voxel feature maps at `[1,1,1]` using PyRadiomics.
2. Normalize foreground voxel feature vectors per case in `[1,1,1]` feature-map space.
3. Downsample normalized feature maps by block mean for K-means and silhouette coefficient selection.
4. Upsample the downsampled cluster labels back to `[1,1,1]` feature-map space with nearest neighbor.
5. Back-project `[1,1,1]` habitat labels to original image/label space using the original-label bbox route.
6. Save multi-class and binary habitat masks in original image space.
7. Extract radiomics for each binary habitat mask using the same image-level MR radiomics settings as whole-tumor radiomics.
8. Add a weighted-average row to each case's habitat radiomics table.

## Primary Paths

- Patient list:
  `/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12.xlsx`
- Original image/label:
  `/host/e/D/Data/Habitats/Jishuitan/original_data/{Patient_set}/{Patient_index}/img.nii.gz`
  `/host/e/D/Data/Habitats/Jishuitan/original_data/{Patient_set}/{Patient_index}/label.nii.gz`
- Voxel feature maps:
  `/host/d/projects/Habitats/radiomics/voxels/{Patient_set}/{Patient_index}/feature_maps/*.nii.gz`
- Habitat outputs:
  `/host/d/projects/Habitats/radiomics/habitats/{Patient_set}/{Patient_index}/`
- Whole-tumor feature min/max list:
  `/host/d/projects/Habitats/radiomics/whole_image/radiomics_features_list.xlsx`

## Step 1: Voxel-Based PyRadiomics Feature Maps

Notebook section: `step 1: voxel-based radiomics`.

Use:

```python
paramPath = '/host/d/Github/Osteosarcoma/radiomics_settings/MR_setting_voxel.yaml'
extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
```

Then execute:

```python
result = extractor.execute(img_p, msk_p, voxelBased=True)
```

Keep only PyRadiomics outputs that are `SimpleITK.Image` objects and are not diagnostics. Save each feature map as:

```text
/host/d/projects/Habitats/radiomics/voxels/{Patient_set}/{Patient_index}/feature_maps/{feature_name}.nii.gz
```

The current skip check uses the sentinel feature map:

```text
original_ngtdm_Coarseness.nii.gz
```

The voxel feature table paths are present in the notebook as:

```text
voxel_feature_table_Wang26.xlsx
voxel_feature_table_Wang26.csv
```

## Step 2: Normalize Features and Find Habitats

Notebook section: `Step2: normalize features and K-means to find out habitats`.

Current core settings:

```python
K_candidates = [3, 4, 5]
random_state = 0
downsample_block_size = (3, 3, 3)
n_init = 30
```

### Foreground Definition

Foreground in `[1,1,1]` voxel feature-map space is defined from the feature maps themselves:

```python
foreground_111 = (~np.all(feature_stack_111 == 0, axis=-1)) & (~np.all(np.isnan(feature_stack_111), axis=-1))
```

Do not require all feature values to be nonzero. The intent is: background equals all features zero; foreground equals not all features zero.

Then:

- remove voxels with NaN/Inf in any feature;
- remove constant features for that case;
- apply `StandardScaler` per case using foreground voxels only;
- write normalized values back to a 4D `[1,1,1]` feature stack, with background left as zero;
- save per-case normalization parameters as `normalization_parameters_111.xlsx`.

### Downsampling and K-Means

Use `skimage.measure.block_reduce(..., func=np.mean, cval=0)` on normalized continuous feature maps.

Do not downsample foreground separately by max for the final K-means mask. After feature downsampling, re-derive foreground directly from downsampled feature maps:

```python
foreground_down = (~np.all(feature_stack_down == 0, axis=-1)) & (~np.all(np.isnan(feature_stack_down), axis=-1))
```

Run K-means on valid downsampled foreground voxels:

```python
KMeans(n_clusters=K, random_state=random_state, n_init=n_init)
```

Select `best_K` by maximal `silhouette_score` among `[3,4,5]`.

### Step 2 Outputs

Per case habitat folder saves:

```text
silhouette_score_K3_K5.png
silhouette_score_K3_K5.xlsx
best_K_summary_downsampled.xlsx
habitats_downsampled.nii.gz
normalization_parameters_111.xlsx
habitats_space111.nii.gz
```

`best_K_summary_downsampled.xlsx` includes identifiers, `best_K`, `best_silhouette_score`, foreground voxel counts, raw/used feature counts, `downsample_block_size`, `random_state`, and `n_init`.

`habitats_downsampled.nii.gz` uses label `0` for background/invalid voxels and labels `1..best_K` for habitats.

Upsample `habitats_downsampled.nii.gz` back to the `[1,1,1]` feature-map shape with nearest neighbor:

```python
habitats_zoomed = zoom(habitats_down, zoom=zoom_factors, order=0).astype(np.uint8)
```

Crop/pad from origin to exactly match the feature-map shape, then restrict to `foreground_111` and save `habitats_space111.nii.gz`.

## Step 3: Back-Project Habitat Masks to Original Image Space

Notebook section: `step 3: put habitat mask into original image space`.

Reason: PyRadiomics voxel feature maps appear to represent a resampled/cropped bbox around the label, not a full-image resample. Direct zoom from feature-map shape to full label shape is not appropriate.

Current settings:

```python
bbox_buffer = 0
target_spacing_111 = (1.0, 1.0, 1.0)
```

Procedure:

1. Load original `label.nii.gz` and `habitats_space111.nii.gz`.
2. Find bbox for `label == 1` in original label space.
3. Crop original label to bbox and save `label_bbox_original_spacing.nii.gz` with a bbox affine.
4. Resample bbox label to `[1,1,1]` using `Data_processing.resample_nifti(order=0, mode="constant", cval=0, in_plane_resolution_mm=1, slice_thickness_mm=1)` and save `label_bbox_resampled_111.nii.gz`.
5. Align the `[1,1,1]` habitat map to the bbox-resampled `[1,1,1]` shape using the project helper:

   ```python
   habitat_aligned_111 = Data_processing.crop_or_pad(habitat_111_arr, bbox_shape_111, 0)
   ```

6. Save the aligned `[1,1,1]` habitat as `habitats_space111_aligned.nii.gz` with the bbox-resampled affine/header.
7. Resample aligned habitats back to bbox original spacing with nearest neighbor using `Data_processing.resample_nifti(order=0, mode="constant", cval=0, in_plane_resolution_mm=original_dx, slice_thickness_mm=original_dz)`.
8. Crop/pad to original bbox shape.
9. Clip outside the original bbox label.
10. Paste the bbox habitat array back into the full original image shape at the saved bbox coordinates.
11. Final clip with original `label == 1` to clean edge leakage.
12. Save the final multi-class habitat mask as `habitats_original_space_final.nii.gz`.
13. Save per-habitat binary masks as `habitat_original_space_final_{k}.nii.gz`.

Important alignment rule: do not use the older custom `crop_or_pad_to_shape_from_origin()` unless explicitly requested. Do not apply the older intermediate clipping step in `[1,1,1]` bbox-resampled space:

```python
habitat_aligned_111[bbox_label_resampled_arr == 0] = 0
```

Current visual checks show outputs look acceptable with `Data_processing.crop_or_pad` and without this intermediate `[1,1,1]` bbox-label clipping. Keep the later original-space clipping with the original manual label.

### Step 3 Outputs

Per case habitat folder saves:

```text
label_bbox_original_spacing.nii.gz
label_bbox_resampled_111.nii.gz
habitats_space111_aligned.nii.gz
habitats_bbox_original_spacing.nii.gz
habitats_original_space_final.nii.gz
habitat_original_space_final_1.nii.gz
habitat_original_space_final_2.nii.gz
...
habitat_original_space_final_{best_K}.nii.gz
```

## Step 4: Extract Habitat Radiomics Features

Notebook section: `step 4: extract habitat radiomics features`.

Use image-level MR settings, not voxel settings:

```python
paramPath = "/host/d/Github/Osteosarcoma/radiomics_settings/MR_setting_image.yaml"
extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
```

Find binary habitat masks with the current final naming convention:

```python
target_file_name = ["habitat_original_space_final_*.nii.gz"]
habitat_files = ff.find_all_target_files(target_file_name, case_habitat_folder)
habitat_files = ff.sort_timeframe(habitat_files, num_of_dots=2, start_signal="_", end_signal=".")
```

This works because the habitat ID is the final number before `.nii.gz`, for example `habitat_original_space_final_4.nii.gz`.

For each habitat mask:

- parse `k` using `ff.find_timeframe(..., num_of_dots=2, start_signal="_", end_signal=".")`;
- count pixels with `np.sum(habitat_arr > 0)`;
- compute `Habitat_pixel_fraction = Habitat_pixel_num / Total_habitat_pixel_num`;
- run `extractor.execute(image_path, habitat_file, voxelBased=False)`;
- remove diagnostics;
- save one table per case.

Per-case table:

```text
/host/d/projects/Habitats/radiomics/habitats/{Patient_set}/{Patient_index}/habitat_radiomics_measurements.xlsx
```

Column order:

```text
Patient_set
Patient_index
Image_filepath
Mask_filepath
k
Habitat_pixel_num
Total_habitat_pixel_num
Habitat_pixel_fraction
<radiomics features...>
```

## Step 4b: Weighted-Average Habitat Radiomics

Notebook section: `step 4b: get weighted averaged radiomics`.

For each case, open `habitat_radiomics_measurements.xlsx`, remove any existing `k == "avg"` row, and recalculate weighted-average features:

```python
weighted_feature = np.sum(feature_values * Habitat_pixel_fraction)
```

Append the result back into the original per-case file as a final row with:

```text
Patient_set = patient_set
Patient_index = patient_index
Image_filepath = original image path
Mask_filepath = ""
k = "avg"
Habitat_pixel_num = ""
Total_habitat_pixel_num = ""
Habitat_pixel_fraction = ""
```

Then overwrite the same original per-case file:

```text
habitat_radiomics_measurements.xlsx
```

## Aggregated Weighted-Average Table

A downstream aggregation can collect the `k == "avg"` row from every per-case `habitat_radiomics_measurements.xlsx` and save:

```text
/host/d/projects/Habitats/radiomics/habitats/habitat_radiomics_measurements_avg.xlsx
```

For this aggregated table, keep only these non-feature columns before the radiomics features:

```text
Patient_set
Patient_index
Image_filepath
Mask_filepath
```

Drop habitat-specific columns such as `k`, `Habitat_pixel_num`, `Total_habitat_pixel_num`, and `Habitat_pixel_fraction`.

## Whole-Tumor Min/Max Normalization Check

Whole-tumor radiomics feature min/max values are stored in:

```text
/host/d/projects/Habitats/radiomics/whole_image/radiomics_features_list.xlsx
```

Columns:

```text
feature_name
feature_min
feature_max
```

A full read-only check over 330 cases found that weighted-average habitat radiomics values mostly fall inside whole-tumor min/max:

```text
334,950 total feature values checked
317,425 in range
17,525 out of range
overall in range: 94.77%
median case in range: 95.57%
```

Using whole-tumor min/max to normalize weighted-average habitat features is acceptable, but apply clipping after min-max scaling because some features exceed the whole-tumor range:

```python
x_norm = (x - feature_min) / (feature_max - feature_min)
x_norm = np.clip(x_norm, 0, 1)
```

Common out-of-range features include `original_shape_SurfaceVolumeRatio`, `original_shape_Sphericity`, and several wavelet texture features.

## Interpolation Rules

- Continuous normalized feature maps downsample: average pooling via `block_reduce(..., func=np.mean)`.
- Habitat labels/masks upsample or resample: nearest neighbor (`zoom(..., order=0)` or `Data_processing.resample_nifti(order=0)`).
- Feature-map resampling experiments may use linear interpolation (`order=1`), but the agreed workflow does not resample feature maps directly to original spacing before K-means.

## Current Best-K Snapshot

Across the 330 processed set_1 + set_2 cases, a check found:

```text
best_K != 3: 48 cases
K=4: 29 cases
K=5: 19 cases
```

Best-K is read from `best_K_summary_downsampled.xlsx`. Habitat fractions are read from `habitat_radiomics_measurements.xlsx`.

## Step 2 Feature Selection for Habitat Weighted-Average Radiomics

Current habitat feature-selection notebook: `/host/d/Github/Osteosarcoma/habitats/step2_feature_selection.ipynb`.

Goal: keep habitat weighted-average radiomics aligned with the whole-tumor feature-selection pipeline.

Use whole-tumor PCC-selected features from:

```text
/host/d/projects/Habitats/radiomics/whole_image/radiomics_measurements_PCC.xlsx
```

This table has 286 columns: 4 non-feature columns plus 282 selected radiomics features. The non-feature columns are:

```text
Patient_set
Patient_index
Image_filepath
Mask_filepath
```

For habitat modeling, remove shape features because habitat shape is not stable enough for the biological question. The 7 removed shape features are:

```text
original_shape_Elongation
original_shape_LeastAxisLength
original_shape_MajorAxisLength
original_shape_Maximum2DDiameterSlice
original_shape_MeshVolume
original_shape_Sphericity
original_shape_SurfaceVolumeRatio
```

Final habitat feature count after removing shape features: 275.

Input table:

```text
/host/d/projects/Habitats/radiomics/habitats/habitat_radiomics_measurements_avg_normalized.xlsx
```

Output table:

```text
/host/d/projects/Habitats/radiomics/habitats/habitat_radiomics_measurements_avg_PCC.xlsx
```

Optional feature-list audit file:

```text
/host/d/projects/Habitats/radiomics/habitats/habitat_PCC_feature_list_without_shape.xlsx
```

## Whole-Image ML Scripts to Reuse for Habitat ML

Whole-image model scripts live in:

```text
/host/d/Github/Osteosarcoma/whole_image/
```

There are 5 model families, each with a `.py` and `.sh`:

```text
LR.py / LR.sh
SVM.py / SVM.sh
RF.py / RF.sh
KNN.py / KNN.sh
XGBoost.py / XGBoost.sh
```

There is also:

```text
summarize.py
```

Shared design:

- Label column: `Prognosis_label`.
- Non-feature columns: `Patient_set`, `Patient_index`, `Image_filepath`, `Mask_filepath`.
- Patient list: `/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12.xlsx`.
- Stratified 5-fold split by `Prognosis_label`.
- Split files are saved/reused as `/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12_5fold_prognosis_random{random_state}.xlsx`.
- Random states used by shell scripts: `0`, `30`, `60`.
- Model scripts load PCC radiomics, merge with labels/folds one-to-one, perform supervised feature selection, then run grid search and out-of-fold evaluation.
- Outputs per experiment include `best_params.json`, `grid_search_results.xlsx`, `predictions.xlsx`, `fold_metrics.xlsx`, `summary.json`, and a ROC PDF.
- `summary.json` includes fold AUCs, mean/std fold AUC, overall out-of-fold AUC, best threshold by max sensitivity+specificity, accuracy, sensitivity, specificity, precision, F1, and confusion matrix counts.
- `summarize.py` discovers model output folders and writes a model summary workbook.

Whole-image input/output constants to change when porting to habitat:

```python
PCC_RADIOMICS_PATH = "/host/d/projects/Habitats/radiomics/whole_image/radiomics_measurements_PCC.xlsx"
WHOLE_IMAGE_RADIOMICS_OUT_DIR = "/host/d/projects/Habitats/radiomics/whole_image"
WHOLE_IMAGE_MODEL_OUT_DIR = "/host/d/projects/Habitats/models/whole_image"
```

For habitat ML, these should become habitat-specific paths, especially:

```text
/host/d/projects/Habitats/radiomics/habitats/habitat_radiomics_measurements_avg_PCC.xlsx
/host/d/projects/Habitats/radiomics/habitats/
/host/d/projects/Habitats/models/habitats/
```

### Model Families and Feature Selectors

LR:

- Script: `LR.py`.
- Feature selector: LASSO via `LogisticRegressionCV(penalty="l1", solver="liblinear", scoring="roc_auc")` inside a `StandardScaler` pipeline.
- Shell combinations: `random_state in [0,30,60]`, `top_k in [None,15,20,25]`.
- If `top_k=None`, keep all non-zero LASSO features, but skip if more than `LASSO_MAX_FEATURES=35`.
- Grid search: `clf__C = [0.001,0.01,0.1,1,10,100]`, `clf__tol = [1e-4,1e-3]`.
- Selected feature tables: `radiomics_measurements_LR_random{seed}_lasso_{none/topK}_selected.xlsx`.
- Model folder: `LR/random{seed}_lasso_{none/topK}`.

SVM:

- Script: `SVM.py`.
- Linear SVM pipeline: `StandardScaler` + `SVC(kernel="linear", class_weight="balanced", probability=True)`.
- Feature selectors: `rfe`, `sfs`, `rfecv`.
- Shell combinations: for `rfe/sfs`, `random_state in [0,30,60]` and `top_k in [15,20,25]`; for `rfecv`, no top_k.
- `RFECV_MAX_FEATURES=35`; skip if RFECV selects too many features.
- Grid search: `clf__C = [0.001,0.01,0.1,1,10,100]`, `clf__tol = [1e-4,1e-3]`.
- Selected feature tables: `radiomics_measurements_SVM_random{seed}_{selector}_top{K}_selected.xlsx` or `..._rfecv_selected.xlsx`.
- Model folder: `SVM/random{seed}_{selector}_top{K}` or `SVM/random{seed}_rfecv`.

RF:

- Script: `RF.py`.
- Classifier: `RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1)`.
- Feature selectors supported: `rfe`, `sfs`, `rfecv`; current `RF.sh` runs `rfe` for top 15/20/25 and `rfecv`.
- Feature-selection RF uses `n_estimators=300`, `max_depth=5`, `max_features="sqrt"`.
- `RFECV_MAX_FEATURES=35`; skip if RFECV selects too many features.
- Grid search: `n_estimators = [100,300,500]`, `max_depth = [None,3,5]`, `max_features = ["sqrt","log2"]`.
- Selected feature tables: `radiomics_measurements_RF_random{seed}_{selector}_top{K}_selected.xlsx` or `..._rfecv_selected.xlsx`.
- Model folder: `RandomForest/random{seed}_{selector}_top{K}` or `RandomForest/random{seed}_rfecv`.

KNN:

- Script: `KNN.py`.
- Pipeline: `StandardScaler` + `KNeighborsClassifier`.
- Feature selector: `sfs` only.
- Shell combinations: `random_state in [0,30,60]`, `top_k in [15,20,25]`.
- SFS estimator defaults to `KNeighborsClassifier(n_neighbors=5, weights="uniform")` in a scaler pipeline.
- Grid search: `clf__n_neighbors = [3,5,7,9,11]`, `clf__weights = ["uniform","distance"]`.
- Selected feature tables: `radiomics_measurements_KNN_random{seed}_sfs_top{K}_selected.xlsx`.
- Model folder: `KNN/random{seed}_sfs_top{K}`.

XGBoost:

- Script: `XGBoost.py`.
- Classifier: `XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", n_jobs=1, scale_pos_weight=n_negative/n_positive)`.
- Feature selectors: `rfe`, `sfs`, `rfecv`.
- Shell combinations: for `rfe/sfs`, `random_state in [0,30,60]` and `top_k in [15,20,25]`; for `rfecv`, no top_k.
- Feature-selection XGB uses `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`.
- `RFECV_MAX_FEATURES=35`; skip if RFECV selects too many features.
- Grid search: `n_estimators = [50,100,200]`, `max_depth = [3,4,5]`, `learning_rate = [0.03,0.1]`.
- Selected feature tables: `radiomics_measurements_XGBoost_random{seed}_{selector}_top{K}_selected.xlsx` or `..._rfecv_selected.xlsx`.
- Model folder: `XGBoost/random{seed}_{selector}_top{K}` or `XGBoost/random{seed}_rfecv`.

## Habitat ML Porting Guidance

When building habitat ML scripts, preserve the whole-image evaluation design and change only dataset-specific constants/names:

1. Input radiomics table should be `habitat_radiomics_measurements_avg_PCC.xlsx`.
2. Radiomics selected-feature tables should be saved under `/host/d/projects/Habitats/radiomics/habitats/` with names beginning `habitat_radiomics_measurements_...`.
3. Model outputs should go under `/host/d/projects/Habitats/models/habitats/`.
4. Keep the same patient split files so whole-tumor and habitat models are evaluated on identical folds.
5. Keep `Prognosis_label`, `N_SPLITS=5`, random states `[0,30,60]`, top_k `[15,20,25]`, and the same model grids unless explicitly changed.
