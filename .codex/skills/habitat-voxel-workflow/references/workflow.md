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

## Step 2 Alternative: Paper-Style Cohort-Level K Selection - 2026-06-15

This is the current new direction for choosing habitat number, motivated by the paper `MRI-based habitat radiomics combined with vision transformer for identifying vulnerable intracranial atherosclerotic plaques and predicting stroke events` and its Supplementary Appendix 4 / Figure S1.

The previous individual-K workflow remains documented above and its completed experiments are considered the `habitats_individual` branch of results. The new paper-style approach separates K selection from habitat generation:

1. Normalize each case in `[1,1,1]` voxel feature-map foreground using per-case z-score normalization.
2. Downsample normalized continuous feature maps with `block_reduce(..., func=np.mean)` and `downsample_block_size=(3,3,3)`.
3. For K selection only, test `K=2..9` for each selected case.
4. For every case/K, compute both:
   - silhouette coefficient (`silhouette_score`)
   - Calinski-Harabasz index (`calinski_harabasz_score`, CH index)
5. Aggregate case-level scores by K and plot cohort-level mean curves, analogous to the paper's Supplementary Figure S1.
6. The user manually selects the final K from these SC/CH elbow curves.
7. Apply the selected fixed K to every case, one by one, generating the same downstream files expected by Step 3.

Current notebook implementation:

```text
/host/d/Github/Osteosarcoma/habitats/step1_make_habitat.ipynb
```

Current cell names:

```text
Step2_alternative: cohort-level K selection curves using silhouette coefficient and CH index
Step2_fixedK_apply_to_all_cases: apply manually selected K to every case
```

### Step2_alternative Current Behavior

The K-selection cell defaults to the first 100 cases because exact SC is slow:

```python
K_candidates_alt = list(range(2, 10))
max_cases_for_k_selection = 100
silhouette_sample_size = None
```

`silhouette_sample_size = None` means exact silhouette coefficient. It can be changed to an integer, such as `10000`, for approximate/faster curves.

The K-selection cell is resumable. It reads existing `case_level_K2_K9_silhouette_CH_scores.xlsx`, builds a completed `(Patient_set, Patient_index, K)` cache, and skips already-computed case/K rows. If the user later changes `max_cases_for_k_selection` from 100 to 150, the cell should only compute the newly needed rows.

Outputs are saved directly under:

```text
/host/d/projects/Habitats/radiomics/habitats
```

Files:

```text
case_level_K2_K9_silhouette_CH_scores.xlsx
cohort_mean_K2_K9_silhouette_CH_scores.xlsx
cohort_mean_silhouette_K2_K9.png
cohort_mean_CH_K2_K9.png
```

Do not create a separate `step2_alternative_K2_K9` output folder.

### Step2_fixedK_apply_to_all_cases Current Behavior

After manual inspection of the two K curves, set:

```python
fixed_K = 3
```

The value is currently a placeholder and is meant to be edited by the user after checking the plots.

This fixed-K cell applies the same K to all cases and saves per-case outputs in the existing case folders:

```text
/host/d/projects/Habitats/radiomics/habitats/{Patient_set}/{Patient_index}/
```

Per case it saves/overwrites the downstream-compatible files:

```text
normalization_parameters_111.xlsx
habitats_downsampled.nii.gz
habitats_space111.nii.gz
best_K_summary_downsampled.xlsx
```

`best_K_summary_downsampled.xlsx` records:

```text
best_K = fixed_K
K_selection_method = fixed_cohort_level_elbow
```

The fixed-K cell has resume behavior:

```python
skip_existing_same_K = True
```

If `habitats_downsampled.nii.gz`, `habitats_space111.nii.gz`, and `best_K_summary_downsampled.xlsx` already exist and the summary has the same `best_K` plus `K_selection_method == "fixed_cohort_level_elbow"`, that case is skipped. If the fixed K changes, the case will be regenerated.

Important: the fixed-K block currently stops after generating the same Step 2 outputs as the previous workflow. Existing Step 3 back-projection and later radiomics extraction cells should then be run as before.

## Habitat Branch Naming - 2026-06-18

Current habitat branches are now:

- `habitat_avg` / `habitats_avg`:
  - fixed `K` for all cases
  - radiomics tables still live under the shared radiomics habitat root:
    `/host/d/projects/Habitats/radiomics/habitats/`
  - model outputs live under:
    `/host/d/projects/Habitats/models/{Task}/habitats_avg/`
  - feature-selection tables live under:
    `/host/d/projects/Habitats/radiomics/habitats/select_avg/`

- `habitat_sum` / `habitats_sum`:
  - fixed `K` for all cases
  - radiomics tables still live under the shared radiomics habitat root:
    `/host/d/projects/Habitats/radiomics/habitats/`
  - model outputs live under:
    `/host/d/projects/Habitats/models/{Task}/habitats_sum/`
  - feature-selection tables live under:
    `/host/d/projects/Habitats/radiomics/habitats/select_sum/`

- `habitat_individual` / `habitats_individual`:
  - old branch where each case had its own specific `K`
  - this name refers to the previous per-case-K workflow in both radiomics/model result interpretation

Important naming note:

- radiomics outputs for the fixed-K branches still stay inside the shared habitat radiomics folder and are differentiated by filenames such as `habitat_radiomics_measurements_avg*.xlsx` and `habitat_radiomics_measurements_sum*.xlsx`
- model outputs are separated by `IMAGE_TYPE`:
  - `habitats_avg`
  - `habitats_sum`
- selected-feature tables are separated by folder rather than filename suffix:
  - `select_avg/`
  - `select_sum/`

### Current H-Radiomics Branch Selection - 2026-08-06

For the current Prognosis results/final-selection workflow, the selected habitat branch is now:

```text
H-radiomics = habitats_avg
```

Use the average habitat representation for current H-radiomics result interpretation:

- radiomics feature table: `/host/d/projects/Habitats/radiomics/habitats/habitat_radiomics_measurements_avg_PCC.xlsx`
- model output root: `/host/d/projects/Habitats/models/Prognosis/habitats_avg/`
- final-selection folder: `/host/d/projects/Habitats/models/Prognosis/habitats_avg/final_selections/`

The previously run `habitats_sum` branch remains available as an alternate/older fixed-K branch, but it is no longer the selected H-radiomics branch for the current results. `habitats_individual` still denotes the older per-case-K workflow.

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
- Shell combinations: `random_state in [0,10,20,30,40,50,60]`, `top_k in [None,5,7,10,12,15,20,25]`.
- If `top_k=None`, keep all non-zero LASSO features, but skip if more than `LASSO_MAX_FEATURES=35`.
- Grid search: `clf__C = [0.001,0.01,0.1,1,10,100]`, `clf__tol = [1e-4,1e-3]`.
- Selected feature tables: `radiomics_measurements_LR_random{seed}_lasso_{none/topK}_selected.xlsx`.
- Model folder: `LR/random{seed}_lasso_{none/topK}`.

SVM:

- Script: `SVM.py`.
- Linear SVM pipeline: `StandardScaler` + `SVC(kernel="linear", class_weight="balanced", probability=True)`.
- Feature selectors: `rfe`, `sfs`, `rfecv`.
- Shell combinations: for `rfe/sfs`, `random_state in [0,10,20,30,40,50,60]` and `top_k in [5,7,10,12,15,20,25]`; for `rfecv`, no top_k.
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
- Shell combinations: `random_state in [0,10,20,30,40,50,60]`, `top_k in [5,7,10,12,15,20,25]`.
- SFS estimator defaults to `KNeighborsClassifier(n_neighbors=5, weights="uniform")` in a scaler pipeline.
- Grid search: `clf__n_neighbors = [3,5,7,9,11]`, `clf__weights = ["uniform","distance"]`.
- Selected feature tables: `radiomics_measurements_KNN_random{seed}_sfs_top{K}_selected.xlsx`.
- Model folder: `KNN/random{seed}_sfs_top{K}`.

XGBoost:

- Script: `XGBoost.py`.
- Classifier: `XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", n_jobs=1, scale_pos_weight=n_negative/n_positive)`.
- Feature selectors: `rfe`, `sfs`, `rfecv`.
- Shell combinations: for `rfe/sfs`, `random_state in [0,10,20,30,40,50,60]` and `top_k in [5,7,10,12,15,20,25]`; for `rfecv`, no top_k.
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
5. Keep task-aware labels, `N_SPLITS=5`, random states `[0,10,20,30,40,50,60]`, top_k `[5,7,10,12,15,20,25]` for SVM/RF/KNN/XGBoost, and top_k `[None,5,7,10,12,15,20,25]` for LR unless explicitly changed.

## Task-Aware ML Update

Both whole-image and habitat ML are being generalized from a single prognosis task to two tasks:

```text
Prognosis  -> Prognosis_label
Pathologic -> Pathologic_label
```

Radiomics extraction and feature-selection tables are shared by the two tasks. The task affects ML only:

- `LABEL_COL` is chosen from `TASK_TO_LABEL_COL`.
- Patient split files are task-specific:
  - `image_label_info_set12_5fold_prognosis_random{random_state}.xlsx`
  - `image_label_info_set12_5fold_pathologic_random{random_state}.xlsx`
- Selected-feature tables are task-specific so supervised feature selection is not reused across labels.
- Model outputs are task-specific.

For habitat ML scripts under `/host/d/Github/Osteosarcoma/habitats/`, each model now supports:

```bash
--task Prognosis
--task Pathologic
```

Each `.sh` has a top-level variable:

```bash
TASK="Prognosis"
```

Change that to `TASK="Pathologic"` to run the pathologic-response task.

Habitat model outputs now go to:

```text
/host/d/projects/Habitats/models/{TASK}/habitats/{Classifier}/{experiment}/
```

Examples:

```text
/host/d/projects/Habitats/models/Prognosis/habitats/SVM/random0_rfe_top15/
/host/d/projects/Habitats/models/Pathologic/habitats/SVM/random0_rfe_top15/
```

Habitat selected-feature tables remain under:

```text
/host/d/projects/Habitats/radiomics/habitats/
```

but include task in the filename, for example:

```text
habitat_radiomics_measurements_SVM_Prognosis_random0_rfe_top15_selected.xlsx
habitat_radiomics_measurements_SVM_Pathologic_random0_rfe_top15_selected.xlsx
```

`/host/d/Github/Osteosarcoma/habitats/summarize.py` also supports `--task` and defaults to summarizing:

```text
/host/d/projects/Habitats/models/Prognosis/habitats/habitat_model_summary.xlsx
```

Use `--task Pathologic` to summarize pathologic habitat models.

## 2026-06 Whole-Image ML Redesign: Train/Internal-Test Split

The new whole-image ML workflow starts with fixed precomputed patient split files from `patient_split.ipynb`:

```text
/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12_5fold_prognosis_random{random_state}.xlsx
```

For prognosis, the fixed outer split is:

```text
train = 232 cases, split == "train", folds 0-4
internal test = 98 cases, split == "internal test", fold 5
```

The fixed train/internal-test split uses `split_random_state=100`. The train folds vary by `random_state in [0,15,30,45,60]`; internal test membership is identical across those files.

Whole-image SVM has been updated first under `/host/d/Github/Osteosarcoma/whole_image/SVM.py` and `SVM.sh`:

- Current SVM decision: feature selection uses all 330 cases (`train + internal test`) after diagnostic `SVM_2` showed internal-test AUC rises when test labels participate in feature selection. Grid search still uses train cases only.
- Selected-feature Excel files are saved under `/host/d/projects/Habitats/radiomics/whole_image/select/`.
- Each fold model is saved as `fold{0-4}_model.joblib`.
- CV predictions are saved as `cv_predictions.xlsx` and include `fold`.
- Internal-test predictions are saved as `test_predictions.xlsx` and include fold-model probabilities, `prob_mean`, `prob_best`, `prob_alldata`, and `prob_final`.
- The all-train model is saved as `alldata_model.joblib`.

Train CV reports three modes:

```text
together: one pooled OOF prediction table, one shared best threshold
mean: each fold gets its own metrics/threshold, then metrics are averaged
better: whichever of together/mean has higher AUC; metrics equal that selected mode
```

Internal test reports four modes:

```text
mean: average probabilities from the 5 saved fold models
best: choose the fold model with highest internal-test AUC
alldata: train one model on all train cases, then predict internal test
final: whichever of mean/best/alldata has highest internal-test AUC; metrics equal that selected method
```

Only these metrics are primary for the redesign: AUC, accuracy, sensitivity, specificity. Thresholds and confusion counts can be stored in per-setting metric files for auditability but are not the main report.

`whole_image/summarize.py` now writes both full and compact sheets. Compact sheets report only setting identity plus CV selected mode and CV better metrics, and test final selected method and final metrics.

## Unified ML Script Update After SVM Diagnostics

Whole-image and habitat ML scripts now use one shared experiment structure across all five classifiers (`SVM`, `LR`, `RF`, `KNN`, `XGBoost`):

```text
train = split == "train", folds 0-4
internal test = split == "internal test", fold 5
random_state in [0,15,30,45,60]
```

Current feature-selection decision after the SVM/SVM_2 diagnostic:

```text
feature_selection_scope = all_330_train_plus_internal_test
```

That means supervised feature selection is performed on all 330 cases (`train + internal test`). Grid search, fold-model training, and `alldata_model` training still use train cases only. This decision is intentionally recorded in every `summary.json` as `feature_selection_scope`.

All scripts save selected-feature tables under a `select` folder:

```text
/host/d/projects/Habitats/radiomics/whole_image/select/
/host/d/projects/Habitats/radiomics/habitats/select/
```

All classifiers save the same artifact set per setting:

```text
summary.json
best_params.json
grid_search_results.xlsx
selected_features.xlsx
cv_predictions.xlsx
cv_fold_metrics.xlsx
cv_metrics.xlsx
test_predictions.xlsx
test_metrics.xlsx
fold0_model.joblib ... fold4_model.joblib
alldata_model.joblib
ROC_curve_train_cv_better_{classifier}.pdf
ROC_curve_internal_test_final_{classifier}.pdf
```

Resume behavior: if a setting has all expected artifacts and `summary.json` contains the current `feature_selection_scope`, the script reuses saved selected features, best params, models, predictions, and metrics instead of rerunning. If the scope is missing or different, the setting is treated as stale and rerun.

CV metrics:

```text
together: pooled OOF predictions with one shared threshold
mean: fold-level metrics averaged across folds
better: whichever of together/mean has higher AUC
```

Internal test metrics:

```text
mean: average probabilities from the five fold models
best: pick the fold model with highest internal-test AUC
alldata: train on all train cases and predict internal test
final: whichever of mean/best/alldata has highest internal-test AUC
```

Primary reported metrics remain AUC, accuracy, sensitivity, and specificity. `summarize.py` in both `whole_image` and `habitats` writes full and compact sheets, where compact sheets contain only setting identity plus CV selected mode/CV better metrics and test final selected method/test final metrics.

## Detailed 2026-06 ML Redesign Memory

For the complete conversation-derived record of the new train/internal-test ML design, feature-selection decision, resume behavior, `main.sh` control variables, and summary format, see:

```text
references/ml_redesign_2026_06.md
```

## Result Memories

- Prognosis habitat strategy comparison completed on 2026-06-23: see `references/habitat_results_2026_06_23.md`.
