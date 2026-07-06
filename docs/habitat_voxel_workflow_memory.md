# Habitat Voxel Workflow Memory

Current notebook: `/host/d/Github/Osteosarcoma/habitats/step1_make_habitat.ipynb`.

This memory mirrors the `habitat-voxel-workflow` skill and records the finalized workflow decisions for the Osteosarcoma habitat pipeline.

## Final Step1 Pipeline

1. Build patient lists from `/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12.xlsx`.
2. Extract PyRadiomics voxel feature maps using `/host/d/Github/Osteosarcoma/radiomics_settings/MR_setting_voxel.yaml` and `voxelBased=True`.
3. Save voxel feature maps to `/host/d/projects/Habitats/radiomics/voxels/{Patient_set}/{Patient_index}/feature_maps/`.
4. For K-means, load `[1,1,1]` feature maps, define foreground as not all feature values zero and not all NaN, remove invalid voxels and constant features.
5. Normalize per case using `StandardScaler` inside `[1,1,1]` foreground only.
6. Downsample normalized feature maps using `block_reduce(..., func=np.mean)` with current `downsample_block_size=(3,3,3)`.
7. Recompute foreground from downsampled feature maps and run K-means for `K_candidates=[3,4,5]` with `random_state=0`, `n_init=30`.
8. Select best K by maximal silhouette coefficient and save `silhouette_score_K3_K5.png`, `silhouette_score_K3_K5.xlsx`, `best_K_summary_downsampled.xlsx`, `habitats_downsampled.nii.gz`, and `normalization_parameters_111.xlsx`.
9. Upsample downsampled labels back to feature-map `[1,1,1]` shape with `zoom(..., order=0)`, crop/pad to target shape, clip by `foreground_111`, and save `habitats_space111.nii.gz`.
10. Back-project habitats to original image space with the original-label bbox route.
11. Crop original label bbox, save `label_bbox_original_spacing.nii.gz`, resample bbox label to `[1,1,1]`, and save `label_bbox_resampled_111.nii.gz`.
12. Align habitat `[1,1,1]` to bbox-resampled `[1,1,1]` shape with `Data_processing.crop_or_pad(habitat_111_arr, bbox_shape_111, 0)`.
13. Do not clip `habitats_space111_aligned.nii.gz` by `label_bbox_resampled_111.nii.gz` in `[1,1,1]` space.
14. Resample aligned habitat back to bbox original spacing with nearest neighbor, crop/pad to bbox original shape, clip by original bbox label, paste back to full original label space, and final-clip by original manual label.
15. Save `habitats_original_space_final.nii.gz` and binary masks `habitat_original_space_final_{k}.nii.gz`.
16. Extract per-habitat radiomics using `/host/d/Github/Osteosarcoma/radiomics_settings/MR_setting_image.yaml` and `voxelBased=False`.
17. Find masks with `ff.find_all_target_files(["habitat_original_space_final_*.nii.gz"], case_habitat_folder)` and sort by `ff.sort_timeframe(..., num_of_dots=2, start_signal="_", end_signal=".")`.
18. Save one per-case table `habitat_radiomics_measurements.xlsx` with basic metadata, `k`, habitat pixel count/fraction columns, and radiomics features.
19. Add or recalculate an in-file weighted-average row where `k == "avg"`; each feature is `sum(feature_k * Habitat_pixel_fraction_k)`.
20. A downstream combined table can collect all avg rows into `/host/d/projects/Habitats/radiomics/habitats/habitat_radiomics_measurements_avg.xlsx`, keeping only `Patient_set`, `Patient_index`, `Image_filepath`, and `Mask_filepath` as non-feature columns.

## Important Current Naming

- Final multi-class original-space habitat: `habitats_original_space_final.nii.gz`
- Final binary original-space habitats: `habitat_original_space_final_{k}.nii.gz`
- Per-case habitat radiomics table: `habitat_radiomics_measurements.xlsx`
- Aggregated avg table: `habitat_radiomics_measurements_avg.xlsx`

## Checks Already Done

- Across 330 cases, `best_K != 3` in 48 cases: 29 with K=4 and 19 with K=5.
- Weighted-average habitat radiomics mostly fall inside whole-tumor min/max from `radiomics_features_list.xlsx`: 94.77% of feature values in range. Use whole-tumor min/max normalization with clipping to `[0,1]`.

## Step2 Feature Selection Memory

`/host/d/projects/Habitats/radiomics/whole_image/radiomics_measurements_PCC.xlsx` has 282 radiomics features after whole-tumor PCC selection. Habitat feature selection reuses those features, then removes 7 shape features:

- original_shape_Elongation
- original_shape_LeastAxisLength
- original_shape_MajorAxisLength
- original_shape_Maximum2DDiameterSlice
- original_shape_MeshVolume
- original_shape_Sphericity
- original_shape_SurfaceVolumeRatio

Final habitat PCC feature count: 275. Input is `/host/d/projects/Habitats/radiomics/habitats/habitat_radiomics_measurements_avg_normalized.xlsx`; output is `/host/d/projects/Habitats/radiomics/habitats/habitat_radiomics_measurements_avg_PCC.xlsx`.

## Whole-Image ML Memory for Habitat Port

Whole-image scripts are in `/host/d/Github/Osteosarcoma/whole_image`: `LR.py`, `SVM.py`, `RF.py`, `KNN.py`, `XGBoost.py` plus matching `.sh` files and `summarize.py`. They use `Prognosis_label`, 5-fold stratified splits, seeds `0/30/60`, top_k `15/20/25`, and output predictions/fold metrics/summary JSON/ROC curves under `/host/d/projects/Habitats/models/whole_image`. Port habitat ML by changing the input to `habitat_radiomics_measurements_avg_PCC.xlsx`, selected-feature outputs to `/host/d/projects/Habitats/radiomics/habitats`, and model outputs to `/host/d/projects/Habitats/models/habitats`, while keeping the same folds and grids.

## Task-Aware Habitat ML Memory

Habitat ML scripts now support two tasks through `--task Prognosis` or `--task Pathologic`. `TASK_TO_LABEL_COL` maps `Prognosis` to `Prognosis_label` and `Pathologic` to `Pathologic_label`. Shell scripts have `TASK="Prognosis"` at the top; change it to `Pathologic` to run that task.

Task controls the label column, split file name, selected-feature filename, and model output root. Habitat model outputs now go under `/host/d/projects/Habitats/models/{TASK}/habitats/`, while selected-feature tables stay under `/host/d/projects/Habitats/radiomics/habitats/` but include the task name in the filename. `habitats/summarize.py` also accepts `--task` and summarizes `/models/{TASK}/habitats` by default.

## 2026-06-15 New K Selection Direction

The project now has a paper-style alternative for selecting habitat number, based on the intracranial plaque Habitat + ViT paper Supplementary Appendix 4 / Figure S1. The old per-case SC-max workflow is now treated as the `habitats_individual` result branch. The new approach is:

1. Run `Step2_alternative` in `habitats/step1_make_habitat.ipynb`.
2. For each selected case, normalize foreground voxel features per case, downsample normalized feature maps with block mean, test K=2..9, and compute both silhouette coefficient and Calinski-Harabasz index.
3. Save case-level cached scores to `/host/d/projects/Habitats/radiomics/habitats/case_level_K2_K9_silhouette_CH_scores.xlsx`.
4. Save cohort mean scores and plots directly under `/host/d/projects/Habitats/radiomics/habitats`:
   - `cohort_mean_K2_K9_silhouette_CH_scores.xlsx`
   - `cohort_mean_silhouette_K2_K9.png`
   - `cohort_mean_CH_K2_K9.png`
5. Current default is `max_cases_for_k_selection = 100`, with resumable case/K caching. Increasing this number later should only compute missing case/K rows.
6. After the user manually chooses final K from the SC/CH elbow curves, run `Step2_fixedK_apply_to_all_cases` with `fixed_K = <chosen K>`.
7. The fixed-K block applies one K to all cases and saves the downstream-compatible Step 2 outputs per case: `normalization_parameters_111.xlsx`, `habitats_downsampled.nii.gz`, `habitats_space111.nii.gz`, and `best_K_summary_downsampled.xlsx`.
8. Fixed-K summaries record `K_selection_method = fixed_cohort_level_elbow`; same-K completed cases are skipped when `skip_existing_same_K = True`.

After fixed-K Step 2, run the existing Step 3 back-projection and later radiomics extraction cells as before.
