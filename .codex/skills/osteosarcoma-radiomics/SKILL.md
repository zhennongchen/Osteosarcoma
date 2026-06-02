---
name: osteosarcoma-radiomics
description: Use when working on the Osteosarcoma MRI habitat/whole-image radiomics project, especially A-line whole-image ML experiments, SVM/RFE/SFS/RFECV workflows, patient split reuse, and the convention that each classifier has its own dedicated .py and .sh runner.
---

# Osteosarcoma Radiomics Project

## Project Location

- Repository: `/host/d/Github/Osteosarcoma`
- A-line whole-image code: `/host/d/Github/Osteosarcoma/whole_image`
- Patient labels: `/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12.xlsx`
- Whole-image radiomics outputs: `/host/d/projects/Habitats/radiomics/whole_image`
- Whole-image model outputs: `/host/d/projects/Habitats/models/whole_image`

## Current Project Convention

Each ML method should have its own dedicated Python file and shell runner in `whole_image`, instead of one giant shared script.

Current example:

- `whole_image/SVM.py`
- `whole_image/SVM.sh`

Follow the same pattern for future methods, such as `LR.py`/`LR.sh`, `RF.py`/`RF.sh`, `XGBoost.py`/`XGBoost.sh`, and `KNN.py`/`KNN.sh`.

## A-Line Workflow

A-line means whole-image radiomics. The early filtering step is separate:

- `whole_image/step2_feature_selection_ICC_PCC.ipynb`

This step performs ICC reproducibility filtering and PCC redundancy filtering, then writes the PCC-filtered feature table.

Dedicated classifier scripts should start from the ICC/PCC-filtered table:

- `/host/d/projects/Habitats/radiomics/whole_image/radiomics_measurements_PCC.xlsx`

Do not assume row order when joining labels and radiomics. Merge by `Patient_set` and `Patient_index` whenever possible.

## Patient Split Rule

Patient split is the first step in each classifier script. It should be controlled by `--random_state` and default to `0`.

Split output path template:

```text
/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12_5fold_prognosis_random{random_state}.xlsx
```

Use stratified 5-fold split on `Prognosis_label`. If the split file already exists and has a `fold` column, load it instead of regenerating it. If it exists but has no `fold` column, regenerate it.

## Reuse Existing Intermediate Results

Classifier scripts should be restartable. Before running expensive steps, check whether the expected output file already exists.

For patient split:

- Load existing split file if present and valid.

For feature selection:

- Load existing selected-feature table if present and it contains feature columns.
- Only rerun feature selection if the file is missing or invalid.

This matters because shell runners may sweep many variable combinations, and repeated runs should resume rather than recompute everything.

## SVM Branch

SVM is currently treated as the likely strongest A-line method and should be optimized carefully.

SVM feature-selection arguments:

- `--svm_feature_selector rfe|sfs|rfecv`
- `--top_k` applies to `rfe` and `sfs`; `rfecv` chooses automatically.
- RFE uses `step=1` fixed in code. Do not expose `--rfe_step` unless the user explicitly revises this.
- RFECV has a hard-coded maximum feature count of `35`. If RFECV selects more than 35 features, skip the current experiment, write `SKIPPED.json`, and continue the shell runner.

SVM model configuration:

- Use linear kernel.
- Use `class_weight="balanced"` always. Do not grid-search class weight.
- Use `probability=True` because OOF AUC uses probabilities.
- Grid-search `clf__C` and `clf__tol`.

Recommended SVM grid:

```python
param_grid_svm = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "clf__tol": [1e-4, 1e-3],
}
```

Keep GridSearchCV and final fold evaluation consistent: the same best parameters should be used in the final 5-fold OOF evaluation.

## SVM Shell Runner Pattern

`SVM.sh` should run `python3 whole_image/SVM.py` across selected variable combinations.

The discussed search pattern includes:

- random state sweeps as needed; originally default was `0`, current `SVM.sh` may include `0 30 60`.
- feature selectors: `rfe`, `sfs`, and `rfecv`.
- top-K values for RFE/SFS. The initial discussed values were `10, 15, 20`; current user-edited runner may use `15, 20, 25`.

Always respect the current user-edited script if it differs from earlier discussion.

## LR Branch

Logistic Regression uses a dedicated pair:

- `whole_image/LR.py`
- `whole_image/LR.sh`

LR feature selection is currently limited to LASSO:

- `--lr_feature_selector lasso`
- `--top_k 15|20|25|None`

When `top_k` is `None`, use all LASSO non-zero features. If the number of non-zero features exceeds the hard-coded limit of `35`, skip the current experiment, write `SKIPPED.json`, and continue the shell runner. LASSO uses `LogisticRegressionCV` with `penalty="l1"`, `solver="liblinear"`, and `Cs=30` unless the user explicitly revises this.

Final LR classifier configuration:

- Use `penalty="l2"`.
- Use `solver="lbfgs"`.
- Use `class_weight="balanced"` always. Do not grid-search class weight.
- Grid-search `clf__C` and `clf__tol`.

Recommended LR grid:

```python
param_grid_lr = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "clf__tol": [1e-4, 1e-3],
}
```

`LR.sh` should sweep random states and `top_k` values as requested by the user. The current requested top-K set is `15, 20, 25, None`.

## XGBoost Branch

XGBoost uses a dedicated pair:

- `whole_image/XGBoost.py`
- `whole_image/XGBoost.sh`

XGBoost feature-selection options:

- `--xgb_feature_selector rfe|sfs|rfecv`
- `--top_k` applies to `rfe` and `sfs`; do not hard-limit accepted top-K values in the Python parser.
- `rfecv` chooses feature count automatically and does not use `top_k`.
- RFECV has a hard-coded maximum feature count of `50`. If RFECV selects more than 50 features, skip the current experiment, write `SKIPPED.json`, and continue the shell runner.
- Starting with XGBoost and later classifiers, do not print every selected feature to stdout; print only the selected feature count and output path.

XGBoost model configuration:

- Use `objective="binary:logistic"`.
- Use `eval_metric="auc"`.
- Use `tree_method="hist"`.
- Use `n_jobs=1` for XGBoost and GridSearchCV in this Docker environment.
- Use `scale_pos_weight = n_negative / n_positive`.

Current XGBoost grid:

```python
param_grid_xgb = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.1],
}
```

`XGBoost.sh` should sweep random states, RFE/SFS top-K values, and RFECV according to the user's current experiment plan.

## Random Forest Branch

Random Forest uses a dedicated pair:

- `whole_image/RF.py`
- `whole_image/RF.sh`

Random Forest feature-selection options:

- `--rf_feature_selector rfe|sfs|rfecv`
- `--top_k` applies to `rfe` and `sfs`; do not hard-limit accepted top-K values in the Python parser.
- `rfecv` chooses feature count automatically and does not use `top_k`.
- RFECV has a hard-coded maximum feature count of `35`. If RFECV selects more than 35 features, skip the current experiment, write `SKIPPED.json`, and continue the shell runner.
- Do not print every selected feature to stdout; print only selected feature count and output path.

Random Forest model configuration:

- Use `class_weight="balanced"`.
- Use `random_state` from the script argument.
- Use `n_jobs=-1` unless the user reports parallel instability.

Current Random Forest grid:

```python
param_grid_rf = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 3, 5],
    "max_features": ["sqrt", "log2"],
}
```

`RF.sh` should sweep random states, RFE/SFS top-K values, and RFECV according to the user's current experiment plan.

## KNN Branch

KNN uses a dedicated pair:

- `whole_image/KNN.py`
- `whole_image/KNN.sh`

KNN feature selection is currently limited to SFS:

- `--knn_feature_selector sfs`
- `--top_k` controls SFS-selected feature count; do not hard-limit accepted top-K values in the Python parser.

KNN cannot use RFE/RFECV directly because it has no `coef_` or `feature_importances_`. Use `SequentialFeatureSelector` with a standardized KNN pipeline.

KNN model configuration:

- Use `StandardScaler` before `KNeighborsClassifier`.
- KNN has no classifier-level random seed; `random_state` still controls patient split and SFS CV.

Current KNN grid:

```python
param_grid_knn = {
    "clf__n_neighbors": [3, 7, 9, 11],
    "clf__weights": ["uniform", "distance"],
}
```

`KNN.sh` should sweep random states and SFS top-K values according to the user's current experiment plan.

## Reporting

For each experiment combination, save outputs in parameter-specific directories so results do not overwrite each other.

Typical saved outputs:

- selected feature table
- `best_params.json`
- `grid_search_results.xlsx`
- `predictions.xlsx`
- `fold_metrics.xlsx`
- `summary.json`

Primary metric is AUC, especially overall out-of-fold AUC for internal 5-fold development comparison.
