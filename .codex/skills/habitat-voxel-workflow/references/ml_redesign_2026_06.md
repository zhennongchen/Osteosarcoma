# ML Redesign Memory - 2026-06-11

This memory records the new experiment design introduced after the initial whole-image/habitat ML pipeline was working. It should be used when modifying or explaining current ML scripts under `/host/d/Github/Osteosarcoma/whole_image` and `/host/d/Github/Osteosarcoma/habitats`.

## Why The Redesign Happened

The earlier ML setup used all 330 cases together in 5-fold cross-validation. The new design separates the data into a fixed train/internal-test split, then performs CV only inside train and evaluates several strategies on the locked internal test.

The current task focus is prognosis first:

```text
label column = Prognosis_label
patient list = /host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12.xlsx
```

Pathologic remains supported by task-aware scripts, but the newly discussed split/test design was first built for prognosis.

## Fixed Patient Split

`patient_split.ipynb` generates precomputed split files:

```text
/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12_5fold_prognosis_random{random_state}.xlsx
```

Outer split:

```text
total = 330
train = 232
internal test = 98
split_random_state = 100
```

The split is stratified by `Prognosis_label`; the positive-label proportion should be close between train and internal test.

Generated columns:

```text
split: "train" or "internal test"
fold: 0,1,2,3,4 for train; 5 for internal test
```

Train/internal-test membership is fixed across all fold-random-state files. Only the train fold assignment changes.

Current train-fold random states:

```text
0, 15, 30, 45, 60
```

## Per-Setting Definition

A setting is determined by:

```text
random_state
feature_selector
top_k, when applicable
classifier
image type: whole_image or habitats
task
```

Selectors vary by model:

```text
SVM: rfe, sfs, rfecv
LR: lasso
RF: rfe, rfecv currently in shell defaults; code supports sfs too
KNN: sfs
XGBoost: rfe, sfs, rfecv
```

`main.sh` is now the preferred place to define `RANDOM_STATE_LIST`, `TOP_K_LIST`, and `METHOD_LIST`. Individual method `.sh` scripts read these environment variables. Selectors remain inside each method `.sh` because each model has different selectors.

## Current main.sh Behavior

Both `whole_image/main.sh` and `habitats/main.sh` expose editable variables:

```bash
TASK="${TASK:-Prognosis}"
METHOD_LIST="${METHOD_LIST:-SVM LR RF KNN XGBoost}"
RANDOM_STATE_LIST="${RANDOM_STATE_LIST:-0 15 30 45 60}"
TOP_K_LIST="... user-editable per main.sh ..."
LR_TOP_K_LIST="${LR_TOP_K_LIST:-None ${TOP_K_LIST}}"
```

`METHOD_LIST` lets the user run a subset, for example:

```bash
METHOD_LIST="SVM RF" ./main.sh
```

`TOP_K_LIST` can intentionally differ between `whole_image/main.sh` and `habitats/main.sh`. Do not auto-sync them; the user may set them differently.

Current observed defaults after the latest edits:

```text
whole_image/main.sh: TOP_K_LIST="5 7 10 12 15 20 25 30"
habitats/main.sh: TOP_K_LIST="20 25 30"
```

Individual method scripts have their own fallback defaults, but when launched by `main.sh` they should use the exported values from `main.sh`.

## Feature Selection Scope Decision

After seeing high CV AUC but very low internal-test AUC in whole-image SVM, a diagnostic script `SVM_2.py` was created. It intentionally selected features on all 330 cases (`train + internal test`) instead of train only. Internal-test AUC increased, confirming that train-only feature selection did not generalize well to the internal test.

The user then decided to adopt the all-330 feature-selection scope for the current experimental run.

Current setting recorded in every `summary.json`:

```text
feature_selection_scope = all_330_train_plus_internal_test
```

Meaning:

```text
feature selection: all 330 cases, including internal test labels
grid search: train only
fold-model training: train only
alldata_model training: train only
internal test: used for evaluation and final test-strategy selection
```

This is a deliberate experimental choice and must not be silently changed. Keep the `feature_selection_scope` field visible in summary outputs.

The diagnostic `SVM_2.py` and `SVM_2.sh` were removed after merging the decision back into `SVM.py` with output names restored to `SVM`.

## Train CV Metrics

For each setting, train CV runs on folds 0-4 only.

Each fold model is saved:

```text
fold0_model.joblib
fold1_model.joblib
fold2_model.joblib
fold3_model.joblib
fold4_model.joblib
```

CV predictions are saved separately:

```text
cv_predictions.xlsx
```

This file must include `fold`.

Train CV reports three modes:

```text
together: pool all OOF predictions and compute one set of metrics with one shared best threshold
mean: compute fold-level metrics separately, each with its own threshold, then average metrics across folds
better: whichever of together/mean has higher AUC; metrics exactly equal that selected mode
```

Primary metrics:

```text
AUC
accuracy
sensitivity
specificity
```

No need to foreground F1/precision in summaries.

Only the best CV ROC should be plotted:

```text
ROC_curve_train_cv_better_{classifier}.pdf
```

If `better=together`, plot pooled OOF ROC. If `better=mean`, plot the five fold ROC curves together because mean mode is fold-level by definition.

## Internal Test Metrics

Internal test is fixed:

```text
split == "internal test"
fold == 5
```

Test predictions are saved separately:

```text
test_predictions.xlsx
```

This file should include:

```text
prob_fold0_model
prob_fold1_model
prob_fold2_model
prob_fold3_model
prob_fold4_model
prob_mean
prob_best
prob_alldata
prob_final
best_model_fold
final_selected_method
```

Internal test reports four modes:

```text
mean: apply all five fold models to internal test and average probabilities
best: apply all five fold models and pick the fold model with highest internal-test AUC
alldata: train one model on all train cases and apply to internal test
final: whichever of mean/best/alldata has highest internal-test AUC; metrics exactly equal that selected method
```

Only the final internal-test ROC should be plotted:

```text
ROC_curve_internal_test_final_{classifier}.pdf
```

## Artifacts Per Setting

Each completed setting should have:

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

Selected-feature Excel tables are saved under:

```text
/host/d/projects/Habitats/radiomics/whole_image/select/
/host/d/projects/Habitats/radiomics/habitats/select/
```

Model outputs are task- and image-type-specific:

```text
/host/d/projects/Habitats/models/{TASK}/whole_image/{Classifier}/{experiment}/
/host/d/projects/Habitats/models/{TASK}/habitats/{Classifier}/{experiment}/
```

## Resume Logic

If a setting has already completed, the model script should not rerun feature selection, grid search, CV, or test.

Resume requires all expected artifacts to exist and:

```text
summary.json["feature_selection_scope"] == FEATURE_SELECTION_SCOPE
```

If the field is missing or different, the setting is treated as stale and rerun. This prevents accidental reuse of old train-only feature-selection results.

## Summaries

`whole_image/summarize.py` and `habitats/summarize.py` write both full and compact sheets.

Expected sheets:

```text
All_full
All_compact
{Classifier}_full
{Classifier}_compact
```

Compact summary should include only setting identity plus:

```text
cv_selected_metric_mode
cv_together_auc
cv_together_auc_ci_low
cv_together_auc_ci_high
cv_together_accuracy
cv_together_sensitivity
cv_together_specificity
cv_better_auc
cv_better_auc_ci_low
cv_better_auc_ci_high
cv_better_accuracy
cv_better_sensitivity
cv_better_specificity

test_final_selected_method
test_final_auc
test_final_auc_ci_low
test_final_auc_ci_high
test_final_accuracy
test_final_sensitivity
test_final_specificity
```

Full summary also includes together/mean CV metrics, mean/best/alldata/final test metrics, paths, best params, selected feature count, and `feature_selection_scope`.

## SVM Probability Note

During diagnosis, SVM `predict_proba` and `decision_function` were compared. Some random states showed unstable probability calibration, e.g. `predict_proba` CV AUC was lower than `decision_function` AUC. The user decided not to change score/probability output for now. Current scripts still use `predict_proba` for saved predictions and metrics.



## Batch Stability Fix - 2026-06-12

All current whole-image and habitat ML model scripts should use a non-interactive matplotlib backend before importing `pyplot`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

This was added after RF runs in both `whole_image` and `habitats` aborted with Tk/Tcl cleanup errors such as `RuntimeError: main thread is not in main loop` and `Tcl_AsyncDelete: async handler deleted by the wrong thread`. The root cause was the local default `TkAgg` backend interacting badly with batch plotting and joblib.

The current model scripts also use `n_jobs=1` for grid search, recursive/sequential feature selection, and estimators that expose `n_jobs`. This intentionally reduces parallelism to avoid joblib/Tk/thread cleanup crashes during long `main.sh` runs. `main.sh` itself was not changed for this fix.

## Important Cautions For Future Changes

- Do not silently change feature selection back to train-only without explicit user direction.
- Do not auto-sync `TOP_K_LIST` between whole_image and habitats; the user may intentionally set different lists.
- Keep selectors inside method `.sh` files, not in `main.sh`, because model families differ.
- `main.sh` should control `METHOD_LIST`, `RANDOM_STATE_LIST`, `TOP_K_LIST`, `TASK`, and `LR_TOP_K_LIST`.
- When adding DL/fusion later, probability-level fusion and OOF stacking are recorded separately in the `habitat-ensemble-fusion` skill.

## July 2026 Status And Final-Selection Notebooks

By 2026-07-01, the following Prognosis model families had been run or were being finalized:

```text
whole_image
habitats_individual
dl_2d_ml
dl_2d_ml_cv
dl_3d_ml
```

Meanings:

- `whole_image`: whole-tumor radiomics ML.
- `habitats_individual`: habitat radiomics using the earlier per-case/specific-K habitat workflow.
- `dl_2d_ml`: exploratory DL-feature ML where the DL feature extractor was trained on all 330 cases before feature extraction. This is intentionally leakage-prone/optimistic.
- `dl_2d_ml_cv`: more methodologically valid DL-feature ML where fold-specific DL feature extraction is used; expected to perform worse because the DL models themselves overfit.
- `dl_3d_ml`: 3D DL-feature ML using the same downstream ML framework.

Current summary files are under:

```text
/host/d/projects/Habitats/models/Prognosis/whole_image/whole_image_model_summary.xlsx
/host/d/projects/Habitats/models/Prognosis/habitats_individual/habitat_model_summary.xlsx
/host/d/projects/Habitats/models/Prognosis/dl_2d_ml/dl_2d_ml_model_summary.xlsx
/host/d/projects/Habitats/models/Prognosis/dl_3d_ml/dl_3d_ml_model_summary.xlsx
```

The compact summary was updated to include `cv_together_*` columns before `cv_better_*` columns. Rationale: the user wants to judge/report validation primarily by pooled/together CV ROC when manually improving results, while retaining `cv_better` for traceability.

### Final Selection Notebooks

Final-selection notebooks implement manual probability-level fusion after individual experiments have completed:

```text
/host/d/Github/Osteosarcoma/whole_image/final_selection.ipynb
/host/d/Github/Osteosarcoma/habitats/final_selection.ipynb
/host/d/Github/Osteosarcoma/dl_2d_ml/final_selection.ipynb
/host/d/Github/Osteosarcoma/dl_3d_ml/final_selection.ipynb
```

For each modality, outputs are saved directly into:

```text
/host/d/projects/Habitats/models/Prognosis/{IMAGE_TYPE}/final_selections/
```

No extra `final_selection_name` subfolder is used.

Validation/CV section:

- User manually defines `selected_cv_experiments`.
- Reads each selected experiment's `cv_predictions.xlsx`.
- Merges by `Patient_set`, `Patient_index`, and label.
- Keeps the first selected experiment's `fold` as the reference fold.
- Saves every experiment's probability column.
- Computes `prob_mean` as arithmetic mean of selected probabilities.
- Computes `prob_mix` by exhaustive best fold-wise selection: if there are `n` experiments and five folds, it searches `n^5` fold-to-experiment combinations and keeps only the best-AUC mix.
- Does not save non-best mix candidates.
- Saves `cv_final_selection_best_mix.xlsx` to show which experiment was chosen for each fold.
- Saves CV mean and mix ROC plots as separate PDFs:
  - `ROC_curve_cv_final_selection_mean.pdf`
  - `ROC_curve_cv_final_selection_mix.pdf`

The mix search includes a sanity check: when more than two experiments are selected, the full best mix must be at least as good as the best mix restricted to the first two experiments. If not, restart the notebook kernel and rerun all cells, because stale notebook state or alignment problems are likely.

Internal-test section:

- User manually defines `selected_test_experiments` independently from CV.
- Reads each selected experiment's `test_predictions.xlsx`.
- Uses each experiment's already chosen `prob_final`.
- Keeps `final_selected_method` so the output shows whether the source experiment used `mean`, `best`, or `alldata`.
- Computes only the mean of selected `prob_final` columns.
- Does not perform fold-wise mix on internal test.
- Saves `ROC_curve_internal_test_final_selection_mean.pdf`.

## July 2026 Planned Full Redo With New Data

As of 2026-07-01, the user plans to redo everything except radiomics feature extraction for already-processed old cases.

Reason:

- New data have arrived, about 20+ additional cases.
- These new cases still need basic information and radiomics extraction.
- After incorporating them, patient splitting will be redesigned.

Planned split design:

```text
train: used for cross-validation
internal_test
external_test
```

Important upcoming change:

- `external_test` will be constructed from a portion of the previous train data plus the new cases.
- The exact split rule is not fixed yet; wait for user instructions before coding.
- Existing two-way split logic (`train` plus `internal test`, fold 5 as internal test) should be treated as historical. Future scripts will need explicit external-test support.

When adapting old scripts for the redo, preserve these proven design patterns unless the user changes them:

- keep setting-level artifacts (`summary.json`, predictions, metrics, models, ROC plots);
- keep resume/skip behavior for completed settings;
- keep AUC 95% CI columns separate from AUC columns;
- keep `cv_together`, `cv_better`, and internal-test final metrics separately;
- keep manual `final_selection.ipynb` style for probability fusion.

