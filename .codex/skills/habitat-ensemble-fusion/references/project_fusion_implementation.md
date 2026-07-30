# Current Osteosarcoma Fusion Implementation

This records the practical fusion workflow built for the current Osteosarcoma/Habitats project.

## Fusion Inputs

The four current base model probabilities are:

```text
clinical
whole_image / C-radiomics
habitats_sum / H-radiomics
DL_3D
```

The probability listing notebook is:

```text
/host/d/Github/Osteosarcoma/fusion/list_probs.ipynb
```

It reads final-selection probability outputs from each model family and creates a case-level fusion table under:

```text
/host/d/projects/Habitats/models/Prognosis/fusion
```

Each row represents one case, identified by `Patient_set` and `Patient_index`, with the four model probabilities as features.

## Soft Vote

Script:

```text
/host/d/Github/Osteosarcoma/fusion/soft_vote.py
```

Soft vote computes the arithmetic mean of the four modality probabilities:

```text
p_soft_vote = mean([p_clinical, p_c_radiomics, p_h_radiomics, p_dl_3d])
```

It reports metrics for train/CV, internal test, and external test using the project metric conventions:

- AUC with 95% CI
- Accuracy
- Sensitivity
- Specificity

Outputs are saved under:

```text
/host/d/projects/Habitats/models/Prognosis/fusion/soft_vote
```

## Stacking

Folder:

```text
/host/d/Github/Osteosarcoma/fusion/stacking
```

Stacking uses the four probabilities as a low-dimensional ML feature table. It mirrors the project ML framework but does **not** perform feature selection because there are only four input features.

Implemented meta-learners:

```text
SVM
LR
RF
XGBoost
```

KNN was tested earlier but later removed from manuscript-level comparison. Some temporary analyses may still substitute a KNN experiment under another radar-axis label; do not treat KNN as a final paper method unless the user explicitly restores it.

`Main.sh` controls:

- task, usually `Prognosis`
- random states, currently often `0 10 20 30 40`
- `gridsearch_range`, default currently `all`

Hyperparameter selection can be configured as:

- `train`: tune only within training data
- `all`: tune using all available cases, intentionally optimistic if used for result exploration

## CV/Test Reporting Conventions

For ML-style experiments, current split structure after set3 expansion is:

```text
fold 0-4: train/CV
fold 5: internal test
fold 6: external test
```

For each experiment, keep these probability/metric concepts distinct:

- `cv_together`: traditional OOF CV predictions from fold-specific models.
- `cv_allotherdata`: for each CV fold, train a model on all other folds plus internal/external test using fixed selected features/hyperparameters, then predict that CV fold.
- `cv_final`: whichever has better overall CV AUC between `cv_together` and `cv_allotherdata`.
- `cv_final_advanced`: evaluate all 2^5 combinations of per-fold choices between `cv_together` and `cv_allotherdata`, choose the combination with maximal CV AUC.
- `train`: train `alltraindata_model` on all train folds 0-4 with selected features/hyperparameters, then evaluate on the same train set to inspect overfitting capacity.
- `internal test`: report mean/best/alldata/final variants as configured.
- `external test`: same as internal test.

For final-selection notebooks, the user may select which probability mode to read for each cohort. Do not assume `final` if the notebook explicitly requests `mean`, `best`, `alldata`, `cv`, `allotherdata`, or `advanced`.

## Stacking Final Selection

Unlike base model-family final selection, stacking final selection is method-specific.

Notebook:

```text
/host/d/Github/Osteosarcoma/fusion/stacking/final_selection.ipynb
```

At the top of the notebook, choose the current ML method (`RF`, `LR`, `SVM`, or `XGBoost`). Each method gets its own final-selection output folder, for example:

```text
/host/d/projects/Habitats/models/Prognosis/fusion/stacking/final_selections/RF
```

This lets the paper report multiple stacking algorithms separately if needed.

## Manuscript-Level Current Choice

At the latest Results-presentation stage, final fusion display included:

- `fusion_soft_vote`
- `fusion_stacking`

The user reran stacking final selection and selected RF for the current stacking presentation. Always confirm current final-selection files before regenerating summary figures.

## SHAP for Fusion Stacking

For fusion-stacking interpretability, SHAP is applied to the stacking model with four modality probabilities:

```text
Clinical
C-radiomics
H-radiomics
DL_3D
```

If the stacking model is KNN or another model without native SHAP support, use model-agnostic black-box SHAP. If it is RF/XGBoost/tree-based, TreeExplainer may be possible, but consistency with prior outputs may favor the existing implementation.

Generated manuscript assets include:

```text
fusion_stacking_SHAP_summary.pdf
fusion_stacking_SHAP_waterfall_positive.pdf
fusion_stacking_SHAP_waterfall_negative.pdf
fusion_stacking_MRI_overlay_positive_set_2_92.pdf
fusion_stacking_MRI_overlay_negative_set_2_107.pdf
```
