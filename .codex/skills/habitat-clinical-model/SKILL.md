---
name: habitat-clinical-model
description: Use when designing, explaining, or implementing clinical-variable models for the Osteosarcoma/Habitats project, especially baseline tables, univariate and multivariate logistic regression, binary clinical ML models, clinical feature screening, and fusion-ready clinical probabilities for Prognosis_label or Pathologic_label tasks.
---

# Habitat Clinical Model

Use this skill for clinical-variable modeling in the Osteosarcoma/Habitats project. Read `references/clinical_model_workflow.md` when writing code or explaining details of baseline tables, univariate analysis, multivariate logistic regression, or binary clinical ML pipelines.

Related skills:

- `habitat-papers` for local paper paths and literature context.
- `habitat-voxel-workflow` for current train/internal-test split and ML output conventions.
- `habitat-ensemble-fusion` for probability-level fusion with whole-image, habitat, and DL models.

## Current Decision

The current project labels are binary labels such as `Prognosis_label` and `Pathologic_label`. For the prognosis task, the current working definition has been updated to a 2-year DFS event label as recorded below. We do not currently use Cox regression for the first clinical model branch.

## Current Prognosis Label Definition - Updated 2026-08-04

For the next redo of the prognosis task, define the outcome as:

- `label = 1`: 2-year disease-free survival (DFS) event.
- Exclude patients whose available follow-up is shorter than 1 year 9 months if they have no observed event.
- If `Follow_up_time` is earlier than `Admission_time`, treat this as a likely date-ordering error and swap the two dates before calculating follow-up duration.
- The analysis cohort should be restricted to patients with corresponding image data.
- Under this rule, an original 3-year event occurring after 2 years should be treated as 2-year `label = 0`, because the patient was event-free at the 2-year horizon.

The decision was made because some patients labeled as 3-year negative did not have sufficient follow-up to confirm 3-year event-free status. A 2-year DFS endpoint with exclusion of follow-up shorter than 1 year 9 months is the current accepted binary-label strategy.

Use binary classification clinical modeling instead:

1. baseline characteristics table: label 0 vs label 1, with group-comparison p values
2. univariate logistic regression: one clinical variable at a time, report OR, 95% CI, p value
3. multivariate logistic regression: candidate variables together, identify independent predictors
4. clinical ML model: train classifiers on clinical variables or selected clinical predictors, output class-1 probability

## Paper-Informed Pattern

The preferred first-pass approach is closest to the Ki-67 ccRCC paper and the Li 2026 osteosarcoma recurrence habitat paper:

- binary outcome
- clinical variables first assessed by univariate/multivariate logistic regression or ML feature screening
- clinical-only model built with ML classifiers
- output probabilities used for ROC AUC and later ensemble fusion

Wang 2025 spinal tumor is more Cox/survival-oriented and should be treated as clinical-variable reference only unless time-to-event data become available.

## Coding Preference

For the first implementation, keep clinical preprocessing leakage-safe:

- split-aware imputation
- train-fold-only scaling for continuous variables
- train-fold-only one-hot encoding for categorical variables
- feature screening fit only on training data within each CV fold when used for model evaluation
- output probabilities compatible with existing ML/ensemble summaries

See `references/clinical_model_workflow.md` for the full workflow and terminology.

For the current set123 redo, see `references/clinical_model_workflow.md` section "Set123 Clinical Workflow - 2026-07-08". It records the latest train/internal/external split-aware baseline, univariate, multivariate, and ML-ready clinical table conventions.

## Current Clinical Variable Pool - 2026-06-27

The user-defined clinical variable pool for the Osteosarcoma/Habitats clinical model is now:

- `Age`
- `Sex`
- `Lesion_site`
- `Pathologic_fracture`
- `Height_at_visit`
- `Weight_at_visit`
- `BMI` derived from height/weight
- `WBC`
- `HGB`
- `PLT`
- `CRP`
- `ALP`
- `Total_cholesterol`
- `Triglycerides`
- `LDL`
- `LDH`
- `PT`
- `APTT`
- `Fibrinogen`
- `D-dimer`
- derived tumor AP diameter, longitudinal diameter, transverse diameter, and tumor volume

This pool supersedes the earlier minimal literature-aligned clinical set for implementation. Manual `Length`, `Width`, and `Height` are not the primary derived size features; the preferred derived tumor dimensions/volume should be calculated from `label.nii.gz` when possible. `Side` is not included in the current user-defined pool.

## Lesion Site Grouping Rule - 2026-06-27

For the current clinical-variable implementation, `Lesion_site` should be translated to English and grouped as follows:

- keep `Femur`, `Tibia`, and `Humerus` as explicit major lesion-site categories
- collapse all other lesion sites into `Others`

Baseline tables should report all four lesion-site rows: `Femur`, `Tibia`, `Humerus`, and `Others`.

Univariate/multivariate logistic regression and ML-ready design matrices should only explicitly encode the three major lesion-site indicators: `Lesion_site_Femur`, `Lesion_site_Tibia`, and `Lesion_site_Humerus`. `Others` is retained in the processed clinical table and baseline table, but is the implicit reference category for modeling and should not be a separate model feature.

## Current Lesion Site Grouping Rule - Updated 2026-06-28

For the current clinical-variable implementation, `Lesion_site` should be translated to English and grouped as:

- `Femur`
- `Tibia and fibula` (combines Tibia and Fibula)
- `Others` (all remaining sites, including Humerus, Radius, Pelvis, Ilium, Ulna, Scapula, Popliteal fossa, etc.)

Baseline tables should use paper-style rows: one parent row `Tumor location` with a single global multi-category p value, followed by the three category rows with n (%). Logistic regression / ML-ready matrices should use dummy-category terms with `Others` as the reference group: `Lesion_site_Femur` and `Lesion_site_Tibia_and_fibula`.

The grouping helper must be idempotent: rerunning it on an already grouped value `Tibia and fibula` must keep it as `Tibia and fibula`, not collapse it into `Others`.

## Strict 2-Year DFS Redo Status - Updated 2026-08-04

The strict 2-year DFS prognosis label update has been applied to patient-list Excel files. Use this state for the next experiment redo unless the user explicitly changes it.

Applied label/case rules:

- `Prognosis_label = 1`: DFS event occurred within 24 months after admission/imaging.
- Original 3-year positive cases with event time greater than 24 months were recoded to `0`.
- No-event cases with follow-up shorter than 21 months were excluded.
- If `Follow_up_time < Admission_time`, the two dates were swapped before calculating follow-up duration.
- Analysis tables are restricted to cases with corresponding image data.

Updated files:

- `label_info_set1.xlsx`, `label_info_set2.xlsx`, `label_info_set3.xlsx`
- `label_info_set1+2_原始.xlsx`, `label_info_set3_原始.xlsx`
- `image_label_info_set123.xlsx`, `image_label_unmatched_set123.xlsx`
- `image_label_info_set123_resampled_bbox_1x1x3.xlsx`
- `image_label_info_set123_clinical_variables.xlsx`
- `image_label_info_set123_clinical_variables_processed.xlsx`
- `largest_slice_info_set123.xlsx`

Excluded cases due to insufficient no-event follow-up:

- `set_3/26`
- `set_3/27`
- `set_3/28`

Cases recoded from original `Prognosis_label=1` to strict 2-year `Prognosis_label=0`:

- `set_1/34`, `set_1/111`, `set_1/119`, `set_1/131`, `set_1/132`, `set_1/136`, `set_1/139`
- `set_2/10`, `set_2/20`, `set_2/43`, `set_2/44`, `set_2/61`, `set_2/105`, `set_2/120`, `set_2/126`

Current matched cohort after update:

- `set_1`: 99 cases, 20 positives (20.20%)
- `set_2`: 231 cases, 63 positives (27.27%)
- `set_3`: 18 cases, 7 positives (38.89%)
- total: 348 cases, 90 positives (25.86%)

Current generated split files:

```text
/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set123_5fold_prognosis_random{0,10,20,30,40}.xlsx
```

Split settings:

- set1+2 split into `n_train=188`, `n_internal_test=96`, `n_external_from_set12=46`
- all set3 cases are external test: `expected_set3_n=18`
- CV uses fold 0-4, internal test is fold 5, external test is fold 6
- fixed train/internal/external split seed selected by search: `107`
- fold random states: `[0, 10, 20, 30, 40]`

Final split counts:

- train: 188 cases, 47 positives (25.00%)
- internal test: 96 cases, 24 positives (25.00%)
- external test: 64 cases, 19 positives (29.69%)

External test composition:

- set1: 10 cases, 2 positives (20.00%)
- set2: 36 cases, 10 positives (27.78%)
- set3: 18 cases, 7 positives (38.89%)

The external-test positive fraction is higher mainly because all set3 cases are assigned to external test and set3 has a higher positive fraction.
