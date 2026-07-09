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

The current project labels are binary labels such as `Prognosis_label` and `Pathologic_label`. We do not currently have recurrence/progression time plus censoring columns, so do not use Cox regression for the first clinical model branch.

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

