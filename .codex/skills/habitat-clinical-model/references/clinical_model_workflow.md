# Clinical Model Workflow Memory - Osteosarcoma/Habitats

Date recorded: 2026-06-18

This note preserves the project decisions about clinical-variable modeling for later code implementation.

## Why Cox Is Not First-Pass

Cox regression is for time-to-event data. It needs at least:

- `time`: recurrence/progression/death/follow-up time
- `event`: whether the event happened, usually 1/0

Current project tasks are binary labels such as:

- `Prognosis_label`
- `Pathologic_label`

Because we currently have labels but not survival/progression time plus censoring information, the first clinical model should be a binary classification model, not a Cox model.

Cox can produce risk scores and, with a baseline survival function and a specified time point, survival/event probabilities. But that is not the natural setup for the current label-only tasks.

## Baseline Table vs Logistic Regression

A common paper table has columns like:

```text
Variable        label=0        label=1        p value
Age             mean/sd        mean/sd        t-test p
Sex             n/%            n/%            chi-square p
```

This is a baseline characteristics table. It answers:

- Are the two label groups different in this variable's distribution?

Typical tests:

- continuous normal variable: t-test
- continuous non-normal variable: Mann-Whitney U test
- categorical variable: chi-square test or Fisher exact test

Univariate logistic regression answers a slightly different question:

- Does this one variable predict the probability/odds of `label=1`?

It reports:

- coefficient
- odds ratio: `OR = exp(coef)`
- 95% CI
- p value

Example:

```text
logit(P(label=1)) = b0 + b1 * Tumor_size
```

Baseline table and univariate logistic often reach similar conclusions, but the modeling direction differs:

- baseline comparison: `Variable ~ Label`
- logistic regression: `Label ~ Variable`

## Univariate Logistic Regression

Univariate means one predictor at a time.

For each clinical variable, fit:

```text
logit(P(label=1)) = b0 + b1 * x
```

For categorical predictors:

- binary categorical variables can be coded 0/1
- multi-class categorical variables need one-hot/dummy encoding

Use the univariate p value and OR to describe association with the binary outcome. Candidate thresholds commonly used for multivariate entry:

- p < 0.05: stricter
- p < 0.10: more inclusive, common for candidate screening

## Multivariate Logistic Regression

Multivariate means multiple predictors in the same logistic model.

Example:

```text
logit(P(label=1)) = b0 + b1 * Age + b2 * Tumor_size + b3 * ALP + b4 * Sex_Male
```

Each variable's p value is interpreted as:

- after controlling for the other variables in this model, does this variable still provide independent association with label=1?

This is how papers identify independent predictors.

Important example:

- ALP can be significant in baseline table/univariate analysis
- but become non-significant in multivariate analysis if tumor size explains the same information

## Paper-Informed Clinical Model Patterns

### Ki-67 ccRCC Paper

Paper: `Integration of multi-scale radiomics and deep learning for Ki-67 prediction in clear cell renal carcinoma`

Outcome: high vs low Ki-67 expression, binary classification.

Clinical workflow:

1. univariate logistic regression
2. multivariate logistic regression
3. significant independent predictors used for clinical model
4. ensemble ML algorithms evaluated

Independent predictors reported:

- PLT
- tumor diameter
- hematuria

The clinical model used these significant clinical/laboratory parameters, and XGBoost performed best among evaluated ensemble algorithms.

### Li 2026 Osteosarcoma Recurrence Habitat Paper

Paper: `Magnetic Resonance Imaging - 2026 - Li - Characterization of Intratumoral Heterogeneity via MRI-Based Radiomic Habitats ...`

Outcome: recurrence vs non-recurrence, binary classification.

Clinical/radiological variables included age, gender, tumor location, pathological type, hemoglobin, ALP, LDH, calcium, sodium, potassium, surgical SSS stage, tumor diameters, bone destruction, periosteal reaction, peritumoral edema, pathological fracture, vascular invasion, and joint invasion.

They did group comparison and built a clinical ML model. Because recurrence events were limited relative to the number of significant clinical variables, traditional multivariate logistic regression was not performed to avoid overfitting. Instead, ML classifiers such as SVM, RF, and LR were used; RF was best for the clinical model.

### Wang 2025 Spinal Tumor PFS Paper

Paper: `wang-et-al-2025-mri-based-habitat-analysis-for-the-prediction-of-progression-free-survival-in-primary-spinal-tumors`

This is primarily survival/PFS oriented and used Cox regression:

1. univariable Cox
2. variables with p < 0.10 enter multivariable Cox
3. clinical model built from independent predictors

It is useful for clinical-variable examples but should not be copied directly unless our project has time-to-event data.

## Recommended First-Pass Project Workflow

For `Prognosis_label` and `Pathologic_label` as binary tasks:

1. Prepare clinical table.
   - Keep patient identifiers: `Patient_set`, `Patient_index`.
   - Include label column for the task.
   - Separate continuous and categorical clinical variables.

2. Baseline table.
   - Produce label=0 vs label=1 summary.
   - Continuous variables: mean/sd or median/IQR; p value by t-test or Mann-Whitney.
   - Categorical variables: counts/percentages; p value by chi-square or Fisher.

3. Univariate logistic table.
   - Fit one variable at a time.
   - Report OR, 95% CI, p value.
   - Use p < 0.10 as a practical candidate threshold unless user chooses otherwise.

4. Multivariate logistic table.
   - Fit candidate variables together.
   - Report adjusted OR, 95% CI, p value.
   - Treat significant variables as independent predictors.

5. Binary clinical ML model.
   - Option A: use all clinically reasonable variables.
   - Option B: use univariate p < 0.10 variables.
   - Option C: use multivariate independent predictors.
   - Compare these settings if useful.

6. Leakage-safe preprocessing.
   - Impute missing values fit only on train fold.
   - Scale continuous variables fit only on train fold.
   - One-hot encode categorical variables fit only on train fold.
   - Any feature screening used for performance evaluation must be fit inside CV training folds, not on validation/test labels.

7. Classifiers.
   - Logistic Regression is the clean interpretable baseline.
   - SVM, RF, XGBoost/Gradient Boosting, KNN can follow the existing ML framework.
   - Output `predict_proba` class-1 probability for AUC and later ensemble fusion.

8. Output should align with existing project ML summaries.
   - prediction tables include probability and fold/split information
   - summary tables report AUC, accuracy, sensitivity, specificity
   - saved clinical model probabilities can later fuse with whole-image radiomics, habitat radiomics, and ViT probabilities

## Practical Cautions

- Do not put too many variables into multivariate logistic when events are few. Events per variable < 10 can cause overfitting or unstable estimates.
- Multi-class categorical variables increase parameter count after one-hot encoding.
- Perfect separation can make logistic regression fail or produce huge ORs; use regularized logistic regression or reduce variables if needed.
- Missing clinical variables need explicit imputation rules.
- Clinical variables collected after treatment should not be used for pre-treatment prediction unless the task explicitly allows post-treatment information.


## Set123 Clinical Workflow - 2026-07-08

This section records the latest completed clinical-variable workflow after adding `set_3` and moving from the old set12 train/internal split to the set123 train/internal/external design.

### Current Inputs And Outputs

Primary notebook:

```text
/host/d/Github/Osteosarcoma/clinical/clinical_features.ipynb
```

Current patient/split files:

```text
/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set123.xlsx
/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set123_5fold_prognosis_random{0,10,20,30,40}.xlsx
```

The file name still contains `5fold`, but the current training CV is 4-fold. Fold meanings are:

```text
train: fold 0,1,2,3
internal test: fold 4
external test: fold 5
```

Current case counts:

```text
total = 351
train = 184, with 46 cases per training fold
internal test = 98
external test = 69
```

Current processed clinical outputs:

```text
/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set123_clinical_variables.xlsx
/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set123_clinical_variables_processed.xlsx
/host/e/D/Data/Habitats/Jishuitan/Patient_lists/clinical_baseline_table_prognosis.xlsx
/host/d/projects/Habitats/radiomics/clinical_variables/clinical_univariate_logistic_prognosis.xlsx
/host/d/projects/Habitats/radiomics/clinical_variables/clinical_multivariate_logistic_prognosis.xlsx
/host/d/projects/Habitats/radiomics/clinical_variables/clinical_variables_raw.xlsx
/host/d/projects/Habitats/radiomics/clinical_variables/clinical_variables_normalized.xlsx
```

### Clinical Feature Construction

Derived tumor features are calculated from original-space `label.nii.gz`, not resampled labels:

```text
Tumor_AP_diameter_mm
Tumor_longitudinal_diameter_mm
Tumor_transverse_diameter_mm
Tumor_volume_mm3
```

Use the original label header/spacing to convert voxel extents/volume into physical units.

Clinical missingness rule:

```text
drop variables with missing percentage > 40%
impute remaining missing values with KNN imputation, n_neighbors = 3
```

Sex and lesion site are translated to English. Current lesion-site grouping:

```text
Femur
Tibia and fibula
Others
```

This grouping must be idempotent: an already grouped `Tibia and fibula` remains `Tibia and fibula`.

### Baseline Table Rule

Baseline is a paper-style label comparison table within each dataset split. It should not compare train vs internal/external directly.

For each split:

```text
training: label 0 vs label 1
internal_test: label 0 vs label 1
external_test: label 0 vs label 1
```

The output columns should include all three groups:

```text
training_label0, training_label1, training_p_value, training_p_method
internal_test_label0, internal_test_label1, internal_test_p_value, internal_test_p_method
external_test_label0, external_test_label1, external_test_p_value, external_test_p_method
```

Continuous variables:

- summarize as median [IQR] plus mean +/- SD when useful;
- use Shapiro normality checks;
- use Welch t-test if both label groups are normal;
- otherwise use Mann-Whitney U test.

Categorical variables:

- summarize as n (%);
- use a single global multi-category p value for a parent categorical variable such as `Tumor location`;
- use chi-square when expected counts are adequate, otherwise Fisher/exact-style fallback where implemented.

### Univariate And Multivariate Logistic Regression

Univariate/multivariate logistic regression uses only the training split:

```text
train_df = clinical_split_df[clinical_split_df["split"] == "train"]
```

This means internal and external tests do not influence paper-style OR/p-value reporting.

Univariate:

- fit one clinical variable at a time;
- report coefficient, OR, 95% CI, p value, and convergence/fit status;
- use the user-defined `p_threshold` to choose candidates for multivariate analysis.

Multivariate:

- fit selected univariate candidates together;
- report adjusted OR, 95% CI, and p value;
- if no variables pass the threshold or the fit is unstable, report that explicitly rather than forcing a model.

For modeling multi-category lesion site:

- use dummy variables with `Others` as the implicit reference;
- current model terms are `Lesion_site_Femur` and `Lesion_site_Tibia_and_fibula`.

### ML-Ready Clinical Table

The final normalized clinical matrix should look like a radiomics feature table:

```text
Patient_set
Patient_index
Image_filepath
Mask_filepath
Prognosis_label
Pathologic_label
split
fold
<normalized clinical feature columns>
```

Use this file as the input for clinical ML:

```text
/host/d/projects/Habitats/radiomics/clinical_variables/clinical_variables_normalized.xlsx
```

For categorical variables:

- binary sex is encoded numerically/dummy-style;
- lesion site is one-hot/dummy encoded with `Others` as reference;
- do not include an explicit `Others` model column unless the user changes the reference-category decision.

## Literature Variable Mapping - Li 2026 and Wang 2025

Recorded after comparing the project cohort table `/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12.xlsx` with clinical variables from:

- Li 2026 osteosarcoma recurrence habitat paper: `Magnetic Resonance Imaging - 2026 - Li - Characterization of Intratumoral Heterogeneity via MRI-Based Radiomic Habitats ...`
- Wang 2025 spinal tumor PFS paper: `wang-et-al-2025-mri-based-habitat-analysis-for-the-prediction-of-progression-free-survival-in-primary-spinal-tumors`

### Li 2026 Osteosarcoma Recurrence Paper

This is the closest paper for our binary osteosarcoma clinical model.

| Literature variable | Chinese | Meaning | Project availability |
|---|---|---|---|
| Age | 年龄 | Patient age at diagnosis/visit | Available: `Age` |
| Gender | 性别 | Male/female | Available: `Sex` |
| Tumor location | 肿瘤部位 | Bone/site involved, such as femur, tibia, humerus | Available: `Lesion_site` |
| Pathological type | 病理类型 | Osteosarcoma pathological subtype, e.g. osteoblastic | Not directly available. Do not substitute `Pathologic_label`; that is an outcome label, not subtype. |
| Hemoglobin | 血红蛋白 | Blood/anaemia marker | Available: `HGB (g/L)` |
| Alkaline phosphatase / ALP | 碱性磷酸酶 | Bone metabolism marker, commonly relevant in bone tumors | Available: `ALP (IU/L)` |
| Lactate dehydrogenase / LDH | 乳酸脱氢酶 | Tissue injury/tumor burden marker | Available: `LDH (IU/L)` |
| Calcium | 血钙 | Electrolyte/bone metabolism marker | Not available |
| Sodium | 血钠 | Electrolyte | Not available |
| Potassium | 血钾 | Electrolyte | Not available |
| Surgical SSS stage | 外科分期 / SSS 分期 | Surgical staging system for bone tumors | Not available |
| Tumor anteroposterior diameter | 肿瘤前后径 | One physical tumor diameter along the anteroposterior direction | Similar manual columns available: `Length`, `Width`, `Height`; can also derive objective label-based size features from `label.nii.gz` |
| Tumor longitudinal diameter | 肿瘤纵径 | Long-axis tumor diameter | Similar manual columns available: `Length`, `Width`, `Height`; can also derive objective label-based size features from `label.nii.gz` |
| Tumor transverse diameter | 肿瘤横径 | Transverse tumor diameter | Similar manual columns available: `Length`, `Width`, `Height`; can also derive objective label-based size features from `label.nii.gz` |
| Type of bone destruction | 骨破坏类型 | Lytic, osteoblastic, mixed, etc. | Not available |
| Periosteal reaction | 骨膜反应 | Imaging sign of periosteal stimulation by tumor | Not available |
| Peritumoral edema | 瘤周水肿 | Edema around tumor in soft tissue/bone marrow | Not available |
| Pathological fracture | 病理性骨折 | Fracture caused by tumor-weakened bone | Available: `Pathologic_fracture` |
| Vascular invasion | 血管侵犯 | Tumor involvement of vessels | Not available |
| Joint invasion | 关节侵犯 | Tumor invasion of adjacent joint | Not available |

### Wang 2025 Spinal Tumor PFS Paper

This paper is survival/PFS oriented and spinal-tumor specific. It is useful as a clinical-variable reference, but many variables should not be forced into the osteosarcoma cohort.

| Literature variable | Chinese | Meaning | Project availability |
|---|---|---|---|
| Sex | 性别 | Male/female | Available: `Sex` |
| Age | 年龄 | Patient age | Available: `Age` |
| Location | 肿瘤部位 | In Wang: cervical/thoracic/lumbar/sacrococcygeal spine location | Similar but not identical: `Lesion_site` is bone/site location in our cohort |
| Gross tumor volume | 总肿瘤体积 | Whole tumor ROI volume | Not directly in table; can derive from `label.nii.gz` as voxel count multiplied by voxel volume |
| Enneking stage | Enneking 分期 | Bone tumor surgical/aggressiveness stage | Not available |
| Surgical method | 手术方式 | e.g. total vertebral resection vs intralesional resection | Not available |
| Extent of resection | 切除范围 | Gross total resection vs subtotal resection | Not available |
| Vertebral compression | 椎体压缩 | Vertebral body compression caused by spinal tumor | Not available and mostly spinal-specific |
| Multivertebral involvement | 多椎体受累 | Involvement of multiple vertebrae | Not available and mostly spinal-specific |
| ICD grade | ICD 分级 | Intermediate vs malignant tumor grade based on ICD coding | Not available |
| Histologic classification | 组织学分类 | Specific histologic tumor type | Not available |
| Bilsky grade | Bilsky 分级 | Epidural spinal cord compression grade | Not available and spinal-specific |
| Paravertebral tumor volume | 椎旁肿瘤体积 | Paraspinal soft-tissue tumor volume | Not available and spinal-specific |
| Chemotherapy | 化疗 | Whether chemotherapy was given | Not available |
| Radiotherapy | 放疗 | Whether radiotherapy was given | Not available |

### Label-Derived Tumor Size Features

The user noted that tumor diameters and tumor volume can be calculated from `label.nii.gz`. This is agreed and should be part of the clinical/shape-size branch.

Recommended label-derived size features:

- `Label_voxel_num`: number of foreground tumor voxels.
- `Label_tumor_volume_mm3`: `foreground_voxel_count * spacing_x * spacing_y * spacing_z`.
- `Label_bbox_x_mm`, `Label_bbox_y_mm`, `Label_bbox_z_mm`: physical bounding-box extents from the tumor mask and image spacing.
- `Label_bbox_max_diameter_mm`: maximum of the three bbox extents.
- Optional: `Label_bbox_product_mm3`: bbox_x * bbox_y * bbox_z, a crude size envelope, not true tumor volume.

Caution:

- AP/longitudinal/transverse diameters are anatomical directions. The NIfTI array axes may not always correspond exactly to AP/longitudinal/transverse directions unless orientation is carefully handled.
- For ML prediction, label-derived physical bbox extents and tumor volume are objective and reproducible, even if they are named generically rather than anatomically.
- Compute these from the original-space label with correct header spacing/affine, not from a resampled mask unless the goal is explicitly resampled-space size.

### Minimal First-Pass Variables Based on Li 2026 + Available Project Data

Use these for a clean, literature-aligned first version:

- `Age`
- `Sex`
- `Lesion_site`
- `Pathologic_fracture`
- `HGB (g/L)`
- `ALP (IU/L)`
- `LDH (IU/L)`
- label-derived tumor size/volume features from `label.nii.gz`

Manual size columns `Length`, `Width`, and `Height` can be retained for comparison, but the label-derived measurements may be more reproducible.

## Current User-Defined Clinical Variable Pool - 2026-06-27

The current clinical-variable modeling pool should use the following variables:

| Variable | Source / handling | Notes |
|---|---|---|
| `Age` | Excel clinical table | continuous |
| `Sex` | Excel clinical table | categorical/binary encoding |
| `Lesion_site` | Excel clinical table | categorical; one-hot encode |
| `Pathologic_fracture` | Excel clinical table | categorical/binary encoding |
| `Height_at_visit` | Excel clinical table | continuous |
| `Weight_at_visit` | Excel clinical table | continuous |
| `BMI` | derived from height/weight | `weight_kg / height_m^2` |
| `WBC` | Excel clinical table | continuous lab variable |
| `HGB` | Excel clinical table | continuous lab variable |
| `PLT` | Excel clinical table | continuous lab variable |
| `CRP` | Excel clinical table | continuous lab variable |
| `ALP` | Excel clinical table | continuous lab variable |
| `Total_cholesterol` | Excel clinical table | continuous lab variable |
| `Triglycerides` | Excel clinical table | continuous lab variable |
| `LDL` | Excel clinical table | continuous lab variable |
| `LDH` | Excel clinical table | continuous lab variable |
| `PT` | Excel clinical table | continuous coagulation variable |
| `APTT` | Excel clinical table | continuous coagulation variable |
| `Fibrinogen` | Excel clinical table | continuous coagulation variable |
| `D-dimer` | Excel clinical table | continuous coagulation variable |
| Tumor AP diameter | derived from label/image | tumor physical size feature |
| Tumor longitudinal diameter | derived from label/image | tumor physical size feature |
| Tumor transverse diameter | derived from label/image | tumor physical size feature |
| Tumor volume | derived from label/image | foreground voxel count * voxel volume |

Implementation notes:

- This user-defined pool supersedes the earlier minimal literature-aligned set for coding.
- `Side` is intentionally not included in the current pool.
- Manual table columns `Length`, `Width`, and `Height` may be useful for comparison, but the current requested derived size variables should preferably come from `label.nii.gz` and image spacing/orientation when possible.
- For ML implementation, continuous variables need imputation/scaling; categorical variables need encoding; BMI and tumor-size features should be calculated before model fitting.

## Lesion Site Grouping Rule - 2026-06-27

Current cohort counts after English translation are dominated by three sites:

- Femur: 175
- Tibia: 94
- Humerus: 36

Rare sites including Fibula, Radius, Pelvis, Ilium, Ulna, Scapula, and Popliteal fossa should be collapsed into `Others`.

Implementation convention:

- processed clinical table: `Lesion_site` contains `Femur`, `Tibia`, `Humerus`, or `Others`
- baseline table: show four rows: `Lesion_site_Femur`, `Lesion_site_Tibia`, `Lesion_site_Humerus`, `Lesion_site_Others`
- uni/multi logistic and clinical ML matrix: encode only `Lesion_site_Femur`, `Lesion_site_Tibia`, and `Lesion_site_Humerus`; do not create `Lesion_site_Others`, because `Others` is the implicit reference group

## Current Lesion Site Grouping Rule - Updated 2026-06-28

For the current clinical-variable implementation, `Lesion_site` should be translated to English and grouped as:

- `Femur`
- `Tibia and fibula` (combines Tibia and Fibula)
- `Others` (all remaining sites, including Humerus, Radius, Pelvis, Ilium, Ulna, Scapula, Popliteal fossa, etc.)

Baseline tables should use paper-style rows: one parent row `Tumor location` with a single global multi-category p value, followed by the three category rows with n (%). Logistic regression / ML-ready matrices should use dummy-category terms with `Others` as the reference group: `Lesion_site_Femur` and `Lesion_site_Tibia_and_fibula`.

The grouping helper must be idempotent: rerunning it on an already grouped value `Tibia and fibula` must keep it as `Tibia and fibula`, not collapse it into `Others`.

