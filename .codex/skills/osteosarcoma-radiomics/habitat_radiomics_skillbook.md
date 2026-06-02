# Habitat + Radiomics Skillbook (Osteosarcoma Project)

## Project scope
- Task family: MRI-based habitat radiomics for osteosarcoma prognosis / NAC response prediction.
- Data structure currently used: set_1 and set_2 cohorts with per-case folders and image-label pairs.

## Evidence base read in this session
- Habitat-based MRI heterogeneity radiomics for predicting neoadjuvant chemotherapy response in osteosarcoma (+ supplement).
- MRI radiomics prediction of NAC response in osteosarcoma (+ supplement, Chinese paper).
- MRI-based habitat analysis for PFS in primary spinal tumors (+ appendix).
- MRI-based habitat radiomics characterization paper (Li, 2026).
- Multi-scale radiomics + deep learning Ki-67 paper (ccRCC) (+ supplements) for transferable methodology.

## Reusable modeling pipeline
1. Cohort split and external validation
- Keep true external center/temporal validation whenever possible.
- Report train/internal/external separately.

2. MRI preprocessing standards
- Use consistent spacing handling and record scanner heterogeneity.
- Preserve label integrity after interpolation (nearest for labels + binary enforcement).
- Validate shape/spacing consistency and segmentation continuity before feature extraction.

3. ROI and habitat construction
- Start with whole-tumor ROI quality control.
- Habitat partitioning via unsupervised clustering (commonly k-means).
- Typical strategy from papers: 2-4 habitats; one study selected 3 habitats via SC analysis.
- Advanced pattern: nested habitat refinement to isolate highest-risk microregion.

4. Feature extraction and robustness
- Use PyRadiomics with fixed parameter YAML.
- Include original + filtered feature families where justified.
- Perform reproducibility filtering (ICC > 0.75 commonly used).
- Remove unstable/redundant features before modeling.

5. Feature selection and model building
- Common sequence: correlation/robustness filtering -> mRMR or equivalent -> LASSO.
- Evaluate multiple classifiers; SVM repeatedly strong in habitat-radiomics papers.
- Build and compare: conventional radiomics, habitat radiomics, combined model.
- Current A-line whole-image decision update (2026-06-01): use classifier-specific feature selection rather than one shared LASSO-selected feature set for every classifier; each classifier-specific feature set should use top 20 selected features by default.
- Current classifier comparison panel remains: LR, SVM, RF, XGBoost, and KNN.
- Rationale for KNN as the fifth classifier: it represents instance/distance-based learning and is less redundant with LR (linear), SVM (margin-based), RF (bagging tree ensemble), and XGBoost (boosting tree ensemble) than LightGBM or ExtraTrees.
- SVM-specific feature selection uses SVM-RFE with top 20 features; SVM uses a linear kernel.
- Global random_state policy: use random_state = 0 everywhere unless the user explicitly changes it.

6. Evaluation and reporting
- Main metrics: AUC/ROC, calibration, decision curve analysis, external validation performance.
- Statistical comparison: DeLong test for AUC differences.
- Survival tasks: Kaplan-Meier + Cox, report independent risk value of habitat score.
- Interpretability: SHAP to quantify habitat feature contribution.

## Practical lessons to reuse in code tasks
- Avoid key mismatches in merges; normalize Patient_set and Patient_index consistently.
- Keep one-case-one-row logic by enumerating via img.nii.gz path.
- Track unmatched rows explicitly when joining image and label tables.
- Preserve separate IDs: Patient_index vs medical record number vs registration number.
- For batch processing pipelines, skip already-processed outputs and log counts.

## Expected high-value outputs for this repo
- Habitat-only vs conventional vs combined model benchmark tables.
- External validation-focused result sheets.
- QC reports: missing labels, spacing mismatch, discontinuity, merge-unmatched keys.
- Reader2 label availability tables and derived cohort subsets.

## Evidence file paths (updated 2026-06-01)
- /host/d/projects/Habitats/papers/Habitat-based MRI heterogeneity radiomics for predicting neoadjuvant chemotherapy response in osteosarcoma.pdf
- /host/d/projects/Habitats/papers/Habitat-based MRI heterogeneity radiomics for predicting neoadjuvant chemotherapy response in osteosarcoma - supplement.docx
- /host/d/projects/Habitats/papers/Integration of multi-scale radiomics and deep learning for Ki-67 prediction in clear cell renal carcinoma.pdf
- /host/d/projects/Habitats/papers/Integration of multi-scale radiomics and deep learning for Ki-67 prediction in clear cell renal carcinoma_supplements.pdf
- /host/d/projects/Habitats/papers/Magnetic Resonance Imaging - 2026 - Li - Characterization of Intratumoral Heterogeneity via MRI‐Based Radiomic Habitats in.pdf
- /host/d/projects/Habitats/papers/mri预测骨肉瘤新辅助化疗.pdf
- /host/d/projects/Habitats/papers/mri预测骨肉瘤新辅助化疗_supplements.docx
- /host/d/projects/Habitats/papers/wang-et-al-2025-mri-based-habitat-analysis-for-the-prediction-of-progression-free-survival-in-primary-spinal-tumors.pdf
- /host/d/projects/Habitats/papers/wang-et-al-2025-mri-based-habitat-analysis-for-the-prediction-of-progression-free-survival-in-primary-spinal-tumors-appendix.pdf

## Parameter evidence snapshot (only explicit mentions)
- Habitat-based MRI heterogeneity radiomics for predicting neoadjuvant chemotherapy response in osteosarcoma (+ supplement)
- binWidth: 25 (explicit in main text and supplement statement with R=1, binWidth=25).
- LoG sigma: sigma=2.0 mm appears in selected feature names (for example, log-sigma-2-0-mm-3D...).
- normalizeScale: not explicitly reported.
- Integration of multi-scale radiomics and deep learning for Ki-67 prediction in clear cell renal carcinoma (+ supplements)
- binWidth: evaluated at 20, 25, 30; reference bin width = 25; retained robust features by CCC.
- LoG sigma: not explicitly reported as sigma values in extracted text.
- normalizeScale: not explicitly reported.
- Magnetic Resonance Imaging (2026) Li - Characterization of Intratumoral Heterogeneity via MRI-Based Radiomic Habitats
- normalizeScale: not explicitly reported in extracted text.
- binWidth: not explicitly reported in extracted text.
- LoG sigma: not explicitly reported in extracted text.
- mri预测骨肉瘤新辅助化疗 (+ supplements)
- binWidth: 10 (derived image description) and 25 (image setting block with resampled voxel setting).
- LoG sigma: 2 mm, 4 mm, 6 mm (explicit in supplement).
- normalizeScale: not explicitly reported.
- wang-et-al-2025-mri-based-habitat-analysis-for-the-prediction-of-progression-free-survival-in-primary-spinal-tumors (+ appendix)
- normalizeScale: not explicitly reported in extracted text.
- binWidth: not explicitly reported in extracted text.
- LoG sigma: not explicitly reported in extracted text.

## Locked extractor settings (from MR_setting_image.yaml, 2026-06-01)
- file: /host/d/Github/Osteosarcoma/radiomics_settings/MR_setting_image.yaml
- imageType:
- Original: enabled
- LoG sigma: [2.0, 4.0]
- Wavelet: enabled
- setting:
- normalize: true
- normalizeScale: 100
- interpolator: sitkBSpline
- resampledPixelSpacing: [1,1,1]
- binWidth: 25
- voxelArrayShift: 300
- label: 1
- geometryTolerance: 0.0001
- status: user confirmed and locked for current A/B experiments.

## Locked A-line feature-selection and classifier policy (2026-06-01)
- Scope: whole-image radiomics / conventional radiomics branch (A-line), starting from the filtered feature table after ICC and PCC steps.
- Shared early filtering remains: ICC reproducibility filtering -> PCC/correlation redundancy filtering.
- Current primary feature-selection policy: each classifier has its own classifier-specific feature-selection path, rather than all classifiers sharing one LASSO-selected feature set. Use top 20 features for each classifier-specific selected feature table unless the user explicitly changes this.
- LR feature-selection path: LASSO logistic regression is used for LR-specific supervised feature selection.
- SVM feature-selection path: SVM-RFE is used for SVM-specific feature selection; save/use top 20 SVM-RFE features by default. SVM-SFS was tested but performed worse, showing that RFE vs SFS can materially change results.
- RF feature-selection path: RF-RFE is used for RF-specific feature selection; save/use top 20 RF-RFE features by default.
- XGBoost feature-selection path: XGBoost-RFE is used for XGBoost-specific feature selection; save/use top 20 XGBoost-RFE features by default.
- KNN feature-selection path: KNN cannot directly use RFE because it has no coef_ or feature_importances_; use KNN-SFS top 20 features by default.
- SVM kernel policy: SVM uses a linear kernel unless the user explicitly changes it.
- Primary classifier comparison panel: LR, SVM, RF, XGBoost, and KNN.
- KNN rationale: distance/instance-based classifier, chosen for representational diversity.
- Do not replace KNN with LightGBM or ExtraTrees in the primary five-method panel unless a new protocol revision is made; LightGBM is redundant with XGBoost, and ExtraTrees is redundant with RF.
- Global random_state policy: use random_state = 0 everywhere unless the user explicitly changes it.
- Important implementation note: before fitting feature selectors or classifiers, merge radiomics features and labels by Patient_index rather than assuming row order when possible.

## Locked internal tuning and testing policy (2026-06-01)
- Current accepted strategy for A-line ML modeling: select one fixed hyperparameter set for each classifier using the development cohort/current available training data, then run internal 5-fold CV using those fixed hyperparameters.
- Interpretation: internal 5-fold CV AUC is a development-cohort reference estimate and may be optimistic because hyperparameters were selected from the same development data.
- Final generalization evidence should come from an independent testing/external validation cohort that is not used for feature selection, scaler fitting, hyperparameter selection, threshold selection, or model comparison decisions.
- For final testing: refit the selected classifier with the fixed hyperparameters on the full development cohort, then evaluate once on the independent testing cohort.
- Testing data should not be repeatedly queried to tune model settings.

## Current classifier-specific update (2026-06-01)
- The project has moved from a single shared LASSO-selected feature set to classifier-specific feature selection for A-line experiments, with top 20 features as the default selected feature count.
- SVM branch: use SVM-RFE top 20 features for feature selection and linear-kernel SVM for modeling.
- random_state is standardized to 0 across notebooks and models unless the user explicitly changes it.

## Locked top-K feature count (2026-06-01)
- Current A-line classifier-specific feature-selection default: top 20 features.
- LR branch: use LASSO-selected top 20 features for LR-specific modeling when top-K export is needed.
- SVM branch: use SVM-RFE top 20 selected features. SVM-SFS was tried and performed worse.
- RF branch: use RF-RFE top 20 selected features.
- XGBoost branch: use XGBoost-RFE top 20 selected features.
- KNN branch: use KNN-SFS top 20 selected features because KNN does not expose coef_ or feature_importances_ for RFE.
- If RFECV reports a different best_N, keep it as exploratory evidence, but saved classifier-specific feature tables for current experiments should use top 20 unless the user explicitly changes K.

## Whole-image radiomics feature-selection outcome (2026-06-01)
- Whole-image A-line radiomics feature selection has been completed with classifier-specific selectors.
- LR: LASSO logistic regression.
- SVM: RFE-based selection, top 20 features; linear-kernel SVM for modeling unless explicitly changed.
- RF: RFE-based selection, top 20 features.
- XGBoost: RFE-based selection, top 20 features.
- KNN: SFS-based selection, top 20 features, because KNN cannot directly use RFE without coef_ or feature_importances_.
- SVM-SFS was tested and performed worse than SVM-RFE, confirming that RFE vs SFS is an important tunable design choice for future experiments.

