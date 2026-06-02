# Osteosarcoma A-B-C-D-E Experiment Protocol (v1)

Last updated: 2026-06-01
Owner: User + Copilot

## 1) Overall objective
- Build 5 model lines: A (whole-image radiomics), B (single-level habitat radiomics), C (DL-2D/3D), D (clinical), E (fusion).
- Target performance pattern:
- E should outperform A/B/C/D whenever possible.
- B should ideally outperform A.

## 2) Key references used
- Wang et al. 2025: MRI-based habitat analysis for progression-free survival in primary spinal tumors.
- Integration of multi-scale radiomics and deep learning for Ki-67 prediction in clear cell renal carcinoma.

## 3) Data scope and current constraints
- Cohorts: set_1 + set_2.
- Clinical table source: E:/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12.xlsx.
- Clarified canonical preprocessing rule:
- Resample image and mask to isotropic 1x1x1 mm before radiomics extraction.
- A branch (whole-image) runs on this 1x1x1 standardized space.
- B branch (habitat) uses 1x1x1-based patch-wise computation, then k-means to split high-risk and low-risk ROI.
- Any previous original-data efficiency shortcut is treated as historical implementation detail, not default protocol.

## 4) Model line definitions

### A. Whole-image radiomics model
- Step A1: Extract whole-tumor radiomics features.
- Step A2: Standard feature-selection pipeline (stability + redundancy control + supervised selection).
- Locked early feature filtering: ICC reproducibility filtering -> PCC redundancy filtering.
- Updated A-line feature-selection policy (v1.5): each classifier uses its own classifier-specific feature-selection path rather than a single shared LASSO-selected feature set; use top 20 selected features by default.
- LR uses LASSO logistic regression for classifier-specific supervised feature selection.
- SVM uses SVM-RFE top 20 features for feature selection and a linear kernel for modeling unless the user explicitly changes it; SVM-SFS was tested and performed worse.
- RF uses RF-RFE top 20 features for classifier-specific feature selection.
- XGBoost uses XGBoost-RFE top 20 features for classifier-specific feature selection.
- KNN uses KNN-SFS top 20 features because KNN cannot directly use RFE without coef_ or feature_importances_.
- Step A3: Train and compare five representative ML classifiers: LR, SVM, RF, XGBoost, and KNN, each with its corresponding top 20 selected feature set.
- Classifier rationale: LR = linear baseline; SVM = linear margin-based radiomics standard; RF = bagging tree ensemble; XGBoost = boosting tree ensemble; KNN = distance/instance-based method with low redundancy against the other four.
- Global random_state policy: use random_state = 0 unless explicitly changed by the user.
- LightGBM and ExtraTrees are not in the locked primary five-method panel because LightGBM overlaps with XGBoost and ExtraTrees overlaps with RF; they may be used only as supplementary analyses or after a protocol revision.
- Step A4: Select the best method by predefined primary metric.
- A-final definition:
- A-final features = top 20 selected feature set corresponding to the best-performing classifier.
- A-final algorithm = best-performing classifier among LR, SVM, RF, XGBoost, and KNN.

### B. Habitat radiomics model (single-level)
- Step B1: Tumor patchification (single-level only; no nested-2-stage habitat split).
- Step B2: Patch-wise scoring using A-side selected feature space + chosen algorithm logic.
- Step B3: K-means clustering into 2 habitats:
- High-risk ROI
- Low-risk ROI
- Step B4: Rebuild radiomics model inside high-risk ROI:
- Re-extract/aggregate features for high-risk region.
- Re-run feature selection.
- Train model and produce B-final.
- Open decision B-ALG:
- Option 1: Reuse A-final algorithm.
- Option 2: Re-search algorithm inside habitat branch.

### C. Deep learning image model
- Build DL-2D and DL-3D models using tumor image input.
- Architecture and training recipe follow Ki-67 paper spirit but adapted to osteosarcoma task.

### D. Clinical model
- Build an independent clinical-only model using variables from image_label_info_set12.
- Exact variable subset and algorithm to be finalized.

### E. Fusion model
- Combine A/B/C/D outputs into one integrated model.
- Candidate strategies (to discuss later): score-level fusion, stacked meta-learner, hybrid late-fusion.
- Primary goal: E > each single branch.

## 5) Required comparison panel
- Compare at least: A vs B vs C vs D vs E.
- Mandatory checks:
- E compared against A/B/C/D.
- B compared against A.
- Keep same split protocol when comparing models.

## 6) Evaluation policy (v1)
- Use one primary metric consistently (default: AUC).
- Also track secondary metrics (accuracy, sensitivity, specificity, calibration metrics when available).
- Report confidence intervals where feasible.
- Preserve train/validation/test or internal/external boundaries clearly.
- Current A-line accepted tuning/evaluation strategy (v1.4): choose one fixed hyperparameter set per classifier using the development cohort, run internal 5-fold CV with those fixed hyperparameters, then assess final generalization on an independent testing/external validation cohort.
- Internal 5-fold CV AUC should be treated as development-cohort reference performance and may be optimistic because hyperparameters come from the same development data.
- Independent testing/external validation data must not be used for feature selection, scaler fitting, hyperparameter tuning, threshold selection, or model comparison decisions.
- Final testing procedure: refit the chosen final model on the full development cohort using the fixed hyperparameters, then evaluate once on the independent testing cohort.

## 7) Reproducibility and outputs
- Save for each branch:
- selected feature list
- trained model configuration
- per-case predictions
- summary metric table
- Save integrated comparison table for A/B/C/D/E.
- Record random seeds and split files.

## 8) Immediate next implementation order
1. Freeze A-final using classifier-specific feature selection and the five-classifier panel (LR, SVM, RF, XGBoost, KNN).
2. Freeze B decision on algorithm reuse vs re-search.
3. Start C baseline (2D and 3D minimal reproducible training).
4. Define D variable shortlist and baseline model.
5. Implement E fusion and generate final comparison table.

## 9) Change log
- v1: Protocol aligned to user-stated strategy; references and constraints fixed.
- v1.1: Clarified that both A and B branches must be based on isotropic 1x1x1 mm resampling before radiomics extraction; original-data patch shortcut marked non-default.

## 10) Parameter lock (v1.2)
- Source config: /host/d/Github/Osteosarcoma/radiomics_settings/MR_setting_image.yaml
- Locked settings for current A/B runs:
- LoG sigma: [2.0, 4.0]
- binWidth: 25
- normalize: true
- normalizeScale: 100
- resampledPixelSpacing: [1,1,1]
- interpolator: sitkBSpline
- Note: Any future parameter sweep must be versioned as a new protocol revision before use.

- v1.2: Recorded locked extractor settings after user confirmation (LoG sigma [2.0,4.0], binWidth 25).
- v1.3: Locked A-line feature selection as ICC -> PCC -> LASSO logistic regression; locked primary classifier panel as LR, SVM, RF, XGBoost, and KNN.
- v1.4: Recorded accepted A-line tuning/evaluation strategy: fixed hyperparameters from development cohort, internal 5-fold CV as reference performance, and independent testing/external validation as final generalization evidence.
- v1.5: Updated A-line policy to classifier-specific feature selection; standardized random_state=0; locked SVM to linear kernel unless explicitly changed.
- v1.6: Locked current A-line classifier-specific selected feature count to top 20 features by default, including SVM-RFE top20 and XGBoost-RFE top20.
- v1.7: Recorded whole-image radiomics classifier-specific feature-selection outcome: LR=LASSO; SVM/RF/XGBoost=RFE top20; KNN=SFS top20; SVM-SFS tested worse than SVM-RFE.
