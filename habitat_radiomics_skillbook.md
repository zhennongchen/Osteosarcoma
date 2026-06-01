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
