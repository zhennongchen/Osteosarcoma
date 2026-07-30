# Osteosarcoma Results Presentation Reference

## Output Root

All final manuscript/summary assets are saved in:

```text
/host/d/projects/Habitats/results
```

Portable HTML assets are stored in:

```text
/host/d/projects/Habitats/results/html_assets
```

The HTML report should reference files relatively so the whole `results` folder can be copied to another computer and opened by double-clicking the HTML.

## Final Model Families

The current manuscript-level model families are:

- `clinical`
- `C-radiomics` = whole-image radiomics
- `H-radiomics` = habitat radiomics, especially `habitats_sum`
- `DL_3D` = deep-learning features followed by ML
- `fusion_soft_vote`
- `fusion_stacking`

When writing legends or tables, use compact names where useful:

```text
Clinical
C-radiomics
H-radiomics
DL_3D
Soft_vote
stacking
```

## Final-Selection Source Pattern

Most model-family folders have a `final_selections` subfolder containing selected per-case probabilities for CV/train/internal/external cohorts. Use these rather than raw experiment folders when presenting final results.

For `dl_2d_ml` and `dl_3d_ml`, final-selection notebooks allow separate probability modes for:

- CV: `cv`, `allotherdata`, `final`, or `advanced`
- internal test: `final`, `mean`, `alldata`, or `best`
- external test: `final`, `mean`, `alldata`, or `best`

For clinical/whole-image/habitats final-selection notebooks, defaults at the time of writing were:

- CV default: `final_advanced`
- internal/external test default: `final`

For final manuscript display, always inspect the notebook/settings used for the generated final-selection file.

## Performance Heatmap

Notebook: `AUC_reports.ipynb` section 1.

Rows:

- cohorts: `train`, `internal test`, `external test`
- models: Clinical, C-radiomics, H-radiomics, DL_3D, fusion_soft_vote, fusion_stacking

Columns:

- Cohort
- Model
- Algorithm
- 95% CI for AUC
- AUC
- Accuracy
- Sensitivity
- Specificity

Rules:

- Round metrics to 3 decimals.
- Algorithm is the dominant/final selected algorithm from CV selection. If multiple settings are fused, use the most frequent algorithm; for soft vote use `N/A`.
- Save as `performance_heatmap.pdf`.

## ROC and DCA Curves

Notebook: `AUC_reports.ipynb` section 2.

Generate:

- individual ROC PDFs for train/internal/external
- individual DCA PDFs for train/internal/external
- combined `ROC_DCA_combined.pdf` in a 2 x 3 layout

ROC legends should include `Model: AUC xxx`.
DCA legends should include model names only, not AUC.

DCA interpretation: a model is useful across threshold probabilities where its net benefit curve is above treat-all and treat-none, and preferably above competing models.

## DeLong Heatmaps

Notebook: `AUC_reports.ipynb` section 3.

Generate:

- train DeLong heatmap
- internal-test DeLong heatmap
- external-test DeLong heatmap
- combined `DeLong_combined.pdf` in a 1 x 3 layout

Style:

- Lower triangle contains pairwise p-values.
- Upper triangle and diagonal are set to 1.
- Colormap follows the referenced paper: low p-values are dark/purple, p=1 is yellow.
- Use white text on dark cells, black text on yellow/light cells.
- Include a colorbar for each individual plot and one shared colorbar for the combined plot.

## Kaplan-Meier Curves

Notebook: `KM_plots.ipynb`.

Inputs:

- Raw label tables for hospital/imaging date and follow-up/event date.
- `Prognosis_label`: label 0 is censored; label 1 is an event.
- If an event date is earlier than hospital date, configurable handling defaults to swapping dates.
- Use a 36-month horizon because the prognosis label was treated as a 3-year event endpoint.

Risk grouping:

- Use model final probability.
- High/low split uses the threshold maximizing sensitivity + specificity.

Plots:

- Six individual KM curves, one per final model family.
- One `KM_combined.pdf` with 2 x 3 layout.
- Each subplot includes internal low/high and external low/high risk curves.
- Include log-rank p-values for internal and external cohorts.
- Include hazard ratio as `HR xx.xx (ll.ll, uu.uu)` with superscript `*` if HR p<0.05.

## SHAP Summary and Waterfall

Notebook: `SHAP.ipynb`.

Current SHAP is for fusion stacking, especially when final stacking is RF/KNN or another black-box model. Use model-agnostic SHAP if the stacking algorithm lacks a native TreeExplainer path or if consistency is preferred.

SHAP summary:

- Use training data for global SHAP interpretation.
- Inputs are four modality probabilities: Clinical, C-radiomics, H-radiomics, DL_3D.
- Save as `fusion_stacking_SHAP_summary.pdf`.

Waterfall candidates:

- Candidate table: `individual_waterfall_candidates.xlsx`.
- Positive and negative representative examples can be selected by SHAP direction.
- Current filtering preference included cases where modality effects follow:

```text
DL_3D >= H-radiomics > C-radiomics > Clinical
```

Waterfall-style plots:

- Positive example used `set_2`, patient `92`.
- Negative example used `set_2`, patient `107`.
- Use Times New Roman, black labels, no gray text.
- Y-axis shows model names only, not feature values.
- Place bold `E[f(X)]` and `f(x)` above the horizontal axis on the same level.

MRI overlays for waterfall examples:

- Select the slice with largest tumor area.
- Overlay tumor label in red with transparency.
- No text/title on image.
- Positive `set_2/92`: WL=1200, WW=2500.
- Negative `set_2/107`: WL=216, WW=433.

## Habitat Selection Figure

Notebook: `habitats_selection.ipynb`.

Data source:

```text
/host/d/projects/Habitats/radiomics/habitats/cohort_mean_K2_K9_silhouette_CH_scores.xlsx
```

Plot:

- x-axis: number of clusters K, usually 2-9.
- Left y-axis: silhouette coefficient (SC).
- Right y-axis: Calinski-Harabasz (CH) index.
- Use separate axis ranges because SC and CH have very different scales.
- K=4 was selected from the elbow-style interpretation of SC/CH curves.
- Save as `habitat_K_selection_SC_CH.pdf`.

Representative habitat overlay:

- Case: `set_1/1`.
- Slice: index 13, zero-based.
- Raw image WL=250, WW=500.
- Figure 1: original image with tumor label overlay.
- Figure 2: center-cropped image around tumor with habitat labels overlaid.
- Apply vertical flip if needed for display consistency.
- Save as `habitat_representative_set_1_1_slice13.pdf`.

## Grad-CAM

Notebook: `Grad_CAM.ipynb`.

Current model example:

```text
/host/d/projects/Habitats/models/Prognosis/resnet18_3D_FTall_AUGfull_96x96x64_nomed_adam/random0_all_fold56/models/model-12.pt
```

Representative cases:

- Positive: `set_2/58`
- Negative: `set_1/145`

Figure:

- 1 x 3 layout: MRI with tumor mask overlay, Grad-CAM heatmap, MRI + Grad-CAM overlay.
- Use the most informative slice, generally near the maximal tumor slice or maximal CAM-over-tumor slice.
- Colorbar ticks every 0.2 from 0 to 1.
- No title/text unless requested.

Interpretation caution:

- For a negative case, the heatmap should still ideally focus on tumor-relevant anatomy if the model is using clinically meaningful information. A heatmap away from the tumor suggests shortcut learning or poor localization.

## ML Methods Radar Plots

Notebook: `ML_methods_comparison/ML_methods_comparison.ipynb`.

Purpose: compare CV AUC among RF, LR, XGBoost, and SVM inside each model family.

Current model families:

- Clinical
- C-radiomics
- H-radiomics
- DL_3D
- fusion_stacking

Current design:

- One radar plot per model family; do not overlay all families in one radar if differences are too compressed.
- Each plot uses its own radial AUC range, tightened around that model family’s four AUCs.
- Save five individual PDFs under `results/ML_methods_comparison`.
- Add these figures to the HTML report in a compact grid.

## Portable HTML Report

Notebook: `results_presentation_html.ipynb`.

Output:

```text
/host/d/projects/Habitats/results/results_presentation.html
```

Main title:

```text
多尺度 radiomics 与 deep learning 融合用于骨肉瘤三年预后预测
```

Language:

- Chinese for explanatory text.
- Keep professional terms in English.
- Do not translate embedded figure/table text.

Current sections:

1. Cohort split
2. Methods
3. Results
   - demographics
   - clinical univariate/multivariate logistic regression
   - performance heatmap
   - ROC/DCA
   - DeLong
   - KM survival analysis
   - SHAP/waterfall/MRI overlays
   - ML methods comparison

For HTML table display:

- Convert NaN to blank.
- Mark p<0.05 in red for demographics and clinical logistic tables.
- Drop unnecessary columns such as `comparison_or_unit` from the clinical logistic display when requested.
- Keep layout compact; avoid oversized figures.
