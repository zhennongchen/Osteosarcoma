---
name: habitat-results-presentation
description: Use when generating, updating, or explaining Results-section figures, tables, final-selection summaries, ROC/DCA/DeLong/KM/SHAP/Grad-CAM plots, ML-method radar plots, or portable HTML reports for the Osteosarcoma/Habitats project.
---

# Habitat Results Presentation

Use this skill for the final Results presentation layer of the Osteosarcoma/Habitats project.

Detailed project-specific plotting and report conventions live in `references/results_presentation.md`. Read that file before editing notebooks under:

```text
/host/d/Github/Osteosarcoma/results_presentation
```

## Core Rules

- Save generated deliverables to `/host/d/projects/Habitats/results`.
- Use Times New Roman for manuscript-style figures.
- Prefer vector PDF for final figures. PNG previews are acceptable only for HTML display assets.
- For portable HTML, use relative paths and store preview images under `results/html_assets`.
- Do not edit text embedded inside source figures/tables unless explicitly requested.
- Main report language is Chinese, while professional terms can remain English, e.g. `radiomics`, `deep learning`, `ROC`, `DCA`, `DeLong`, `SHAP`, `Grad-CAM`.

## Current Results Notebooks

- `demographics.ipynb`: demographics/baseline table and clinical uni/multivariate logistic regression table.
- `AUC_reports.ipynb`: performance heatmap, ROC/DCA curves, DeLong heatmaps.
- `KM_plots.ipynb`: Kaplan-Meier curves, log-rank p-values, hazard ratio with 95% CI.
- `SHAP.ipynb`: fusion-stacking SHAP summary, waterfall candidate selection, waterfall-style plots, MRI overlays.
- `habitats_selection.ipynb`: SC/CH fixed-K selection plot and representative habitat overlay.
- `Grad_CAM.ipynb`: 3D ResNet Grad-CAM representative cases.
- `ML_methods_comparison/ML_methods_comparison.ipynb`: ML-method CV AUC radar plots.
- `results_presentation_html.ipynb`: portable static HTML summary.

## Preferred Workflow

1. Load final-selection outputs from model folders under `/host/d/projects/Habitats/models/Prognosis`.
2. Recompute metrics from saved per-case probabilities when possible, instead of copying numbers manually.
3. Save figures/tables to `/host/d/projects/Habitats/results`.
4. If a report page is needed, regenerate `results_presentation.html` from `results_presentation_html.ipynb`.
5. Keep the notebook modular: each section should be independently runnable.
