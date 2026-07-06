---
name: habitat-voxel-workflow
description: Use when working on the Osteosarcoma voxel-based habitat pipeline, including PyRadiomics voxel feature maps, per-case normalization, block_reduce downsampling, individual-K SC selection, paper-style cohort-level SC/CH fixed-K selection, original-space habitat masks, per-habitat radiomics, weighted-average habitat radiomics, habitat feature selection, and current whole-image/habitat/DL-feature ML scripts, fixed train/internal-test split history, all-330 feature selection, CV/test modes, resume logic, summarize/main shell workflows, final-selection probability fusion notebooks, and the July 2026 planned train/internal/external-test redo.
---

# Habitat Voxel Workflow

For Osteosarcoma voxel-based habitat questions, read `references/workflow.md` first. This skill captures the agreed workflow and naming decisions from the completed `step1_make_habitat.ipynb` pipeline.

For the current ML redesign, read `references/ml_redesign_2026_06.md` when the request involves model scripts, patient splits, `main.sh`, CV/test metrics, resume behavior, feature-selection scope, or summaries.

For the 2026-06-15 paper-style cohort-level K selection update, read `references/workflow.md` section "Step 2 Alternative".

Primary project notebook: `/host/d/Github/Osteosarcoma/habitats/step1_make_habitat.ipynb`.

Related literature/path skill: `habitat-papers`.
Related probability-level fusion skill: `habitat-ensemble-fusion`.
