---
name: habitat-voxel-workflow
description: Use when working on the Osteosarcoma voxel-based habitat pipeline, including PyRadiomics voxel feature maps, per-case normalization, block_reduce downsampling, K-means/SC selection, original-space habitat masks, per-habitat radiomics, weighted-average habitat radiomics, habitat feature selection, and current whole-image/habitat ML scripts with fixed train/internal-test split, all-330 feature selection, CV/test modes, resume logic, and summarize/main shell workflows.
---

# Habitat Voxel Workflow

For Osteosarcoma voxel-based habitat questions, read `references/workflow.md` first. This skill captures the agreed workflow and naming decisions from the completed `step1_make_habitat.ipynb` pipeline.

For the current ML redesign, read `references/ml_redesign_2026_06.md` when the request involves model scripts, patient splits, `main.sh`, CV/test metrics, resume behavior, feature-selection scope, or summaries.

Primary project notebook: `/host/d/Github/Osteosarcoma/habitats/step1_make_habitat.ipynb`.

Related literature/path skill: `habitat-papers`.
Related probability-level fusion skill: `habitat-ensemble-fusion`.
