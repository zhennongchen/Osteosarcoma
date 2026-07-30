---
name: habitat-ensemble-fusion
description: Use when designing, explaining, or implementing multimodal ensemble fusion for the Osteosarcoma/Habitats project, especially fusing whole-tumor radiomics, habitat radiomics, clinical models, and future deep learning models using SoftVote-EQ, HardVote, or OOF stacking meta-learners.
---

# Habitat Ensemble Fusion

Use this skill when the user asks how to combine multiple model streams in the Habitat/Osteosarcoma project, including whole-image radiomics, habitat radiomics, clinical models, and future DL models.

Core reference: `references/fusion.md`.

## Preferred Fusion Level

Fuse at the probability level unless the user explicitly asks to concatenate raw features.

Each base learner should output a calibrated or at least comparable positive-class probability for each case, for example:

```text
p_whole_radiomics
p_habitat_radiomics
p_dl
p_clinical
```

These probabilities become either:

- direct inputs to voting rules, or
- meta-features for stacking.

## Methods To Consider

1. **SoftVote-EQ**: arithmetic mean of base learner probabilities.
2. **HardVote**: binarize each probability at 0.5, then majority vote; resolve ties by SoftVote-EQ.
3. **Stacking**: train a second-level meta-learner on base learner probabilities.

For stacking, always protect validation integrity with out-of-fold (OOF) predictions on the training set. Never train the meta-learner on predictions from base models that were trained on the same cases being predicted.

## When Implementing

Read `references/fusion.md` for formulas, examples, and the recommended training/test procedure.

For this project, likely base learners are:

- whole-tumor radiomics model
- habitat weighted-average radiomics model
- DL model when added
- optional clinical model

For stacking meta-learners, reasonable candidates following the Ki-67 ccRCC paper are RF, AdaBoost, XGBoost, LightGBM, and ExtraTrees. Prefer strong regularization and shallow/controlled models because the meta-feature space is low-dimensional.


## Current Project Implementation

For the current Osteosarcoma/Habitats project implementation, also read `references/project_fusion_implementation.md` before editing `/host/d/Github/Osteosarcoma/fusion` or fusion final-selection notebooks.
