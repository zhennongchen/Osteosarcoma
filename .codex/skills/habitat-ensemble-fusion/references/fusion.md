# Ensemble Fusion Notes

This note records the multimodal fusion logic discussed from:

`/host/d/projects/Habitats/papers/Integration of multi-scale radiomics and deep learning for Ki-67 prediction in clear cell renal carcinoma.pdf`

The paper builds unimodal models first, then combines their predicted probabilities. It tested voting-based ensembles and stacking ensembles. Its final best fusion model was an ExtraTrees stacking model using probabilities from top-performing base learners such as Habitat, PERI1mm, DL_2D, and Clinical models.

## Inputs

For one patient/case, each base learner outputs a positive-class probability:

```text
p_whole = whole-tumor radiomics probability
p_habitat = habitat radiomics probability
p_dl = deep learning probability
p_clinical = clinical model probability, if available
```

Example with three models:

```text
p_whole = 0.3
p_habitat = 0.5
p_dl = 0.8
```

Example with four models:

```text
p_habitat = 0.50
p_peri_or_whole = 0.30
p_dl = 0.80
p_clinical = 0.40
```

## SoftVote-EQ

Equal-weight soft voting computes the arithmetic mean of all base learner probabilities.

Formula:

```text
p_final = mean([p_1, p_2, ..., p_m])
```

Three-model example:

```text
p_final = (0.3 + 0.5 + 0.8) / 3 = 0.5333
```

If the classification threshold is 0.5:

```text
0.5333 >= 0.5 -> positive class
```

Properties:

- no extra training
- every base model has equal weight
- cannot learn that one model is more reliable than another
- useful baseline and tie-breaker for HardVote

## HardVote

Hard voting first binarizes each base probability using a 0.5 threshold, then takes the majority class.

Rule:

```text
vote_i = 1 if p_i >= 0.5 else 0
final_class = majority(vote_1, vote_2, ..., vote_m)
```

Three-model example:

```text
p_whole = 0.3 -> 0
p_habitat = 0.5 -> 1
p_dl = 0.8 -> 1

votes = [0, 1, 1]
final_class = 1
```

Four-model tie example:

```text
p_habitat = 0.50 -> 1
p_peri_or_whole = 0.30 -> 0
p_dl = 0.80 -> 1
p_clinical = 0.40 -> 0

votes = [1, 0, 1, 0]
```

This is a tie. The paper resolves ties with the SoftVote-EQ average:

```text
p_soft = (0.50 + 0.30 + 0.80 + 0.40) / 4 = 0.50
```

Then apply the project's threshold rule. Be explicit in code about whether `0.5` itself is positive:

```text
p >= 0.5 -> positive
```

or

```text
p > 0.5 -> positive
```

Prefer `>= 0.5` unless the user chooses otherwise.

## Stacking

Stacking trains a second-level model, called a meta-learner, using base learner probabilities as input features.

For one case:

```text
X_meta = [p_whole, p_habitat, p_dl]
```

The meta-learner outputs:

```text
p_final = meta_model.predict_proba(X_meta)[:, 1]
```

Using the example:

```text
X_meta = [0.3, 0.5, 0.8]
```

The stacking output is not necessarily the average `0.5333`. It might be 0.72, 0.41, or another value, depending on what the meta-learner learned. Stacking can learn interactions such as:

- DL high but habitat low may not always mean high risk.
- habitat and whole-radiomics both high may be more reliable than either alone.
- clinical high but imaging low may deserve lower confidence.

The ccRCC Ki-67 paper evaluated RF, AdaBoost, XGBoost, LightGBM, and ExtraTrees as meta-learners. ExtraTrees stacking performed best in that paper.

## OOF Training To Prevent Leakage

Do not train the meta-learner using base model predictions from models that already saw the same cases during training.

Use out-of-fold predictions on the training set:

1. Split the training set into 5 stratified folds.
2. For fold 1, train each base model on folds 2-5 and predict probabilities for fold 1.
3. For fold 2, train each base model on folds 1,3,4,5 and predict fold 2.
4. Continue until every training case has OOF probabilities.
5. Concatenate OOF predictions into a training matrix for the meta-learner.

Meta-training table example:

```text
Patient | p_whole_oof | p_habitat_oof | p_dl_oof | Label
1       | 0.31        | 0.52          | 0.77     | 1
2       | 0.18        | 0.22          | 0.41     | 0
3       | 0.67        | 0.74          | 0.60     | 1
```

Then train:

```text
meta_model.fit(X_meta_oof, y_train)
```

## Validation/Test Prediction Procedure

After the meta-learner is trained:

1. Retrain each base learner on the entire training set.
2. Generate base probabilities for validation/test cases.
3. Stack these probabilities into `X_meta_test`.
4. Apply the trained meta-learner.

Example:

```text
base probabilities for a test case:
p_whole = 0.3
p_habitat = 0.5
p_dl = 0.8

X_meta_test = [0.3, 0.5, 0.8]
p_final = meta_model.predict_proba(X_meta_test)[:, 1]
```

## Project Guidance

For the Osteosarcoma/Habitats project, start with probability-level fusion:

```text
whole-image radiomics model
habitat weighted-average radiomics model
future DL model
optional clinical model
```

Recommended reporting:

- compare each base model alone
- SoftVote-EQ
- HardVote
- stacking with at least one simple regularized meta-learner
- optional comparison among RF/Ada/XGB/LGBM/EXT meta-learners

Important safeguards:

- keep train/validation/test separation intact
- generate OOF probabilities only within the training set
- do not let validation/test labels affect base model selection or meta-learner fitting
- save per-case base probabilities and final probabilities for auditability
- when using a 0.5 threshold, state whether equality maps to positive
