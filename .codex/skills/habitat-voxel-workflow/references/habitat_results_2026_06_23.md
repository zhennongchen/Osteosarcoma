# Habitat Model Result Memory - Prognosis - 2026-06-23

Source directory:

`/host/d/projects/Habitats/models/Prognosis`

Summary files read:

- `/host/d/projects/Habitats/models/Prognosis/habitats_individual/habitat_model_summary.xlsx`
- `/host/d/projects/Habitats/models/Prognosis/habitats_avg/habitat_model_summary.xlsx`
- `/host/d/projects/Habitats/models/Prognosis/habitats_sum/habitat_model_summary.xlsx`

Interpretation note:

- `cv_better_auc` is the best CV AUC according to the current CV mode selection (`together` vs `mean`).
- `test_final_auc` is the internal-test AUC according to the selected test method (`mean`, `best`, or `alldata`).
- The setting with best CV AUC and the setting with best internal-test AUC may differ.
- For choosing among the three habitat strategies, internal-test `test_final_auc` is the more relevant ranking criterion.

## habitats_individual

Older branch: patient-specific K selected per case.

Best CV AUC:

- AUC: `0.7962121212`
- classifier/model: `SVM / Linear SVM`
- experiment: `random30_rfe_top20`
- selected features: `20`
- CV mode: `mean`
- same setting internal-test final AUC: `0.7491254373`

Best internal-test AUC:

- AUC: `0.8350824588`
- classifier/model: `SVM / Linear SVM`
- experiment: `random60_rfecv`
- selected features: `28`
- CV AUC for this setting: `0.7289720696`
- test selected method: `best`
- best fold model: `fold 4`

## habitats_avg

Fixed-K branch using weighted-average habitat radiomics.

Best CV AUC:

- AUC: `0.7986555112`
- classifier/model: `SVM / Linear SVM`
- experiment: `random30_rfe_top17`
- selected features: `17`
- CV mode: `mean`
- same setting internal-test final AUC: `0.7331334333`

Best internal-test AUC:

- AUC: `0.7661169415`
- classifier/model: `SVM / Linear SVM`
- experiment: `random30_rfe_top30`
- selected features: `30`
- CV AUC for this setting: `0.7708926490`
- test selected method: `best`
- best fold model: `fold 2`

## habitats_sum

Fixed-K branch using summed habitat radiomics.

Best CV AUC:

- AUC: `0.7907321845`
- classifier/model: `SVM / Linear SVM`
- experiment: `random15_rfe_top30`
- selected features: `30`
- CV mode: `mean`
- same setting internal-test final AUC: `0.7881059470`

Best internal-test AUC:

- AUC: `0.8173413293`
- classifier/model: `SVM / Linear SVM`
- experiment: `random30_rfe_top30`
- selected features: `30`
- CV AUC for this setting: `0.7631816101`
- test selected method: `best`
- best fold model: `fold 3`

## Ranking

By best CV AUC:

1. `habitats_avg`: `0.7986555112`
2. `habitats_individual`: `0.7962121212`
3. `habitats_sum`: `0.7907321845`

By best internal-test AUC:

1. `habitats_individual`: `0.8350824588`
2. `habitats_sum`: `0.8173413293`
3. `habitats_avg`: `0.7661169415`

Current conclusion:

- If ranking by locked internal-test performance, `habitats_individual` is the best of the three.
- `habitats_sum` is close and performs clearly better than `habitats_avg` on internal test.
- `habitats_avg` has the highest peak CV AUC, but its best internal-test AUC is the lowest among the three, so it is not preferred as the final habitat strategy based on generalization.
