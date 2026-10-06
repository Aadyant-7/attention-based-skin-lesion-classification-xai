# S68/S69 metadata with the retained ensemble — closeout

## Decision

**Keep S53 image-only.** Neither fixed metadata adjustment passed the meaningful-improvement gate. No strength, bin, category or threshold search follows. No GPU training or original locked-test evaluation was performed.

## Matched results

|Method|Exploratory accuracy|Macro precision|Macro recall|Macro-F1|Melanoma recall|
|---|---:|---:|---:|---:|---:|
|S53 image only|93.6128%|0.915781|0.865636|0.886857|82.0359%|
|s68_age|93.4797%|0.913917|0.860172|0.882840|80.2395%|
|s69_age_sex_location|93.0805%|0.907911|0.856537|0.877659|80.8383%|

- s68_age: 7 correct predictions gained / 9 lost / net -2; material gate failed.
- s69_age_sex_location: 5 correct predictions gained / 13 lost / net -8; material gate failed.

Material gate fixed before scoring: at least8 net additional correct images, at least+0.005 absolute accuracy, no macro-F1 or melanoma-recall decline. The existing five CNNs, checkpoint selections, equal weights and224px identity preprocessing were unchanged.

## What was fitted

This is an interpretable **train-only metadata likelihood adjustment**, not a jointly trained multimodal neural network or a validation-fitted stacker. Existing out-of-fold training probabilities for all S53 members were unavailable; using in-sample outputs to fit a fusion head would not give a reliable estimate of deployment behavior.

Only age, sex and lesion localization were parsed for the8,512 permitted development images. Original locked-test clinical rows were skipped before parsing. Labels came from the development training manifest. Image/lesion IDs were used for alignment/group exclusions only; diagnosis-confirmation type and diagnosis-derived metadata were excluded as predictors.

- **4,923 distinct training lesions** fitted the conditional categorical likelihood tables.
- **773 training images** were excluded because their lesion appeared in validation; metadata fitting has zero validation-lesion overlap.
- Contradictory categorical metadata images excluded: 0.
- Age bins fixed:0–19,20–39,40–59,60–79,80–120. Sex/localization vocabularies came only from metadata-fit training lesions.
- One record per training lesion, Laplace alpha1 smoothing, fixed adjustment strength0.25.
- Missing/unseen values add no class evidence. Missingness is not a feature, and no validation-driven imputation/binning is fitted.

Formula: `P_new = softmax(log(P_S53) + 0.25 * sum_fields log(P(field | class)))`. Class priors are not multiplied again. The categorical conditional-likelihood construction follows [categorical naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes); the independence assumptions and fixed tempered fusion are approximations, not an exact proven posterior for these CNNs. Age/sex/location correlate with signals already encoded by images, so added likelihoods can double-count information or override a correct image decision.

## Class behavior

|Candidate|Class|Reference recall|Candidate recall|Reference F1|Candidate F1|Net correct|
|---|---|---:|---:|---:|---:|---:|
|s68_age|akiec|0.7143|0.6939|0.7865|0.7727|-1|
|s68_age|bcc|0.9740|0.9610|0.9375|0.9250|-1|
|s68_age|bkl|0.8545|0.8667|0.8785|0.8827|+2|
|s68_age|df|0.7647|0.7647|0.8667|0.8667|+0|
|s68_age|mel|0.8204|0.8024|0.8405|0.8349|-3|
|s68_age|nv|0.9791|0.9801|0.9681|0.9676|+1|
|s68_age|vasc|0.9524|0.9524|0.9302|0.9302|+0|
|s69_age_sex_location|akiec|0.7143|0.6735|0.7865|0.7586|-2|
|s69_age_sex_location|bcc|0.9740|0.9610|0.9375|0.9136|-1|
|s69_age_sex_location|bkl|0.8545|0.8606|0.8785|0.8765|+1|
|s69_age_sex_location|df|0.7647|0.7647|0.8667|0.8667|+0|
|s69_age_sex_location|mel|0.8204|0.8084|0.8405|0.8333|-2|
|s69_age_sex_location|nv|0.9791|0.9752|0.9681|0.9646|-4|
|s69_age_sex_location|vasc|0.9524|0.9524|0.9302|0.9302|+0|

The observed harm, especially melanoma/akiec tradeoffs, rejects these two recipes. It does not prove that every carefully trained multimodal model would fail. Earlier metadata work on a different cohort also failed to beat its image-only reference; no costly multimodal training is justified by this new probe.

## Limits and next decision

This is post-test exploratory development on the repeatedly used1,503-image validation cohort. Training-only metadata fitting avoids fitting the clinical adjustment on these labels, but the base CNN checkpoints and previous ensemble choices were selected on this cohort. These scores are not independent generalization or new test evidence. The original test report remains86.7598% and is not rerun.

External-data pooling, confidence calibration and metadata adjustments have not produced a material retained-method gain. Do not start another long run from these negative results or repeat small voting/metadata weight searches. Any next performance proposal needs a substantive image-representation/training hypothesis, a matched control, bounded compute and evidence that distinguishes it from techniques already rejected. The strict320px inference-only resolution screen also failed, so merely resizing an unchanged checkpoint is not a justified shortcut.

## Artifacts

`results/data_assisted/s68_s69_metadata_screen/` saves the fixed plan, source hashes, fitted categorical model/counts, metadata-fit/exclusion manifests, reference and both candidate metrics, predictions/probabilities, gain/loss details, class comparison and raw/normalized confusion matrices with PNG/PDF figures. `comparison_figures/` saves the model comparison. Registry entries: `s68_age_metadata_likelihood_exploratory_seed42` and `s69_age_sex_location_metadata_likelihood_exploratory_seed42`.

Numerical/class-order/identity/lesion-exclusion checks passed, and all reported metrics/confusion matrices were independently recomputed from saved predictions. Existing probabilities/checkpoints and the retained method were preserved. Runtime: 10.0s CPU processing, excluding Python import startup. The metadata model is saved as JSON; there is no new CNN checkpoint or epoch training curve.
