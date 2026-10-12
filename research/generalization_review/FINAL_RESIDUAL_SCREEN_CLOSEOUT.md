# Final S83 learned residual study

12 October 2026. Two substantive CPU-only hypotheses were screened once after
the user authorized a final accuracy investigation. **Neither improved overall
accuracy or passed the advancement gate. Retain frozen S83.** No candidate was
scored on the previously assessed test cohort.

## Why these hypotheses

Earlier unrestricted stacking, feature-fusion classifiers, calibration and
naive-Bayes metadata adjustments had failed. This study keeps S83's strong image
probabilities as a fixed offset and learns only a penalized correction:

1. **Clinical residual:** jointly learn age/sex/localization corrections to
   `log(P_S83)`. Training-fold age normalization and category vocabularies prevent
   held-out rows from setting the clinical encoder. Missing/unseen values add no
   clinical feature evidence. A matched intercept-only residual separates generic
   class calibration from added clinical information.
2. **Top-two reranker:** only for member disagreement and leading/runner-up
   probability margin at most 0.5, learn a binary residual to the original log
   odds. Predictors are the two class identities and six member probability
   differences. Preserve their joint mass, every other class probability and all
   noneligible predictions. Labels define training targets only, never routing
   eligibility or inference features.

Every fit starts at zero correction and uses mean unweighted cross-entropy plus
`0.01/2 * squared correction parameters`, with fixed L-BFGS-B settings. Fifteen
small CPU fits cover five intercept, five clinical and five reranker heads. There
is no parameter grid, threshold search, additional candidate or rescue fit.

## Evaluation and actual results

Five fixed stratified lesion-group folds, seed 42, use the original **1,503
validation images / 1,429 lesions**. Each image receives exactly one prediction
from a correction that did not fit that image or lesion. All images stay included.

| Method | Accuracy | Macro-F1 | Melanoma recall | akiec recall | Fixed / broken | Positive folds |
|---|---:|---:|---:|---:|---:|---:|
| Frozen S83 reference |**94.0120%**|0.895201|82.0359%|71.4286%|0 / 0|—|
| Intercept-only control |93.8789%|0.897436|82.0359%|77.5510%|4 / 6|2 / 5|
| Clinical residual |93.7458%|0.887418|82.6347%|73.4694%|5 / 9|2 / 5|
| Top-two reranker |93.9454%|0.898092|80.8383%|75.5102%|4 / 5|2 / 5|

The clinical head loses four net correct images and also falls below its
intercept control. The reranker loses one net correct image; its higher macro-F1
comes with two fewer correctly recognized melanoma images. No trade-off is
presented as an accuracy improvement.

The predeclared gate required at least eight net additional correct images,
at least 0.5 percentage-point accuracy gain, nondecreasing macro-F1 and melanoma/
akiec recall, and positive gains in at least four folds. Clinical information
also had to improve accuracy over the intercept control. **Both gates failed.**

## Limits and decision

This is **post-development meta-OOF validation**. The base CNNs/checkpoints and
earlier ensemble decisions were already selected using this validation cohort;
some related lesions are shared with original CNN training. Grouped cross-fitting
protects the new residual fitting only. It is not independent CNN cross-validation
or untouched-test evidence.

Clinical parsing was restricted to the 1,503 allowed validation originals before
age/sex/site columns were parsed. No old-test clinical rows or candidate test
predictions were loaded. Image IDs and lesion IDs only align data and define
groups; neither is a predictor. Diagnosis-confirmation type is excluded.

The fitting functions genuinely train and evaluate small CPU heads. The frozen
plan/summary's `new_model_inference: false` means **no new CNN/backbone inference**;
the verification receipt explicitly records CPU head fitting/prediction. Zero
training epochs means zero additional CNN epochs, not absence of CPU optimization.

Preserve the existing final results: **94.011976% exploratory single-image
validation / 0.895201 macro-F1**, **87.558217% recorded single-image test audit /
0.807031**, and the separate **90.071556% retrospective multi-image lesion
assessment / 0.809079 on 1,118 previously assessed lesions**. No checkpoint,
ensemble weight, preprocessing, earlier prediction or XAI artifact changed.

No full-development deployment head is fitted after these failures. No GPU run,
new test assessment or further search is scheduled by this finite study.

## Reproducibility and artifacts

[Saved package](../../results/diagnostics/s83_final_residual_screen_2026-10-12/):

- Frozen source/code hashes and plan; fold assignments and complete fold parameters.
- All four full prediction/probability packages, macro/class scores, raw/normalized
  matrices, matrix PNGs and a publication PNG/PDF comparison.
- Fold-by-fold comparisons, changed-image identities, verification and inventory.
- These CPU heads have optimizer iterations/objectives rather than epoch curves.

[Runner](final_residual_screen.py), [clinical fitting](clinical_residual.py),
[reranker fitting](top2_residual.py). The completed-run guard prevents repeats;
cached source arrays are local and hash-authenticated, not new checkpoints.
