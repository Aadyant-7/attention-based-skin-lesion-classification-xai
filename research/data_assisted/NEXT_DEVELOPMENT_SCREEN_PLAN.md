# Bounded data screen — original plan

Executed as S66/S67 on2026-10-06. Both gates failed; no pooled-data GPU run is recommended. Results and interpretation: `research/data_assisted/S66_S67_CLOSEOUT.md`. The original fixed plan below is preserved.

## Question

Does additional curated external dermoscopy training data improve difficult-class representation in a matched frozen-feature classifier, enough to justify any GPU fine-tuning? This is a data hypothesis, not another architecture or voting-weight search.

## Preconditions

Full acquisition and pixel quarantine must finish. Use only `pixel_screened_candidate_manifest.csv`, its recorded hash and original fixed partitions. Retain the seven-class order. Report AK subset mapping and source shift. Patient-level independence and unknown aliases of original test images remain unverified, so this is development evidence only.

## Paired CPU experiment

1. Reuse verified S64 square224 ImageNet-1k ConvNeXt-Tiny training/validation feature caches and identity order. Do not reuse a fine-tuned checkpoint trained on evaluated labels.
2. Extract the same frozen FP32 features for screened external images, with the same square224 preprocessing. No backpropagation or hyperparameter search.
3. Fit two classifiers with the same fixed weighted logistic-regression recipe: HAM development-training only versus HAM training plus the external-training partition. Fit scaling and weights on each training set only. Use `C=1`, seed42 and all seven classes, as in S64.
4. Preserve the external preflight-validation partition for descriptive source-shift checks. It contributes no training examples, learned fusion coefficients or classifier selection search.
5. Evaluate paired predictions on the existing exploratory validation cohort. Then require a train-only lesion-group consistency check before a GPU recommendation, because the first S64 validation gain did not replicate consistently across groups.

## Fixed advancement gate

- At least +1.0 percentage point exploratory accuracy versus the matched HAM-only frozen-feature control.
- Macro-F1 improves by at least 0.01; melanoma recall does not decline.
- Pooled train-only group-CV accuracy gain at least +1.0 percentage point, with improvement in at least two of three fixed folds and no pooled melanoma-recall decline.
- No parameter, source subset or ensemble-weight search to rescue a failed gate.

These gates measure whether the data deserves further compute; passing does not guarantee a fine-tuned ensemble gain. Failing means stop and inspect domain/label shift before training.

## Artifacts and compute

Prepare a fixed experiment ID/config only after the final screened manifest exists. Save source/cache hashes, class/source counts, model recipe, per-image predictions/probabilities, metrics, class scores, confusion matrices and PNG/PDF comparisons. Add a model registry row only when an actual classifier has been evaluated. Expected CPU feature extraction: several minutes, hardware dependent; no 20–50 epoch run is authorized by this plan. Any later GPU proposal must give a bounded schedule, matched control, monitoring command and checkpoint/result paths first.

Metadata fusion is deliberately deferred to a separate matched experiment.
