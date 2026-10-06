# S66/S67 external-data CPU screen — closeout

## Decision

**FAIL: do not spend GPU time training this fixed combined-data recipe.**

The current fine-tuned ensemble stays unchanged at93.6128% exploratory validation accuracy. These are frozen-feature diagnostic classifiers, not new ensemble scores or locked-test results.

## Fixed setup

- Fresh ImageNet1k ConvNeXt-Tiny features, CPU FP32, square224 bilinear identity preprocessing.
- Same weighted logistic regression, C=1, seed42, train-only scaling and normalized sqrt inverse-frequency sample weights.
- Control:7,009 HAM training images. Candidate: same HAM training plus2,823 fixed curated BCN training lesions.
- Exploratory validation: same1,503 HAM development images; no fit on these images.
- Confirmation: three fixed train-only lesion-group folds; the same external training partition is added to each candidate fit. No HAM lesion crosses fit/heldout boundaries within these folds.
- External500-lesion preflight validation: descriptive source-shift check only, never fitted.
- No parameter/source-subset/weight search, backpropagation, metadata fusion or GPU training. Historical S64 control metrics reproduced exactly.

## Results

|Diagnostic method|Accuracy|Macro precision|Macro recall|Macro-F1|
|---|---:|---:|---:|---:|
|S66 ham_only|79.3081%|0.642028|0.664278|0.650621|
|S66 ham_plus_external|78.5762%|0.646424|0.655765|0.649117|
|S67 ham_only|75.8025%|0.582825|0.567880|0.573473|
|S67 ham_plus_external|74.4471%|0.533637|0.542825|0.537126|

S66: **-0.7319 percentage points**, macro-F1 -0.001504; 78 correct predictions gained / 89 lost / net -11. Melanoma recall 62.8743% → 58.0838%.

S67 pooled: **-1.3554 percentage points**, macro-F1 -0.036347; 458 gained / 553 lost / net -95. Melanoma recall 50.2571% → 47.9434%; 0/3 folds improve.

|Group fold|HAM-only accuracy|HAM+external accuracy|Net correct change|
|---|---:|---:|---:|
|1|74.8823%|74.1977%|-16|
|2|75.5993%|74.1866%|-33|
|3|76.9264%|74.9572%|-46|

### Class behavior on exploratory validation

|Class|HAM-only recall / F1|HAM+external recall / F1|
|---|---:|---:|
|akiec|0.6327 / 0.6019|0.5918 / 0.5631|
|bcc|0.6883 / 0.6127|0.6364 / 0.5799|
|bkl|0.6061 / 0.6061|0.6303 / 0.6190|
|df|0.4118 / 0.4516|0.4706 / 0.5161|
|mel|0.6287 / 0.5769|0.5808 / 0.5465|
|nv|0.8729 / 0.8956|0.8709 / 0.8899|
|vasc|0.8095 / 0.8095|0.8095 / 0.8293|

Full class-wise precision/recall/F1/support and raw/normalized confusion matrices are saved for both protocols and the external preflight check. Paired gain/loss image identities are saved; no visual/test-label review is required to obtain these counts.

### External preflight domain check

|Method|Accuracy on external500|Macro-F1|
|---|---:|---:|
|ham_only|46.4000%|0.308729|
|ham_plus_external|58.8000%|0.447269|

These source-specific scores have different case composition and cannot be compared as if they were matched HAM validation results. Learning BCN performance does not by itself prove improved HAM generalization. AK is subset supervision for HAM akiec; unknown test aliases and patient overlap remain unverified. This is post-test exploratory development, not a replacement test evaluation.

## Gate and next action

- S66 accuracy gain below +1.0 percentage point
- S66 macro-F1 gain below +0.01
- S66 melanoma recall declined
- S67 pooled accuracy gain below +1.0 percentage point
- S67 pooled macro-F1 declined
- S67 pooled melanoma recall declined
- S67 fewer than two improving folds

Keep the curated external pool and all evidence, but reject GPU fine-tuning of this simple pooled-data recipe. Inspect source/class-shift evidence before a revised hypothesis; do not rescue this screen through subset, source-weight or hyperparameter searches. Metadata remains a separate deferred option.

## Saved evidence

- `results/data_assisted/s66_s67_frozen_data_screen/`: plans, source/cache hashes, summary and verification.
- `s66/ham_only/`, `s66/ham_plus_external/`: exploratory metrics, predictions/probabilities, confusion/class PNG/PDF figures; external-preflight subfolders contain the source-shift reports.
- `s67/ham_only/`, `s67/ham_plus_external/`: train-only meta-OOF metrics, predictions/probabilities and figures; folds/assignments/class comparisons saved in `s67/`.
- `s66/comparison_figures/`, `s67/comparison_figures/`: paper-ready comparison CSV/PNG/PDF.
- `.cache/s66_s67_frozen_data_screen/`: reusable external features and all fitted CPU classifiers (local only).
- `results/master_experiment_registry.csv`: four completed diagnostic method entries, explicitly separate from CNN/ensemble/test performance.

Runtime: 4.79 minutes. Original locked-test images/labels never loaded; no new original-test evaluation.


## Descriptive source-shift audit

The added pool differs substantially in class composition and source. Raw HAM image counts also include repeated lesion views, whereas the external pool uses one view per lesion. The same class-weight formula is recomputed on each training set; class weights and effective regularization/data mass therefore change with the added cohort. This is a fixed data-recipe comparison, not proof that sample count alone caused the change.

|Class|HAM train images / lesions|External train lesions|HAM image share|External share|Combined share|
|---|---:|---:|---:|---:|---:|
|akiec|229 / 166|188|3.27%|6.66%|4.24%|
|bcc|359 / 244|802|5.12%|28.41%|11.81%|
|bkl|769 / 541|285|10.97%|10.10%|10.72%|
|df|81 / 57|32|1.16%|1.13%|1.15%|
|mel|778 / 478|433|11.10%|15.34%|12.32%|
|nv|4693 / 3926|1052|66.96%|37.27%|58.43%|
|vasc|100 / 74|31|1.43%|1.10%|1.33%|

Learning external cases while harming HAM validation and all three HAM group folds is consistent with source/domain and case-composition shift. It does not establish one exact cause, and frozen features cannot rule out all future adaptation methods. The practical decision is to avoid a speculative GPU run of this rejected pooled recipe. Audit CSV/JSON are saved beside the results.
