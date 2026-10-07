# S73: matched supervised-contrastive CPU feature-head preflight

Completed 7 October 2026. **Gate failed; no GPU run or development-validation scoring follows.** The retained S53 ensemble and original frozen strict/test method remain unchanged.

## Research question and method

Does supervised contrastive regularization improve learned class separation beyond the identical nonlinear head with weighted cross-entropy alone?

Reused the existing ImageNet-1k ConvNeXt-Tiny frozen square224 FP32 feature cache. Both arms use the same three train-only lesion-group folds, fold-local scaling, seeds, architecture, optimizer, class weights, feature dropout and fixed 20-epoch endpoint. No evaluated lesion enters that fold's fitting set. No CNN forward pass, image loading or GPU operation is performed.

- Head: feature dropout0.10; Linear768->256; GELU; dropout0.20; Linear256->128; Linear128->7 classifier.
- Two independent stochastic feature-dropout views in **both** arms.
- AdamW LR0.001, weight decay0.0001; cosine scheduling to0.00001; batch128; gradient clip1.
- Control: train-fold sqrt inverse-frequency weighted CE on both views.
- Candidate: same CE +0.1 supervised contrastive loss, temperature0.1, normalized128-dimensional embeddings. All nonself same-class examples are positives; each anchor has at least its second view.
- Epoch20 is fixed. No heldout-epoch checkpoint selection or loss-weight/temperature search.

[Khosla et al., Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) motivates grouping same-class embeddings and separating different classes. This cheap feature-dropout, joint-loss experiment is **not** replication of the paper's image augmentation or full-network/two-stage training procedure. A failed probe does not establish that every end-to-end SupCon recipe would fail; it supplies insufficient evidence to spend GPU time on this proposal.

## Results: training-only group out-of-fold, 7,009 images

| Matched head | Accuracy | Macro precision | Macro recall | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|---:|---:|
| CE only | 79.0840% | 0.617817 | 0.591215 | 0.603310 | 49.3573% |
| CE + SupCon | 79.1126% | 0.621041 | 0.582965 | 0.600324 | 49.1003% |

Candidate gains126 predictions and loses124: **net+2 of7,009, +0.0285 percentage points**. Fold net changes: -8, -1, +11. Only one of three folds improves accuracy. Macro-F1 drops0.002986; melanoma recall drops0.2571 percentage points; akiec recall drops48.9083%->48.0349%.

The predeclared gate required +1 percentage point pooled accuracy, +0.01 macro-F1, no melanoma-recall decline and positive net accuracy in at least two folds. It failed. No rescue search, GPU pilot, ensemble addition or S53 validation score is produced.

The CE-only MLP has higher grouped accuracy/F1 than S72's linear frozen-feature head (75.8025% /0.573473), but melanoma recall is slightly lower than that head (50.2571%). That cross-study comparison is descriptive and changes the classifier/optimization; it does not isolate SupCon, satisfy a minority-class continuation gate or demonstrate any improvement over the93.61% ensemble.

## Verification and artifacts

Loss checks passed: independent scalar reference, nonself denominator, permutation invariance and finite nonzero backward gradients. A masked selection avoids `0 * -inf` NaNs. IDs/class order, full OOF coverage, zero fold lesion overlap, train-fold-only scaler/weights and source-cache integrity passed. Metrics and confusion matrices were independently recomputed from saved predictions.

`results/short_screening/s73_matched_contrastive_head_cpu/` contains the fixed plan, log, loss preflight, fold assignments/comparisons, training histories and curves, complete OOF predictions/probabilities, class scores, raw/normalized confusion matrices, PNG/PDF figures, source/checkpoint hashes, verification and summary.

Six **CPU feature-head** final checkpoints and three fold scalers are preserved in `checkpoints/short_screening/s73_matched_contrastive_head_cpu/`. These are not full ConvNeXt CNN checkpoints. The registry uses a distinct CPU feature-head training record kind and train-group protocol. Original checkpoints remain untouched.

Runtime:40.3seconds processing, excluding interpreter startup. No development-validation feature cache or original-test labels/images were loaded. Original structured test accuracy remains86.7598%; retained single-image exploratory accuracy remains93.6128%.

## Next research decision

S70-S73 do not establish a material retained-model gain. Do not extend them into more loss/temperature/voting searches or automatically launch a long CNN run. A distinct next cheap hypothesis is **lesion-normalized training influence**: repeated views currently contribute multiple training terms, so equal image weighting may overrepresent frequently photographed lesions. It must first be compared with the same CE-only head on train-only grouped folds, with weights derived only inside each training fold, and require accuracy, macro-F1 and melanoma recall gains. This is a proposed training-sampling/loss hypothesis, not an implemented improvement or proof of the plateau's cause. No new GPU run is prepared or launched by this closeout.
