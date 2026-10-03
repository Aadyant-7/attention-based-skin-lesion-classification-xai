## S18 completed; research paused for tonight

S18 finished20epochs: best92.22%/.8549 at17; final91.28%/.8398. Modest accuracy gain versus Tiny with lower macro-F1; retain candidate for future ensemble review. No new ensemble run. User-approved15/20continuations superseded the old epoch8gate; future plan undecided until tomorrow. See research/phase3/S18_CLOSEOUT.md (root-relative). Historical proposals below remain preserved.

## Current decision after S15 (3 October 2026)

S15: 89.62%/.8075. One fixed S17 fusion: 92.61%/.8771, only two extra correct versus FP32 reference with lower macro-F1. DenseNet excluded from selected ensemble; S12 FP32 reference retained. Next proposal: staged ConvNeXt-Small S18, with 8/12/20-epoch approval gates. S16 B3/Swin deferred; S14 remains deferred. No new GPU run approved. See research/phase3/S15_CLOSEOUT.md and research/phase3/S18_PROPOSAL.md (paths relative to root).

# High-performance heterogeneous ensemble phase

Priority changed by the user on3 October2026. **S14 is deferred and unlaunched. No new experiment is authorized.** This replaces inference-tweak prioritization; S01–S13 results and S14 preparation remain preserved. Expanded Phase3 now targets two strong new backbones, with one conditional transformer slot, then bounded heterogeneous fusion. Strict confirmation waits until this block is complete; original locked test untouched.

## Evidence and candidate choice

Existing review remains authoritative: `research/literature/review_table.md`, `scispace_analysis/INTEGRATED_REVIEW.md`, all50 workbook rows/49 distinct DOI candidates, base paper and HAM10000 authorities. New selected-source checks supplement that snapshot without rewriting extraction decisions or historical results. See [focused primary-source update](../literature/HIGH_PERFORMANCE_ENSEMBLE_EVIDENCE.md).

| Priority | Candidate | Why / admission question | Status |
|---|---|---|---|
| 1 | DenseNet201 | Dense feature concatenation/reuse differs from ConvNeXt's residual depthwise/LayerNorm blocks and V2-S/B0's MBConv/SE. Strong HAM10000 precedent; test whether this gives a competitive standalone and useful new errors. | S15 config/code ready; approval pending |
| 2 | EfficientNet-B3 | Substantially larger compound-scaled V1 model than B0; direct strong ensemble precedent. Native SE retained, no added CBAM. A stronger candidate, not a B0 tweak. | S16 config/code ready; re-evaluate after S15; separate approval |
| 3, conditional | Swin-T | Shifted-window hierarchical transformer adds a distinct attention representation. CNN/transformer fusion is supported by existing review; exact standalone HAM10000 expectation not established. | Only if first two do not deliver sufficient strength/diversity; adapter/config/budget proposal still required |

Do not train ResNet101, B3, B4, DenseNet201 and transformers as a sweep. ResNet101 is a reserve alternative if dense connectivity proves unhelpful; B4 is not an automatic extra EfficientNet run. Swin-T is selected over a larger ViT-B/16 for a plausible8GB cost/parameter envelope, not because proven superior on our cohort. Architectural labels are hypotheses; complementarity must be measured.

**Next: S15 DenseNet201.** B3 has higher author-reported standalone scores in the new reference, but DenseNet offers the missing dense family first and is the clearest new test of complementarity with our strong ConvNeXt. No guarantee it beats ConvNeXt, achieves95%, or adds more than a weak model would. Our ImageNet weight version, split/head/loss/preprocessing/epoch budget differ from those papers.

## Transfer learning and comparability

For S15/S16, retain the successful registered `exploratory_screening_v1` recipe: same7009train/1503validation/1503originaltest,224px direct bilinear resize/ImageNet normalization, seed42, AdamW3e-5backbone/1e-4head, effectivebatch32, all layers trainable from pretrained weights, dropout.2/GAP/linear7, AMP training,20-epoch cap/5-stale-accuracy stop, accuracy-selected primary and separately saved macro-F1 winner. This is a comparison of pretrained model packages, not topology alone.

DenseNet201 source: `DenseNet201_Weights.IMAGENET1K_V1`; seven-class adapter18,106,375 parameters, features→native ReLU→pool→head. EfficientNet-B3 source: `EfficientNet_B3_Weights.IMAGENET1K_V1`; seven-class10,706,991 parameters. B3's official300px crop differs from the common224px research input and must be disclosed; the verified recent ensemble paper also chose224px across backbones. No automatic native-resolution run.

Training-only sqrt inverse-frequency weights, normalized to mean1, weighted CE; natural validation population retained. No oversampling of validation/test. Do not silently add paper focal/Mixup/SMOTE or heavy augmentation: our historical focal/strong-augmentation results regressed, so evidence does not justify adopting them as default for new backbones. Literature suggests these can help under other recipes, not that they universally do. If a strong new model demonstrably underfits/overfits or misses minority classes, propose one explicit training-package amendment with a research question and comparator; do not start a tweak queue. Added CBAM remains optional evidence-driven work, not an obligatory accuracy claim. XAI stays a final-method explanation task.

Review learning curves before interpreting a poor result as a universal architecture rejection. No automatic extra epochs/restarts; a nonconverged run would require a separately documented budget amendment/proposal.

## Bounded ensemble sequence

1. Close each approved run; verify all selected/latest checkpoints, primary/secondary metrics and aligned validation probabilities. Compare with S06 and S12/its FP32 control on the same manifest; separately disclose different source packages/costs/numerical amendments.
2. Measure errors recovered/broken relative to ConvNeXt and S12, joint errors, disagreement, minority-class recall/F1 and confidence. Oracle unions are diagnostic upper bounds only. A weak model needs unusually useful diversity to be admitted; architecture name or existence of a checkpoint is insufficient.
3. Retain at most three useful new models. Predeclare **at most four equal-voting candidates** before computing fusion metrics: typically ConvNeXt+DenseNet, ConvNeXt+B3, ConvNeXt+DenseNet+B3 and (only if admitted) the same three plus Swin-T. Replace/reject candidates if standalone/error analysis justifies doing so before fusion evaluation. Existing V2-S/B0 may replace a weak newcomer; not a combinatorial sweep.
4. On only one justified member set after equal voting, permit **two extra global weight templates maximum**: pair(.6,.4)/(.4,.6); triple(.5,.25,.25)/(.4,.4,.2); four(.4,.2,.2,.2)/(.3,.3,.2,.2). Order members before weighted evaluation using standalone accuracy, macro-F1 and cost; first template favours the chosen strongest anchor; the second is one predeclared alternative. Exact member order/templates frozen in a candidate manifest before scores. Equal is the baseline. No per-image/class-correctness weights, continuous optimization, iterative refinement, label-dependent oracle routing or hidden discarded trials.
5. Maximum six fusion candidates in this phase, all disclosed including failures. Admit3/4members only if complementary errors justify cost. Keep individual models if fusion regresses. Report weights as validation-selected development decisions; these are not independent-test claims. Do not use the original test to select anything.
6. Operational **material-gain target**: at least1 percentage point over92.4817% (15+ net additional correct out of1503), with macro-F1 at least.877454 and no concealed minority-class collapse. This defines what would satisfy the priority change, not a promised result or a statistical significance threshold. Smaller gains are still recorded accurately but do not trigger a decimal-tuning phase. Aim to investigate movement toward95%; do not infer that papers' numbers will transfer.
7. After this bounded block, freeze strongest accuracy/balanced methodology for later fresh strict training and final locked-test evaluation. No strict confirmation now.

## Artifacts and permissions

Every serious run: config/question/pretraining/environment/source/split hashes, logs/progress/full history, best/latest/best_macro_f1 states with resume data, selected prediction probabilities, accuracy/macroP/R/F1/loss, class scores/supports, matrices, training curves, registry/closeout and paper figures. Secondary class/matrix plots and fair comparison/error figures verified at closeout. CPU fusion references parent checkpoints/curves and saves its own probabilities/metrics/matrices/class scores/plots/registry; never invents epochs/new weight checkpoints.

The immediate executable proposal is [S15](S15_PROPOSAL.md). Each GPU run needs separate human approval. Once approved: independent launch, confirm log and atomic checkpoints, return monitor/resume/location instructions, then stop agent activity until user returns. No S14/S15/S16 or CPU fusion launched during preparation.

Progress uses the newly expanded high-performance block:10% planning/preparation; overall project remains76% (estimates).
