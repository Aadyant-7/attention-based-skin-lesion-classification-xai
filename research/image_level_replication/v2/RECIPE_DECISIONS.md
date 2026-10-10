# Phase 2: B3 recipe decisions

10 October 2026. Prepared first baseline; no GPU run approved or launched.

Sources: [DermAI 1.0, §§3.2–3.6](https://pmc.ncbi.nlm.nih.gov/articles/PMC10573070/) and [official torchvision B3 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b3.html). Exact reproduction remains impossible from the available description. Paper-reported accuracy is not a forecast for this implementation.

| Choice | Evidence / actual implementation |
|---|---|
| Backbone | Paper: ImageNet B3. Here: fresh torchvision `EfficientNet_B3_Weights.IMAGENET1K_V1`, local original-weight checksum verified. |
| Resolution | Paper: 112×112. Here: direct RGB bilinear resize to 112, overriding torchvision's native 300px evaluation crop. |
| Normalization | Paper unspecified. Here: external weight's ImageNet mean/std; no dataset-fitted mean/std. |
| Pooling | Paper unspecified. Here: native global average pooling gives1536 features. |
| Head | Paper: two 4096 ReLU dense layers, dropout0.5, seven-class softmax. Here: 1536→4096/ReLU→4096/ReLU→Dropout0.5→7logits; CE provides log-softmax internally. |
| Fine-tuning | Paper does not give freeze schedule. Here: all layers train from epoch 1. |
| Optimizer | Paper unspecified. Here: Adam, all layers 1e-4, betas 0.9/0.999, epsilon 1e-8, no weight decay. |
| Batch / allowance | Paper: 16 / 50 epochs. Here: batch 16, max 50; no automatic extension. |
| Balancing | Paper describes minority augmentation and class weights without counts/formula. Here: each training class has exactly 5,430 virtual exposures/epoch, generated from training originals only; each original appears at least once. |
| Loss weights | Class weights from the equalized exposure distribution are all1. Do not also apply inverse weights from the original imbalanced counts. Validation uses natural proportions and ordinary CE. |
| Augmentation | Paper lists crops/expansion/flips/color/shift/blur without magnitudes. Our explicitly frozen mild parameters are in `b3_fold00.json`; validation has none. No augmented copies enter assessment. |
| Attention | No added CBAM in this baseline; B3's native squeeze-and-excitation stays. Paper attention equations are unresolved; later CBAM requires its own named comparison. |
| Numerical policy | BF16 training forward, FP32 CE/validation; no FP16. Nonfinite inputs/logits fail, nonfinite gradient updates are skipped with BN buffers restored, three consecutive skips stop with diagnostics. No silent recipe/precision change. |
| Checkpoint selection | Earliest raw accuracy maximum plus separate earliest raw macro-F1 maximum. Inner validation only. |
| Scheduling / stopping | Our additions: meaningful ReduceLROnPlateau; min 25/max 50; +0.002 accuracy or +0.003 F1 meaningful anchors; patience 10; late guard 30+ with 8 stale epochs, flat trend and LR settling. Raw numerical bests remain separate. |
| Excluded package | No focal loss, Mixup, added CBAM, TTA, resolution search, legacy warm-start or ensemble search in this run. |

## Research question and limits

This is one fresh paper-inspired baseline on **predeclared fold 0 inner development**, using 8,111 fitting / 902 validation originals. It tests the combined recipe under the new protocol; it is not a pure head, resolution or augmentation ablation. The 902-image inner-validation result is not comparable directly to prior 1,503-image exploratory scores.

The reference's smaller input can lose lesion detail and its large dense head can overfit. Balancing changes the effective class prior. These are measured risks, not reasons to quietly change the recipe during training. CPU checks verify implementation and recovery; they cannot establish a likely accuracy score.

An outer assessment, outer 90% refit, other fold, another backbone or ensemble needs a separate freeze/proposal. No original-project fine-tuned weights or probabilities are eligible for this new split. A full K10 claim requires all ten same-method outer folds, OOF predictions and declared aggregation. Internal post-development image-level evaluation with lesion overlap remains the reporting label.

## Epoch policy amendment

The Phase 1 plan's 30-epoch working budget was provisional. Phase 2 fixes a **50-epoch allowance with a 25-epoch minimum and meaningful stopping**, following the reported allowance and the user's request to allow adequate time. No absolute accuracy threshold kills a run; no numerical fluctuation resets patience unless it accumulates to the meaningful delta from its last meaningful anchor.
