# Why performance stalled: evidence audit, 4 October 2026

## Decision

Keep the expensive queue stopped. Do not repeat the original test, start another backbone, or adopt another combined paper recipe. Recover the known-good transfer-learning recipe as the control. The evidence identifies several distinct problems; it does not prove one universal cause or guarantee 92% unseen-test accuracy.

Audit used saved metrics, saved development predictions, partition identities and source/configuration reads. No new image inference, raw test-label access, training, checkpoint modification or split modification occurred. Raw outputs: `results/aggressive_enhanced/v1/root_cause_audit/`.

## 1. We were comparing different evaluation problems

| Result | Accuracy | Meaning |
|---|---:|---|
| Frozen exploratory ensemble | 92.4817% | Image-level development validation; repeated model selection |
| Strongest numerical exploratory alternative | 92.6148% | Different selected ensemble, same exploratory cohort; not frozen final method |
| Frozen strict validation | 90.1530% | New lesions; separate cohort and fresh training |
| Original held-out test | 86.7598% | First/only frozen test evaluation |

Exploratory-to-strict gap: 2.3287 percentage points. Strict-validation-to-test gap: 3.3932 points. These gaps do not establish an architectural ceiling.

The exploratory split has 596 validation images from lesions also represented in training. However, the saved FP32 ensemble scored 90.94% on that subgroup and 93.50% on the 907 images without a training-lesion match. This does NOT show that overlap was harmless: unseen subgroup NV prevalence is 691/907 = 76.19%, versus 316/596 = 53.02% in the seen subgroup. Melanoma recall is 69.81% unseen versus 83.33% seen. Reweighting each subgroup's class recalls to the full exploratory validation class mix reverses the aggregate ordering: 92.38% seen versus 91.34% unseen. This diagnostic standardization still does not control lesion difficulty or make small-class estimates precise. Do not blame the entire gap on leakage or interpret either subgroup as a new test result.

Many validation comparisons and selecting the best epochs can produce optimistic validation estimates. Lesion-disjoint test difficulty, cohort variation and selection optimism are plausible contributors, not individually measured causes. Class supports in strict validation and test are very similar, so a changed overall class mix alone does not explain the full 3.39-point gap.

## 2. The actual difficult classes were not fixed

Original saved test confusion matrix: 199 errors; melanoma 75, benign keratosis 46, nevi 45. MEL+BKL contribute 121/199 = 60.80% of errors. The largest pair is melanoma predicted as nevi (57 images). These are descriptive findings from the already-published result, not permission to tune on test examples.

The same bottleneck was visible before testing: strict validation had 65 melanoma errors and 32 BKL errors, 97/148 = 65.54% of all errors; melanoma recall was 60.84%. Therefore, the development evidence already called for improving MEL/BKL discrimination, rather than adding generic capacity or concentrating all imbalance measures on the rarest classes.

Reaching 92% on a cohort of 1,503 with the original result would require 79 additional correct predictions, roughly 40% of current errors, without losing existing correct predictions. That is a material generalization improvement; rounding, FP32 changes or tiny voting adjustments cannot supply it.

## 3. The aggressive weight adaptation changed the learning priorities

Our original recipe used square-root inverse-frequency weights. The aggressive adaptation used inverse frequency clamped to [1,20], then mean-normalized. Clamping raised NV's unnormalized weight from about .213 to 1.0 and compressed the moderate-minority contrast.

| Class relative to NV | Proven sqrt weights | Aggressive bounded weights |
|---|---:|---:|
| MEL | 2.45 | 1.29 |
| BKL | 2.47 | 1.30 |
| DF | 7.66 | 12.52 |
| VASC | 6.85 | 10.01 |

That formulation was explicit and tested, but it was a poor hypothesis for our observed bottleneck. Normalizing the weights to mean one does not restore the lost ratios. Focal modulation further depends on sample difficulty, so these ratios alone do not measure final gradient contributions.

At DenseNet's saved accuracy winner, DF recall was 94.44% but precision only 32.69% (17 true DF among 52 DF predictions); melanoma recall was 42.17%, versus 60.84% for the proven strict ensemble. This is consistent with misplaced emphasis, not a causal ablation. The B3 accuracy winner also had melanoma recall only 45.78%.

## 4. We changed too many things simultaneously

| Proven strict recipe | Aggressive package |
|---|---|
| ImageNet normalization | Training-derived normalization |
| Native/simple head, dropout .2 | Random 512/256 BN head, dropout .65/.55/.45 |
| Sqrt-weighted cross-entropy | Bounded weighted focal, gamma2.2 |
| Horizontal flip | Affine/rotation/color/erasing plus Mixup |
| Immediate full fine-tuning | Two head-only + three partial epochs |
| Backbone LR3e-5 / head1e-4 | Same LR across head/backbone per model |
| Weight decay .0001 | .0015: fifteen times higher |
| Validation-reactive LR scheduler | Fixed StepLR |

The effects are confounded. Heavy regularization, changed normalization of ImageNet inputs, five delayed full-fine-tuning epochs, altered class emphasis and a newly initialized head are plausible reasons for slow adaptation. We cannot assert that focal loss, augmentation or a particular backbone universally failed.

Same strict data at epoch19: ConvNeXt-Tiny had already achieved 88.62% best accuracy and .8089 best macro-F1; aggressive B3 83.30% / .7019; incomplete aggressive DenseNet 82.37% / .6829. This is not a backbone-controlled comparison, but it rejects the claim that the new package was clearly better. DenseNet stopped for budget at19; eventual performance remains unknown.

## 5. The runtime increase was measurable, but not constant

Proven ConvNeXt strict median epoch: 58 seconds. Historical exploratory DenseNet median: 112 seconds. Aggressive DenseNet full-fine-tuning median: 271 seconds, with a 112–329 second range. B3 full-fine-tuning median: 107 seconds, with occasional 408-second epochs.

DenseNet peak allocated memory rose from about1.98GB to4.35GB. Its explicit FP32 batch-normalization layers return FP32 activations; DenseNet repeatedly concatenates feature maps, adding memory/casting work compared with the earlier AMP implementation. Strong augmentation and the custom head add work too. Both old and new recipes used microbatch16/effective32, so gradient accumulation alone does not explain the slowdown. Variable runtime also suggests transient resource conditions; logs cannot identify the exact cause of every slow epoch. The previous single 97% utilization reading established active computation at that moment, not a complete performance profile.

## 6. Paper numbers are hypotheses, not reproduced baselines

The [2026 source](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2026.1847649/full) reports metrics on a 1,103-image single-image-lesion validation subset, despite also describing an 8,012/2,003 partition. It uses timm B3, validation-selected fusion and class multipliers. We used a different cohort, torchvision weights, added CBAM and data-only weights. This was an adaptation, not exact reproduction.

The [2025 source](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2025.1699960/full) describes duplicate filtering, approximately 2,000 training samples per class, different backbones and Bayesian tuning. Our on-the-fly augmentation preserved the original sampling frequencies; it did not reproduce that balanced exposure. Exact split IDs and reproducible reference code/checkpoints are not established by this audit. Neither paper's high aggregate score establishes the expected score on our locked test.

## Recovery: no more blind run sequence

1. Retain the proven strict ensemble as the honest baseline. Retain all exploratory results separately. No new locked-test evaluation or retrospective tuning on its images/labels.
2. Stop using the combined aggressive package as the default. For any subsequent development experiment, keep ImageNet normalization, simple dropout.2 head, discriminative learning rates and the proven CE recipe as the control. Do not stack focal/Mixup/heavy augmentation without local evidence.
3. The first controlled question worth answering is whether training-only class emphasis or sampling can improve MEL/BKL discrimination without excessive NV losses, using only development data. Restore original sqrt weights as the control; do not derive new weights from test errors. Inspect saved probabilities and training examples before training. Hypothesis strength is not evidence of achieved improvement.
4. Prefer a diagnostic warm start from a preserved, same-protocol checkpoint over a new backbone from scratch. Use separately saved control/treatment runs from the same checkpoint, seeds and epoch window if a training ablation becomes justified. Label these post-test development work. A 5-epoch screening result is not grounds for claiming a fresh architecture failed; its role is to establish directional benefit at modest cost.
5. Require a predeclared finite compute budget, matched comparisons and worthwhile gain in accuracy AND difficult-class behavior. Evaluate saved probabilities on a reserved development audit cohort where feasible; a subset repeatedly used during prior selection is not newly independent data. Patience cannot extend a pilot beyond the fixed cap.
6. If no material gain is supported, end performance work and publish the actual protocol gap and negative evidence. Another 50-epoch queue does not resolve an unidentified cause.

No GPU run was launched in this audit. There is no promise that these findings will yield 92% independent test accuracy. The specific next training decision should follow a controlled hypothesis and affordable budget, not the desire to match a paper's headline number.
