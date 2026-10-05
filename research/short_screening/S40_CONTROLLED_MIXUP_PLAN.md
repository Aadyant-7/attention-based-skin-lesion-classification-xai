# S40: address generalization from the start

## Decision

Stop the inference-tweak sequence. Run ONE20-epoch ConvNeXt-Tiny training ablation with Mixup as the only learning change. User authorizes autonomous research and automatic closeout of short runs; long GPU training follows launch-and-stop. No next model or large search is queued.

### Evidence

S29 fits the training images almost perfectly by epoch15 while strict validation remains around88%. Latest late fine-tuning reached99.8% training accuracy without improvement. Targeted extra exposure failed its control;320px inference lost accuracy and melanoma recall. These findings support investigating generalization from early training, not additional fitting of a saturated checkpoint. They do not prove Mixup will improve HAM10000 accuracy.

The aggressive package's Mixup was confounded with focal loss, different class weights, a new heavy classifier, normalization and other augmentation. It was not this controlled ablation.

References: original Mixup paper https://arxiv.org/abs/1710.09412 ; official ConvNeXt recipe https://github.com/pytorch/vision/blob/main/references/classification/README.md . Evidence is general classification, not a guarantee of our HAM10000 test score.

## Fixed protocol and budget

- Post-test STRICT DEVELOPMENT:7009train/1503val, original test excluded. Existing strict validation is already selection-exposed; no fresh independent claim.
- ConvNeXt-Tiny, fresh cached `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`; same native LayerNorm/dropout.2 seven-class head.
- RGB square224 bilinear antialias/ImageNet normalization; same horizontalflip.5.
- Only change: full effective-batch32 Mixup with Beta(.2,.2), independent random stream, soft targets from the same permutation. No focal, CutMix, new sampling, heavy augmentation, crop, TTA or weight search.
- Same train-only sqrt class weights. Mixed weighted CE numerator divided by total effective-batch target weight. Permutation makes mixed total weight equal to original total weight. CPU test compares soft-label loss and gradients with the accumulated16+16 implementation, including the lambda1 ordinary-CE limit.
- Same AdamW3e-5backbone/1e-4head, decay1e-4; same plateau scheduler factor.5/patience2/threshold0. Scheduler unchanged to isolate Mixup. AMP training, FP32 validation, clip1, deterministic seed42.
- Exactly20 epochs maximum, no validation-driven extension or automatic continuation. Meaningful accuracy+.002/F1+.003 anchors saved for transparency; they cannot extend the hard budget.
- Estimate20–30minutes, roughly1.5–3GB allocated VRAM on RTX4060; actual runtime recorded.

## Comparisons and decision

Historical S29 through20 epochs: accuracy winner epoch19,88.6228% / .808912 macro-F1. Compare S40 through the same epoch window. Same recipe/seed/initialization lineage; historical control, not a contemporaneous replicated trial. Mixup training accuracy is lambda-weighted mixed-label accuracy and is not ordinary training accuracy.

Also compare with the FULL preserved S29 checkpoint89.0220%/.814678 and frozen full strict ensemble90.1530%/.843486, clearly labelled longer-budget references. One fixed equal B0/S40/V2S comparison uses the standalone-accuracy winner; no ensemble-epoch or weight search.

Material adoption gate: >=+0.5pp over the frozen full ensemble, macro-F1 nondecrease, MEL recall nondecrease, NV recall decline<=1pp. Failure closes this route; no further coefficient search or automatic new training. No promise of92% independent test accuracy. Do not reevaluate the original test.

## Artifacts and commands

```powershell
.\.venv\Scripts\python.exe -m research.short_screening.controlled_mixup --check
.\.venv\Scripts\python.exe -u -m research.short_screening.controlled_mixup --run
Get-Content '.\results\short_screening\mixup_v1\s40_convnext_tiny_mixup_strict_dev_seed42\train.log' -Tail 15 -Wait
# Recover only from the last committed epoch, unchanged source/config:
.\.venv\Scripts\python.exe -u -m research.short_screening.controlled_mixup --run --resume
```

Results/logs: `results/short_screening/mixup_v1/s40_convnext_tiny_mixup_strict_dev_seed42/`.
Checkpoints: `checkpoints/short_screening/mixup_v1/s40_convnext_tiny_mixup_strict_dev_seed42/`.

Initial/latest full optimizer/scheduler/scaler/RNG/independentMixup RNG states; raw accuracy best, raw F1 best; metrics, predictions, probabilities, LR history, meaningful counters, curves, class scores, matrices and PNG/PDF comparison figures. Record explicit cap/stopping reason and registry entries. The worker closes out and attempts to publish results, then stops; publication errors preserve a pending report. No independent process restarts itself or launches another experiment.

Policy going forward: bounded short inference/CPU studies are completed and verified in the same Codex turn, rather than requiring a user round trip. Each must answer a defined question and have a finite scope; failed studies do not trigger endless variants. Long GPU runs launch-and-stop after logging/checkpoint confirmation.
