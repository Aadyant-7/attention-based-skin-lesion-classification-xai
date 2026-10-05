# S56/S57: stronger pretrained representation, bounded training

## Why this direction

S06 and S18 ended near 99.7% training accuracy, while their best exploratory validation accuracies were 91.75% and 92.22%. The current equal-five candidate reaches 93.6128%, but recent averaging, routing, TTA and PanDerm frozen-head additions did not produce a material gain. This supports investigating pretrained feature quality rather than more epochs or more fusion variants. It does not isolate every source of generalization error.

S53's 96 validation errors include 30 melanoma, 24 benign keratosis, 21 nevus and 14 akiec errors. The largest single confusion is melanoma -> nevus (27). Future complementarity analysis should inspect these errors, not just overall accuracy.

## Candidate and verified sources

**ConvNeXt-V2 Tiny**, exact timm identifier `convnextv2_tiny.fcmae_ft_in22k_in1k`, uses author-provided FCMAE pretrained weights with ImageNet-22k and ImageNet-1k fine-tuning. V2 introduces Global Response Normalization. This is a different representation/training history in the proven ConvNeXt family, not another B0 variation.

- [Official implementation and model table](https://github.com/facebookresearch/ConvNeXt-V2)
- [Exact checkpoint model card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k)
- [Paper](https://arxiv.org/abs/2301.00808)

These sources support the architecture and pretraining rationale, not a guaranteed HAM10000 accuracy. The seven-class implementation has 27,871,879 parameters. The author checkpoint SHA-256 is recorded in `results/short_screening/convnextv2_transfer_v1/preflight.json`.

## S56: cheap screening before training

Extract identity-view frozen features for only the existing 7,009 train and 1,503 exploratory validation images. Compare fresh ImageNet-1k ConvNeXt-Tiny with the V2/22k candidate. Fit the same standardized logistic regression (C=1, max_iter=1000, seed42, training-only sqrt inverse-frequency sample weights) to each model's training features. No head tuning.

Predeclared continuation gate: V2 accuracy improves by at least 1 percentage point, OR macro-F1 improves by at least 0.02 with no accuracy decline. The gate tests representation usefulness; it cannot predict a fine-tuned score or guarantee ensemble gains.

Results, predictions, scores, confusion matrices and comparison figures: `results/short_screening/convnextv2_transfer_v1/s56_frozen_feature_screen/`. Ignored feature caches and shallow heads: `.cache/convnextv2_transfer_v1/`.

The first probe invocation hit the development loader's train/val guard before feature extraction. It was corrected by retaining the train partition flag and removing random flips for frozen train feature extraction. No split rows were relabeled; the preserved failure note records this.

## S57: launch only after gate passes

- Fresh exact author-pretrained candidate; seven-class head, full fine-tuning.
- Same exploratory split, seed42, square224 bilinear/ImageNet normalization, horizontal flips only, head dropout0.2.
- Weighted cross-entropy, training-only normalized sqrt inverse-frequency weights.
- AdamW, backbone3e-5/head1e-4, weight decay1e-4; batch16, accumulation2, effective32.
- AMP/GradScaler training, finite-gradient checks and norm1 clipping; FP32 identity validation.
- ReduceLROnPlateau, factor0.5, patience2, meaningful accuracy threshold0.002 absolute.
- Hard **20-epoch maximum**. After epoch15, permit plateau stopping only after 6 epochs without cumulative +0.002 accuracy or +0.003 macro-F1 and at least one LR reduction. Tiny improvements can update raw best checkpoints without resetting meaningful counters.
- Preserve raw accuracy best, macro-F1 best, atomic latest including optimizer/scaler/scheduler/RNG/history, LR history, predictions, class metrics, confusion matrices, curves and registry entries.
- No Mixup, focal loss, oversampling, CBAM, TTA or resolution changes in this run.

This changes architecture/pretraining and scheduler/validation precision versus historical S06; it is a method comparison, not a pure architecture-only ablation. All outcomes remain exploratory development following earlier test reporting.

Commands from project root:

```powershell
.\.venv\Scripts\python.exe -m research.short_screening.convnextv2_transfer --check
.\.venv\Scripts\python.exe -m research.short_screening.convnextv2_transfer --probe
.\.venv\Scripts\python.exe -m research.short_screening.convnextv2_transfer --run
# Recovery from latest committed epoch, no reset:
.\.venv\Scripts\python.exe -m research.short_screening.convnextv2_transfer --run --resume
```

Training output: `results/short_screening/convnextv2_transfer_v1/s57_convnextv2_tiny_22k_exploratory_seed42/`.
Checkpoints: `checkpoints/short_screening/convnextv2_transfer_v1/s57_convnextv2_tiny_22k_exploratory_seed42/`.

RTX4060 8GB expectation: roughly 15–30 minutes for 20 epochs, about 2–4GB allocated training VRAM, subject to actual throughput. These are estimates from prior Tiny runs, not measured S57 runtime.

## After completion

Collect standalone and class-wise performance, compare errors with S06 and S53, and decide whether one fixed ensemble comparison is justified. Do not auto-search ensemble weights or select checkpoints using ensemble validation scores. Do not launch another training job automatically. The original frozen strict/test methodology and first/only locked-test result remain preserved; no locked-test loader is created.

## Screening outcome: S57 not launched

S56 completed in 122.9 seconds: ImageNet1k control 79.2415% / 0.643682 macro-F1; ConvNeXt-V2 candidate 78.8423% / 0.657982. The original gate failed and was not lowered. S57 was not launched. This rejects the candidate under this cheap screen, not all possible V2 fine-tuning outcomes. One additional supervised-22k candidate is documented separately in `S58_S59_SUPERVISED22K_PLAN.md`.
