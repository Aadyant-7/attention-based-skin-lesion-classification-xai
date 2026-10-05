# S58/S59: supervised ImageNet-22k ConvNeXt-Tiny transfer

## Evidence and decision

The V2/FCMAE frozen-feature candidate failed S56's predeclared gate; no V2 training was launched. A second targeted candidate kept ordinary ConvNeXt-Tiny, our proven model family, and changed to author-provided supervised ImageNet-22k -> ImageNet-1k weights. Exact timm identifier: `convnext_tiny.fb_in22k_ft_in1k`.

- [Author implementation and published pretrained models](https://github.com/facebookresearch/ConvNeXt)
- [Exact model card](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k)
- Author weight URL and SHA-256: `results/short_screening/supervised22k_transfer_v1/preflight.json`.

S58 reused S56's exact frozen ImageNet1k control features. The candidate required one additional feature-extraction pass. Both used square224 identity images, normalized pooled features and the same training-only weighted StandardScaler + logistic regression (C1, seed42). No classifier or augmentation search was conducted.

| Frozen-feature probe | Accuracy | Macro-F1 |
|---|---:|---:|
| Fresh ImageNet1k ConvNeXt-Tiny control | 79.2415% | 0.643682 |
| Supervised22k ConvNeXt-Tiny candidate | 80.1065% | 0.665805 |

S58 took 67.7 seconds and passed the unchanged gate via **macro-F1 gain >=0.02 with no accuracy decline**. Its accuracy gain of 0.865 percentage points did not meet the alternative +1-point accuracy branch. This is modest supportive evidence to try fine-tuning, not proof of a material improvement over our 93.6128% trained ensemble or a forecast of its final performance.

## Authorized S59 run

One fresh pretrained model with a new seven-class head; full fine-tuning, not restarting any previous experiment. Same exploratory development split (7,009 train / 1,503 validation), seed42 and class order. Locked-test images/labels are excluded from loaders; the original test result remains unchanged.

- 224x224 bilinear square resize, ImageNet normalization, horizontal flips only, head dropout0.2.
- Weighted cross-entropy, normalized training-only sqrt inverse-frequency weights.
- AdamW: backbone3e-5, head1e-4, weight decay1e-4, batch16/accumulation2/effective32.
- AMP/GradScaler training, finite-gradient checks and norm1 clipping; FP32 identity validation.
- ReduceLROnPlateau: factor0.5, patience2, absolute accuracy threshold0.002.
- Hard20 maximum. After epoch15, stop only if neither cumulative +0.002 accuracy nor +0.003 macro-F1 has occurred for six epochs and the LR has been reduced. Tiny raw gains may update best checkpoints but cannot reset the meaningful counters.
- Separate raw accuracy best, raw macro-F1 best and atomic latest (including optimizer, scheduler, scaler, RNG, history and counters); preserve earliest ties.
- Save metrics, probabilities/predictions, class scores, raw/normalized confusion matrices, training curves, comparison figures, LR history, stop reason and registry row.
- No focal loss, Mixup, oversampling, CBAM, TTA or multi-resolution in this run.

Compared with historical S06, this changes pretraining/timm implementation, validation precision and meaningful scheduler threshold. It is not a pure architecture-only paired ablation.

Approximate RTX4060 8GB cost: 15–30 minutes, 2–4GB allocated VRAM. Actual throughput can differ. Once launched and first optimizer update/checkpoint/logging are confirmed, stop interacting and let the process run independently. Do not launch another model or ensemble automatically.

## Paths and commands

- Results/log: `results/short_screening/supervised22k_transfer_v1/s59_convnext_tiny_supervised22k_exploratory_seed42/`
- Checkpoints: `checkpoints/short_screening/supervised22k_transfer_v1/s59_convnext_tiny_supervised22k_exploratory_seed42/`
- Screening artifacts: `results/short_screening/supervised22k_transfer_v1/s58_frozen_feature_screen/`
- Config: `research/short_screening/supervised22k_transfer_v1.json`

From project root:

```powershell
.\.venv\Scripts\python.exe -m research.short_screening.supervised22k_transfer --run
# If interrupted, resume only from the latest atomic epoch boundary:
.\.venv\Scripts\python.exe -m research.short_screening.supervised22k_transfer --run --resume
```

Manual monitor from any PowerShell directory:

```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\short_screening\supervised22k_transfer_v1\s59_convnext_tiny_supervised22k_exploratory_seed42\train.log' -Tail 20 -Wait
```

After the user returns, close out S59, inspect class-wise errors and complementarity with S06/current S53, and consider one fixed replacement of the old Tiny member if evidence supports it. Do not select epochs using ensemble scores. These remain post-test exploratory development results, not a new untouched test estimate.
