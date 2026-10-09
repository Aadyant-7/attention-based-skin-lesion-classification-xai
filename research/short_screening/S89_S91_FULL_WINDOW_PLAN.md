# Finish Swin fairly; train the larger DeiT III Base

9 October2026. User requested a useful larger model rather than selecting Small for concurrent memory limits, and asked to allow20 epochs instead of cutting all runs at15. This bounded batch has exactly two sequential stages. No architecture sweep, third model, automatic30-epoch extension, ensemble or test evaluation.

## S89 epoch15 evidence and continuation

Best accuracy **90.6853%**, best macro-F1 **0.860448**, both at15; macro precision0.910125, macro recall0.819586. Melanoma recall66.4671%; akiec73.4694%. Checkpoints, optimizer state, metrics/predictions/confusion matrices/class scores/curves passed CPU verification.

| Epoch | Accuracy | Macro-F1 |
|---|---:|---:|
| 10 | 89.7538% | 0.839256 |
| 11 | 89.3546% | 0.828859 |
| 12 | 88.3566% | 0.825057 |
| 13 | 88.6893% | 0.818590 |
| 14 | 89.2881% | 0.831187 |
| 15 | **90.6853%** | **0.860448** |

LR reduced at13;15 is the strongest region. S06 Tiny's first15 best was91.1510% /0.856052. S89 is0.4657percentage points below that accuracy and has higher macro-F1, passing the original continuation criterion. It remains below S06's full20 accuracy91.7498%; do not claim improvement yet. Weighted validation loss rose to1.029913 at15 despite the classification gain, so confidence/generalization has not conclusively improved. More epochs are justified here, not guaranteed to help.

**Resume exact latest15 to20**, keeping configuration, LR/scheduler/scaler/optimizer/RNG, recipe, split, seed and selection rules unchanged. Preserve raw winners/latest and whole pilot15 evidence:

- Result snapshot: `results/short_screening/swin_transfer_v1/s89_swin_t_none_exploratory_seed42/pilot_epoch15/`.
- Checkpoint copies: `checkpoints/short_screening/swin_transfer_v1/s89_swin_t_none_exploratory_seed42/pilot_epoch15/`.
- Snapshot `closeout_verification.json` records hashes, finite optimizer/model state, strict CPU state loading and independently reconstructed metrics.

Expected remaining cost approximately6–12minutes, unmeasured. No extension to30: judge the completed20-epoch trend first and record any later amendment separately.

## S90 Small superseded before launch; S91 Base instead

DeiT III Small's source/CPU preflight remain historical preparation; **it was never trained and has no accuracy**. User interrupted before its GPU launch. No source/checkpoint is deleted.

**S91 exact model:** `deit3_base_patch16_224.fb_in22k_ft_in1k`. [Official model-maintainer card](https://huggingface.co/timm/deit3_base_patch16_224.fb_in22k_ft_in1k) documents86.6M original ImageNet parameters,17.6GMACs,224 input and author ImageNet22k->1k weights. [Author paper](https://arxiv.org/abs/2204.07118) supports supervised ViT transfer. Seven-class replacement reduces the parameter count; CPU preflight records the exact count and pretrained SHA. No source establishes a HAM10000 score for our protocol.

**Research question:** can higher-capacity global CLS-token attention provide stronger standalone and useful complementary predictions for our CNN/CBAM/Swin ensemble? Architectural/pretraining diversity is a rationale, not a prediction of95% or a guarantee that larger is better.

- Same7009train/1503validation exploratory manifest/hash, seed42, seven classes,224RGB square bilinear/ImageNet normalization, horizontal training flips.
- Full fine-tuning; native attention/LayerNorm/LayerScale; seven-class head dropout0.2; stochastic depth0.1. No addedCBAM on this first Base run; current S83 contains the CBAM branch.
- Same training-only normalized sqrt inverse-frequency weightedCE, AdamW3e-5backbone/1e-4head, weight decay1e-4. No focal, oversampling, Mixup, metadata, TTA or resolution change.
- **Microbatch4×accumulation8=effective32**. This preserves model capacity and the combined weighted CE denominator, not the smaller architecture. Record microbatch/dropout realization differences; not a pure matched architecture ablation.
- AMP training/GradScaler, finite loss/gradient handling and norm1 clipping; FP32 identity validation; raw accuracy and F1 winners independently saved, earliest ties.
- ReduceLROnPlateau factor0.5/patience2/absolute delta0.002; meaningful accuracy+0.002/F1+0.003 counters and LR history recorded.
- **Minimum20 and maximum20:** evaluate the full common window, no epoch15 pause or automatic extension. Counter history is descriptive during this fixed window.
- Start only after the Swin child exits successfully at20. Require at least6GiB free at model allocation;85% per-process allocator cap. Estimated standalone peak3–6.5GiB and runtime60–120minutes for20epochs; these are estimates until measured. If it fails, preserve evidence and stop the batch; no silent replacement.

All work remains further exploratory development on reused validation after earlier test outcomes; not independent confirmation. No test loader/images/labels are used, and no new test score is produced.

## Exact launch and monitoring

Batch from project root:

```powershell
.\.venv\Scripts\python.exe -B -u -m research.short_screening.run_s89_s91_completion
```

This invokes, in order:

```powershell
.\.venv\Scripts\python.exe -B -u -m research.short_screening.s89_swin_transfer --run --resume --stop-after-epoch 20
.\.venv\Scripts\python.exe -B -u -m research.short_screening.s91_deit3_base_transfer --run --stop-after-epoch 20
```

Master:

```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\short_screening\s89_s91_completion_v1\master.log' -Tail 20 -Wait
```

Swin while active:

```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\short_screening\swin_transfer_v1\s89_swin_t_none_exploratory_seed42\train.log' -Tail 20 -Wait
```

Base after its master START message:

```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\short_screening\deit3_base_transfer_v1\s91_deit3_base_none_exploratory_seed42\train.log' -Tail 20 -Wait
```

## Artifact and recovery paths

- Master code: `research/short_screening/run_s89_s91_completion.py`.
- Master state/plan/logs/stage consoles: `results/short_screening/s89_s91_completion_v1/`.
- Base runner/config: `research/short_screening/s91_deit3_base_transfer.py`, `s91_deit3_base_transfer_v1.json`.
- Base CPU preflight: `results/short_screening/deit3_base_transfer_v1/preflight.json`.
- Base results/logs/history/LR history/probabilities, accuracy/F1/final class/confusion PNG/PDF, training/comparison figures: `results/short_screening/deit3_base_transfer_v1/s91_deit3_base_none_exploratory_seed42/`.
- Base checkpoints: `checkpoints/short_screening/deit3_base_transfer_v1/s91_deit3_base_none_exploratory_seed42/` — raw accuracy best, rawF1 best and atomic latest with full optimizer/scheduler/scaler/RNG/counters.
- Master skips completed20 stages and resumes existing latest checkpoints. After interruption first verify no master/child is still alive; a surviving master lock must be inspected before recovery. Do not duplicate a running worker.
- Registry updates remain locked/atomic. The assistant confirms first resumed Swin update/log/checkpoint and stops; the background helper waits independently and starts only the prepared Base. Full analysis/complementarity waits for the user's return.
