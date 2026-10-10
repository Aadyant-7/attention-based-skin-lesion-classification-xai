# S97: final bounded ConvNeXt warm-start proposal

Prepared 10 October 2026. **GPU training has not started; awaiting approval of this concrete longer-run proposal.** Architecture screening is over. This is one optimization experiment, followed by one predefined CPU fusion comparison.

## Verified starting evidence

| Method | Exploratory validation accuracy | Macro-F1 | Decision |
|---|---:|---:|---|
| S95 Tiny, decay 0.05, best epoch 17 | 91.6168% | 0.859748 | No improvement over original S06 |
| S96 fixed S95 replacement in S83 | 93.2136% | 0.874849 | Reject: gained 3, lost 15 |
| S79 Tiny + CBAM, accuracy-selected epoch 33 | 93.0140% | 0.877434 | Strongest saved Tiny accuracy checkpoint |
| S83 existing equal-six ensemble | **94.0120%** | **0.895201** | Retained numerical leader |

S95 checkpoints, predictions, class scores, matrices, curves and final epoch20 were independently verified. S96 used exactly one planned replacement. Details: `S95_S96_REGULARIZATION_CLOSEOUT.md`.

S79's original40-epoch trajectory already flattened: training accuracy reached about99.87%, meaningful improvement last occurred at33, and LR fell at25/37. More epochs alone are not evidence of better generalization. The proposed hypothesis is whether a **bounded optimizer restart from its actual best weights** finds a better basin. No guarantee of95%. On this1503-image cohort, S83 needs at least15 additional net correct predictions to reach95%.

## Exact initialization and epoch accounting

- Initialize from `checkpoints/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/best.pt`, epoch33, SHA256 `18bdf9c6237a4653c91fdd76bfa51dacb20c368a355e2692a57ca944ad66afc7`.
- This selected checkpoint contains weights, not optimizer/RNG state. S97 therefore uses **fresh AdamW moments, scheduler, AMP scaler and deterministic seed42**. It is a separate warm-start branch, not an uninterrupted resume of S79.
- The unchanged source `latest.pt` at40 supplies exact historical records1–33 only; its optimizer, RNG and epoch40 weights are not loaded into S97. All source files/checkpoints remain intact.
- Carry history1–33 as ancestry; train new branch epochs34–70. **70 cumulative branch epochs =33 inherited + at most37 new**, not70 new epochs.
- Minimum cumulative50 means17 new epochs before plateau stopping. Thereafter meaningful accuracy +0.002 or macro-F1 +0.003 resets patience8. Require a child-period LR reduction and two settling epochs. Tiny numerical gains still save raw best files but cannot reset meaningful counters. Hard cap70 always applies.
- Initial raw accuracy and F1 winners both come from parent33: accuracy0.9301397205588823/F1 0.8774343983529919. Parent F1-best35 remains external reference evidence and is not borrowed into the child's starting record.

## Retained successful recipe

ImageNet1k ConvNeXt-Tiny with CBAM on the final768-channel spatial map; full fine-tuning immediately;224 square bilinear RGB/ImageNet normalization; train-only horizontal/vertical flips and quarter turns; natural shuffled training with training-derived square-root inverse-frequency weighted cross-entropy; dropout0.2; original AdamW decay0.0001; effective batch32 (micro16, accumulation2).

Backbone/head-attention initial LRs1.5e-5/5e-5 are copied from the parent33 post-scheduler values. Fresh ReduceLROnPlateau references parent33 accuracy, factor0.5/patience3/meaningful threshold0.002/minLR1e-7. Safe AMP training with initial scaler1024, gradient clipping1, skipped overflow updates and diagnostic stop after3 consecutive overflows; FP32 identity validation, no TF32. Original two-epoch head warmup is already ancestral and is not repeated.

No stronger decay0.05, focal loss, oversampling, Mixup, EMA, new backbone, TTA or multi-resolution is added: none has demonstrated a useful improvement to this winning local recipe.

## Fixed post-training comparison

After the user returns, verify S97's best accuracy/F1/latest checkpoints, final metrics, prediction probabilities, class scores, raw/normalized confusion matrices and annotated curves. Training curves mark the optimizer/RNG restart after33 and separate new versus ancestral epoch counts.

Run **one** equal-six fusion: replace S83's existing S79 CBAM F1-selected35 member with S97's standalone F1-selected winner. Keep the other five saved FP32 sources and weights1/6 unchanged. No member, weight or ensemble-epoch search. If the S97 winner remains inherited33, label it inherited evidence rather than a new training achievement. Retain S83 if replacement is worse. The predeclared material gate requires at least8 net additional correct, accuracy gain>=0.005, and nondecreasing macro-F1/melanoma recall.

## Runtime and saved artifacts

Historical S79 median was77.37 seconds/epoch and peak allocated VRAM about1.6GiB. Budget **25–60 minutes** for17–37 additional epochs plus setup/closeout; approximately2–3GiB working VRAM on the RTX4060 8GB. Timing is an estimate.

- Config: `research/short_screening/s97_final_convnext_warmstart_v1.json`.
- Frozen proposal/source hashes and CPU verification: `results/short_screening/final_convnext_warmstart_v1/`.
- Results/log: `results/short_screening/final_convnext_warmstart_v1/s97_convnext_tiny_cbam_best_warmstart_exploratory_seed42/`.
- Checkpoints: `checkpoints/short_screening/final_convnext_warmstart_v1/s97_convnext_tiny_cbam_best_warmstart_exploratory_seed42/`.
- `latest.pt` contains full optimizer/scheduler/scaler/RNG, selected weights/predictions, LR history and meaningful counters. `best.pt`/`best_macro_f1.pt` contain selected weights/metrics for inference. Resume from **latest**, not a selected weights-only file.
- Root result package is raw-accuracy-selected; `macro_f1_selected/` is F1-selected; `final_epoch/` is final committed validation. `history.csv` includes ancestry; `ancestral_history.csv`/`child_history.csv` separate periods. Summary records inherited/new/total epochs, stopping reason, runtime and overflow evidence. PNG/PDF figures are generated on closeout.

## PowerShell commands

After approval, from the project root:

```powershell
Set-Location 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
.\.venv\Scripts\python.exe -u -B -m research.short_screening.s97_final_convnext_warmstart --start
```

Manual monitoring after launch:

```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\short_screening\final_convnext_warmstart_v1\s97_convnext_tiny_cbam_best_warmstart_exploratory_seed42\train.log' -Tail 20 -Wait
```

Recovery, only after the worker has exited:

```powershell
.\.venv\Scripts\python.exe -u -B -m research.short_screening.s97_final_convnext_warmstart --resume
```

Recovery verifies frozen metadata and latest full state; partial epochs replay from the last committed boundary. A setup failure before the initial checkpoint can be recovered only with matching metadata and no committed history/checkpoints; evidence is preserved. Once launched and first accepted GPU update plus logging/checkpointing are confirmed, Codex stops interacting. No automatic ensemble, test or next experiment is launched.

## Evaluation boundary

Use the existing image-level exploratory development split:7009 training/1503 validation. Related lesions can occur in both development partitions, so these results are not lesion-independent estimates. **No test inference is part of S97.** The original held-out cohort has already been evaluated. Only after final decisions freeze and explicit approval may it receive a repeated post-development audit; that result must not be called the first untouched independent test. No method changes follow that audit.
