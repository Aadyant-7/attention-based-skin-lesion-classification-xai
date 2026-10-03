# S14 proposal: fixed224/320 ConvNeXt inference inside S12

**Deferred by the user on3 October2026; not launched.** [High-performance backbone/ensemble plan](HIGH_PERFORMANCE_ENSEMBLE_PLAN.md) replaces this next-run priority. Original proposal/config preserved; requires explicit future approval.
**Prepared only. Wait for GPU inference approval; no training or resolution search.**

## Question and rationale

Can averaging224px and320px ConvNeXt predictions improve the current S12 ensemble over its matched FP32 identity-only evaluation? Keep B0 and EfficientNetV2-S at224px; same three best checkpoints, same equal member weights. ConvNeXt receives equal224/320 resolution weights. Identity only: no flips, crops or other changes.

S13 fixed flips lost14 correct images versus FP32 identity, including net2 akiec and8 melanoma cases. Reject TTA. S12's diverse models remain useful, and ConvNeXt is the strongest standalone reference. A larger input can preserve finer lesion detail and changes feature-map sampling while avoiding another20-epoch training cost. This is a hypothesis: the saved ConvNeXt was trained at224px, so320px may cause a scale mismatch and regress. No guaranteed gain or percentage target.

[Verified Gessert multi-resolution paper](https://arxiv.org/abs/1910.03910) supports testing resolution diversity in skin-lesion ensembles. Its ISIC2019 dataset, balanced accuracy, training/cropping/subset search differ from ours; this is a small inspired inference intervention, not reproduction or a comparable accuracy claim.

## Frozen intervention and control

- Proposed aggregate: `(B0_224 + V2S_224 + 0.5*ConvNeXt_224 + 0.5*ConvNeXt_320)/3`.
- Reuse authenticated FP32 identity probabilities from S13 for all224px inputs; infer only ConvNeXt320px on1503 validation images.
- Control: S12 FP32 identity92.4817%/.877454; also disclose original S12 AMP92.4152%/.877023. Both already saved, not reselected.
- Exactly one fixed resolution candidate. No224/288/320/384 grid, weight tuning, flipped views, model retraining or checkpoint replacement.
-320px inference batch8 versus cached224px batch16, both FP32. BatchNorm in eval mode; disclose possible tiny implementation rounding differences. ImageNet normalization/direct-square bilinear resize unchanged.
- Standalone320px metrics saved as a diagnostic of the same inference collection, not an additional tuned candidate. Primary selection concerns the frozen equal-resolution ensemble.
- Same exploratory image-level split7009/1503/1503;563 shared train/validation lesions remain disclosed. Test images never loaded; later strict confirmation freshly trains selected finalists.

## Resources and launch commands

RTX4060 8GB, FP32, batch8, one ConvNeXt loaded. Estimated **1–3 minutes /2–4GB VRAM**, unbenchmarked at320px. S13 all12 passes took2.79min including plotting; this proposal needs one new pass at roughly2.04x224px pixel count. CPU hash/loading/plot overhead affects timing. Deployment uses4 passes instead of S12's3; the development run reuses the other saved probabilities.

After approval, from project root:

```powershell
.\.venv\Scripts\python.exe -u -m research.multires --config research/configs/phase3/s14_s12_convnext_224_320_exploratory_seed42.json
```

Manual monitor in another PowerShell:

```powershell
Set-Location 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
Get-Content .\results\structured_experiments\s14_s12_convnext_224_320_exploratory_seed42\run.log -Tail 30 -Wait
```

Recovery: same start command with `--resume`. Reuses validated complete320px CSV if available; otherwise recomputes the320px pass. Requires identical config/source signatures/parent hashes. No training optimizer or new weight checkpoint exists.

## Paths and evidence saved

- Config: `research/configs/phase3/s14_s12_convnext_224_320_exploratory_seed42.json`.
- Runner: `research/multires.py`; `--check` is CPU metadata/hash/probability validation only.
- Result/log: `results/structured_experiments/s14_s12_convnext_224_320_exploratory_seed42/`, `run.log`.
- Checkpoints unchanged: `checkpoints/structured/s03_efficientnet_b0_none_exploratory_seed42/best.pt`, `s06_convnext_tiny_none_exploratory_seed42/best.pt`, `s10_efficientnet_v2_s_none_exploratory_seed42/best.pt`. Only S06 needs loading for new inference; no new checkpoint files.
- Save probabilities, aggregate and320px diagnostic accuracy/macroP/R/F1, class scores, matrices/PNG300dpi+PDF figures, environment/source hashes, progress/log/record/closeout verification and master registry row.
- Comparisons: `results/model_comparison/structured/s14_multires/`; compare S12 original AMP, FP32 identity control and frozen resolution aggregate, including gains/losses and class trade-offs.

CPU preflight, three synthetic composition/guard tests and a320px random-model CPU shape/finite-output check passed. No actual320px validation inference or GPU fit/performance check performed.

If no worthwhile gain, retain S12 and stop resolution variations; consider one genuinely diverse stronger pretrained model as a separately justified proposal. GPU launch policy remains independent launch, confirm first saved batch/log/checkpoint loading, give manual monitor, then stop agent interaction until user returns.
