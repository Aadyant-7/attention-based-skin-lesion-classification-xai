# S13 proposal: fixed four-flip TTA on the S12 ensemble

**Historical approved proposal: completed. [Closeout](S13_CLOSEOUT.md) rejects four-flip TTA; original prelaunch plan below is retained.**
**Prepared only; GPU inference awaits approval. No retraining.**

## Question and choice

Does a single fixed geometry-averaging intervention improve S12's92.42%/.8770 on the identical exploratory validation cohort? B0/ConvNeXt/V2-S already provide complementary errors. Test inference invariance before spending another20-epoch backbone run. The integrated literature review's preprocessing/augmentation section supports investigating TTA as its own intervention; legacy TTA evidence is context, not a promised gain.

Freeze four224px views: identity, horizontal flip, vertical flip, both flips. Equal mean of views within each model, then equal mean of the three accuracy-selected parent models. No crop/resolution/weight/view tuning. Include a freshly inferred **FP32 identity-only ensemble control** to separate the precision effect from TTA; compare saved S12 AMP, FP32 identity and FP32 TTA explicitly. All forwards FP32 because S10 exposed FP16 overflow. This changes inference precision, not training; disclose both effects. If FP32 logits are nonfinite, stop and preserve evidence; never sanitize outputs.

Same exploratory image-level manifest:7009/1503/1503, validation only; no test loader. S10 selected model is epoch15; original recovery remains disclosed. Three fixed parent checkpoints, no new weights/training/checkpoint selection. Twelve model passes per image instead of three; any gain must justify that cost. If no useful gain, retain S12 and stop TTA variations.

## Budget and commands

RTX4060 8GB, one model loaded at a time, batch16, FP32; approximate **5–15 minutes**, **3–5GB VRAM**. Estimate is unbenchmarked; can differ with system load. About5MB saved per-view CSVs plus aggregate predictions/plots; no new training checkpoints.

From project root, after approval:

```powershell
.\.venv\Scripts\python.exe -u -m research.tta --config research/configs/phase3/s13_s12_four_flip_tta_exploratory_seed42.json
```

Manual monitor from a second PowerShell:

```powershell
Set-Location 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
Get-Content .\results\structured_experiments\s13_s12_four_flip_tta_exploratory_seed42\run.log -Tail 30 -Wait
```

Recovery: same start command with `--resume`. Completed parent/view CSVs are validated and reused; interrupted view recomputed in full. Config/source signature and parent hashes must match. No partial weight checkpoint exists because this is inference only.

## Storage / readiness

- Config: `research/configs/phase3/s13_s12_four_flip_tta_exploratory_seed42.json`.
- Runner: `research/tta.py`; `--check` checks metadata/hashes on CPU without CUDA or image inference.
- Results/log: `results/structured_experiments/s13_s12_four_flip_tta_exploratory_seed42/`.
- Existing checkpoint directories: `checkpoints/structured/s03_efficientnet_b0_none_exploratory_seed42/`, `s06_convnext_tiny_none_exploratory_seed42/`, `s10_efficientnet_v2_s_none_exploratory_seed42/`; each `best.pt`, fixed by SHA in config. Originals unchanged.
- Outputs:12 per-model/view probability CSVs, TTA and FP32 identity metrics/predictions/class scores/matrices/PNG+PDF figures, progress/log/environment/source signature/record/closeout, registry entry and `results/model_comparison/structured/s13_tta/`.
- Training curves remain the parents' curves; no invented new epochs/curves/checkpoints.

CPU preparation checks passed; no actual GPU inference/accuracy measured yet. Launch policy: independently start after approval, confirm log and parent checkpoint loading/output progress, return manual monitor, then stop agent activity until user returns. No further GPU run authorized.
