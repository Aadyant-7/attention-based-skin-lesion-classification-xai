# S79 Phase 2 complete: ready for training approval

8 October 2026. The dedicated runner is implemented and preflight passed. **Full S79 training has not started.** The Phase 1 configuration, policy and source hashes remain unchanged. No original test labels/images were loaded, no experiment registry row was added, and no actual S79 training folders/checkpoints exist yet.

## Implemented behavior

- Explicit `--start`, `--resume` and CPU-only `--check`; OS run lock prevents simultaneous runs. Existing run folders cannot be overwritten by a fresh start. A completed resume returns without GPU initialization or further training.
- Fresh cached ImageNet ConvNeXt-Tiny, CBAM before pooling, the fixed two-epoch warm-up/full-stage transition, natural sampling, moderate geometric augmentation and training-only sqrt-weighted CE. Microbatch16/effective32 loss is normalized by the entire effective batch's target-weight sum.
- AMP/GradScaler training; validation/inference and equal fusion use FP32. The numerical implementation uses initial scaler1024, disables TF32, and stops/preserves diagnostics for nonfinite inputs/logits/loss/gradients/state rather than silently changing the recipe. These numerical choices fill unspecified Phase 1 runtime settings and are recorded in the launch manifest; no Phase 1 config was edited.
- Fixed40-epoch ceiling/minimum20, meaningful accuracy/F1 counters, LR scheduling and late plateau. Raw records save independent accuracy/F1 winners without resetting patience for tiny changes.
- Atomic `latest.pt` saves model/optimizer/scheduler/scaler, CPU/CUDA RNG, deterministic per-epoch loader seed policy, history, meaningful counters, both best snapshots, predictions and runtime before any CSV/registry mutation. Partial epochs replay from the previous committed boundary; there is no mid-epoch continuation.
- Resume repairs CSVs, prediction packages, best checkpoints and the registry from committed state before continuing. Full config/source signatures and finite state are checked. Failures preserve `latest.pt`; numerical diagnostics are separate, explicitly non-resumable snapshots.
- At completion, save accuracy/F1/final-epoch result packages, probability CSVs, per-class scores, raw/normalized confusion matrices, curves and PNG/PDF figures. Automatically close out one fixed CPU S80 comparison replacing only S53's Tiny with the accuracy-selected S79 checkpoint. No further training, inference combination search or XAI run follows automatically.

## Matched reference precision

The S76 FP32 inference caches already hold predictions from the exact five frozen reference checkpoints. Their checkpoint/cache/signature hashes and validation IDs/class order are checked. The reproduced reference is93.6128% /0.886857 macro-F1. S80 uses these cached FP32 probabilities for the other four unchanged components, avoiding new GPU work and inherited AMP probabilities. Equal averaging is FP32; no weights are selected from validation labels.

## Preflight evidence

`results/final_cbam_development/v1/phase2/gpu_preflight.json` records a16.32-second disposable synthetic probe on RTX4060:

- AMP warm-up/full-stage updates and FP32 inference finite.
- CUDA RNG/optimizer/scheduler/scaler recovery reproduced the next update bitwise.
- Single-model peak allocated GPU memory:1,514.33 MiB. The separate two-model recovery peak is also recorded; these figures exclude driver/display and other processes' memory.
- Synthetic full-epoch compute estimate:67.77 seconds, excluding real loading/augmentation/checkpoint/reporting costs. Budget roughly60–90 minutes for40 epochs; convergence may stop earlier after20. This is an estimate, not a deadline.
- No accuracy was measured; probe checkpoints were temporary and deleted. The original registry remained unchanged.

Two CPU tests passed: committed-checkpoint repair and complete artifact closure without optimizer work; deterministic zero/two-worker loader replay and forbidden-partition rejection. Artifact fixtures reused saved predictions in temporary folders and were deleted; they are not new research results. `train --check` also passed and did not initialize CUDA. Evidence/launch hashes: `results/final_cbam_development/v1/phase2/`.

## Start command (after approval)

Open PowerShell at the project root:
```powershell
Set-Location 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
.\.venv\Scripts\python.exe -B -u -m research.final_cbam_development.train --start
```

This starts the full bounded run, not a screening pilot. If Codex launches it, use a hidden detached process; confirm logging/initial checkpoint creation, provide PID/monitor path, then stop interacting as instructed. The runner performs its own artifact and fixed CPU ensemble closeout after convergence.

Monitor from a separate PowerShell window:
```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\final_cbam_development\v1\s79_convnext_tiny_cbam_exploratory_seed42\train.log' -Tail 20 -Wait
```

Results: `results/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/`.

Checkpoints: `checkpoints/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/` (`latest.pt`, `best.pt`, `best_macro_f1.pt`). Initial `latest.pt` is committed at epoch0; selected best checkpoints appear after the first successful validation epoch. S80 metrics/predictions/figures live in the run's `fixed_ensemble/` folder.

Resume only after the old process has exited:
```powershell
Set-Location 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
.\.venv\Scripts\python.exe -B -u -m research.final_cbam_development.train --resume
```

The command resumes the latest committed epoch, retaining the same max40/config. A power cut may lose the incomplete epoch. Do not use diagnostic snapshots, delete folders, relabel an old checkpoint, modify signed code or restart from scratch. If startup fails before epoch0 exists, preserve the error and inspect it; there is no checkpoint to resume yet. Numerical failures stop for diagnosis; the runner does not automatically switch precision or alter LR/loss.

## Reporting gate

No preflight result predicts accuracy. S79's selected checkpoint follows earliest maximum standalone accuracy; the original Tiny reference was F1-selected. This is a combined training-package comparison. The material ensemble gate remains at least8 net correct/+0.005 accuracy with nondecreasing macro-F1 and melanoma recall. A failed gate preserves both the attention variant's real score and the stronger reference; it does not trigger a search. Original locked-test report stays immutable. Grad-CAM and the user's report formatting requirements are later phases.
