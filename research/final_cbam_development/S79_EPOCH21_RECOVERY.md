# S79 epoch21 gradient-overflow recovery

The run stopped at effective batch117 of epoch21 when gradient clipping detected a nonfinite total norm. Epoch20 was the latest fully committed checkpoint (4,380 successful optimizer updates). The diagnostic snapshot contained116 further successful updates before the failing batch. Both model and optimizer tensors in the valid checkpoint and failure snapshot are finite. The loss scale was4096; the original input/logit/loss guards had passed before backward/clipping failed. The exact offending gradient tensor/batch was not retained by the original diagnostic snapshot, so AMP backward overflow is the supported explanation, not a proven identification of a particular module or image.

The implementation incorrectly made a recoverable AMP gradient overflow fatal before GradScaler's normal skip/backoff handling. The numerical amendment skips that optimizer update, lowers the scale through GradScaler, records affected gradient parameter names/image IDs/scales, and retains strict finite forward/loss/model/optimizer checks. Three consecutive gradient overflows stop for diagnosis. Actual successful optimizer updates and skipped updates are tracked separately; the full history/closeout/registry notes record this amendment. No automatic precision or hyperparameter change occurs.

A disposable tiny GPU test injected infinite gradients, verified unchanged model/optimizer state on the skipped update, scale4096->2048, and a successful subsequent finite update. It measured no skin-lesion accuracy. The config, split, class order, architecture, loss, LR, scheduling, augmentation, checkpoint-selection rules and40-epoch ceiling remain unchanged. This is a transparent numerical-policy amendment; it can alter the number of successful updates and must be disclosed, rather than called numerically identical to the interrupted implementation.

Resume uses `latest.pt` at epoch20 and replays the uncommitted epoch21. An exact backup is `checkpoints/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/recovery_epoch20_original.pt`. Both hashes are `ca508dcee16327a6315134e1b354bef6e5117dde5dcd3ccc046043aa5d9b6bb3`. Original failure snapshots/logs and execution-source copies are preserved. The original Phase2 launch manifest is not overwritten: `results/final_cbam_development/v1/phase2/amp_recovery_manifest.json` records the explicit old/new signatures and checkpoint hash. Further resumes use the amended signature in newly committed checkpoints.

From project root:
```powershell
.\.venv\Scripts\python.exe -B -u -m research.final_cbam_development.train --resume
```

Monitor:
```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\final_cbam_development\v1\s79_convnext_tiny_cbam_exploratory_seed42\train.log' -Tail 20 -Wait
```

The run's `amp_overflow_events.json`, history skipped-update counts and final training summary preserve numerical evidence. No new model or original-test evaluation is authorized by this recovery.
