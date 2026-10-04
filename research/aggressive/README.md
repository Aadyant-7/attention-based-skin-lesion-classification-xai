# Bounded enhanced development study

This post-test study is separate from the original untouched-test result (86.7598%). It uses only the existing 7,009-image training and 1,503-image lesion-disjoint validation cohorts. It does not reopen that test.

- Frozen recipe: `config.json` and `ENHANCED_PROTOCOL_FREEZE.md`.
- Evidence and adaptations: `VERIFIED_METHOD_AUDIT.md`.
- Start/recover from the project root: `.\.venv\Scripts\python.exe -u -m research.run_aggressive_enhanced_pipeline`.
- State/logs/results: `results/aggressive_enhanced/v1/`.
- Checkpoints: `checkpoints/aggressive_enhanced/v1/` (local, not uploaded to Git).
- Recovery verifies source/config/runtime hashes, skips completed stages and resumes committed checkpoints; no recipe change or discarded evidence.
- The master sequentially trains the three prescribed models, closes out fixed fusion and validation XAI, publishes results when available, and stops regardless of score.

Final prelaunch verification: nine CPU tests passed, including stable focal/Mixup, complete sample coverage, checkpoint guards, actual Grad-CAM/CBAM/occlusion and isolated result closeout. Disposable BF16 and full-FP32 optimizer preflights were finite for all three models. These probes were discarded and measured no validation accuracy.

Final microbatch is 16, effective batch 32. Earlier resource estimates in the chronological freeze appendices are superseded by the final fit report: approximately 4.1 hours of synthetic optimizer time at the maximum epoch allowance; plan roughly 4–7 hours including validation, augmentation, I/O and XAI. This is an estimate, not a deadline. BF16 uses no gradient loss scaling; loss, batch normalization and validation are FP32. The predefined full-FP32 recovery was also verified to fit.
