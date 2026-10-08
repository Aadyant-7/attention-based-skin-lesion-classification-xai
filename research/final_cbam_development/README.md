# Final attention-inclusive development: Phase 1

Prepared 8 October 2026 on `structured-research`. Phase 1 is complete. No GPU preflight, training run, new accuracy result or locked-test evaluation has started. The full GPU runner and its artifact closeout still require Phase 2 implementation and launch approval.

## Frozen experiment

Config: `research/configs/final_cbam_development/s79_convnext_tiny_cbam_v1.json`.

S79 is one fresh ImageNet ConvNeXt-Tiny + CBAM training package, not another architecture search. CBAM uses reduction16 and spatial kernel7 on the final768-channel spatial features before global pooling. The native ConvNeXt normalization, dropout0.2 and seven-class linear head are retained. Two warm-up epochs train the attention/head; full fine-tuning follows using the same optimizer. Maximum40 total epochs, minimum20 before convergence stopping; finite-value failures may stop earlier.

Use the existing7,009/1,503 exploratory development split, seed42,224px square RGB and ImageNet normalization. Training-only horizontal/vertical flips and uniform90-degree rotations provide moderate geometric augmentation. Keep natural sampling and training-only sqrt inverse-frequency weighted CE. AdamW backbone LR3e-5, attention/head1e-4, weight decay1e-4, microbatch16/effective32 and gradient clip1.0. FP16 AMP training is proposed; validation/final inference is FP32 identity-only. CPU preparation does not verify GPU AMP safety or memory.

Patience8 resets only for cumulative accuracy gain>=0.002 or macro-F1 gain>=0.003 from the last meaningful reference. Retain independent raw-best checkpoints. Accuracy-reactive ReduceLROnPlateau has factor0.5/patience3/absolute threshold0.002. Convergence stopping requires an actual LR reduction and two epochs to settle. A flat late plateau after30 is recorded separately;40 is the hard epoch cap. Every rule is recorded in the config and tested in the shared `protocol.py`.

## Fixed comparison

Reference S53:93.6128% exploratory validation accuracy /0.886857 macro-F1. Its exact five checkpoint and prediction hashes are in the frozen manifest. After S79 training, replace only Tiny with S79's earliest accuracy-selected checkpoint; retain Small, DenseNet201, EfficientNetV2-S and B0, all weights0.2. No ensemble epoch/member/weight search. Material gate: at least+0.005 accuracy and eight net additional correct predictions, with macro-F1 and melanoma recall nondecreasing.

S53's Tiny is macro-F1-selected; S79 is accuracy-selected under the new predeclared rule. Duration, augmentation, attention and selection differ. This is a package comparison, not a causal CBAM ablation. The existing matched S03/S05 study supplies separate CBAM evidence. If the attention-inclusive final system is weaker, report its own result and retain S53 as the stronger reference. A mandatory attention component does not justify relabelling S53's score as its score.

## Completed CPU checks

`results/final_cbam_development/v1/phase1/phase1_report.json` records:

- Cached explicit ImageNet weights verified by hash and loaded tensors; five source checkpoint/prediction hashes verified unchanged.
- Correct class order,7009/1503 development counts, training-only class weights and zero original-test image/lesion overlap. Only test identity/group columns were inspected for exclusion; no test labels/images were loaded.
- Existing exploratory overlap documented:563 train/validation shared lesions,596 affected validation images. This protocol is not lesion-independent.
- CPU finite forward and attention gradients; warm-up backbone unchanged; full-stage backbone updates and carried optimizer state verified.
- Temporary atomic checkpoint round-trip and RNG/optimizer recovery reproduced the next update bitwise. Mismatched config/signature, diagnostic snapshots, wrong counters and invalid epoch boundaries rejected.
- Minimum training window, tiny raw records, cumulative gains, macro-F1 gains, LR opportunity, late plateau, hard cap and invalid-metric stopping checks passed.
- Deterministic validation, seeded augmentation, forbidden-partition loader rejection and effective-batch weighted-loss normalization verified.

Disposable CPU optimizer probes used synthetic inputs; their losses/history are explicitly probe data, not experiment metrics. Probe files were removed, the experiment registry stayed byte-identical, and no actual S79 checkpoint/result folder was created. CUDA was never initialized. CPU recovery verifies the CPU path; CUDA RNG/scaler/loader recovery remains a GPU runner requirement.

Freeze manifest: `results/final_cbam_development/v1/phase1/frozen_manifest.json`. It records configuration, shared code, pretrained weights, split and ensemble source hashes. Subsequent code amendments need an explicit new freeze/amendment before training; do not alter this evidence silently.

Recheck Phase 1 from project root (completed probes are not repeated):
```powershell
.\.venv\Scripts\python.exe -B -u -m research.final_cbam_development.prepare_phase1
```

## Phase 2 requirements before launch

Implement the dedicated runner using the frozen protocol; do not pass the new recipe to the historical runner, which rejects unregistered changes. Validate GPU memory/timing and AMP on disposable inputs, then freeze runner hashes. Announce exact start/monitor/resume commands and the measured resource estimate; wait for training approval. After launch/logging/checkpoint confirmation, follow the established launch-and-stop rule.

Future outputs (reserved; not yet created):

- `results/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/`: logs, resolved config, history/LR/stopping counters, summaries, both selected checkpoint prediction/metric packages, final-epoch metrics, class/confusion/curve PNG/PDF and registry closeout.
- `checkpoints/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/`: independent raw accuracy/F1 winners and latest full state; immutable historical checkpoints elsewhere stay preserved.

Commit the full latest epoch state atomically before updating history/registry/plots. Resume must repair bookkeeping from committed state without repeating an epoch. Save CUDA/Python/NumPy/Torch and loader seed/RNG state, optimizer, scheduler, scaler, both best records, meaningful counters, LR history and configuration/source hashes. Stop and preserve diagnostics on nonfinite inputs/logits/loss or model/optimizer state. FP32 validation is the primary policy, not a fallback-selected result.

## Reporting and XAI

The original strict test report remains86.7598% /0.794473 macro-F1. This new exploratory run cannot supply another untouched score on that cohort. Report its selected epoch as best exploratory validation, not average accuracy. Repeated validation selection remains a limitation. CPU preparation measured no accuracy gain.

Later Grad-CAM must cover the final component models including CBAM, correct/incorrect cases and melanoma/akiec; verify target layers and label maps by originating model. Combine maps only with an explicitly documented visualization rule. Final figures, tables and paper formatting follow the user's report guidelines when supplied.
