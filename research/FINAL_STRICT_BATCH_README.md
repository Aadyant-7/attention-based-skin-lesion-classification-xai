# Final strict batch v1 — authorized, frozen architecture

One launch / recovery command from project root:

```powershell
.\.venv\Scripts\python.exe -u -m research.run_final_strict_training
```

Queue: S28 EfficientNet-B0 → S29 ConvNeXt-Tiny → S30 EfficientNetV2-S → S31 fixed equal FP32 ensemble → STOP. Sequential execution on RTX 4060 8GB. Fresh original ImageNet weights only; no exploratory state.

Config files: `research/configs/final_strict/`. Shared frozen recipe: `research/configs/final_strict_recipe_v1.json`. New runner isolated in `research/strict_train.py`; historical runner and outputs remain unchanged.

Strict split: `data/splits/split_assignments.csv`; 7009/1503/1503; identity-only test overlap verification, zero lesion overlap. Test rows excluded before label parsing, including raw metadata. No test Dataset/DataLoader or inference exists in this workflow.

Max 50, minimum review 15, patience 12; weighted CE, AdamW3e-5/1e-4, decay1e-4, clip1, plateau accuracy factor.5/patience2,224/ImageNet normalization/Hflip.5,AMP training and FP32 validation from start. No Mixup/focal/CBAM/TTA/weight tuning.

AMP safety: after unscaling, clip only finite gradients. Initial scaled-gradient overflow uses standard GradScaler rejection/scale reduction, explicitly logged; samples remain counted, and no NaN/Inf is clamped. Nonfinite forward loss still stops with diagnostics. GPU fit-check discovered and verified this clipping/scaler interaction before actual training.

**Checkpoint selection clarification:** latest explicit user instruction selects each member's independent earliest maximum validation-accuracy checkpoint, then fixed equal averaging. This supersedes the earlier proposal to select a composite ensemble epoch. Independent best macro-F1 and latest/resume states are retained for analysis, not substituted into the final ensemble.

Results/logs: `results/structured_experiments/s28_efficientnet_b0_final_strict_seed42/`, `s29_convnext_tiny_final_strict_seed42/`, `s30_efficientnet_v2_s_final_strict_seed42/`. Checkpoints under corresponding `checkpoints/structured/` IDs. Master/verification/ensemble/comparison figures: `results/final_strict/v1/`. Final report auto-generated: `research/FINAL_STRICT_VALIDATION_RESULTS.md`. Master registry preserves all historical rows and adds protocol-labelled strict entries.

Resume: run same queue command. OS master lock prevents duplicate masters; live workers reattach, compatible committed latest checkpoints resume; verified completed models skip. Logs append; atomic epoch boundaries preserve optimizer/scheduler/scaler/RNG. A numerically failed record stops safely with diagnostics and requires diagnosis rather than silently retrying/clamping/skipping. Interrupted initialization without a checkpoint is preserved and stops for inspection. No test evaluation or extra model is scheduled after closeout.

Standalone artifacts include accuracy/F1/latest metrics and predictions/probabilities, checkpoint identities, normalized/raw matrices, per-class scores, PNG/PDF curves, LR history, runtime/epoch/parameter summaries. Exploratory table remains separate. Strict ensemble always uses FP32 identity predictions with exactly 1/3 each.
