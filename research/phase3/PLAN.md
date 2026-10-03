# Phase 3 — evidence-led experiments

**Prepared on 3 October 2026; GPU approval pending. No structured training launched.**

Literature input: [integrated review](../literature/scispace_analysis/INTEGRATED_REVIEW.md), [all 50 screening decisions](../literature/scispace_analysis/screening.csv), and the retained Phase 2 primary references/base-paper audit. Phase 2 recipe v1 remains unchanged; original Phase 2 files are preserved as the planning snapshot.

## Decisions after the expanded review

1. **First: EfficientNet-B0 without added CBAM.** It is the affordable ImageNet anchor and the missing matched control for our historical B0+CBAM work. Research question: what balanced validation performance does the common transfer-learning recipe achieve before added attention? This is a new seven-class common-head run, not retraining the historical weighted experiment.
2. **Next proposal after evidence: MobileNetV3-Large**, the base-paper lightweight comparator. Assess curves, class recall, learning-rate changes and cost before deciding the next architecture. Residual ResNet50, dense DenseNet121 and modern ConvNeXt-Tiny remain the planned distinct-family comparisons. No automatic multi-model launch. B2 remains optional after controls and budget review.
3. **Then matched CBAM**, on one/two strongest feasible backbones, identical head/recipe/seed with native SE disclosed. Review minority-class changes and inference cost. Repeat leading control/attention pair at seed43 if budget permits; +.005 macro-F1 remains a practical threshold, not significance.
4. **One diagnosed improvement at a time:** training-only balancing alternative if minority recall is weak; moderate geometry/colour augmentation if curves/errors justify it; resolution/multiscale intervention if lesion-detail loss is evident. No simultaneous balancing, hair-removal, segmentation, activation and transformer additions. Record a versioned amendment before altering v1; rerun matched controls when necessary.
5. **Small ensemble only with complementary errors:** use saved validation probabilities, inspect disagreement and class-specific errors, start equal two-member probability averaging. One fixed third member only if justified. Any learned weights/stacker require development-only fitting and honest validation-selection disclosure; never fit using locked-test labels. IGIA/custom six-model/GAN pipelines are deferred.
6. **XAI/final:** review labelled correct/incorrect development Grad-CAM cases and target-layer behavior; freeze selection/configs/member weights before the single final test evaluation. Future test/XAI runners need separate review and approval.

## First proposed run

| Field | Proposal |
|---|---|
| ID | `s01_efficientnet_b0_none_strict_seed42` |
| Config | `research/configs/phase3/s01_efficientnet_b0_none_strict_seed42.json` |
| Config SHA-256 | `71ced4b777f9ecd64d18779d1e90f730d16f2b8eab9878d3ae8b326477cb8d82` |
| Runner | `research/train.py`; `--check` uses metadata/filenames only and no model/CUDA/downloads/output registry rows. |
| Model | ImageNet `EfficientNet_B0_Weights.IMAGENET1K_V1`, no added CBAM, native SE retained; GAP/dropout.2/linear7; all layers fine-tuned. |
| Data | Frozen strict lesion-disjoint manifest;7,009 training references and 1,503 natural validation images; no test loader. |
| Recipe | Existing v1:224 RGB/ImageNet normalization, training flip.5, sqrt inverse-frequency weighted CE, AdamW backbone3e-5/head1e-4, AMP, deterministic settings. |
| Batch/budget | Microbatch16 × accumulation2, effective32;20-epoch cap,5 stale macro-F1 epochs. First epoch is the real measured feasibility/timing observation. No separate unapproved GPU probe. |
| Selection | Earliest highest full-validation macro-F1; also save accuracy, macro precision/recall and seven-class scores/support. |
| Resource estimate | One RTX 4060 8GB. Budget roughly20–60 minutes and up to20 epochs; planning estimate from historical B0 runs, **not a benchmark of this new runner**. Actual GPU fit/time are unmeasured; first-epoch logs provide an estimate. Allow1–2GB for checkpoint snapshots/temporary atomic writes, plus cached ImageNet weights. |

Start from the project root **only after explicit approval**:

```powershell
.\.venv\Scripts\python.exe -u -m research.train --config research/configs/phase3/s01_efficientnet_b0_none_strict_seed42.json
```

Live monitor in another PowerShell at the project root, after the log exists:

```powershell
Get-Content .\results\structured_experiments\s01_efficientnet_b0_none_strict_seed42\train.log -Tail 30 -Wait
```

Compact current status (rerun whenever needed):

```powershell
Get-Content .\results\structured_experiments\s01_efficientnet_b0_none_strict_seed42\progress.json
```

Read-only readiness check, safe before approval:

```powershell
.\.venv\Scripts\python.exe -m research.train --config research/configs/phase3/s01_efficientnet_b0_none_strict_seed42.json --check
```

## Saved artifacts and failure behavior

```text
checkpoints/structured/s01_efficientnet_b0_none_strict_seed42/
  latest.pt                 atomic epoch boundary: model, optimizer, scheduler,
                            scaler, RNG, best state/metrics/probabilities, history,
                            config/code/runtime identities and cumulative time
  best.pt                   selected model + config/class order/metrics
results/structured_experiments/s01_efficientnet_b0_none_strict_seed42/
  config.json, environment.json, pretraining.json, record.json
  train.log, progress.json, history.csv, training_summary.json
  validation_metrics.json, validation_predictions.csv
  figures/                  raw/normalized confusion matrix CSV+PNG+PDF,
                            class scores/support and training curves PNG+PDF
results/master_experiment_registry.csv
                            new era=structured training_run only on actual launch
```

Log status includes epoch/step and best macro-F1; epoch history includes timing, measured peak allocated VRAM, LR and optimizer-update count. Validation timing is end-to-end including loading/transfers, not pure model inference latency. Parameter count and exact pretrained download hash are recorded. Checkpoints/logs remain local/ignored; metrics/configs/figures are portable tracked evidence after a completed run. Run `.\.venv\Scripts\python.exe -m research.compare` for the common-recipe comparison table and PNG/PDF graph once at least two comparable saved results exist. It filters completed structured v1 controls and never pools historical exploratory rows. No comparison output is created with zero completed runs.

Preparation verification: 21 CPU tests passed, including all six backbone interfaces, effective-batch weighted gradient equivalence, test-partition rejection, RNG restore, interrupted checkpoint preservation, resume reconciliation and completed-run GPU bypass. `python -m research.verify_layout --phase3-preparation` verified 247 immutable historical files and registry references without opening images. See `results/audit/phase3_preservation_verification.json`. **Actual CUDA fit, AMP execution, training throughput and complete GPU lifecycle remain untested until approval.**

Failures mark the run failed and preserve atomic epoch checkpoints. Resume explicitly with the same command plus `--resume`; code/config/runtime mismatch is rejected. Resume repairs history/registry from the committed checkpoint, including after a final bookkeeping failure. Partial epochs rerun; completed IDs exit without GPU/training. A process lock prevents duplicate concurrent resumes. If interrupted before the first committed epoch, no resumable checkpoint exists: inspect retained logs and propose a separate ID rather than deleting evidence. OOM does not silently lower the batch: stop and propose a common microbatch8/accumulation4 recipe amendment before comparison runs.

## Review after each approved run

Report selected epoch and best accuracy separately if different; retain macro-F1 checkpoint selection. Compare minority recall, confusion patterns, convergence and cost. Check whether an apparent gain just increases nevus accuracy. Diagnose underfitting/overfitting or recipe limitations before spending on more backbones. Validation is reused for development; acknowledge selection bias and reserve the untouched test for the final frozen methodology. No promised accuracy target or literature score replaces measured evidence.

Phase 3 progress counts preparation as a portion of the work; the model comparisons remain pending approval and actual results.
