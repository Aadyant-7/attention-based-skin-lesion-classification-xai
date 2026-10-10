## Current status: S95 ConvNeXt regularization prepared for authorized launch

10 October2026: user approved proceeding with the proven ConvNeXt direction. S95 tests AdamW decay0.05 versus historical S06's0.0001 using the same fresh ImageNet1k ConvNeXt-Tiny, full fine-tuning, exploratory cohort,224/horizontal flip/weightedCE/effective32 and20-epoch budget. Existing CBAM/S83 sources remain unchanged. Modern guarded FP32 validation is disclosed as a numerical amendment, so this is not a pure causal decay estimate. CPU pretrained forward/backward, actual optimizer step/group coverage, checkpoint/optimizer/RNG restoration, source hashes and test-loader rejection passed. No candidate accuracy exists yet. Live launch/completion status belongs to the registry and `results/short_screening/convnext_regularization_v1/`, not this static preparation note.

Plan/commands/resources/gates: `research/short_screening/S95_CONVNEXT_REGULARIZATION_PLAN.md`. No extension beyond20, new model, automatic weighting search or test evaluation. Once launch is confirmed, stop interaction. After the user returns, verify the standalone run and one fixed equal-six S83 replacement before deciding whether to keep it. Current observed exploratory leader remains S83 **94.0120% /0.895201**.

## Previous completed closeout: S89/S91 and S92–S94

9 October 2026 closeout: S89 Swin-T best90.6853% /0.860448 at15; its remaining five epochs did not improve the best. S91 DeiT III Base best90.5522% at13 (tied17), macro-F1 best0.850535 at17. Both completed20 and all accuracy/F1/final checkpoints and prediction packages were independently verified. S90 Small was superseded before training.

Three bounded CPU-only equal-probability additions used existing standalone macro-F1 winners: S92 S83+Swin93.6793% /0.889231 (8gained/13lost), S93 S83+DeiT93.6793% /0.889811 (7/12), S94 S83+both93.4132% /0.889986 (8/17). All lost net correct predictions and reduced melanoma recall. Retain S83 **94.0120% /0.895201** as the numerical exploratory leader. The transformers do recover some errors individually, but their actual equal-weight fusion does not improve this ensemble. No weighting search or extension beyond20 was performed; no next GPU run is queued. Report writing remains paused.

Full closeout: `research/short_screening/S89_S91_TRANSFORMER_CLOSEOUT.md`; complete comparison/class diagnostics/changed image IDs/PNG/PDF metrics packages: `results/short_screening/s92_s94_transformer_addition_cpu/`. Individual training folders and checkpoints remain unchanged except for normal completion through20; original Swin epoch15 snapshots are preserved. No new test scoring or training was performed during closeout. These are repeatedly selected exploratory validation results, not test results.

## Historical launch plans (superseded by completed results above)

Latest9October decision: S89 pilot15 passed its continuation gate (**90.6853% /0.860448**, both best at15 after LR reduction). Epoch15 results and all three checkpoints were independently verified and copied into `pilot_epoch15/` snapshot folders before any resume. Continue unchanged to20. User superseded the unlaunched S90 Small plan with a larger model/full20 window: **S91 DeiT III Base ImageNet22k->1k**, full fine-tuning, microbatch4×accumulation8=effective32, sequential after Swin exits. Exact two-stage commands/resources/artifact/recovery paths: `research/short_screening/S89_S91_FULL_WINDOW_PLAN.md`. No automatic30 epochs, third model, ensemble or test scoring. Live master state: `results/short_screening/s89_s91_completion_v1/`. Static notes do not imply either stage has finished.

S89 has subsequently been launched with an epoch15 boundary. On9October the user explicitly requested a concurrent new model. S90 DeiT III Small (global patch attention, author ImageNet22k->1k weights) is authorized for one epoch15 pilot alongside Swin. Independent runner/config, microbatch8×accumulation4=effective32, minimum3GiB initial free VRAM and per-process allocator cap40%. S89's recipe/checkpoints are not changed. Full rationale/start/monitor/recovery/artifact paths: `research/short_screening/S90_DEIT3_CONCURRENT_PILOT.md`. Live state is in each run's progress/logs and registry; this static note is not a completion claim. No automatic extension/third model/test scoring.

9 October 2026: S86/S87 audited four unused saved candidates and added only the two recovering the most S83 errors. Equal-seven S83+ResNet101 and S83+PanDerm Large each reached **93.7458%**, losing4 net correct versus S83; macro-F1 was0.891351 and0.888905. Neither passed the directional gate, so no joint equal-eight or all-model fusion was run. Full evidence and independently recomputed verification are saved under `results/short_screening/s86_s87_saved_member_addition/`. S83 remains the numerical leader at **94.0120% /0.895201**. No test or GPU inference/training was used for these checks.

**Next prepared experiment, awaiting explicit GPU approval:** S89 fully fine-tuned ImageNet Swin-T. Isolated runner/config, same exploratory cohort, weightedCE, FP32 validation; mandatory pilot pause15, hard20 cap and meaningful plateau counters. CPU pretrained forward/backward and split safety preflight are recorded in `results/short_screening/swin_transfer_v1/`. Proposal, gating rationale, estimated cost, start/monitor/recovery commands and artifact paths: `research/short_screening/S89_SWIN_PROPOSAL.md`. No training launched; existing frozen/historical training code remains unchanged. S88's conditional joint fusion was not executed.

## Earlier completed CPU stability checks

9 October 2026: two fixed CPU-only follow-ups did not improve S83. S84 softened only CBAM probabilities at temperature 2: **93.8789% /0.893894**, zero gained and two lost predictions. S85 averaged CBAM's saved accuracy/F1 winners within its existing one-sixth ensemble slot: **93.9454% /0.894622**, zero gained and one lost prediction. Reject both for accuracy development; retain S83 as the numerical leader (**94.0120% /0.895201**). All metrics, prediction probabilities, class scores, confusion/comparison PNG/PDF figures and registry rows are preserved. No training, new inference or test evaluation.

The diagnostic found at least one component was correct on 68 of S83's 90 errors, but that is label-based complementarity, not a deployable oracle. These two fixed fusion changes recovered none. Avoid extending this into temperature/checkpoint/weight sweeps. The next useful direction needs evidence of a train-only difficult-class specialist or genuinely stronger representation, reviewed against the already failed rescue/stacking/feature studies before any GPU proposal. No new GPU run is prepared or authorized. See `research/short_screening/S84_S85_CBAM_STABILITY_CLOSEOUT.md`.

At the user's request, accuracy development resumed9October2026. Keeping original ConvNeXt-Tiny and adding Tiny+CBAM is stronger than replacing Tiny. Fixed equal-six S82 reached93.9454% /0.893190; S83 uses the already-saved macro-F1-selected CBAM checkpoint35 and reached **94.0120% /0.895201**,1413/1503correct, with melanoma recall82.0359% unchanged versus S53. S83 is the highest observed exploratory validation accuracy. Its +6correct/+0.3992percentage-point gain still fails the predeclared material gate against S53; keep that policy reference and the numerical leader distinct. No GPU training/inference or test scoring occurred.

Read `research/short_screening/S82_S83_NEXT_DECISION.md` and the two closeouts. Both full prediction/class/confusion/comparison PNG/PDF packages, source hashes and registry rows are saved and verified. Do not launch a long run just to reproduce saved probability averaging. The next useful hypothesis concerns difficult-class specialist errors, after reviewing previous rescue studies; no costly run is prepared or approved.

S80/S81 remain unchanged: earlier five-model CBAM validation93.3466% /0.877017 and previously evaluated held-out-cohort audit87.0925% /0.801646. S80's10-case XAI/tables are complete; their labels/checkpoints cannot be reassigned to S83. Original S31 test remains86.7598% /0.794473. No further test evaluation is queued. Guide formatting references are preserved privately, but report writing is paused. Historical plans below are records, not launch authorization.

## Completed Phase 1 preparation

User-authorized Phase 1 completed on8October2026. S79 ConvNeXt-Tiny+CBAM specification, existing ensemble source hashes and development protocol are frozen. CPU forward/gradient/staging, meaningful-stopping and checkpoint/RNG/optimizer recovery checks passed. No GPU training/preflight, new accuracy score or test evaluation started. Full dedicated GPU runner and launch approval are still pending. Start at `research/final_cbam_development/README.md`; evidence is in `results/final_cbam_development/v1/phase1/`.

## Earlier completed screening: 8 October 2026

S76 actual trained-feature fusion was rejected (91.3506% versus reproduced S53 93.6128%). S77 matched published PanDerm Large frozen transfer reached88.9554% versus Base88.5562%; its predefined gate failed, so S78 was not run. No further model training or weight search is queued. S53 remains the retained single-image exploratory candidate (93.6128% /0.886857 macro-F1), not a new test result. Original strict locked-test result stays86.7598% /0.794473. Do not reevaluate or tune against that test.

Read `research/short_screening/S76_FEATURE_FUSION_CLOSEOUT.md`, `research/short_screening/S77_PANDERM_LARGE_CLOSEOUT.md` and `research/short_screening/RESULTS_ARTIFACT_INDEX.md`. Both substantive representation checks are complete, with full evidence preserved and no long training justified by their results. Any future costly proposal needs a concrete supported change and its own bounded protocol; no automatic small-variation loop. Earlier dated plans below are historical and do not authorize launches.

## Historical: S18 completed; research paused for tonight

S18 finished20epochs: best92.22%/.8549 at17; final91.28%/.8398. Modest accuracy gain versus Tiny with lower macro-F1; retain candidate for future ensemble review. No new ensemble run. User-approved15/20continuations superseded the old epoch8gate; future plan undecided until tomorrow. See research/phase3/S18_CLOSEOUT.md (root-relative). Historical proposals below remain preserved.

## Current decision after S15 (3 October 2026)

S15: 89.62%/.8075. One fixed S17 fusion: 92.61%/.8771, only two extra correct versus FP32 reference with lower macro-F1. DenseNet excluded from selected ensemble; S12 FP32 reference retained. Next proposal: staged ConvNeXt-Small S18, with 8/12/20-epoch approval gates. S16 B3/Swin deferred; S14 remains deferred. No new GPU run approved. See research/phase3/S15_CLOSEOUT.md and research/phase3/S18_PROPOSAL.md (paths relative to root).

# Research plan: 3–17 October 2026

**Active priority (3 October2026):** [High-performance heterogeneous ensemble phase](research/phase3/HIGH_PERFORMANCE_ENSEMBLE_PLAN.md): DenseNet201 next ([S15 proposal](research/phase3/S15_PROPOSAL.md)), EfficientNet-B3 second subject to evidence/approval, conditional Swin-T third; equal voting then a small bounded global-weight study. **S14 deferred by user; no new experiments launched.** Reference remains S12 92.42%/.8770, FP32 identity92.48%/.8775; S13 TTA rejected. Same exploratory split; strict confirmation postponed; original locked test untouched.

## Original Phase2 plan (historical)

The two-week target is a planning window, not a promise of a score. Final architecture is open to controlled evidence. **Phases 1–2 are complete; no new training or test evaluation was launched.** Start at [the Phase-2 handoff](research/phase2/README.md), [fixed recipe v1](research/phase2/EXPERIMENTAL_SETUP.md) and [planned model configs](research/configs/backbone_comparison/comparison_plan.csv). Phase 3 begins with CPU runner implementation/review; announce every GPU preflight/run before launching it.

## Phases and gates

| Phase | Work | Gate / output |
|---|---|---|
| 1 | Audit, preserve legacy, common registry/model interface/artifact logger, dataset counts and saved-result figures | Completed infrastructure audit; history unchanged; test images unopened. |
| 2 | Verify literature rows, dataset/protocol text, research question and comparison recipe | Completed focused primary-source table, research gap, verified protocol and recipe v1; unknowns explicit. |
| 3 | Implement/review common training runner, then sequential transfer-learning comparisons | 4–6 completed comparable run packages; no test selection. |
| 4 | Paired no-attention vs CBAM on 1–2 strongest backbones | Same head/recipe/seed; show class-wise changes and compute cost, including negative results. |
| 5 | One or two hypothesis-led improvements on strongest candidates | Predeclared metric/compute gates; avoid another broad tuning sweep. |
| 6 | Small set of two-model probability ensembles; third member only if justified | Members/weights fixed from development evidence; report inference cost. |
| 7 | Choose final single model or ensemble and freeze methodology | Exact configs, checkpoint hashes, class order, split and preprocessing written down. |
| 8 | One explicit locked-test evaluation | Only after freeze; save full metrics, predictions, matrix and plots. |
| 9 | XAI target-layer verification and representative explanations | Correct/incorrect melanoma, common/minority classes, confidence and true/predicted labels. |
| 10 | Paper tables/figures, results/discussion/limitations and full draft | Guide review; venue/template formatting after venue is known. |

Start literature writing in Phase 2 and methods/setup drafting alongside training; reserve final claims until Phases 7–9. This overlap saves time without inventing results. Phase 9 may use development cases to validate XAI implementation earlier, but final test examples follow Phase 8.

## Shortlist for Phase 3

The installed environment was checked: RTX 4060, 8 GB VRAM; PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128. `research/backbone_shortlist.csv` records exact installed weight-enum names and original ImageNet parameter counts; those are not HAM10000 results.

| Candidate | Role | Compute decision |
|---|---|---|
| EfficientNet-B0 | Small anchor linking to historical work | New no-attention common-head baseline; historical B0+CBAM remains a reference. |
| MobileNetV3-Large | Lightweight alternative | Useful deployment/compute comparator; do not assume it beats B0. |
| ResNet50 | Residual CNN comparator | Distinct family, explicit ImageNet weight version; smaller microbatch if needed. |
| DenseNet121 | Dense-connectivity comparator | Complementary family; memory checkpointing only if documented consistently. |
| ConvNeXt-Tiny | Modern convolutional comparator | Larger compute cost; include only after a short memory/time feasibility check. |
| EfficientNet-B2 | Same-family scale check | Optional sixth; do not add if four/five runs already answer the comparison or deadlines tighten. |

Choose five initial candidates; B2 is optional. These are feasible *candidates*, not verified training-memory fits. No model weights were downloaded to make this recommendation. Official model/weight references: [torchvision models](https://docs.pytorch.org/vision/stable/models.html), [EfficientNet-B0](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html), [ConvNeXt implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/convnext.html). A larger EfficientNetV2-L/ViT-L sweep is not a sensible first use of this GPU or deadline. Domain-specific pretraining remains allowed, with overlap/license caveats.

## Fair-comparison recipe fixed in Phase 2

The detailed versioned specification in `research/phase2/EXPERIMENTAL_SETUP.md` and `recipe_v1.json` is authoritative. The earlier `backbone_example.json` is a historical proposal; Phase 3 uses the six planned configs under `research/configs/backbone_comparison/`. Before changing an optimizer/batch/transform decision, record a recipe amendment and maintain matched controls.

- Use the exact existing lesion-disjoint manifest, seven classes, fixed seed and macro-F1 checkpoint selection; report accuracy, macro precision/recall/F1 and class metrics together.
- Start with 224-square ImageNet-normalized input, one documented augmentation recipe, training-only weighted CE, common seven-class GAP/dropout/linear head and no added CBAM. Record native architectural normalization and pretrained recipe differences.
- Use a common effective batch size (proposed 32), 20-epoch cap, five stale-epoch patience, AdamW and documented backbone/head learning rates. Run a short timing/memory preflight per backbone and adjust microbatch/accumulation rather than silently changing effective batch size. BatchNorm microbatch differences remain a limitation; avoid batch size one.
- Log exact weights, hashes, environment, seed, trainability/freeze stages, parameter counts, epochs and wall time. Similar epoch budgets do not imply identical optimization quality; disclose this rather than overclaiming a universal best architecture.
- Implement and record deterministic settings/worker seeds in the new runner; seed 42 alone is insufficient. Explicitly handle already-completed runs and safe resume so post-training bookkeeping never depends on an uninitialized epoch.
- The planned configs are not an executable training runner. Phase 3 must implement and verify that runner, resume/checkpoint semantics, effective-batch weighted-loss reduction and validation-only loading before launching.
- Historical B0 runs took roughly 10–43 minutes; different backbones/heads/augmentation can take longer. Derive a real remaining budget from preflight timings; do not claim an unmeasured exact ETA.
- If useful, repeat the top paired attention comparison with a second seed. A single seed cannot establish statistical significance.

## Suggested allocation

Days 1–2: Phase 1/2 and protocol freeze. Days 3–6: model comparison. Days 7–9: attention and limited improvements. Days 10–11: ensembles and final selection. Days 12–13: one test evaluation and XAI. Day 14: paper evidence pack and guide review. Reduce optional runs before sacrificing controls or reporting accuracy.

## What is needed from the team

Enough data, split manifests, local checkpoints and dependencies are already available for Phase 1 and planning. No user file is required to complete this phase. Before later final paper formatting, a target journal/conference template would help; before scheduling all runs, the actual daily GPU availability/meeting date would refine the budget. Neither blocks Phase 2. Do not ask the team to recollect material already in the repository.

## Commands available now

```powershell
# Rebuild indexes/tables/figures only; no model execution
.\.venv\Scripts\python.exe -m research.build
# Verify infrastructure only, using saved development evidence/synthetic tensors
.\.venv\Scripts\python.exe -m unittest discover -s research/tests -v
```

Phase-2 validation/table refresh: `.\.venv\Scripts\python.exe -m research.phase2.prepare` (CPU only). No manual command is required now. Next is Phase-3 CPU runner implementation and review, followed by a concrete announced GPU preflight. Do not launch historical training or `src.evaluate --confirm-locked-test`.
