# Attention-Based Skin Lesion Classification with Explainable AI

B.Tech CSE Minor Project · HAM10000 · seven classes · active branch: **structured-research**.

## Active study — 10 October 2026

[Separate paper-guided image-level workflow V2](research/image_level_replication/v2/README.md): **revised Phase 1 complete**, stratified **ten-fold** outer assignments with inner validation, seed 42, natural lesion overlap allowed. Actual fitting/inner-validation/assessment sizes are approximately **81/9/10**; a later fresh outer-90% refit is optional and requires a separate proposal. [Saved plan](research/image_level_replication/v2/PLAN.md) · [split audit and preservation checks](research/image_level_replication/v2/PHASE1_CLOSEOUT.md) · [paper-method review](research/image_level_replication/v2/PAPER_PROTOCOL_REVIEW.md). No new model trained and no ten-run queue scheduled. Next is one inner-fold B3 recipe proposal.

The earlier [70/15/15 V1](research/image_level_replication/README.md) is preserved as an inactive preparation, with all manifests, code and figures unchanged. The user relaxed the split requirement before Phase 2; V2 follows the reference's K10 structure as a documented adaptation, not a certified exact reproduction.

Previous S83 results remain **94.011976% exploratory validation / 87.558217% original-cohort test audit** ([report](research/FINAL_S83_TEST_AUDIT.md)). Fresh external pretrained weights are required for the new study; original results/checkpoints/manifests are preserved.

## Start here

| Open | What it contains |
|---|---|
| [Research workspace](research/README.md) | Active code, result contract and navigation. |
| [Phase 2 handoff](research/phase2/README.md) | Literature, dataset protocol and frozen comparison plan. |
| [S01 completed result](research/phase3/S01_CLOSEOUT.md) |85.63% strict validation accuracy/.7791 macro-F1; verified artifacts and historical context. |
| [Two-stage strategy](research/phase3/TWO_STAGE_STRATEGY.md) | Exploratory discovery, then fresh strict confirmation; locked test preserved. |
| [S02 completed result](research/phase3/S02_CLOSEOUT.md) |86.03% exploratory accuracy/.7861 macro-F1; verified artifacts and same-protocol context. |
| [S03 completed result](research/phase3/S03_CLOSEOUT.md) |86.36% exploratory accuracy/.7707 macro-F1; matched MobileNet comparison and error analysis. |
| [S04 completed fusion](research/phase3/S04_CLOSEOUT.md) |88.56% exploratory accuracy/.8057 macro-F1; fixed 50/50 probabilities with no new training. |
| [S05 attention result](research/phase3/S05_CLOSEOUT.md) | CBAM tied B0 accuracy; criterion/class trade-offs documented. |
| [ConvNeXt result](research/phase3/S06_CLOSEOUT.md) | 91.75% exploratory accuracy; strongest standalone so far. |
| [Bounded ensemble results](research/phase3/S07_S09_CLOSEOUT.md) | Three fixed fusions; no accuracy gain, S09 macro-F1 .8691. |
| [S10–S12 closeout](research/phase3/S10_S12_CLOSEOUT.md) | Recovery verified; fixed S12 ensemble reaches92.42%/.8770. |
| [Folder map](research/FOLDER_MAP.md) | Active, historical and local folders explained. |
| [Next steps](NEXT_STEPS.md) | Remaining phases and training gates. |
| [Experiment registry](results/master_experiment_registry.csv) | Historical evidence and actual structured run records; protocols kept separate. |
| [Dataset table](results/datasets/strict_lesion_disjoint/class_counts.md) | Verified per-class total/train/validation/test counts. |
| [Saved figures](results/figures/figure_index.csv) | Training curves, confusion matrices and class-score charts. |
| [Legacy research](legacy/README.md) | Preserved code/results/reports and previous navigation documents. |

## Historical status — 3 October 2026

**Active priority (3 October2026):** [High-performance heterogeneous ensemble phase](research/phase3/HIGH_PERFORMANCE_ENSEMBLE_PLAN.md): DenseNet201 next ([S15 proposal](research/phase3/S15_PROPOSAL.md)), EfficientNet-B3 second subject to evidence/approval, conditional Swin-T third; equal voting then a small bounded global-weight study. **S14 deferred by user; no new experiments launched.** Reference remains S12 92.42%/.8770, FP32 identity92.48%/.8775; S13 TTA rejected. Same exploratory split; strict confirmation postponed; original locked test untouched.

Historical validation leaders: **87.69% strict accuracy**, **0.7837 strict macro-F1** (different combinations), and **90.75% exploratory accuracy / 0.8573 macro-F1**. These are validation-selected observations, not final test scores. Historical comparison figures are not controlled architecture ablations.

New work uses `research/`, `results/structured_experiments/`, `results/model_comparison/structured/`, and local `checkpoints/structured/`. Historical source/results stay at original paths because saved recipes depend on them. Raw images, weights, caches and private references remain Git-ignored.

## Organization cleanup

Historical reports/notebooks/archive/old Grad-CAM folders now live under `legacy/`. Historical code/results/checkpoints retain required executable paths. [Cleanup report and root map](legacy/organization/README.md) explain all moves and checks. SciSpace workbook: `research/literature/incoming/SciSpace Literature Review.xlsx` (local-only; unanalysed).

## Safe commands available now

```powershell
# Verify organization/preservation without model execution
.\.venv\Scripts\python.exe -m research.verify_layout
# Refresh saved-evidence tables only; no model execution
.\.venv\Scripts\python.exe -m research.build --skip-figures
# Validate the Phase 2 plan and regenerate literature tables (CPU only)
.\.venv\Scripts\python.exe -m research.phase2.prepare
```

To present saved tables/PNG/PDF figures, simply open them; Python, raw images and checkpoints are unnecessary. A clone without ignored historical assets can rebuild evidence with `--portable`. The shared training runner is Phase 3 work, so there is no valid new training command yet. Before any future GPU run we will state its question, reason, exact start/monitor commands and output paths.

The long historical README and two old handoff guides are preserved under [legacy/navigation](legacy/navigation/README.md). The six existing Word/PDF reports remain under `legacy/reports/` as historical reports; their earlier fixed-architecture statements are superseded.
