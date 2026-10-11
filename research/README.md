# Research workspace

## Current decision

**Training and performance searches are closed.** Retain [S83's six-model equal FP32 identity ensemble](FINAL_EXPLORATORY_ENSEMBLE.md): **94.011976% exploratory validation accuracy / 0.895200837 macro-F1**. Its [post-development audit](FINAL_S83_TEST_AUDIT.md) scored **87.558217% / 0.807031200** on the previously evaluated test cohort.

[Latest R203 closeout](convnext_base_original/CLOSEOUT.md): ConvNeXt-V2 Base finished 30 epochs on the original exploratory split; accuracy best **93.213573%** at raw epoch 20, macro-F1 best **0.890102** at EMA epoch 27. Two fixed CPU fusions did not improve S83 accuracy. No further run is queued.

## Current reading order

1. [Reporting workspace](../reporting/README.md) and [evidence index](../reporting/evidence/INDEX.md).
2. [Study index](STUDY_INDEX.md): research questions, useful conclusions and preserved negative studies.
3. [Verified literature](literature/README.md), [review table](literature/review_table.md), [integrated SciSpace screening](literature/scispace_analysis/INTEGRATED_REVIEW.md), [bibliography](literature/references.bib).
4. [Retained exploratory methodology](FINAL_EXPLORATORY_ENSEMBLE.md) and [its test audit](FINAL_S83_TEST_AUDIT.md).
5. [Separate strict validation](FINAL_STRICT_VALIDATION_RESULTS.md) and [original strict test](FINAL_LOCKED_TEST_RESULTS.md).
6. [Current exact S83 explainability](xai/README.md), with completion receipts; earlier S80 explanations remain a distinct historical workflow.
7. [Folder map](FOLDER_MAP.md) and [remaining work](../NEXT_STEPS.md).

## Authoritative evidence

| Path | Contents |
|---|---|
| [Master experiment registry](../results/master_experiment_registry.csv) | Preserved legacy records and structured training/inference candidates with protocol, selector and artifact paths. |
| [S83 validation package](../results/short_screening/s83_cbam_f1_addition_cpu/) | Metrics, predictions/probabilities, class scores, confusion matrices and comparison figures. |
| [S83 frozen method](../results/final_exploratory_freeze/v1/frozen_method.json) | Six checkpoint paths/hashes, selected epochs, class order, preprocessing and weights. |
| [S83 test audit](../results/final_exploratory_test_audit/v1/) | Audit metrics, member outputs, figures, source verification and completed receipts. |
| [Strict methodology](../results/final_strict/v1/) | S28/S29/S30 summaries, frozen S31 validation and comparisons. |
| [Original strict test](../results/final_locked_test/v1/) | Frozen-method test outputs and first-evaluation evidence. |
| [Exact S83 XAI](../results/final_s83_xai/v1/README.md) | Current six-model explanation package: 12 predetermined validation cases, gallery, atlas and verification/completion receipts. |
| [S80 reporting/XAI](../results/final_cbam_reporting/v1/) | Historical five-model reporting tables, audit and ten validation explanation cases. |
| [R203/R204/R205](../results/structured_experiments/convnext_base_original_v1/) | Latest Base standalone, fixed fusions and their verification. |
| [Dataset/protocol audits](../results/datasets/) | Saved class counts and manifest audits. |
| [Portable legacy evidence](../results/legacy/) | Immutable earlier inventory and small portable metric/history/config copies. |

A registry row is saved evidence, not necessarily an independently trained model. Missing fields mean unrecorded, not zero. The original exploratory cohort has 7,009 training and 1,503 validation images; related lesions can cross its development partitions. Strict results use a different lesion-disjoint methodology. Later test audits do not restore first-test independence.

## Shared implementation: preserve paths

- `common.py`: project root, classes, protocol paths, hashes and atomic writes.
- `models.py`: backbone adapters and optional CBAM; imports preserved `src/cbam.py`.
- `train.py`, `strict_train.py`, `strict_protocol.py`: development loaders, training/recovery and protocol safeguards.
- `experiment.py`, `registry.py`, `plots.py`: artifact contract, experiment indexing and figures.
- `final_cbam_development/`, `short_screening/`, `aggressive/`, `phase3/`: completed studies that also contain shared runtime/reporting helpers.
- `image_level_replication/v2/`: separate study whose model primitives are reused by R203.
- `tests/`: implementation/recovery checks; importing source does not authorize training.

Imports, configuration paths and frozen signatures require these paths. Completed runners, proposals and recovery helpers remain historical implementation evidence; [the study index](STUDY_INDEX.md) identifies their status. Rebuilding registries/closeouts can write artifacts, so it is not a generic read-only browsing step.

## Historical navigation and local storage

Earlier planning is preserved unchanged in [the navigation archive](../legacy/navigation/cleanup_2026-10-11/research/README.md). Its dated pending-run/test-lock statements describe the state when written and do not supersede this page.

Physical root: **`E:\Minor Project\Project Development`**; [migration record](migration/LOCAL_DRIVE_MIGRATION.md). Frozen manifests and local checkpoints are preserved. Raw images/checkpoints/caches/private references are excluded from Git and are not all available from a GitHub clone. Existing presentation evidence can be read without retraining.
