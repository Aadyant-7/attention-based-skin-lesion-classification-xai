# Saved evidence index

Links open original files. This is navigation/provenance, not a finished report or new experiment. [source_manifest.csv](source_manifest.csv) records exact source paths and SHA-256 hashes. Accuracy below is percent; macro-F1 is a fraction.

[Original organization readiness audit](FINAL_READINESS_AUDIT.md): dated preservation snapshot. [Current S83 final-product readiness](S83_FINAL_PRODUCT_READINESS.md): exact-method XAI completed and verified.

## Current outcomes: separate protocols

| Method/cohort | Accuracy | Macro-F1 | Source |
|---|---:|---:|---|
| S83 retained equal-six, exploratory validation | **94.011976%** | **0.895200837** | [Metrics](../../results/short_screening/s83_cbam_f1_addition_cpu/validation_metrics.json) |
| S83 post-development audit, previously evaluated test cohort | 87.558217% | 0.807031200 | [Metrics](../../results/final_exploratory_test_audit/v1/ensemble/test_metrics.json) |
| S31 strict equal-three, lesion-disjoint validation | 90.153027% | 0.843486392 | [Metrics](../../results/final_strict/v1/ensemble/validation_metrics.json) |
| S31 original strict held-out test | 86.759814% | 0.794472645 | [Metrics](../../results/final_locked_test/v1/ensemble/test_metrics.json) |
| R203 ConvNeXt-V2 Base, original exploratory validation, raw accuracy selector epoch 20 | 93.213573% | 0.888528134 | [Metrics](../../results/structured_experiments/convnext_base_original_v1/r203_convnextv2_base_original_exploratory_seed42/best_accuracy/validation_metrics.json) |

94.01% is extensively reused image-level **validation**, with related lesions shared between development partitions; it is not test or strict lesion-independent accuracy. The S83 test cohort had previously been evaluated. S31 is a separately strict-trained three-model methodology. R203 has no test score from its closeout.

## Frozen method and local checkpoints

- [Current S83 method](../../research/FINAL_EXPLORATORY_ENSEMBLE.md): six separately trained models and their selectors; equal 1/6 FP32 identity probability fusion.
- [S83 freeze](../../results/final_exploratory_freeze/v1/frozen_method.json): checkpoint paths/hashes, epochs, source probabilities, preprocessing and class order.
- [Completed audit status](../../results/final_exploratory_freeze/v1/audit_status.json), [recorded S83 test closeout](../../research/FINAL_S83_TEST_AUDIT.md).
- [Strict S31 freeze](../../results/final_strict/v1/ensemble/frozen_ensemble.json), [strict pre-test verification](../../results/final_locked_test/v1/pre_inference_verification.json).

S83: S06 ConvNeXt-Tiny/F1, S18 ConvNeXt-Small/F1, S15 DenseNet201/F1, S10 EfficientNetV2-S/F1, S03 EfficientNet-B0/accuracy, S79 ConvNeXt-Tiny+CBAM/F1. Local weights stay under `checkpoints/`; GitHub source/results do not include them. The original freeze receipt's pending status is historical; use adjacent audit status for completion. Preserve both.

## Figures, predictions and class scores

| Evidence | Original package |
|---|---|
| S83 validation, class scores, probability/prediction CSVs, raw/normalized matrices | [S83 package](../../results/short_screening/s83_cbam_f1_addition_cpu/) and [PNG/PDF figures](../../results/short_screening/s83_cbam_f1_addition_cpu/figures/) |
| S83 recorded test | [Ensemble package](../../results/final_exploratory_test_audit/v1/ensemble/) and [validation-to-test comparison](../../results/final_exploratory_test_audit/v1/validation_test_comparison/) |
| S31 strict validation | [Package](../../results/final_strict/v1/ensemble/) and [figures](../../results/final_strict/v1/ensemble/figures/) |
| S31 original strict test | [Package](../../results/final_locked_test/v1/ensemble/) and [figures](../../results/final_locked_test/v1/ensemble/figures/) |
| R203 training/validation/LR curves | [Figures](../../results/structured_experiments/convnext_base_original_v1/r203_convnextv2_base_original_exploratory_seed42/figures/), [history](../../results/structured_experiments/convnext_base_original_v1/r203_convnextv2_base_original_exploratory_seed42/history.csv), [LR history](../../results/structured_experiments/convnext_base_original_v1/r203_convnextv2_base_original_exploratory_seed42/lr_history.csv) |
| R203 selector/class/matrix evidence | [Best accuracy](../../results/structured_experiments/convnext_base_original_v1/r203_convnextv2_base_original_exploratory_seed42/best_accuracy/), [best F1](../../results/structured_experiments/convnext_base_original_v1/r203_convnextv2_base_original_exploratory_seed42/best_macro_f1/), plus run's raw_accuracy/raw_macro_f1/ema_accuracy/ema_macro_f1 folders |

Best accuracy/F1 can belong to different checkpoints; do not imply both come from one selected model. [R203 closeout](../../research/convnext_base_original/CLOSEOUT.md) records raw accuracy at epoch 20 and EMA F1 at27. [Two fixed fusion comparisons](../../results/structured_experiments/convnext_base_original_v1/ensemble_comparison/) did not improve retained S83 accuracy; class/error trade-offs remain saved.

## Model comparisons and honest progress narrative

- [Master registry](../../results/master_experiment_registry.csv): filter protocol, evaluation_split, record_kind, recipe and checkpoint selector. Includes CPU candidates, not only GPU training.
- [Historical three-run ledger](../../results/experiments.csv): early EfficientNet-B0+CBAM imbalance studies.
- [Structured run packages](../../results/structured_experiments/): S02 MobileNet, S03 B0, S05 B0+CBAM, S06 ConvNeXt-Tiny, S10 V2-S, S15 DenseNet201, S18 ConvNeXt-Small, S19 ResNet101, and their controlled ensembles.
- [S79 attention package](../../results/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/) and [F1 selector](../../results/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/macro_f1_selected/).
- [Earlier model table](../../results/model_comparison/final_architecture_selection/model_vs_accuracy.csv) and [later standalone table](../../results/final_cbam_reporting/v1/paper_tables/standalone_exploratory.csv) are **historical stage snapshots**, not current complete leaderboards. Newer runs can be absent; old status text was true at creation.

Suggested later writing order: dataset/protocols → matched backbones → complementary soft voting → attention role/ablation → retained exploratory framework → strict/recorded-test generalization → XAI and limitations. Keep meaningful unsuccessful backbones and neutral CBAM/TTA findings in a concise comparison/appendix. Do not imply every trial improved or mix different protocols into one rising curve.

## Current XAI: exact frozen S83 six-model method

[Completed package](../../results/final_s83_xai/v1/README.md) and [fixed plan](../../research/final_s83_xai/PLAN.md) identify the current explanations. Twelve SHA-256-ranked original exploratory-validation cases cover all seven true classes, correct/incorrect melanoma and akiec, plus additional errors. All six branches target the saved S83 predicted class through `(1/6) × branch probability[target]`; S79 uses its frozen F1 epoch 35 checkpoint and actual CBAM gates. Artifacts include72 branch maps, overlays/arrays, 12 case panels, [HTML gallery](../../results/final_s83_xai/v1/gallery.html) and [12-page atlas](../../results/final_s83_xai/v1/S83_GradCAM_Atlas.pdf).

[Completion manifest](../../results/final_s83_xai/v1/manifest.json) and [passed verification](../../results/final_s83_xai/v1/verification.json) record 72 nondegenerate CAMs, 12 CBAM cases, 192 PNGs, 25 PDFs and zero changed labels. Branch/ensemble replay maximum differences are 2.980232e-6/5.215406e-7; checkpoints, methodology and earlier XAI/test evidence are unchanged. All 12 atlas pages and a representative CBAM PDF passed visual layout review. XAI adds no training, test evaluation or new accuracy result.

[S80 historical manifest](../../results/final_cbam_reporting/v1/xai/manifest.json) and [ten earlier case packages](../../results/final_cbam_reporting/v1/xai/) remain distinct FIVE-model evidence, weights 0.2 and S79 accuracy epoch 33. Do not relabel them as S83. Grad-CAM is qualitative attribution; mean normalized maps are display composites, not segmentation or causal clinical reasoning.

[Final product handoff](../FINAL_PRODUCT_HANDOFF.md) gives the exact saved-results/gallery/atlas opening sequence. The linked prior readiness audit describes the state before this new exact-method XAI task.

## Later pre-test review: 12 October 2026

[Saved-validation generalization review](../../research/generalization_review/README.md)
decomposes lesion exposure and class mix, checks late overfitting, and records one
rejected training-loss-derived correction. A separate R202 EMA20 image-level
assessment is CPU-prepared, **not evaluated**. It has no test score and does not
replace S83 or its exact XAI package.

## Dataset and literature

- [Strict split](../../data/splits/split_assignments.csv), [exploratory split](../../data/splits/exploratory/image_level_dev_v1.csv), [class order](../../data/splits/label_mapping.json).
- [Split audit](../../results/datasets/split_audit.json), [two-stage audit](../../results/datasets/two_stage_protocol_audit.json), [class counts](../../results/datasets/two_stage_class_counts.csv).
- [Literature navigation](../../research/literature/README.md), [review table](../../research/literature/review_table.md), [CSV](../../research/literature/review.csv), [verified sources](../../research/literature/sources.json), [BibTeX](../../research/literature/references.bib).
- [Integrated SciSpace review](../../research/literature/scispace_analysis/INTEGRATED_REVIEW.md), [screening](../../research/literature/scispace_analysis/screening.csv), [provenance](../../research/literature/scispace_analysis/provenance.json), [original Excel](../../research/literature/incoming/SciSpace%20Literature%20Review.xlsx).
- [Ensemble evidence](../../research/literature/HIGH_PERFORMANCE_ENSEMBLE_EVIDENCE.md), [evidence caveats](../../research/literature/evidence_notes.md).

[Paper-inspired V2 results](../../results/image_level_replication/v2/) use a different cohort/protocol and stay separate from original exploratory comparisons. Preserve base/foundational references and comparability caveats. Cleanup did not analyze literature anew or alter splits.
