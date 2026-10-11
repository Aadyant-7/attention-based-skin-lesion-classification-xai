# Research study index

Updated 11 October 2026. **Training and performance searches are closed.** Work is grouped by research question rather than execution order. Negative studies remain negative; different protocols/recipes are not one controlled comparison.

## Current retained methodology

[S83 specification](FINAL_EXPLORATORY_ENSEMBLE.md): ConvNeXt-Tiny, ConvNeXt-Small, DenseNet201, EfficientNetV2-S, EfficientNet-B0 and ConvNeXt-Tiny + CBAM; equal 1/6 soft voting, FP32 identity, 224 × 224 RGB/ImageNet preprocessing.

- Exploratory validation: **94.011976% accuracy / 0.895200837 macro-F1**; [original package](../results/short_screening/s83_cbam_f1_addition_cpu/).
- Post-development test audit: **87.558217% / 0.807031200**; [report](FINAL_S83_TEST_AUDIT.md), [artifacts](../results/final_exploratory_test_audit/v1/).
- [Frozen checkpoint manifest](../results/final_exploratory_freeze/v1/frozen_method.json): exact selectors, epochs, hashes and preprocessing.

Validation was reused extensively and can share related lesions across development partitions. The audit cohort had already been evaluated. Neither number is a new independent estimate of clinical performance.

## Study groups

| Research question | Work and entry point | Status / useful conclusion |
|---|---|---|
| What does the foundational transfer-learning/CBAM pipeline achieve, and how does imbalance handling affect it? | Preserved `src/`, `scripts/`, root `configs/`; [legacy index](../legacy/README.md), [old training ledger](../results/experiments.csv). | Historical baselines and weighted/focal/oversampling investigations; preserved evidence, not the final method. |
| What literature and evaluation differences justify the setup? | [Verified review](literature/README.md), [review table](literature/review_table.md), [SciSpace screening](literature/scispace_analysis/INTEGRATED_REVIEW.md), [Phase-2 protocol](phase2/DATASET_PROTOCOL.md). | Literature integrated with foundational references. Spreadsheet claims are screened, not assumed verified or comparable. Phase-2 plans are historical. |
| Which backbones work under the matched exploratory recipe? | [Phase-3 plan](phase3/PLAN.md), S02/S03/S06/S10/S15/S18/S19 closeouts under `phase3/`; [model table](MODEL_VS_ACCURACY.md), [packages](../results/structured_experiments/). | ConvNeXt is the strongest early family. Budgets/selectors and numerical amendments differ where disclosed. All use transfer learning already. |
| Does added CBAM improve a matched baseline? | [S05 closeout](phase3/S05_CLOSEOUT.md), compared with [S03](phase3/S03_CLOSEOUT.md). | Mixed/near-neutral overall effect; class-level changes matter. S79's longer recipe is not a pure CBAM ablation against S06. |
| Does heterogeneous fusion improve prediction coverage? | [S07–S09](phase3/S07_S09_CLOSEOUT.md), [S10–S12](phase3/S10_S12_CLOSEOUT.md), [S53 equal-five](short_screening/S53_EQUAL_FIVE_CLOSEOUT.md), [S83 addition](short_screening/S83_CBAM_F1_ADDITION_CLOSEOUT.md). | Complementary backbones gave the clearest gains. Actual fusion can introduce errors even when a new model fixes some reference errors. S83 is retained; its small S53 gain missed the original material-improvement gate. |
| Do TTA or bounded weighted fusion justify extra complexity? | [S13](phase3/S13_CLOSEOUT.md), [earlier architecture freeze](FINAL_ARCHITECTURE_SELECTION.md), [selection artifacts](../results/architecture_selection/final_v1/). | Four-view TTA was not retained; numerical alternatives/weighting were judged with class trade-offs and complexity. That earlier three-model freeze predates S83. |
| How does the original frozen architecture transfer to strict evaluation? | [S28–S31 validation](FINAL_STRICT_VALIDATION_RESULTS.md), [original strict test](FINAL_LOCKED_TEST_RESULTS.md). | Separate B0/Tiny/V2-S equal-three: 90.153027% / 0.843486392 validation; 86.759814% / 0.794472645 original test. Preserve first-test evidence. |
| Do heavier recipes and small inference/fusion changes materially help? | [Aggressive recipe audit](aggressive/VERIFIED_METHOD_AUDIT.md), [budget closeout](aggressive/BUDGET_SCREENING_RESULTS.md), [screening catalogue](short_screening/RESULTS_ARTIFACT_INDEX.md). | Rejected/partial studies, diagnostics, precision recovery and compute costs remain evidence. The long catalogue contains dated statuses, not current launch authorization. |
| Can features, metadata or external data repair the bottleneck? | [S76 feature fusion](short_screening/S76_FEATURE_FUSION_CLOSEOUT.md), [S77 PanDerm](short_screening/S77_PANDERM_LARGE_CLOSEOUT.md), [S68/S69 metadata](data_assisted/S68_S69_METADATA_CLOSEOUT.md), [clearance](data_assisted/DATA_CLEARANCE_REPORT.md). | Tested variants did not justify replacing the image-only method. Do not generalize to every multimodal/domain model. External-data/pretrained overlap limits remain disclosed. |
| How do the stronger CBAM component and XAI fit? | [S79/S80](final_cbam_development/S79_S80_CLOSEOUT.md), [S80 handoff](final_cbam_reporting/REPORTING_HANDOFF.md), [XAI status](xai/README.md). | S79 supplies S83's CBAM slot. Verified Grad-CAM/actual CBAM maps explain ten validation cases of S80's different five-model method; exact S83 explanations remain separate. |
| Do transformers, regularization or longer warm-start training improve S83? | [S89/S91 and S92–S94](short_screening/S89_S91_TRANSFORMER_CLOSEOUT.md), [S95/S96](short_screening/S95_S96_REGULARIZATION_CLOSEOUT.md), [S97/S98 decision](FINAL_EXPLORATORY_ENSEMBLE.md). | These bounded additions/replacements did not improve S83 accuracy. Longer training did not guarantee improvement. |
| Does a separate paper-inspired image-level protocol change outcomes? | [V1 preparation](image_level_replication/PHASE1_CLOSEOUT.md), [V2](image_level_replication/v2/README.md), [R201](image_level_replication/v2/R201_CLOSEOUT.md), [R202](image_level_replication/v2/R202_CLOSEOUT.md). | Different-split study stays separate. Ten folds are prepared, but completed ten-fold evaluation/outer assessment is not claimed. R202 cannot be fused directly with original-split models. |
| Does fresh ConvNeXt-V2 Base improve the original-split ensemble? | [R203/R204/R205 closeout](convnext_base_original/CLOSEOUT.md), [packages](../results/structured_experiments/convnext_base_original_v1/). | Base: 93.213573% accuracy / separate best F1 0.890102 in 30 epochs. Equal-seven addition and equal-six Small replacement did not improve S83 accuracy. Retain S83. |

## Using this for reporting

Present each selected question, protocol, actual recipe, saved result and conclusion. Summarize rejected/partial/unlaunched studies briefly in an appendix or limitations; the full [master registry](../results/master_experiment_registry.csv) and original packages stay available.

Use [the current evidence index](../reporting/evidence/INDEX.md) for figures/tables. Keep checkpoint selection, class-wise performance and evaluation labels visible. Do not imply every study added a positive gain, repeated validation is independent testing, or CBAM/Grad-CAM guarantees better accuracy.

Code stays at its original paths because later implementations import its helpers and freezes hash it. [Folder map](FOLDER_MAP.md) · [Current next steps](../NEXT_STEPS.md) · [Archived navigation](../legacy/navigation/cleanup_2026-10-11/)
