# Saved scores and graphs: quick index

All paths below are relative to the project root.

## Latest completed: S89/S91 full20 and S92–S94 transformer additions

- `research/short_screening/S89_S91_TRANSFORMER_CLOSEOUT.md`: final standalone results, best epochs, actual recovered/lost predictions and interpretation.
- `results/short_screening/s92_s94_transformer_addition_cpu/`: fixed three-fusion plan with source hashes, standalone/class complementarity, changed predictions, scores, probabilities, raw/normalized confusion matrices and PNG/PDF comparisons. Equal-seven Swin and DeiT additions both93.6793%; equal-eight joint addition93.4132%. All worse than retained S83 **94.0120% /0.895201**.
- `results/short_screening/swin_transfer_v1/s89_swin_t_none_exploratory_seed42/`: completed20, best90.6853% /0.860448 at15; best-accuracy/F1/final predictions, metrics, class matrices and training curves. `full20_closeout_verification.json` verifies all three checkpoints; immutable original `pilot_epoch15/` evidence remains available. Local checkpoints: `checkpoints/short_screening/swin_transfer_v1/s89_swin_t_none_exploratory_seed42/`.
- `results/short_screening/deit3_base_transfer_v1/s91_deit3_base_none_exploratory_seed42/`: completed20, best accuracy90.5522% at13 (tied17), best macro-F10.850535 at17; same full reporting and verification package. Local checkpoints: `checkpoints/short_screening/deit3_base_transfer_v1/s91_deit3_base_none_exploratory_seed42/`.
- `results/short_screening/s89_s91_completion_v1/`: completed orchestration state, plan and original launch receipt. S90 Small was superseded before training. No new test scoring or GPU work in closeout; no subsequent training queued.

## Historical preparation notes (completed/superseded above)

**Latest amendment:** S89 verified epoch15 score90.6853% /0.860448 and immutable pilot/checkpoint snapshots before continuing to20. S90 Small was never launched and is superseded. S91 is the larger DeiT III Base full20 run, sequential after Swin; plan `research/short_screening/S89_S91_FULL_WINDOW_PLAN.md`. Master logs/state: `results/short_screening/s89_s91_completion_v1/`; Base outputs: `results/short_screening/deit3_base_transfer_v1/s91_deit3_base_none_exploratory_seed42/`; Base checkpoints: `checkpoints/short_screening/deit3_base_transfer_v1/s91_deit3_base_none_exploratory_seed42/`. No new test or ensemble score.

The user subsequently authorized a concurrent S90 **DeiT III Small ImageNet22k->1k** epoch15 pilot. Plan: `research/short_screening/S90_DEIT3_CONCURRENT_PILOT.md`; isolated runner/config: `s90_deit3_transfer.py`, `s90_deit3_transfer_v1.json`; CPU preflight/launcher logs: `results/short_screening/deit3_transfer_v1/`; serious-run outputs: `results/short_screening/deit3_transfer_v1/s90_deit3_small_none_exploratory_seed42/`; checkpoints: `checkpoints/short_screening/deit3_transfer_v1/s90_deit3_small_none_exploratory_seed42/`. No score exists until training validation is committed; inspect live progress/registry rather than treating this static index as a completion claim.

- `results/short_screening/s86_s87_saved_member_addition/`: fixed candidate audit (MobileNet, ResNet101, ImageNet22k Tiny, frozen PanDerm Large), class-wise error complementarity and two rejected equal-seven additions. Both93.7458%; ResNet macro-F10.891351; PanDerm macro-F10.888905. Metrics, predictions/probabilities, raw/normalized confusion matrices, class scores, PNG/PDF/comparison figures, source hashes and independent verification. No conditional joint fusion was executed. Closeout: `research/short_screening/S86_S87_SAVED_ADDITION_CLOSEOUT.md`.
- `research/short_screening/S89_SWIN_PROPOSAL.md`: exact isolated Swin-T model/recipe, epoch15 pilot, hard20 cap, continuation criteria and commands. Runner/config: `s89_swin_transfer.py`, `s89_swin_transfer_v1.json`. CPU safety preflight: `results/short_screening/swin_transfer_v1/preflight.json`. **Prepared only, no GPU run or accuracy result.**

## Latest completed checks: S84/S85

Both rejected against S83: fixed CBAM-only temperature 2 **93.8789% /0.893894**; probability averaging CBAM checkpoints 33/35 within its one-sixth slot **93.9454% /0.894622**. Saved metrics, predictions/probabilities, class metrics, raw/normalized confusion matrices, PNG/PDF figures, changed identities, source/cache provenance and independent verification: `results/short_screening/s84_s85_cbam_stability_cpu/`. Interpretation: `research/short_screening/S84_S85_CBAM_STABILITY_CLOSEOUT.md`. CPU only; no new test or training run. S83 remains the highest observed exploratory validation result at **94.0120% /0.895201**.

## Latest numerical development leader: S82/S83

Fixed equal-six keeps both Tiny and Tiny+CBAM. S82 accuracy-selected CBAM addition93.9454% /0.893190; S83 macro-F1-selected addition **94.0120% /0.895201** with unchanged82.0359% melanoma recall versus S53. S83 is the highest observed exploratory validation accuracy; its six net correct gains fall short of the unchanged material gate, so S53 remains the policy reference. Full metrics/predictions/probabilities, class/confusion/comparison PNG/PDF, gains/losses and hashes: `results/short_screening/s82_cbam_addition_cpu/`, `results/short_screening/s83_cbam_f1_addition_cpu/`. Decision: `research/short_screening/S82_S83_NEXT_DECISION.md`. No new GPU/test run; report writing paused by user.

## Final test audit, XAI and report tables

`research/final_cbam_reporting/REPORTING_HANDOFF.md` indexes the completed fixed S80 post-development test audit87.0925% /0.801646, all-five validation Grad-CAM/actual CBAM maps and standalone/protocol comparison tables. Artifacts: `results/final_cbam_reporting/v1/`. Original S31 test evidence was preserved; this is not a new pristine first test. S80 validation93.3466% and retained S53 validation93.6128% remain separately labelled.

## Completed final CBAM package: S79/S80

S79 standalone93.0140% /0.877434 at accuracy-selected epoch33; macro-F1 winner0.877708 at35. S80 fixed replacement ensemble93.3466% /0.877017, below retained S53. Closeout: `research/final_cbam_development/S79_S80_CLOSEOUT.md`. Full selected/final/ensemble metrics, predictions, probabilities, class/confusion/curve PNG/PDF, logs and verification: `results/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/`. Local best/latest checkpoints: `checkpoints/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/`. Recovery transparently recorded; no original-test evaluation.

## Representation studies: S76-S78

S77 completed: Base88.5562% /0.810305; Large88.9554% /0.820575. Advancement gate failed; S78 was not run. Closeout: `S77_PANDERM_LARGE_CLOSEOUT.md`. Full reporting recovered from saved artifacts after a path typo, without repeating inference/fitting.

- `results/short_screening/s76_trained_feature_fusion/`: actual fine-tuned member features, one fixed training-only nonlinear classifier; 91.3506% / 0.850417, rejected against the freshly reproduced 93.6128% / 0.886857 control. Full metrics, predictions, class/confusion/comparison figures and source/cache verification. Closeout: `S76_FEATURE_FUSION_CLOSEOUT.md`.
- `results/short_screening/s77_panderm_large_transfer/`: matched official PanDerm Base/Large frozen transfer, with at most one conditional S78 B0 replacement. Predeclared scope and commands: `S77_PANDERM_LARGE_PLAN.md`; completion status/results are recorded in that result folder's `summary.json`. Local classifier bundles are under `checkpoints/short_screening/s77_panderm_large_transfer/`; third-party weights/features remain local.

## New bounded checks: S70-S75

- `results/short_screening/s75_convnext_checkpoint_average_cpu/`: one fixed equal parameter average of S06 epochs17/18/20, with fresh CPU FP32 epoch18 control. Standalone91.4172% versus91.4837%; fixed five-model fusion93.6793% versus93.6128%, net+1 but lower melanoma recall; rejected. Four full prediction/class/confusion PNG/PDF result packages, comparison, gain/loss identities, compatibility/source hashes and runtime verification. Local new checkpoint: `checkpoints/short_screening/s75_convnext_checkpoint_average_cpu/averaged.pt`; source checkpoints unchanged. Closeout: `S75_CHECKPOINT_AVERAGE_CLOSEOUT.md`. No training/GPU/test.

- `results/short_screening/s74_lesion_normalized_head_cpu/`: fixed inverse-lesion-view CE weighting on the unchanged CPU feature head. Train-only group-OOF accuracy79.1554% versus79.0840% control, lower macro-F1/melanoma recall; rejected. Full metrics/predictions/probabilities, class/confusion/comparison PNG/PDF, three fold histories/curves, weight manifests and verification. CPU heads/scalers: `checkpoints/short_screening/s74_lesion_normalized_head_cpu/`. Closeout: `S74_LESION_NORMALIZED_CLOSEOUT.md`. No validation/GPU/test run.

- `results/short_screening/s70_s71_same_lesion_multiimage/`: fixed validation-only same-lesion multi-image pooling. Exploratory S70 93.7458% / 0.888382 (rejected); strict-validation S71 91.0845% / 0.854216 (conditional multi-image evidence only). Different input contract; no original-test run or retained-method change. Separate image/lesion metrics, predictions, class/confusion/comparison PNG/PDF, gains/losses, bootstrap intervals and hashes. Closeout: `S70_S72_CLOSEOUT.md`.
- `results/short_screening/s72_nonlinear_frozen_feature_head/`: fixed train-only group-OOF nonlinear frozen-feature classifier, tied control accuracy75.8025% but lower macro-F1; rejected without development-validation scoring. Full metrics/predictions/figures/folds and source checks. Closeout: `S70_S72_CLOSEOUT.md`.
- `results/short_screening/s73_matched_contrastive_head_cpu/`: completed matched CE versus CE+SupCon CPU feature-head preflight. Train-only grouped accuracy79.0840% versus79.1126%; candidate gains only two net images, lowers macro-F1/melanoma recall and fails the gate. No validation or GPU run. Full predictions, metrics, fold/training histories, curves, confusion/class/comparison PNG/PDF and verification. Six CPU head checkpoints/scalers are in `checkpoints/short_screening/s73_matched_contrastive_head_cpu/`; these are not CNN checkpoints. Closeout: `S73_CONTRASTIVE_CLOSEOUT.md`.

## Latest provisional exploratory candidate

S53 equal five-model fusion adds S03 EfficientNet-B0 to the four members below, with weight 0.20 each. It achieved **93.6128% accuracy / 0.886857 macro-F1** (1,407 correct / 96 incorrect). The net gain is only two images; it is not established as a material improvement. S76 freshly reproduced all-five FP32 inference with the same labels and metrics; S79/S80 verification also reproduced the reference from those hash-verified FP32 caches.

Scores, predictions, probabilities, class metrics and PNG/PDF figures: `results/short_screening/s53_equal_five_b0_addition/`. Interpretation: `research/short_screening/S53_EQUAL_FIVE_CLOSEOUT.md`.

## GPU-reproduced four-model reference

**93.4797% accuracy / 0.883680 macro-F1**, 1,405 correct / 98 incorrect on 1,503 exploratory validation images. Equal fusion of ConvNeXt-Tiny, ConvNeXt-Small, DenseNet201 and EfficientNetV2-S using their saved macro-F1-selected checkpoints.

- Original CPU fusion: `results/short_screening/s46_all_f1_checkpoint_fusion/`
- Fresh FP32 identity GPU reproduction (same predicted classes): `results/short_screening/s50_s51_current_ensemble_tta/identity_fp32/`
- Checkpoint paths and hashes: S46 `candidate_manifest.json`; S50/S51 `verification.json` in its parent folder.

Within each result folder:

| File/folder | Contents |
|---|---|
| `validation_metrics.json` | Accuracy, macro precision/recall/F1 and class-wise results |
| `validation_predictions.csv` | Image identities, true/predicted classes and scores |
| `validation_probabilities.csv` | Saved class probabilities |
| `figures/per_class_metrics.csv` | Class-wise precision, recall, F1 and support |
| `figures/confusion_matrix.csv` | Raw confusion matrix |
| `figures/confusion_matrix_normalized.csv` | Normalized confusion matrix |
| `figures/*.png` and `figures/*.pdf` | Confusion matrices, class scores and class support figures |

## Comparisons and rejected checks

- `results/master_experiment_registry.csv`: experiment index and scores.
- `results/short_screening/s46_all_f1_checkpoint_fusion/comparison_figures/`: comparison CSV and PNG/PDF model-comparison plot.
- `results/short_screening/s50_s51_current_ensemble_tta/`: identity control, rejected four-view TTA, verification and summary.
- `results/short_screening/s52_disagreement_resnet_rescue/`: rejected disagreement-only ResNet rule, routing CSV, scores and full figures.
- `results/short_screening/s54_s55_panderm_fusion/`: rejected equal-six PanDerm addition and equal-five B0 replacement; full scores, source manifests, class comparisons and PNG/PDF figures for both.
- `results/short_screening/convnextv2_transfer_v1/s56_frozen_feature_screen/`: failed V2 frozen-feature gate; its 20-epoch S57 run was not launched.
- `results/short_screening/supervised22k_transfer_v1/s58_frozen_feature_screen/`: passed supervised22k frozen-feature gate, scores/predictions and figures.
- `results/short_screening/supervised22k_transfer_v1/s59_convnext_tiny_supervised22k_exploratory_seed42/`: authorized capped fine-tuning run; artifact policy and commands in `research/short_screening/S58_S59_SUPERVISED22K_PLAN.md`.
- S59 is now closed out: best accuracy 90.6853% at epoch20; best macro-F1 0.832974 at epoch18. `research/short_screening/S59_S61_CLOSEOUT.md` records verification and the decision not to extend training.
- `results/short_screening/s60_s61_s59_fusion/`: two rejected S59 addition/replacement ensembles, both 93.0805%, with full metrics, prediction files and figures. S53 remains the provisional highest exploratory accuracy.
- `results/short_screening/s62_s53_error_audit/`: descriptive audit of all 96 S53 errors, six image review sheets, preprocessing comparison, confidence/quality/class CSVs and PNG/PDF figures. Findings and the next bounded calibration proposal are in `research/short_screening/S62_ERROR_AUDIT.md`; no model score changed during this audit.
- `results/short_screening/s63_crossfit_temperature_calibration/`: rejected five-fold lesion-group temperature study, 93.6793% / 0.887038 macro-F1; only one net additional correct prediction and lower melanoma recall. Includes meta-OOF metrics/predictions/probabilities, temperatures, fold/class comparisons, calibration diagnostics, verification and PNG/PDF figures. `research/short_screening/S63_CALIBRATION_CLOSEOUT.md` records the failed material gate. S53 remains the preferred candidate despite S63's slightly higher observed number.
- `research/short_screening/`: experiment scripts, plans and closeout Markdown files explaining decisions.
- `results/short_screening/s64_paired_preprocessing_cpu/`: paired fresh CPU FP32 pretrained-feature square versus official resize/crop screen, 79.3081% versus81.7698%; initial gate passed, not an ensemble score.
- `results/short_screening/s65_preprocessing_train_group_cv/`: fixed three-fold train-only lesion-group consistency check, pooled75.8025% versus76.4160%; only one positive fold and lower melanoma recall, so no crop-based GPU training. Both studies save metrics, predictions/probabilities, confusion/class/comparison PNG/PDF figures, plans and verification. Interpretation: `research/short_screening/S64_S65_PREPROCESSING_CLOSEOUT.md`.
- `results/short_screening/external_isic2019_feasibility/`: metadata-only non-HAM candidate identity manifest, image/lesion counts, source hashes and limitations; not an accuracy experiment. Data license/source attribution in its README. Next action: `research/short_screening/EXTERNAL_DATA_NEXT_DIRECTION.md`.
- `research/short_screening/BOTTLENECKS_AND_NEXT_ACTION.md`: evidence synthesis separating difficult-class errors, fusion limits, generalization gaps and unproven causes.
- `results/data_assisted/clearance_v1/`: completed external BCN data preparation: plans/source hashes, acquired-image integrity manifest, published-name exclusions, exact/near-hash candidate quarantine, and final3323-lesion manifest. This is not an accuracy experiment. `research/data_assisted/DATA_CLEARANCE_REPORT.md` explains label mapping, counts and unresolved independence limits; `METADATA_FOLLOWUP_DEFERRED.md` records the older metadata study and later follow-up.

## Training and historical evidence

- `results/data_assisted/s68_s69_metadata_screen/`: completed fixed metadata follow-up on S53. Age-only93.4797% /0.882840; age+sex+location93.0805% /0.877659; both rejected versus93.6128% /0.886857 image-only. Contains train-only likelihood model, metadata-fit/exclusion manifests, full predictions/probabilities, class/confusion PNG/PDF figures and comparison. No validation-fitting or GPU/test run. Closeout: `research/data_assisted/S68_S69_METADATA_CLOSEOUT.md`.

- `results/data_assisted/s66_s67_frozen_data_screen/`: rejected fixed external-data CPU screen and three-fold train-only lesion-group confirmation. HAM-only versus HAM+external:79.3081%→78.5762% on exploratory validation;75.8025%→74.4471% on pooled group-OOF. Full metrics, predictions/probabilities, class/confusion/comparison PNG/PDF, gain/loss identities, source hashes and classifier hashes are saved. These are diagnostic feature heads, not the93.61% ensemble. Closeout: `research/data_assisted/S66_S67_CLOSEOUT.md`.

- `results/structured_experiments/`: individual structured model runs, training logs/history, validation metrics/predictions and figures.
- `checkpoints/structured/`: preserved model checkpoints, including best raw accuracy, best macro-F1 and latest checkpoints where produced.
- Training curves belong to training runs. CPU fusion/inference-only experiments have result/comparison figures, not new epoch curves.

## Separate strict and original test results

- `research/FINAL_LOCKED_TEST_RESULTS.md`: original held-out test report.
- `results/final_locked_test/v1/ensemble/test_metrics.json`: original frozen ensemble test metrics, **86.7598% accuracy / 0.794473 macro-F1**.

The 93.48% reference and provisional 93.61% candidate are exploratory validation performance from subsequent development; neither is a new locked-test accuracy. No test evaluation is repeated by these studies.
