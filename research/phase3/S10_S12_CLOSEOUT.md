# S10 closeout and bounded S11–S12 ensemble study

Completed 3 October 2026. CPU artifact verification and two fixed fusions only; no new GPU run, retraining or test-image evaluation.

## S10 checkpoints and metrics

| State | Epoch | Accuracy | Macro precision | Macro recall | Macro-F1 | Loss |
|---|---:|---:|---:|---:|---:|---:|
| Best accuracy / best macro-F1 | 15 | 89.4877% | 0.842993 | 0.809949 | 0.823162 | 0.676173 |
| Final latest.pt | 20 | 88.9554% | 0.828502 | 0.806050 | 0.816210 | 4.719814 |

Both selected checkpoint files contain the same verified epoch15 model. latest.pt contains the finite final epoch20 model, optimizer/scaler/scheduler/RNG, complete sequential history and both selected models. Full model/optimizer states and diagnostic snapshots load on CPU. Checkpoint hashes, source hashes, unchanged epoch1–13 history and frozen epoch13 SHA are authenticated in closeout_verification.json. Final-epoch metrics come from latest.pt history; the runner saved prediction CSVs for the two selected winners, not a separate final-epoch prediction CSV. No new inference was used to replace saved metrics.

## Numerical amendment

AMP failure recurred at14,18,20 on ISIC_0025851. Replay hooks first detected FP16 Inf in BatchNorm modules features.6.11.block.1.1, features.6.9.block.1.1 and features.6.10.block.3.1 respectively; final logits/probabilities contained NaNs. Inputs and model/optimizer tensors were finite; the same-batch FP32 probe and entire1503-image FP32 validation passed. This establishes upstream FP16 activation overflow on the resumed failures, rather than loss-only arithmetic or a discarded/corrupt image. Exact original failed epoch14 state was unavailable, so do not claim identical original-operation proof.

Full FP32 validation actually used at14/18/20. All other resumed validations, including both selected winners at15, used AMP. No examples removed, values clamped or model weights reset. Config, training AMP, seed, split, optimizer settings and20-epoch budget remained unchanged. FP32 validation loss enters the ordinary scheduler/selection logic, so the amendment may affect the subsequent trajectory; do not claim bit-identical uninterrupted training or identical evaluation precision across every epoch. LR halved at19 under the existing scheduler. Training accuracy99.03% vs final validation88.96%, and validation loss spikes, indicate poor late generalization/numerical sensitivity rather than a reason to extend training.

## Selected class scores

| Class | Support | Precision | Recall | F1 | S06 F1 | S12 F1 |
|---|---:|---:|---:|---:|---:|---:|
| akiec | 49 | 0.6889 | 0.6327 | 0.6596 | 0.7674 | 0.7865 |
| bcc | 77 | 0.8171 | 0.8701 | 0.8428 | 0.8605 | 0.8902 |
| bkl | 165 | 0.8521 | 0.7333 | 0.7883 | 0.8626 | 0.8553 |
| df | 17 | 0.9286 | 0.7647 | 0.8387 | 0.8485 | 0.9032 |
| mel | 167 | 0.8125 | 0.7006 | 0.7524 | 0.7903 | 0.8123 |
| nv | 1007 | 0.9268 | 0.9682 | 0.9471 | 0.9581 | 0.9613 |
| vasc | 21 | 0.8750 | 1.0000 | 0.9333 | 0.9524 | 0.9302 |

S10 has lower F1 than S06 in every class. It is weaker standalone, but has complementary individual predictions. S10 fixes55 S06 errors; S06 fixes89 S10 errors;69 are jointly wrong. S10 also fixes43 S09 errors. The95.41% oracle union is a label-dependent diagnostic upper bound, **not achieved accuracy**.

## Bounded fusion decision

Predeclared before computing fused metrics: S11 equal S06/S10 and S12 equal S03/S06/S10. S03 recovers25 images both S06/S10 miss, versus19 for S02 and19 for S05; choose S03 as the one third model. No weight/criterion/view sweep. Accuracy-selected parents only, exactly aligned1503 validation identities/probabilities.

| Method | Accuracy | Macro-F1 | Model passes |
|---|---:|---:|---:|
| S06 ConvNeXt | 91.7498% | .862831 | 1 |
| S09 S02/S03/S06 | 91.7498% | .869094 | 3 |
| S10 V2-S | 89.4877% | .823162 | 1 |
| S11 S06/S10 | 92.0825% | .860166 | 2 |
| S12 S03/S06/S10 | **92.4152%** | **.877023** | 3 |

S11 fixes34/breaks29 vs S06: modest accuracy gain, worse macro-F1. S12 fixes37/breaks27 vs S06 and27/breaks17 vs S09:10 additional correct images, +.6653 percentage points and+.007929 F1 vs S09. S12 retains melanoma recall79.04%; class trade-offs remain visible in comparison CSV. This is an observed development-set gain after candidate selection, not an independent/statistically established improvement. Keep S06 as the cheap single-model reference, S12 as the stronger accuracy/balance ensemble, and retain every negative outcome.

## Artifact navigation

- S10 results: `results/structured_experiments/s10_efficientnet_v2_s_none_exploratory_seed42/` (metrics/predictions/history/recovery JSON/diagnostic PT/checks).
- S10 checkpoints: `checkpoints/structured/s10_efficientnet_v2_s_none_exploratory_seed42/{best.pt,best_macro_f1.pt,latest.pt}`.
- Original frozen failure evidence: `results/audit/s10_nonfinite_20261003/`.
- S10 selected figures/curves: run `figures/`; secondary class/matrix figures: `figures_macro_f1/`. PNG/PDF plus numeric matrix/class CSVs verified.
- S11/S12: `results/structured_experiments/s11_s06_s10_equal_probability_exploratory_seed42/` and `s12_s03_s06_s10_equal_probability_exploratory_seed42/`. Each has config, metrics, predictions, four PNG/PDF figure pairs, numeric matrices/class metrics, parent identities, log, record and closeout verification. Fusions have no new checkpoints/training curves; reference parent models/curves.
- Comparisons: `results/model_comparison/structured/s10_backbone_review/`, `s10_bounded_ensembles/` and shared `exploratory_screening_v1_seed42/`. Includes class comparisons/aligned errors/overlap/paired gains/predeclared candidates.
- Registry: `results/master_experiment_registry.csv`;473 historical rows preserved.

## Protocol / next step

Identical exploratory image-level70/15/15 manifest:7009/1503/1503;563 shared train/validation lesions disclosed. Strict scores stay separate; locked original test untouched. Later strict confirmation must freshly train selected finalists from external weights.

Next proposal: [S13 fixed TTA inference](S13_PROPOSAL.md), no new backbone training. Phase3 screening progress92%; total project progress75% (planning estimates; strict confirmation/test/XAI/paper remain).
