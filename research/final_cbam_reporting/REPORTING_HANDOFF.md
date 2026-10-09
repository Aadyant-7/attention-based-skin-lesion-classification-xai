# Final figures/tables and writing handoff — 9 October 2026

## Model versus accuracy: exploratory validation

| Model | Epochs completed | Accuracy checkpoint | Accuracy | Macro-F1 at that checkpoint |
|---|---:|---:|---:|---:|
| MobileNetV3-Large | 20 | 16 | 86.0279% | 0.786111 |
| EfficientNet-B0 | 20 | 20 | 86.3606% | 0.770660 |
| EfficientNet-B0 + CBAM | 20 | 19 | 86.3606% | 0.769301 |
| ConvNeXt-Tiny | 20 | 17 | 91.7498% | 0.862831 |
| EfficientNetV2-S | 20 | 15 | 89.4877% | 0.823162 |
| DenseNet201 | 17 | 12 | 89.6208% | 0.807519 |
| ConvNeXt-Small | 20 | 17 | 92.2156% | 0.854948 |
| ResNet101 | 20 | 19 | 88.0905% | 0.792032 |
| ConvNeXt-Tiny + CBAM (longer package) | 40 | 33 | 93.0140% | 0.877434 |

All use the same7009/1503 image-level development partition. Actual budgets and training-package differences are disclosed; S79 is not a pure CBAM ablation. Macro-F1 here belongs to the accuracy checkpoint, not necessarily each model's separate F1 maximum. No valid comparable completed EfficientNet-B3 score is invented. All backbones already use transfer learning; do not portray these as models trained from scratch.

## Main protocol/result table

| Method and evaluation | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|
| S53 retained equal-five (without added CBAM) | 93.6128% | 0.915781 | 0.865636 | 0.886857 |
| S80 CBAM-inclusive equal-five | 93.3466% | 0.913249 | 0.849629 | 0.877017 |
| S80 fixed post-development test audit | 87.0925% | 0.814371 | 0.795075 | 0.801646 |
| S31 original strict locked-test evaluation | 86.7598% | 0.814010 | 0.778724 | 0.794473 |

Protocol labels in `protocol_results.csv` must remain visible in the paper. The new audit does not supersede the original first evaluation or restore test independence. No score-triggered tuning follows it.

## Recommended results wording

"The retained heterogeneous ensemble achieved93.61% exploratory validation accuracy (macro-F1 0.8869). The separately evaluated CBAM-inclusive ensemble achieved93.35% exploratory validation accuracy (macro-F1 0.8770) and 87.09% in a post-development audit of the previously evaluated held-out cohort (macro-F1 0.8016). Grad-CAM visualizations explain representative predictions without altering the classifier."

Make validation prominent through a clearly labelled comparison figure and table, not an ambiguous general-performance headline. Highest observed tiny-gain/multi-image variants are historical secondary evidence, not retained single-image or test scores. CBAM benefit is mixed, not universally positive.

## Completed explainability

completed; 10 deterministic validation cases, covering melanoma/akiec correct and incorrect categories and remaining classes where available. Every case includes all-five branch Grad-CAM, raw/normalized arrays, overlays, input224, actual CBAM attention weights and label/confidence metadata. Cases: ISIC_0024351, ISIC_0024886, ISIC_0024710, ISIC_0024511, ISIC_0025128, ISIC_0024595, ISIC_0024726, ISIC_0025668, ISIC_0024308, ISIC_0025677.

The equal normalized-map composite is a display summary; neither attention nor Grad-CAM proves lesion segmentation, causal clinical reasoning or model correctness. Explainability does not increase accuracy.

## File/figure index (repository-relative)

- `results/final_cbam_reporting/v1/paper_tables/`: model comparison CSV plus PNG/PDF; protocol result CSV; S80 validation/test class-score CSV.
- `results/final_cbam_reporting/v1/ensemble/`: audit metrics, predictions/probabilities, class scores and raw/normalized confusion PNG/PDF.
- `results/final_cbam_reporting/v1/xai/<image_id>/gradcam_panel.{png,pdf}`: all-five explanations and composite; per-model folders hold overlays/maps and CBAM figures.
- `results/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/figures/`: accuracy/loss/F1 curves, selected confusion and class figures. Its `fixed_ensemble/figures/` is S80 validation.
- `results/short_screening/s53_equal_five_b0_addition/`: retained non-CBAM validation result and figures.
- `results/final_locked_test/v1/`: unchanged original strict test evidence.
- `results/master_experiment_registry.csv`: complete method/results index; S81 is explicitly post-development.
- Local-only report references: `docs/private_reference/report_guidance_2026-10-09/REFERENCE_NOTES.md`; originals and private notes excluded from Git.

## Next writing phase

Use the university A4/report structure after reconciling later guide instructions: preliminary pages; introduction/objectives; literature review; dataset/protocols; requirements/design/methodology; experiments/results; XAI; discussion/limitations; conclusions/future work; references; appendix/individual contributions. Add timeline, roles, constraints, risk and budget where the required Project Plan format calls for them. Journal/conference paper formatting is a separate template. Team identities, actual contributions, signatures, venue, deadlines and any additional guide requirements must come from the user; do not fabricate them.

Remaining work is manuscript/report composition, final layout/citation review and guide feedback. No new training, ensemble search or test evaluation is queued.
