# S54/S55: bounded PanDerm fusion closeout

The preserved PanDerm Base identity-view frozen features + RBF SVM were evaluated on the exact same 1,503 exploratory validation images as S53. No backbone inference, new head fitting, GPU training or test evaluation occurred.

Before evaluation, two fixed combinations were declared: add PanDerm as an equal sixth member; if that failed the directional gate, replace B0 with PanDerm and use equal fifths. No weights, thresholds, views, epochs or alternative heads were searched.

| Method | Accuracy | Macro precision | Macro recall | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|---:|---:|
| S53 reference, equal five | 93.6128% | 0.915781 | 0.865636 | 0.886857 | 82.0359% |
| S54 add PanDerm, equal six | 93.3466% | 0.919036 | 0.859430 | 0.885008 | 80.2395% |
| S55 replace B0 with PanDerm, equal five | 93.2801% | 0.912717 | 0.862674 | 0.884112 | 79.6407% |

PanDerm standalone accuracy was 88.4897%. It correctly classified 28 of the reference's 96 errors but missed 105 images the reference got right. Complementary hard predictions therefore did not translate into a useful equal-probability fusion gain.

- S54 gained 4 correct predictions and lost 8: net -4, 1,403 correct / 100 incorrect.
- S55 gained 8 and lost 13: net -5, 1,402 correct / 101 incorrect.
- Both failed the directional and material gates. Do not adopt either. Preserve S53 (provisional 93.6128%) and GPU-reproduced four-member reference S50 (93.4797%).

This rejects these two fixed uses of the existing frozen PanDerm/SVM; it does not prove that every fine-tuned PanDerm model would be ineffective. It gives no direct evidence to justify an expensive PanDerm training run. Do not expand this into another averaging/weighting search. Any future long training needs a separate capacity/transfer-learning rationale and bounded screening plan.

## Verification and artifacts

Image IDs and labels were aligned with existing validation predictions; only split identities were used to exclude locked-test images. Class order, finite normalized probabilities, source/checkpoint hashes and exact reconstruction of S53 were checked. Scores and confusion matrices were independently recomputed from saved CSVs. Source files and checkpoints were not modified.

`results/short_screening/s54_s55_panderm_fusion/` contains the predeclared plan, summary, comparison PNG/PDF/CSV, and separate `equal_six_panderm_addition/` and `replace_b0_with_panderm/` folders. Each saves metrics, predictions, probabilities, source/checkpoint manifests, class comparisons, class scores, raw/normalized confusion matrices and PNG/PDF figures. The master registry contains both experiments.

These are post-test exploratory development results after repeated validation selection, not independent test results. Original locked-test accuracy remains 86.7598%; the original strict/test methodology was not changed.
