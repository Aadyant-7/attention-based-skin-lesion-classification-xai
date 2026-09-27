# Phase 5B validation-only inference conclusion

The weighted EfficientNet-B0 + CBAM checkpoint was not updated. Its backbone, CBAM, and BatchNorm layers remained in evaluation mode; only the three classifier Dropout layers were stochastic. The 10-, 20-, and 30-pass estimates were formed from cumulative class-probability averages over the same fixed 1,503-image validation partition. Original and horizontal-flipped probabilities received equal weight for flip TTA.

| Protocol | Validation accuracy | Macro F1 |
|---|---:|---:|
| Weighted normal | 0.845642 | 0.756873 |
| Weighted flip TTA | 0.842315 | 0.770361 |
| Best Phase 4 ensemble | 0.843646 | 0.771810 |
| Oversampled flip TTA | 0.857618 | 0.729602 |
| MC dropout 10 | 0.842981 | 0.744330 |
| MC dropout 20 | 0.844977 | 0.752729 |
| MC dropout 30 | 0.843646 | 0.752305 |
| MC dropout 10 + flip | 0.844311 | 0.762049 |
| MC dropout 20 + flip | 0.841650 | 0.756033 |
| MC dropout 30 + flip | 0.841650 | 0.767283 |

The best MC protocol (30 passes + flip) did **not** beat the existing 0.857618 accuracy or 0.771810 macro-F1 validation leaders. The optional diagnostic `Resize(224) → CenterCrop(224)` preserved aspect ratio but scored only 0.816367 accuracy and 0.715878 macro F1. This changes the field of view, so its lower score is diagnostic evidence against adopting that inference transform for the current checkpoint; it does not isolate aspect ratio as the sole cause.

See `mc_dropout_results.csv`, `best_confusion_matrix.json`, `per_class_comparison.csv`, and `aspect_ratio_diagnostic.json` for full validation metrics. No training or test inference occurred.
