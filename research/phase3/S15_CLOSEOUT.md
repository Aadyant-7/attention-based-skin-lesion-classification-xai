# S15 DenseNet201 and S17 fixed fusion closeout

CPU-only verification; same 1503-image exploratory validation cohort; original locked test untouched. Checkpoint tensors finite; best/latest history and prediction identities authenticated; matrices, per-class scores and PNG/PDF curves verified. Both master registry entries closed out; all unrelated registry rows preserved.

| Method/checkpoint | Epoch | Accuracy | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|---:|---:|
| S15 accuracy winner | 12 | 89.6208% | .816913 | .805484 | .807519 |
| S15 F1 winner | 14 | 89.5542% | .823740 | .806132 | .810585 |
| S15 final latest | 17 | 89.4877% | .823281 | .787132 | .802195 |
| S06 ConvNeXt-Tiny | 17 | 91.7498% | .880435 | .853326 | .862831 |
| S10 EfficientNetV2-S | 15 | 89.4877% | .842993 | .809949 | .823162 |
| S12 reference | — | 92.4152% | .902493 | .858557 | .877023 |
| S13 FP32 identity reference | — | 92.4817% | .903260 | .858699 | .877454 |
| S17 equal S06 + S15 | — | 92.6148% | .909743 | .855342 | .877061 |

S15 stopped on patience after17epochs,31.65minutes; peak allocated VRAM1.94GiB. Best loss .599114 at accuracy winner; final loss .667579. Training accuracy99.79% with weaker validation indicates generalization limitations, not insufficient fit. No validation FP32 fallback occurred; no numerical amendment to S15.

DenseNet fixes61of124ConvNeXt errors; ConvNeXt fixes93of156DenseNet errors;63joint errors. DenseNet's class F1 is lower for all7classes; recall lower for6and tied forvasc. These complementary errors alone do not guarantee a useful fusion.

The single fixed50/50fusion fixes36ConvNeXt errors but breaks23(net13). Versus FP32 S12 it fixes25but breaks23(net2): only+.1331percentagepoints, slightly lower macro-F1. Its akiec recall67.35% versus reference71.43%; melanoma recall79.04% unchanged. This is the highest observed accuracy, but fails our material-gain gate; retain S12/FP32 identity as selected reference and exclude DenseNet from the selected final ensemble. No DenseNet weighting or further combination search.

## Artifacts
- `results/structured_experiments/s15_densenet201_none_exploratory_seed42/`: history, log, best-accuracy and best-F1 metrics/predictions, primary and F1 figures, verification.
- `checkpoints/structured/s15_densenet201_none_exploratory_seed42/`: best.pt, best_macro_f1.pt, latest.pt (preserved).
- `results/structured_experiments/s17_s06_s15_equal_probability_exploratory_seed42/`: metrics, probabilities, class metrics/matrices, figures, parent hashes, verification.
- `results/model_comparison/structured/s15_backbone_review/`: comparison table/PNG/PDF, per-class comparison, aligned errors and overlap JSON.
- `results/master_experiment_registry.csv`: authoritative registry.

All results are development/validation-selected exploratory results, not lesion-independent or test performance. Strict confirmation remains deferred.
