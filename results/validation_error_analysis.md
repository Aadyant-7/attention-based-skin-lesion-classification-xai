# Phase 3 validation error analysis

All figures use the saved lesion-grouped **validation** partition (1,503 images). No test predictions were computed. Full per-class metrics and confusion matrices are in `results/validation_inference/*.json`; the protocol summary is in `results/validation_inference_comparison.csv`.

| Best checkpoint | Protocol | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| Weighted | Normal | 0.845642 | 0.756873 | 0.845521 |
| Weighted | Horizontal-flip TTA | 0.842315 | 0.770361 | 0.842506 |
| Oversampled | Normal | 0.852961 | 0.732132 | 0.847110 |
| Oversampled | Horizontal-flip TTA | 0.857618 | 0.729602 | 0.852713 |

The weighted checkpoint with horizontal-flip TTA is the current **validation macro-F1 candidate**. TTA improved its macro F1 by 0.013488, while accuracy fell by 0.003327. TTA lowered the oversampled model's macro F1 by 0.002530, despite increasing accuracy; discard that protocol when prioritizing macro F1. These are validation-only choices, not final test conclusions.

The dominant confusion is between **mel and nv**. For weighted normal inference, 48 mel images were predicted nv and 50 nv images were predicted mel. The oversampled model predicted 56 mel images as nv and 23 nv images as mel. Weighted-model TTA still has 47 mel→nv and 56 nv→mel errors, and mel F1 decreases from 0.558 to 0.544. The model's high nv F1 (0.925–0.932 across these normal runs) therefore masks a persistent melanoma problem.

**bkl** is also confused with both nv and mel: weighted normal inference has 18 bkl→nv and 17 bkl→mel errors; oversampled normal inference has 25 bkl→nv and 9 bkl→mel. For **akiec** and **bcc**, confusion runs in both directions: weighted normal inference has 10 akiec→bcc and 8 bcc→akiec errors. Their F1 scores are 0.640 and 0.741 respectively. These counts come directly from the saved confusion matrices.

Full random oversampling raised peak validation accuracy but reduced macro F1 from 0.756873 to 0.732132 and produced very low training loss (~0.009) against validation loss ~0.701. The next controlled experiment keeps the original 7,009 training images and uses training-frequency-weighted focal loss to reduce the contribution of easy predictions, without repeating minority images. Improvement is a hypothesis to be judged on validation macro F1.
