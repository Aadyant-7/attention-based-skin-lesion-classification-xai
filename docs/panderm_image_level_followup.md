# PanDerm image-level and orientation follow-up

Date: 2026-09-28. The exploratory split uses the original strict development pool only: 7,009 train and 1,503 validation images. Its 1,503 locked strict-test images are excluded. **563 lesion IDs cross the exploratory train/validation boundary; 596 validation images share a lesion with training.** These are image-level validation results, not independent lesion-disjoint or final test scores.

The same [PanDerm Base](https://github.com/SiyuanYan1/PanDerm) frozen 224-pixel features and two fixed shallow heads from the strict probe were fitted afresh to the exploratory training split. The B0 streams are the already saved weighted and stronger-augmentation exploratory checkpoints, each evaluated at 224 and 384 pixels. Four deterministic PanDerm views (identity, horizontal flip, vertical flip, and both flips) are classified by its RBF SVM, and their probabilities are averaged with the four B0 streams at equal 1/8 weights. This requires no new full-network training. The PanDerm SVM head is saved locally under `checkpoints/exploratory/panderm_base_image_level_v1/`; the eight-stream predictor is `scripts/predict_panderm_exploratory_tta.py`.

| Method | Exploratory val accuracy | Macro F1 |
|---|---:|---:|
| Previous four-stream B0 control | 0.881570 | 0.828008 |
| PanDerm SVM alone, normal view | 0.884897 | 0.808220 |
| PanDerm SVM, four-view TTA | 0.890885 | 0.815206 |
| B0 four + PanDerm SVM TTA as one stream | 0.893546 | 0.839978 |
| **B0 four + four PanDerm orientation streams, eight equal** | **0.907518** | **0.857301** |

The eight-stream method is **+2.59 percentage points** above the previous exploratory B0 leader on the same validation images and improves macro F1 by 0.0293. It is selected after inspecting validation variants, so it is not an unbiased held-out estimate. The predictor was checked against cached validation probabilities on a development image: the class matched, and its maximum probability difference was under 0.00037 from GPU batch arithmetic.

The same orientation idea was checked on the lesion-disjoint strict validation split. Its best accuracy variant reached **0.876913 / 0.777119 macro F1**, a marginal accuracy gain over the earlier 0.876248 / 0.778083 strict candidate. Thus the 90.75% result **does not transfer to the strict protocol**. Full per-class results are in `results/exploratory/panderm_base_image_level_v1/`, `results/exploratory/panderm_base_image_level_tta_v1/`, and `results/accuracy_exploration/panderm_base_strict_tta_v1/`.

Reproduce with `python -m scripts.probe_panderm_image_level`, then `python -m scripts.probe_panderm_image_level_tta`. The source and weights setup is documented in [the strict PanDerm probe](panderm_probe.md). Classify one image using `python -m scripts.predict_panderm_exploratory_tta --image path/to/image.jpg` from the repository root. No locked test image or external diagnostic label is loaded by these experiments.
