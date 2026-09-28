# PanDerm Base strict-development probe

Date: 2026-09-28. All reported scores are on the repository's **1,503-image, lesion-disjoint strict validation partition**. The locked 1,503-image test partition was not used for feature extraction, training, prediction, or model selection. All seven HAM10000 classes are included. The separate `accuracy-exploration` branch preserves the earlier strict baseline on `main`.

## Why this method

The authors of [PanDerm](https://github.com/SiyuanYan1/PanDerm), a [Nature Medicine dermatology foundation model](https://pmc.ncbi.nlm.nih.gov/articles/PMC12353815/), provide a ViT-B/16 checkpoint pretrained on a large mixed-modality dermatology collection and a linear-probe recipe. It offers task-relevant visual features unlike the project's ImageNet B0 backbone. The source [checkpoint link](https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB/view) is 343 MB; its SHA-256 is recorded in `results/accuracy_exploration/panderm_base_strict_lp_v1/protocol.json`. The authors' repository is CC BY-NC-ND 4.0 for academic noncommercial research. The third-party source and weights are kept in ignored `.cache/`, and are not committed or redistributed here. We cannot independently verify every pretraining image against HAM10000, so potential pretraining overlap remains a limitation.

The official `linear_eval.py` loads its `test` rows, so it was **not** run. Our scripts filter the fixed development manifest to `train` and `val`, verify lesion separation, and do not construct a test image loader. The original PanDerm Base checkpoint is frozen during feature extraction; its author-provided 256-short-edge resize, 224 center crop, and normalization are used.

## Results and stop decision

| Candidate | Strict val accuracy | Strict val macro F1 | Interpretation |
|---|---:|---:|---|
| PanDerm frozen, author's unscaled logistic head (C=53.76) | 0.838989 | 0.679145 | Hit 1,000-iteration convergence cap; retained as a measured probe, not selected. |
| PanDerm frozen, standardized logistic head (C=1) | 0.845642 | 0.694255 | Head saved locally. |
| PanDerm frozen, standardized RBF SVM (C=10) | 0.846307 | 0.682778 | Head saved locally; complementary errors to B0. |
| PanDerm last-two-block pilot, best epoch 2 | 0.844311 | 0.690947 | Stopped after two epochs under preset 85.5% gate; best/latest checkpoints retained. |
| Four equal streams: PanDerm SVM + standardized linear + weighted B0 + oversampled B0 | 0.874917 | 0.771935 | Frozen-backbone ensemble. |
| **Four equal streams: PanDerm SVM + pilot + weighted B0 + oversampled B0** | **0.876248** | **0.778083** | Best fixed candidate examined on strict validation. |

The previous strict leaders were 0.858949 accuracy and 0.771810 macro F1. The new selected four-stream candidate is **+1.73 percentage points** higher in accuracy and +0.00627 macro F1, but it is selected from validation candidates, so the difference can be optimistic. It is **not** a final held-out result and does not meet the requested 93%. The short fine-tuning pilot regressed on its own and was stopped instead of extending to 20 epochs. Full per-class metrics and confusion matrices are in `results/accuracy_exploration/`.

## Reproduce and use

From the repository root, install `requirements.txt` into the project's environment, then obtain the [PanDerm source](https://github.com/SiyuanYan1/PanDerm) and [authors' Base weights](https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB/view):

```powershell
git clone --depth 1 https://github.com/SiyuanYan1/PanDerm.git .cache/research_panderm
.\.venv\Scripts\python.exe -m gdown "https://drive.google.com/uc?id=17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB" -O ".cache/panderm_bb_data6_checkpoint-499.pth"
.\.venv\Scripts\python.exe -m scripts.probe_panderm_base
.\.venv\Scripts\python.exe -m scripts.probe_panderm_heads
.\.venv\Scripts\python.exe -m scripts.train_panderm_pilot
.\.venv\Scripts\python.exe -m scripts.compare_panderm_ensembles
```

These scripts save feature caches under `.cache/panderm_base_strict_lp_v1/`, local heads and `best.pt`/`latest.pt` under `checkpoints/accuracy_exploration/`, and tracked metrics under `results/accuracy_exploration/`. The checkpoints and source weights are intentionally ignored by Git. After reproduction, classify one image with `python -m scripts.predict_panderm_strict_ensemble --image path/to/image.jpg`. The predictor was checked against the saved four-stream validation probabilities on a development image; maximum probability difference was 0.00021 from GPU batch-size arithmetic and the predicted class matched.
