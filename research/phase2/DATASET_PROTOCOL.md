# HAM10000 dataset and evaluation protocol

Protocol v1, fixed 3 October 2026. Counts and hashes come from saved repository manifests, not paper estimates. [Original dataset paper](https://arxiv.org/abs/1803.10417) describes 10,015 images; [usage notes](https://arxiv.org/html/1803.10417v1) describe repeated views of lesions. The older v1 manuscript has inconsistent provisional counts; repository metadata and the final abstract establish our actual inventory.

## Primary cohort

- Manifest: `data/splits/split_assignments.csv`; SHA-256 `db1ce9f8b83f176001dd89fd332fb1668c7d2ba5f58d77157d293a09e2c2e696`.
- Class index: `akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6`; use `data/splits/label_mapping.json`, never alphabetical assumptions at prediction time.
- 10,015 unique image IDs, 7,470 lesion IDs. Saved assignments match raw metadata diagnoses/lesion IDs; no missing/extra filenames.
- Original creation: `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)`; folds 0–2 test, 3–5 validation, remainder train (`src/data.py`). Reuse assignments; do not recreate them with a newer dependency version.

| Class | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| nv | 6705 | 4694 | 1006 | 1005 |
| mel | 1113 | 779 | 166 | 168 |
| bkl | 1099 | 769 | 165 | 165 |
| bcc | 514 | 359 | 77 | 78 |
| akiec | 327 | 228 | 50 | 49 |
| vasc | 142 | 100 | 21 | 21 |
| df | 115 | 80 | 18 | 17 |
| **TOTAL** | **10015** | **7009** | **1503** | **1503** |

Actual shares: 69.985% / 15.007% / 15.007%. Lesion counts: **5,230 / 1,122 / 1,118**. All train/validation/test pairwise lesion intersections are zero. The primary split is **lesion-disjoint**, not verified patient-disjoint or externally validated. Multiple images per lesion can weight image-level metrics unequally. Low minority supports limit certainty.

[Generated tables/charts](../../results/datasets/strict_lesion_disjoint/class_counts.md) and [split audit](../../results/datasets/split_audit.json) are the machine-backed sources; training-only count/weight values are saved by the Phase-2 generator. Balancing/augmentation applies only to training; validation/test keep natural class proportions.

## Historical exploratory protocol

`data/splits/exploratory/image_level_dev_v1.csv`, hash `75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb`, has the same total partition sizes. It shares **563 lesion IDs** between train/validation; **596 validation images** have a training lesion. Test assignments are identical to primary. Keep its 90.75% validation leader labelled exploratory; no mixing in the new controlled-model ranking.

## Test lock and reporting

Phase-2 verification reads manifest labels/IDs for counts, never test pixels, predictions, embeddings or model metrics. Future runner defaults to train/validation only and fails on a changed manifest. Test access needs a separately reviewed final evaluation command after checkpoint/config/ensemble freeze. Save all test predictions then, evaluate every natural test case once and report the full seven-class matrix.

Report accuracy, macro precision/recall/F1 and class-wise precision/recall/F1/support. Primary endpoint is macro-F1 with all seven classes and zero-division=0; accuracy is secondary. Selection, LR scheduling and early stopping use validation only. Later uncertainty intervals should resample by lesion, not treat repeated views as independent. Historical validation searches are development evidence, not independent estimates of generalization.
