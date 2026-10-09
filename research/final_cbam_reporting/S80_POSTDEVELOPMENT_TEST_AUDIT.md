# S80 fixed CBAM ensemble: post-development test audit

User-authorized on9October2026, with the exact S80 method frozen before inference. This is the first S80 evaluation on this cohort, **not the first evaluation of the cohort**. Original S31 artifacts/receipt/report are preserved unchanged. Earlier test performance was already known before later development, so this score is descriptive post-development evidence, not a pristine independent final-test claim. No test-derived tuning or method change follows it.

Five frozen constituent passes,FP32 identity224RGB/ImageNet normalization; labels withheld during inference; equal0.2 probability averaging. Grad-CAM does not change inference. Exact checkpoint epochs/hashes and zero training/development-to-test image/lesion overlap are recorded in `results/final_cbam_reporting/v1/frozen_audit.json`.

- Accuracy: **87.0925%**, macro precision:0.814371, macro recall:0.795075, macro-F1:**0.801646**, weighted-F1:0.866897.
- Correct:1309; incorrect:194; total1503. Melanoma recall:57.14%; akiec recall:65.31%.
- S80 exploratory validation93.3466% /0.877017; audit minus validation:-6.2542 percentage points.
- Original S31 strict test86.7598% /0.794473; different models and training protocol, so this is not a controlled treatment-effect comparison.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| akiec | 0.7805 | 0.6531 | 0.7111 | 49 |
| bcc | 0.7667 | 0.8846 | 0.8214 | 78 |
| bkl | 0.8000 | 0.7273 | 0.7619 | 165 |
| df | 0.7368 | 0.8235 | 0.7778 | 17 |
| mel | 0.7007 | 0.5714 | 0.6295 | 168 |
| nv | 0.9159 | 0.9532 | 0.9342 | 1005 |
| vasc | 1.0000 | 0.9524 | 0.9756 | 21 |

## Saved evidence

`results/final_cbam_reporting/v1/ensemble/`: metrics, predictions/probabilities, seven-class scores and raw/normalized confusion PNG/PDF. Each member's folder contains the same package from its single original pass. Comparison tables/figures, freeze, execution receipts and original-artifact hashes are in the parent folder.

## Paper wording

The retained non-CBAM S53 achieved93.6128% best exploratory validation accuracy /0.886857 macro-F1. The separately reported CBAM-inclusive S80 achieved93.3466% /0.877017 on exploratory validation. Both development scores use a repeatedly reused image-level split with within-development lesion overlap; neither is a strict independent-test claim. Do not attribute S53's score to S80 or label a highest observed validation variant as test performance.

No further training or alternative test method is authorized by this audit. Next work: reviewed XAI examples, report/paper figures and writing.
