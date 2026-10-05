# S39 resolution-screen closeout

Strict DEVELOPMENT validation, post-test development. No training or test inference. Same immutable S29 checkpoint at224/320, FP32 identity; fixed B0/V2-S224 probabilities for ensemble comparisons.

| Method | Accuracy | Macro-F1 | MEL recall | BKL recall |
|---|---:|---:|---:|---:|
| 224 | 89.0220% | 0.814678 | 0.6627 | 0.8121 |
| 320 | 87.0259% | 0.780225 | 0.5663 | 0.8061 |
| ensemble_224 | 90.1530% | 0.843486 | 0.6084 | 0.8061 |
| ensemble_320 | 89.0220% | 0.821875 | 0.5663 | 0.7879 |

Standalone320 gained 32 correct predictions and lost 62; net -30. Ensemble320 gained 13 and lost 30; net -17.

Runtime 40.7seconds. Material gate FAILED. Retain224 reference. Do not extend this inference screen or start training automatically.

Interpretation: increasing inference resolution alone on a checkpoint trained at224 worsened accuracy and F1. This rejects that immediate change, not resolution-matched training or the general possibility that detail matters. The CPU associations did not predict an achieved training benefit.

Verification passed:1503 aligned predictions per method; finite normalized probabilities; recomputed accuracy/F1;224 decisions match original source; PNG/PDF figures exist; checkpoint hash unchanged. Registry scores verified/filled. All historical checkpoints and original test result preserved.
