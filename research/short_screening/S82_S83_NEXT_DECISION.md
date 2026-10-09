# S82/S83 research decision — 9 October 2026

User paused report writing and explicitly reopened accuracy development. Two bounded CPU experiments used the same saved FP32 exploratory validation probabilities; no training, GPU inference, raw test data or new test scoring. Original S80/S81 evidence is unchanged.

## New finding

The new CBAM model is more useful when **added alongside the original Tiny**, rather than replacing it. Equal-six voting retains Tiny, Small, DenseNet201, EfficientNetV2-S and B0, then adds Tiny+CBAM. Two existing standalone selection rules were compared: S79 accuracy-selected33 and macro-F1-selected35. No other epochs, subsets or weights were tried.

| Result | Accuracy | Macro-F1 | Melanoma recall | Correct /1503 |
|---|---:|---:|---:|---:|
| Retained S53 equal-five | 93.6128% | 0.886857 | 82.0359% | 1407 |
| S80 CBAM replacement, prior validation | 93.3466% | 0.877017 | 79.0419% | 1403 |
| S82 equal-six, accuracy-selected CBAM | 93.9454% | 0.893190 | 80.8383% | 1412 |
| S83 equal-six, macro-F1-selected CBAM | **94.0120%** | **0.895201** | **82.0359%** | **1413** |

S83 macro precision0.922321; macro recall0.874217. Versus S53 it gains12 and loses6 predictions: net+6, +0.3992percentage points accuracy and +0.008343 macro-F1, with unchanged aggregate melanoma recall. BKL gains4 net, NV3, DF1; BCC loses2; akiec/melanoma/vasc counts are unchanged. Within melanoma,3 errors are fixed and3 formerly correct predictions are lost; aggregate recall alone does not show identical cases.

Versus the attention-inclusive S80 validation reference, S83 has10 additional correct predictions, +0.6653percentage points accuracy and +0.018184 macro-F1. This is a descriptive comparison after development selection, not an independent statistical superiority claim.

## Decision and limits

S83 is the **highest observed exploratory validation accuracy so far**, exceeding earlier numerical single-image and multi-image variants. Both additions fail the original material gate against S53 because neither reaches8netcorrect and0.005absolute accuracy. Do not lower that gate retrospectively. Keep S53 as the retained reference under that policy, while preserving S83 as a promising numerical development leader that contains CBAM and improves the observed balanced metrics. A failed material gate is not a fabricated failure of the measured direction; the actual improvement remains recorded.

The new methods have no independent test result. S80's post-development audit remains87.0925% /0.801646 and cannot be assigned to S83. Existing S80 Grad-CAM panels likewise must not be relabelled as S83 explanations because its CBAM checkpoint and fusion differ. Reused validation and known earlier test outcomes limit generalization claims.

## Remaining bottleneck and next useful question

S83 still misses90 validation images: melanoma30, BKL20, NV18, akiec14, BCC4, DF3, vasc1. Melanoma/BKL/akiec account for64/90 errors. The next meaningful question is whether a specialist representation/training intervention can recover these difficult-class errors without losing current correct predictions, not whether another unrestricted weight/epoch sweep can find one more image. Review prior specialist/rescue and feature-head work before proposing such a run; do not repeat a rejected method under a new name. No new long GPU experiment is justified merely to reproduce the saved probability average.

The two-checkpoint CPU sequence is complete. Further costly training needs its own concrete hypothesis/configuration, compute estimate, screening gate and user approval. No training, test reevaluation or further combination search is queued here. Report-writing work remains paused at the user's request.

## Evidence

`results/short_screening/s82_cbam_addition_cpu/` and `s83_cbam_f1_addition_cpu/` contain source plans/hashes, full metric/prediction/probability/class/confusion packages, comparison PNG/PDF, changed identities and verification. Both appear in the master registry. The individual closeouts preserve each predeclared gate and decision.
