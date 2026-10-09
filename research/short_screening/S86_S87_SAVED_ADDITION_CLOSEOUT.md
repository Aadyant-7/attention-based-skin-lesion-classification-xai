# S86/S87 bounded saved-model additions

Four fixed unused candidates were screened for error complementarity; at most two separately added and a joint run allowed only if both passed the directional gate. This is adaptive exploratory validation selection, not independent confirmation. No training, fresh inference or test access.

| Method | Accuracy | Macro-F1 | Gained / lost vs S83 |
|---|---:|---:|---:|
| S83 reference | 94.0120% | 0.895201 | — |
| S86 add_s19 | 93.7458% | 0.891351 | 6 / 10 |
| S87 add_s77 | 93.7458% | 0.888905 | 5 / 9 |

Adding models is conditional on measured probability-fusion performance, not the union of all correct predictions. All source hashes, selection rules, precision differences, candidate/class diagnostics, changed identities, scores and PNG/PDF figures are saved. No weights, checkpoints or temperatures were searched.

## Decision

Both additions are rejected: each loses four net correct predictions versus S83. The joint equal-eight fusion was not run because neither individual addition passed the directional gate. Keep S83's94.0120% /0.895201 as the numerical development leader; do not add all models indiscriminately.

The four-candidate audit found31 recovered S83 errors for ResNet101,30 for frozen PanDerm Large,24 for MobileNet and22 for ImageNet22k ConvNeXt-Tiny. These label-based counts do not mean a practical fusion will recover all of them: the selected candidates also make124 and106 errors respectively on images S83 already gets right. Selection is adaptive on reused validation and therefore cannot establish independent superiority.

Next prepared candidate is S89, a fully fine-tuned Swin-T with an epoch15 pilot boundary and hard20 cap. That tests a new attention-based representation; it does not relabel or rerun the existing frozen PanDerm SVM. See `S89_SWIN_PROPOSAL.md`. GPU training awaits explicit approval.
