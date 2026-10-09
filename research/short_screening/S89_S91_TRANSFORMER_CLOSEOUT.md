# S89/S91 full20 closeout and S92–S94 transformer additions

9 October 2026. Both GPU runs completed their authorized20-epoch window. This closeout uses saved validation probabilities only; no training, fresh inference, or test access.

## Standalone results

| Model | Best accuracy / epoch | Best macro-F1 / epoch | Final accuracy / macro-F1 |
|---|---:|---:|---:|
| S89 Swin-T | 90.6853% / 15 | 0.860448 / 15 | 89.9534% / 0.848050 |
| S91 DeiT III Base | 90.5522% / 13 | 0.850535 / 17 | 89.6208% / 0.831388 |

Swin did not improve its epoch15 best during epochs16–20. DeiT tied its accuracy maximum at17, but the earliest accuracy-selected checkpoint remains13; macro-F1 selects17. Both final checkpoints are weaker than their selected best. These curves do not justify an automatic30-epoch extension. S90 Small was superseded before training, not a failed trained model.

## Standalone error diversity versus S83

| Selected standalone | S83 errors fixed | New errors on S83-correct images |
|---|---:|---:|
| S89 Swin-T epoch15 | 31 | 81 |
| S91 DeiT III Base epoch17 | 27 | 79 |

These are diagnostic overlaps, not an oracle accuracy or guaranteed ensemble recoveries. Actual fixed fusion results follow.

## Actual probability fusion

Each existing S83 constituent is retained; new members receive equal1/7 or1/8 weighting. Use each transformer’s existing standalone macro-F1 winner, selected before this fusion study. No weights or checkpoints searched.

| Method | Accuracy | Macro-F1 | Gained / lost versus S83 | Net | Melanoma recall |
|---|---:|---:|---:|---:|---:|
| S83 reference | 94.0120% | 0.895201 | — | — | 82.04% |
| S92 +swin | 93.6793% | 0.889231 | 8 / 13 | -5 | 80.84% |
| S93 +deit_base | 93.6793% | 0.889811 | 7 / 12 | -5 | 79.64% |
| S94 +both | 93.4132% | 0.889986 | 8 / 17 | -9 | 77.25% |

Material gate: at least8netcorrect and0.005absolute accuracy gain, macro-F1/melanoma recall nondecreasing. Passed by: none.

Decision: keep S83 unchanged. The tested equal-weight transformer additions do not improve the current ensemble. The joint addition improves akiec from 35 to 37 correct and nv from 989 to 992, but reduces melanoma from 137 to 129, bkl from 145 to 142, bcc from 73 to 71 and df from 14 to 13; vasc stays 20. Thus it does not recover new images without worsening others. No additional GPU work is queued.

Checkpoints/optimizer tensors are finite, model state dictionaries load strictly, three selected/final prediction packages per backbone reproduce checkpoint/history metrics, class scores and confusion matrices. Curves, source hashes, LR history and safe AMP-overflow skip counts are preserved in each `full20_closeout_verification.json`. The original Swin epoch15 snapshots remain unchanged. All unrelated historical registry rows are preserved.

Artifacts: `results/short_screening/s92_s94_transformer_addition_cpu/` contains the plan, overlap/class diagnostics, three full metric/probability/class/confusion PNG/PDF packages, changed image IDs and comparison figures. Individual training packages remain in `results/short_screening/swin_transfer_v1/` and `results/short_screening/deit3_base_transfer_v1/`; local checkpoints are in the matching `checkpoints/short_screening/` folders and are not distributed by Git.

Repeated exploratory image-level validation selection, after prior held-out outcomes were known; not independent confirmation or a test score. Same cohort, but native architecture, pretraining and training recipe details differ. No new training, inference, weight or checkpoint search.
