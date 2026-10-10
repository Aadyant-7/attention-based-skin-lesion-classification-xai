# S95/S96 regularization closeout

10 October 2026. S95 completed 20 epochs. CPU-only closeout verified all best-accuracy/best-F1/latest checkpoints, finite model/optimizer states, strict model loading, saved prediction/history metrics, class scores, matrices and curves. No new GPU inference or test scoring. Source code/config/pretraining fingerprints remain unchanged.

| Standalone | Best accuracy | Macro-F1 at accuracy winner | Best macro-F1 | Epoch |
|---|---:|---:|---:|---:|
| S06 historical control | 91.7498% | 0.862831 | 0.868746 | accuracy 17 / F1 18 |
| S95 decay 0.05 | 91.6168% | 0.859748 | 0.859748 | both 17 |

Final epoch 20: 90.6188% / 0.841269. Stronger decay did not improve standalone performance. Historical AMP versus new FP32 validation and the modern numerical safeguards limit a pure causal claim; report this as a negative development experiment.

| Fixed ensemble | Accuracy | Macro-F1 | Gained / lost vs S83 | Melanoma recall |
|---|---:|---:|---:|---:|
| S83 existing equal-six | 94.0120% | 0.895201 | — | 82.04% |
| S96 S95 replaces original Tiny | 93.2136% | 0.874849 | 3 / 15 | 80.84% |

Net -12 correct; material gate passed: False. Decision: reject_no_balanced_gain. S95 alone recovers 19 of S83's 90 errors, but makes 55 errors on previously correct images; these are diagnostics, not an oracle ensemble.

All other members/weights and CBAM stayed fixed. Exactly one predeclared replacement used S95's existing standalone F1 winner; no weights, epochs or alternative combinations searched. Existing S83 and historical rows/checkpoints are preserved. All unrelated committed registry rows were verified unchanged.

Artifacts: `results/short_screening/convnext_regularization_v1/s95_convnext_tiny_wd005_exploratory_seed42/` (training/full 20-epoch verification), matching local `checkpoints/short_screening/convnext_regularization_v1/` folder; `results/short_screening/s96_convnext_replacement_cpu/` (plan/source hashes, metrics, prediction probabilities, changed IDs, class scores, confusion/comparison PNG/PDF figures, verification).

Repeated exploratory validation selection after earlier test outcomes were known; not independent test evidence. S95 uses FP32 validation, unlike original S06 AMP; historical comparison is not a pure causal weight-decay estimate. A later evaluation on the previously evaluated test cohort is a post-development audit, not a new first independent test. Reaching 95% validation is not guaranteed and cannot determine whether unseen-test accuracy exceeds 95%.
