# S77: PanDerm Large did not pass the advancement gate

Completed 8 October 2026. Same exploratory cohort and paired frozen transfer recipe; no backbone updates or original-test inference.

| Method | Accuracy | Macro precision | Macro recall | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|---:|---:|
| Matched PanDerm Base | 88.5562% | 0.850318 | 0.777287 | 0.810305 | 66.4671% |
| Published PanDerm Large | 88.9554% | 0.853826 | 0.795856 | 0.820575 | 73.6527% |
| Retained S53 equal-five | 93.6128% | 0.915781 | 0.865636 | 0.886857 | 82.0359% |

Large gains six net correct predictions over Base (+0.3992 percentage points), with improved macro-F1 and melanoma recall. It misses the predefined >=1-point and >=90% standalone gate. **S78 was not run.** Large fixes 32 of S53's 96 errors but misses 102 cases S53 gets right. These counts show some complementary information; they do not establish that voting can exploit it, or prove every possible fusion would fail.

The paired study used official Base/Large weights, strict encoder loading, native CLS features, FP32 identity inference and author preprocessing. One fixed RBF SVM per encoder was fitted only on training data. Feature extraction took 278.68 seconds; PyTorch peak allocated GPU memory was 1,352.55 MiB. CPU classifier fitting/checkpoint validation took approximately 7 seconds per head. No long GPU training followed, no head/weight search was run, and original test results remain unchanged.

## Verification and recovery

Both classifiers converged and their reloaded bundles reproduced saved probabilities. Metrics were recomputed, all source/cache hashes checked, and all 565 previously committed registry rows remained unchanged. S76 and S77 added four rows in total. Full predictions/probabilities, class scores, raw/normalized confusion matrices and PNG/PDF comparison figures are saved.

After scoring and artifact creation, a final verification typo used `ROOT[path]` instead of `ROOT/path`, raising TypeError. The original execution source, signature and failure log were preserved. `recover_s77_report.py` completed reporting from saved files without repeating inference or fitting, and recorded the amendment. The metrics were not changed. A completed-run guard now prevents accidental rerunning of this finished study.

## Decision and limits

Retain S53; reject automatic PanDerm Large ensemble substitution and another long run based on this frozen screen. S76 also failed, so neither feature concatenation nor a larger frozen domain encoder delivered the requested material improvement. This does not prove complete PanDerm fine-tuning would fail; it supplies no sufficient evidence to spend that training budget now. Stop automatic classifier/weight/model variants after this bounded study.

The evidence points to a generalization problem rather than an identified loading/scoring bug: several trained members already fit the training images almost perfectly, while their validation errors persist. Earlier generic frozen-feature screens were also imperfect predictors of full fine-tuning (S58 passed but S59 failed to improve the ensemble). A cheap positive screen cannot guarantee a worthwhile long run.

All current scores are exploratory validation reused during development. PanDerm's complete pretraining overlap with HAM10000 cannot independently be ruled out. The original strict model's test accuracy remains 86.7598%; this study does not establish 93% test accuracy. Further performance claims need an appropriate genuinely untouched evaluation cohort after methodology selection, not another evaluation-driven adjustment to the old test.

Artifacts: `results/short_screening/s77_panderm_large_transfer/`; local classifiers: `checkpoints/short_screening/s77_panderm_large_transfer/{base,large}/svm.joblib`; features: `.cache/s77_panderm_large_transfer/`. These classifiers require their matching encoder, scaler, training feature kernel and author preprocessing. Source/protocol: `S77_PANDERM_LARGE_PLAN.md`.
