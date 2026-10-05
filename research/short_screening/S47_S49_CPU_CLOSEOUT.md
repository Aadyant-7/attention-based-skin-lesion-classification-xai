# S47–S49 bounded CPU closeout

No GPU, model training, new image inference or test access. All methods use the same four preserved S46 macro-F1-selected exploratory models and validation cohort. Both robust-voting policies and one nonlinear meta-model configuration were recorded before execution; no rescue search.

| Method | Accuracy | Macro-F1 | Gain/loss versus S46 |
|---|---:|---:|---|
| S46 equal soft vote reference |93.4797%|.883680|—|
| S47 hard plurality vote, soft-probability tie-break |93.4132%|.882507|1gained/2lost|
| S48 per-class middle-two probability average |93.3466%|.880522|2gained/4lost|
| S49 nonlinear heldout-meta fusion |92.7478%|.871335|20gained/31lost|

None improved aggregate accuracy/F1. S49 improved melanoma recall to81.44% versus79.04%, but lost11net correct predictions. Preserve the trade-off without presenting it as the best accuracy method. S47/S48 outputs are normalized voting decision scores, not calibrated probabilities; their likelihood scores are not probability-calibration claims.

S49 used5lesion-group meta folds, zero lesion overlap in each fold, fixed100iterations/7leaves/minleaf30/L2=10. Source CNN checkpoints were selected using the same full validation cohort, so this is not independent CNN cross-validation or test performance. It was not fitted and scored on identical meta-training samples. Only1/5folds improved accuracy. No final whole-validation meta-model is fitted or deployed.

Verification:1503aligned unique IDs per result, finite normalized scores, recomputed accuracy/F1/confusion matrices, saved PNG/PDF, class scores and complete prediction/score CSVs. Registry updated; original source predictions/checkpoints unchanged.

Decision: keep S46 at93.4797%/.883680 as the strongest observed exploratory candidate. Close robust-voting and nonlinear-fusion routes at this budget. Do not launch training on the basis of these failures or claim a new independent test. Original86.7598%one-time test remains unchanged.
