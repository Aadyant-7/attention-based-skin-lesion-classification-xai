# R201 B3: user-stopped closeout

The user requested cancellation after the current epoch. Epoch **28** finished
validation and committed its atomic checkpoint and bookkeeping. A verified
external helper then stopped only that training process and its child process.
No epoch-29 update appears in the log. Any uncommitted next-epoch setup was
discarded; epoch 28 is the final state.

| Selection | Epoch | Inner validation accuracy | Macro-F1 |
|---|---:|---:|---:|
| Raw accuracy winner | 21 | 85.8093% | 0.757946 |
| Raw macro-F1 winner | 26 | 85.6984% | 0.759662 |
| Final committed state | 28 | 85.5876% | 0.755297 |

Final training accuracy was 99.8106% over repeated balanced augmented exposures;
final validation CE was 1.316764. Three LR reductions occurred at epochs 11, 19
and 26. No numerical event was recorded. The original meaningful-stopping policy
had not fired: this is **user cancellation**, not an automatic convergence stop.
The frozen maximum-50/minimum-25 configuration and policy remain unmodified.

The saved package includes best-accuracy, best-F1 and final prediction CSVs,
probabilities, class scores, raw/normalized confusion matrices and PNG/PDF figures,
plus history, LR history and training curves. Both registries say `stopped_by_user`.
`latest.pt`, `best.pt` and `best_macro_f1.pt` were retained with unchanged SHA256
hashes during CPU closeout. `final.pt` is a new copy of the committed final state
with administrative-stop provenance. Checkpoints remain local and Git-ignored.

Paths:

- Results: `results/image_level_replication/v2/r201_b3_dermai_adaptation_fold00_seed42/`
- Checkpoints: `checkpoints/image_level_replication/v2/r201_b3_dermai_adaptation_fold00_seed42/`
- Cancellation: `administrative_stop.json` in the result folder.
- Full metrics and checkpoint hashes: `closeout.json` in the result folder.
- Current recipe proposal: [ConvNeXt-Tiny + CBAM](CONVNEXT_NEXT_PROPOSAL.md).

This result does not justify further B3 compute. High repeated-exposure training
accuracy and much lower validation accuracy are consistent with overfitting;
individual causes have not been isolated. More exposure, stronger balancing and
a paper-inspired head did not establish higher accuracy. No outer assessment or
locked-test inference was performed. This is one fold's inner validation, not a
completed K10 experiment. Historical ConvNeXt scores use a different cohort;
they motivate the next package but are not a matched numerical comparison.

No new model, ensemble, TTA or GPU run was launched after cancellation.
