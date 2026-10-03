# S18 ConvNeXt-Small final closeout

Completed20epochs across approved8/15/20epoch boundaries. Original config, source hashes, weights, split, seed, optimizer/loss, scheduler and checkpoint-selection rules unchanged. The original epoch8gate was explicitly superseded by the user's later continuation approvals; no failed-pilot classification retained. No other experiment or ensemble launched during this closeout; original locked test untouched.

| Result | Epoch | Accuracy | Macro precision | Macro recall | Macro-F1 | Val loss |
|---|---:|---:|---:|---:|---:|---:|
| Best accuracy and best F1 |17|92.2156%|.879387|.837148|.854948|.511493|
| Final latest |20|91.2841%|.869795|.819336|.839791|.644037|
| S06 Tiny accuracy winner |17|91.7498%|.880435|.853326|.862831|.529672|

Small improves accuracy by .4657 percentage points (7additional correct/1503), but macro-F1 falls .007882. This is a modest accuracy gain, not a material balanced improvement or proven statistical advantage. Accuracy declines after17: epochs18/19/20=91.35/91.42/91.28%; no further training proposed. Saved Tiny F1-selected checkpoint separately reaches .868746 at18; compare accuracy-selected winners primarily, disclose independent F1 selection.

Class-F1 improves versus Tiny for akiec/bcc/bkl/mel/nv, but falls for df/vasc; macro averaging exposes these minority trade-offs. CPU aligned error analysis: Small fixes57Tinyerrors; Tiny fixes50Smallerrors;67jointwrong. Keep Small as a future ensemble candidate because its performance is competitive and some errors differ; no ensemble gain has been established. No fusion generated tonight.

CPU verified finite best/latest weights, full latest history, both selected metrics/predictions, config/split/source provenance, confusion matrices, class scores and PNG/PDF curves. All unrelated registry rows preserved. Runtime24.50minutes, peak allocatedVRAM2.38GiB; all validationAMP, no FP32 recovery amendment. Natural scheduler learning-rate reductions remain part of the original recipe, not a new intervention.

## Saved artifacts
- results/structured_experiments/s18_convnext_small_none_exploratory_seed42/: config, history, metrics, both selected prediction files, figures/figures_macro_f1, verification and screening snapshots.
- checkpoints/structured/s18_convnext_small_none_exploratory_seed42/: best.pt, best_macro_f1.pt, latest.pt; preserved epoch8/15checkpoint copies in screening_epoch_08/ and screening_epoch_15/.
- results/model_comparison/structured/s18_backbone_review/: matched Tiny/Small comparisons and class/error analysis.
- results/master_experiment_registry.csv: completed20epoch entry, candidate decision.

Overall observed exploratory best remains S17 fixed fusion92.6148%; selected balanced reference remains S12FP32identity92.4817%/.877454. Strict confirmation/test evaluation remain deferred. Tomorrow's next research step requires a new decision; no continuation or other run queued.
