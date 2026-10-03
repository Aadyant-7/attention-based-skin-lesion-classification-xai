# S07–S09 closeout — bounded equal-probability ensembles

**All three CPU candidates completed and independently verified.** Same 1,503 exploratory validation images, same parent accuracy winners, no weight/criterion/grid search, raw images, GPU, test inference or new checkpoint.

Configs and hashes in `predeclared_candidates.json` were written before any new fusion probabilities/metrics were computed. Parent states: S02 epoch16, S03 epoch20, S06 epoch17. No S05 or secondary-F1 checkpoint was silently added.

| Method | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted loss | Model passes |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt standalone | 91.7498% | 0.880435 | 0.853326 | 0.862831 | 0.529672 | 1 |
| S02 + S06 (50/50) | 91.3506% | 0.883446 | 0.839042 | 0.857283 | 0.399437 | 2 |
| S03 + S06 (50/50) | 91.7498% | 0.879179 | 0.853662 | 0.862958 | 0.390022 | 2 |
| S02 + S03 + S06 (equal thirds) | 91.7498% | 0.888868 | 0.855425 | 0.869094 | 0.372957 | 3 |

## Interpretation

- **No accuracy improvement over S06.** S07 loses6 correct images; S08/S09 each retain1,379 correct but change which images are correct. Versus S06, S08 fixes26/breaks26; S09 fixes34/breaks34.
- **S09 is a balanced alternative:** macro-F1 +.006263 over S06 accuracy winner, melanoma recall130/167→132/167 (77.84%→79.04%), akiec33/49→35/49 (67.35%→71.43%). F1 improves akiec,df,mel,nv; bkl falls.8626→.8471 and bcc recall74/77→72/77. Three model passes are a cost, not an automatic preferred final method.
- S06 secondary epoch18 already achieves.868746 macro-F1 at91.48% accuracy with one model. S09 exceeds that F1 by only.000348 while gaining4 correct images; disclose this criterion/cost alternative rather than comparing only the weaker primary F1.
- S08 macro-F1 +.000127 is negligible as evidence for adding an entire model. S07 lowers both accuracy and F1. Preserve all outcomes; no follow-up weight sweep.
- S06 remains the simpler accuracy/cost reference; S09 remains the balanced fusion reference. The old S04 remains88.56%/.8057 and all three new candidates exceed it, but the strongest standalone is the proper current comparator.
- S02/S03 do recover some S06 errors; averaging also introduces errors. Correctness oracle unions in the S06 review are upper bounds using labels, not actual95% ensemble results. A stronger second representation is more useful to test now than arbitrary validation-fitted weights.

## Artifacts and reproducibility

Each `results/structured_experiments/s07_*`, `s08_*`, `s09_*` folder contains config, local run.log, record, metrics, aligned probabilities/predictions, parent_artifacts.json, closeout_verification.json, four PNG/PDF matrix/class/support pairs and three CSVs. Re-run exact configs with `python -m research.fuse --config <config>`; completed IDs preserve existing results. `--repair` is only for a inspected partial artifact write.

`results/model_comparison/structured/s06_bounded_ensembles/` contains the frozen three-candidate list, all-eight-method comparison CSV/PNG/PDF, per-class comparison CSV and paired verification JSON. Individual pair/triple comparison figures live under `s07_fixed_fusion/`, `s08_fixed_fusion/`, `s09_fixed_fusion/`. No fusion history/training curve/best epoch/new checkpoint exists; parent checkpoint/curve paths are explicitly linked.

CPU arrays were rechecked against exact fixed means, training-derived weighted log loss, manifest labels, macro/class metrics and confusion matrices; all figure exports checked. Master registry entries closed out. All non-candidate rows and473 historical rows preserved. Original locked test identity unchanged; no test images loaded. Strict and historical protocols remain separate.

Next proposal: [S10 EfficientNetV2-S](S10_PROPOSAL.md). No GPU launch authorized or performed in this closeout.

Current Phase Progress: 86%
Total Project Progress: 72%
