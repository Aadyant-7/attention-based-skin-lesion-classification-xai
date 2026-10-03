# S05 closeout — matched B0 + CBAM ablation

Completed 20 epochs,3 October 2026; ID `s05_efficientnet_b0_cbam_exploratory_seed42`. Runtime743.02s/**12.38min**,4,221,413 parameters,peak allocated VRAM856.00MiB (not total reservation). Runtime differed from S03; machine load was not controlled, so do not claim CBAM speeds training.

| Saved state | Epoch | Accuracy | Macro precision | Macro recall | Macro-F1 | Loss |
|---|---:|---:|---:|---:|---:|---:|
| S03 no-CBAM accuracy/F1 winner |20 |86.3606% |.753429 |.791633 |.770660 |.630469 |
| **S05 accuracy winner** |**19** |**86.3606%** |.759920 |.781260 |.769301 |.624005 |
| **S05 macro-F1 winner** |**16** |86.2941% |.756883 |.797873 |**.774591** |.602329 |
| S05 final/latest |20 |85.6953% |.742228 |.772634 |.755472 |.637923 |

Exact selected metrics, full precision/recall/class scores and matrices live in primary/secondary JSONs. Do not combine epoch19 accuracy with epoch16 F1 or classes. `best.pt`=19,`best_macro_f1.pt`=16,`latest.pt`=20.

## Did CBAM help?

**No primary accuracy improvement:** both accuracy-selected models correctly classify1,298/1,503. Primary macro-F1 falls0.001359,macro recall falls0.010373,while macro precision rises0.006492 and weighted loss falls0.006463. Secondary criterion comparison gives **+0.003931 best macro-F1**, with one fewer correct image than S03. This is a small criterion-dependent trade-off, not an overall CBAM advantage.

| Class (support) | S03 F1 | S05 accuracy-winner F1 | Delta |
|---|---:|---:|---:|
| akiec (49) |.6458 |.6237 |-.0222 |
| bcc (77) |.7805 |.7564 |-.0241 |
| bkl (165) |.7461 |.7391 |-.0070 |
| df (17) |.7778 |.8333 |+.0556 |
| mel (167) |.6862 |.6997 |+.0135 |
| nv (1,007) |.9321 |.9329 |+.0007 |
| vasc (21) |.8261 |.8000 |-.0261 |

Primary melanoma recall rises117/167→120/167 (**70.06%→71.86%**); bcc recall falls64/77→59/77 and akiec31/49→29/49. df improvement involves only one additional correct image. Secondary F1 winner improves akiec,bkl,df F1 versus control, but lowers bcc,mel,nv,vasc F1. Its melanoma recall122/167 is73.05% with lower precision. CSV explicitly separates selection criteria; do not cherry-pick classes from different winners.

Matched training policy/input/head/pretraining/split/seed; only added CBAM plus administrative recipe/phase/run labels differ. Native SE retained,CBAM before GAP. Scheduler histories differ legitimately: S05 LR reduction before13, S03 none. Same seed does not make head initialization/random training paths identical after adding a module. One seeded validation-selected pair cannot establish significance or universal causal benefit.

Minimum val loss at14; final train94.89% vs val85.70%, with worsening late loss. Four AMP optimizer skips recorded (4,376/4,380 updates). Preserve this negative/mixed attention result; no automatic B0 tuning extension or CBAM inclusion in ensembles. S04 frozen fixed-fusion reference remains88.56%/.8057, but it is a two-model method with a different cost.

## Verified/preserved artifacts

- `results/structured_experiments/s05_efficientnet_b0_cbam_exploratory_seed42/`: config/environment/pretraining/record,log/progress,20-epoch history,summary,primary+secondary metrics and1,503 predictions each,`closeout_verification.json`.
- `figures/`: seven PNG/PDF curves/matrix/class/support pairs and three CSVs;`figures_macro_f1/`: secondary matrix/class/support figures and CSVs. Accuracy curve/normalized matrix visually checked.
- `checkpoints/structured/s05_efficientnet_b0_cbam_exploratory_seed42/`: best/F1/latest CPU-loaded; source/config hashes,model weights,selected predictions,history and metrics verified against committed states.
- `results/model_comparison/structured/s03_s05_cbam_ablation/`: criterion-labelled table/graph and class deltas. Common exploratory training table now includesS02/S03/S05; strict results remain separate.
- Registry completed/closed out; all other rows unchanged including473 historical rows. Original exploratory manifestSHA `75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb`,7,009/1,503/1,503 (~70/15/15),563 shared train/val lesions. Test never loaded; later strict confirmation requires fresh strict training.

Decision: retain both attention checkpoints and class evidence, then [screen a distinct stronger backbone](S06_PROPOSAL.md) and [build bounded diverse ensembles](DIVERSE_BACKBONE_STRATEGY.md). No S06 training launched.

Current Phase Progress: 78%
Total Project Progress: 68%
