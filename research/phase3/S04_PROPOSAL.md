# S04 proposal — fixed equal-probability fusion (CPU only)

**Prepared, not executed.** ID `s04_s02_s03_equal_probability_exploratory_seed42`.

## Question and rationale

Does averaging S02/S03 accuracy-winner probabilities **50/50** improve accuracy and balanced/class-wise performance without another GPU run? Their aligned predictions show 98 images only MobileNet classifies correctly and 103 only B0 classifies correctly. This supports testing complementarity, not a guaranteed gain. Literature review's ensemble/fusion section recommends starting with two models and fixed equal fusion; legacy ensembles also support diversity as a hypothesis.

One candidate only: seven-class probability mean, normal argmax, no weight search, calibration, prior correction, TTA, threshold tuning or new checkpoint selection. Parent accuracy winners: S02 epoch16 and S03 epoch20. Config pins both prediction hashes and checkpoint identities. Macro-F1 winner inputs happen to be identical; no criterion sweep. No fusion result exists yet.

## Protocol and resources

Existing exploratory manifest `data/splits/exploratory/image_level_dev_v1.csv`: 7,009/1,503/1,503 (~70/15/15); 563 train/val shared lesions disclosed. Original test preserved and excluded. Uses the same 1,503 validation rows; results remain exploratory and validation-selected through parent checkpoints. Strict confirmation later requires fresh strict training of the selected recipe/members, followed by a frozen test evaluation.

Approximately **10–30 seconds on CPU**, mostly plotting/imports; no GPU, raw images or checkpoint loading required. Saved probability CSVs, parent configs/records/metrics and split metadata are required. Actual new-image inference would require both trained checkpoints and costs two model passes.

Config `research/configs/phase3/s04_s02_s03_equal_probability_exploratory_seed42.json`. CPU-only readiness and three synthetic prediction-alignment tests passed. No experiment output folder or registry row has been created.

From project root when ready to run the proposed CPU candidate:

```powershell
.\.venv\Scripts\python.exe -u -m research.fuse --config research/configs/phase3/s04_s02_s03_equal_probability_exploratory_seed42.json
```

Optional live log in another PowerShell once created (the short run normally finishes immediately):

```powershell
Get-Content .\results\structured_experiments\s04_s02_s03_equal_probability_exploratory_seed42\run.log -Tail 30 -Wait
```

Results/log: `results/structured_experiments/s04_s02_s03_equal_probability_exploratory_seed42/`: `run.log`,config/record,metrics,predictions,confusion matrices,class scores,PNG/PDF figures,parent artifact references. Comparison table/figure: `results/model_comparison/structured/s04_fixed_fusion/`. Actual execution adds a separately labelled fixed-fusion row to the master registry; common-recipe training comparison remains separate.

**No new checkpoint or training curve exists for probability fusion.** Parent models/checkpoints/curves remain under their S02/S03 folders, linked by `parent_artifacts.json`. Source metrics/predictions/checkpoints are preserved. If partial CPU artifact creation fails, inspect `run.log`, then use the same command with `--repair`; a completed ID exits without recomputation.

After this one result, choose a matched added-CBAM ablation or targeted training-only augmentation on the strongest relevant backbone; do not select a fixed automatic list. Any future GPU proposal must state model/question/rationale/protocol/runtime/start+monitor commands/output paths, wait for approval, and **stop immediately after launch/log/first checkpoints are confirmed**. No polling, epoch updates or background monitoring; wait for the user to return.
