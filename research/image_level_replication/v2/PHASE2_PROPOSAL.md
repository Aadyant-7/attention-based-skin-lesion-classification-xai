# Phase 2 — prepared B3 fold-00 GPU proposal

**Status: prepared pending separate GPU approval. No experiment launched.**

## Proposed run

`r201_b3_dermai_adaptation_fold00_seed42`: fresh ImageNet EfficientNet-B3; 112×112; large two 4096-unit ReLU head/dropout 0.5; full fine-tuning; Adam 1e-4; batch 16; training-only balanced augmented exposures and cross-entropy. No added CBAM, focal loss or Mixup. [Verified recipe and assumptions](RECIPE_DECISIONS.md).

Question: does the reported B3/input/head/balancing approach produce a useful high-performing inner-validation component under the new image-level protocol? This directly investigates the chosen paper's recipe instead of modifying previous predictions. Higher accuracy is not guaranteed.

Fixed fold 0: **8,111 fitting / 902 inner validation**, seed 42. The 1,002 outer assessment images have no dataset/loader or inference path in this runner. Earlier project cohorts/checkpoints/results remain preserved. New scores use a different cohort and must be labelled accordingly.

## Compute expectation and budget

- RTX4060 8GB; about 33.8M trainable parameters including 23.1M in the new head.
- Exactly**38,010 exposures /2,376 batches per training epoch**: each class 5,430; original support 8,111. This is 4.69× the original count, not free balancing.
- Planning estimate **2–6 minutes/epoch; roughly 1–5 hours for 25–50 epochs**, subject to Windows image decoding, augmentation and deterministic kernels. No GPU timing benchmark has been run during preparation.
- Approximate **2–4GB VRAM**, unmeasured until approved launch; reserve **3–5GB disk** for checkpoints/atomic replacement/diagnostics. Local free space checked during preparation:21.2GiB.
- Hard cap 50; no automatic extension. Minimum 25 except numerical failure. Meaningful accuracy delta 0.002 / F1 delta 0.003, patience 10, LR scheduling; late flat-curve guard after 30 uses 8 stale epochs plus an LR reduction and 3-epoch settling. Full policy is frozen in the config.

## Exact commands

Start (only after GPU approval):

```powershell
Set-Location 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
.\.venv\Scripts\python.exe -B -m research.image_level_replication.v2.train_b3 --train
```

Manual live log:

```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\image_level_replication\v2\r201_b3_dermai_adaptation_fold00_seed42\train.log' -Tail 20 -Wait
```

Resume after the old process has exited:

```powershell
Set-Location 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
.\.venv\Scripts\python.exe -B -m research.image_level_replication.v2.train_b3 --resume
```

Resume requires the same config, source hashes, external weights and runtime. `latest.pt` is the authoritative complete-epoch state, including model/Adam/scheduler/RNG/history/winners. A partial epoch replays from the last committed boundary; it is not silently treated as completed. Failure diagnostics remain preserved. `--closeout` repairs only a run already at its stopping boundary, without GPU inference. `--check` is CPU metadata-only and cannot launch training.

If a power cut leaves a registry-lock timeout, first verify no project training/registry writer remains active. Only then remove the specifically reported stale `results/image_level_replication/v2/experiment_registry.lock` or `results/master_experiment_registry.lock`, retaining the CSV/checkpoints, and retry `--resume`. Run locks themselves use OS locks and release automatically when the process exits. Do not remove a lock belonging to an active writer.

## Artifact locations

Result/log root: `results/image_level_replication/v2/r201_b3_dermai_adaptation_fold00_seed42/`.

- `train.log`, `config.json`, `launch_manifest.json`, `environment.json`, `process.json`, `progress.json`.
- `history.csv`, `lr_history.csv`, `meaningful_stopping.json`, `numerical_events.json`, failure evidence.
- `best_accuracy/`, `best_macro_f1/`, `final/`: metrics, complete predictions/probabilities, class scores, raw/normalized confusion matrices and PNG/PDF figures.
- Root `figures/`: train/validation loss/accuracy/F1 curves. `closeout.json`: exact winners/epochs, final metrics, stop reason and checkpoint hashes.
- Local V2 and master registries receive this ID only when training actually starts.

Checkpoint root: `checkpoints/image_level_replication/v2/r201_b3_dermai_adaptation_fold00_seed42/`, containing `best.pt`, `best_macro_f1.pt`, `latest.pt`, `final.pt`. Best files are weights/selection packages; recovery uses `latest.pt`.

CPU evidence and immutable launch signature: `results/image_level_replication/v2/preparation/`. No trained checkpoint or model score is created there.

## After approval / next decision

Launch this one run; confirm active logging, an initialized recovery checkpoint and the first accepted GPU update; then stop interacting. No epoch polling, automatic next model, outer scoring or ten-fold queue. User returns after completion for closeout.

Judge the actual inner-validation curve, class behaviour and overfitting first. A useful B3 could later join a **fresh same-fold ConvNeXt** equal-probability ensemble after that comparator has its own approval. Prior ConvNeXt probabilities cannot be reused. No ensemble or full K10 launch is authorized now.
