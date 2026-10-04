# Final autonomous architecture selection

Approved4October2026; two candidates only: S16EfficientNet-B3 (ImageNet1K_V1), S19ResNet101 (ImageNet1K_V2). Sequential because resource tests found limitedhostRAM despite GPUfit. Full fine-tune, same seed42/cohort/recipe, max20/patience5; no epoch8gate. No third architecture, TTA or resolution run.

Start/recover from project root:
```powershell
.\.venv\Scripts\python.exe -u -m research.run_final_architecture_selection
```
CPU check:
```powershell
.\.venv\Scripts\python.exe -m research.run_final_architecture_selection --check
```
The master survives independently after launch. OS lock prevents duplicate masters; per-model OS locks prevent concurrent resumes. Live orphaned workers are reattached. Completed candidates verified/skipped. Valid config/source/runtime-matching checkpoints resume; only one clearly recoverable interruption retry. Nonfinite/OOM/config failures are preserved, marked failed and bypassed; never restart failed models from scratch. Stale registry lock recovered only if no project training writer exists.

Each child writes its standard artifact package. After both candidates terminate, automatic CPU analysis produces model_vs_accuracy.csv, class/error complementarity, <=6equal/2weighted fusions, metrics/predictions/matrices/figures and final architecture_freeze.json. Freeze fallback is the FP32S12reference when no material balanced gain qualifies. Immutable historical file hashes and every pre-batch registry row are checked before/after. New registry rows append only.

Master outputs: results/architecture_selection/final_v1/status.json and master.log.
Training results: results/structured_experiments/<runID>/; checkpoints: checkpoints/structured/<runID>/.
Final comparison: results/model_comparison/final_architecture_selection/.
Automatic documents: research/MODEL_VS_ACCURACY.md, FINAL_ARCHITECTURE_SELECTION.md, FINAL_HIGH_PERFORMANCE_RECIPE.md, FINAL_50_EPOCH_TRAINING_PLAN.md.
Final artifacts publish to structured-research if branch/staged-state/network permit; publication failure preserves local freeze and records publication.json. No checkpoint/rawdata pushed.

Future strict fresh training and one locked-test evaluation are prepared only, never executed by this batch. Architecture exploration is permanently closed after the freeze report; no new backbone proposal follows.
