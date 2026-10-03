# Historical research index

All pre-3-October experiments remain at their original locations. **No historical result, checkpoint, script, split, report or weight file was moved or deleted.** `artifact_manifest.csv` indexes them with SHA-256, size and original Git tracking state. Existing duplicate-content files remain intact; see `results/audit/duplicate_content.json`.

Start with `results/master_experiment_registry.csv`; filter `era=legacy` and the desired `protocol`. The original `results/experiments.csv` is preserved as its original three-run ledger. Historical subfolders (`phase4`, `phase5*`, `accuracy_exploration`, `exploratory`, `strict_multires_probe`, `validation_inference`, local `runs`) continue to work.

Generated visual evidence is under `results/figures/legacy/`. Source hashes and protocol labels accompany figures. New experiments belong only under `results/structured_experiments/`, with local weights under `checkpoints/structured/`.

Historical reports in `reports/` describe the earlier architecture constraint. The guide's 3-October instruction supersedes that constraint: architecture, attention and ensemble choices are now selected from controlled evidence. Those reports remain historical documentation; use `research/README.md` and `NEXT_STEPS.md` for current direction.
