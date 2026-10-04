# Authorized stopping amendment — 2026-10-04

The user requested convergence stopping while the queue was running. The training recipe, split, loss, augmentation, optimizer, LR schedule, seed and raw checkpoint rankings remain unchanged.

- Maximum 50 epochs; minimum 25 completed epochs.
- Reset patience only for accuracy at least +0.002 or macro-F1 at least +0.003 above that metric's last meaningful reference. Smaller cumulative gains can eventually cross this reference; isolated tiny gains do not reset it.
- Count epochs since the most recent meaningful improvement in either metric, including the minimum training window. Stop after 12 stale epochs, once the minimum window and LR opportunity requirements are satisfied.
- After epoch 30, stop with 10 stale epochs if both metrics' recent 10-epoch slopes and final-versus-first three-epoch means indicate no threshold-sized upward trend.
- Require at least one scheduled LR reduction and three completed epochs since the most recent reduction before plateau/patience stopping. StepLR continues unchanged.
- No absolute accuracy target controls stopping. Numerical best accuracy and macro-F1 checkpoints continue to update independently for any true improvement.
- Save meaningful references, individual/combined stale counters, stopping reason, LR history and original checkpoints/history. Recovery accepts only the explicitly recorded old/new source fingerprints.

B3's Python process had already loaded the old function. To avoid unsafe runtime injection, the amended master reattaches to that worker, checks committed history, and when the new rule triggers hands it off from the exact committed checkpoint to the amended worker for closeout. No completed epoch is repeated; original checkpoint copies and a migration receipt are retained. Any subsequently started partial epoch is discarded and disclosed. If the original worker finishes naturally first, its artifacts are retained and the new counters are recorded as a retrospective audit. Remaining models use the amendment from their first epoch.

This is still post-test development validation, not a new untouched-test result. The original held-out result remains unchanged.
