# Automatic compute budget policy

The user authorized autonomous short screening on 2026-10-04. This supersedes the speculative 50-epoch queue, now disabled by `budget_control.json`. DenseNet was budget-stopped with all committed checkpoints preserved; ResNet101 was cancelled before launch.

Future sequence: saved-probability CPU analysis → only a justified short pilot → bounded continuation. Keep the proven preprocessing and training recipe as the control, changing one justified factor at a time. Compare only matching protocols and checkpoints at the same stage. Do not automatically carry experimental changes into the locked-test model.

Default pilot: 5 epochs; one continuation to 10 only for useful learning trend, then a hard ceiling of 15 screening epochs. Early pilot results are screening evidence, not proof that an architecture is intrinsically poor. An underperforming pilot is archived as incomplete. Full runs require convincing matched-stage evidence and predicted ensemble value; default full ceiling is 20, not 50. Every launch states a finite epoch/time budget in configuration. Patience cannot extend a hard cap.

Patience resets require both a new raw record and a qualifying gain: accuracy at least +.002 or macro-F1 at least +.003 above the meaningful reference. A .001 isolated accuracy bump does not qualify; macro-F1 or cumulative improvement can independently qualify and must be logged. Keep raw-best checkpoints separate from this decision. No broad architecture or weighting searches. Do not change or repeatedly evaluate the revealed original locked test.

Current next action is the CPU screen and recipe audit, not another automatic GPU launch. If there is no supported improvement, keep the proven reference and finish reporting rather than consuming the remaining budget.
