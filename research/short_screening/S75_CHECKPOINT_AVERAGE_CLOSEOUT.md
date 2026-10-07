# S75: fixed ConvNeXt-Tiny checkpoint average — rejected

Completed 7 October 2026. Two full CPU FP32 identity inference passes took 214.77 seconds including reporting; **no CNN training, GPU or original-test inference** occurred.

## Question and fixed method

Test one equal parameter average of all three available distinct late S06 ConvNeXt-Tiny states: accuracy winner epoch17, macro-F1 winner epoch18 and final epoch20. All configuration/class-order/state-shape checks passed; hashes identify three unique model states. Average floating tensors with FP64 accumulation, then store FP32. ConvNeXt has no BatchNorm layers, so there are no averaged running statistics to recalibrate. Original files are preserved.

This is **post-hoc same-run checkpoint averaging**, motivated by [Izmailov et al., Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407). It does not reproduce that paper's constant/cyclical-SGD SWA training: S06 used AdamW and ReduceLROnPlateau, with validation-selected snapshots. The paper supplies a hypothesis, not a guaranteed HAM10000 gain.

Fresh CPU FP32 epoch18 predictions provide the matched control. For fusion, change only the Tiny member of S53, keep the four other original cached member probabilities, and keep all five weights at0.20. The 1,503-image exploratory validation cohort/preprocessing/class order stay fixed. No checkpoint-subset, weight, view or resolution search.

| Exploratory validation method | Accuracy | Macro precision | Macro recall | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|---:|---:|
| Original Tiny epoch18 CPU control | 91.4837% | 0.873533 | 0.866726 | 0.868746 | 82.6347% |
| Averaged Tiny epochs17/18/20 | 91.4172% | 0.878928 | 0.863964 | 0.870132 | 81.4371% |
| Equal-five matched CPU-Tiny control | 93.6128% | 0.915781 | 0.865636 | 0.886857 | 82.0359% |
| Equal-five averaged-Tiny candidate | 93.6793% | 0.917152 | 0.865064 | 0.887186 | 81.4371% |

## Decision

Fusion gains two predictions and loses one: **net+1 / +0.0665 percentage points**. Melanoma recall loses one correct melanoma (137→136 of167). akiec recall stays35/49=71.4286%. Standalone averaging improves akiec recall34→36/49 and macro-F1 slightly, but loses one overall correct prediction and two correct melanomas. It is not a material accuracy improvement.

The predeclared ensemble gate required at least eight net gains/+0.5 percentage points accuracy, nondecreasing macro-F1 and melanoma recall versus both matched control and S53, and +0.5 percentage points versus S53. **Failed. Retain S53**; no subset/weight rescue or GPU training follows. This rejects this single fixed average, not all SWA or regularization methods.

The original CPU Tiny matches S06's recorded epoch18 accuracy/F1/confusion matrix. Its resulting five-member ensemble matches S53's accuracy/F1/confusion matrix. This controls the comparison's Tiny precision change; it is not fresh uniform-FP32 inference of all five members. S06's accuracy-selected epoch17 remains91.7498% /0.862831, a different checkpoint from the S53 epoch18 member.

## Artifacts and scope

- `results/short_screening/s75_convnext_checkpoint_average_cpu/`: fixed plan, compatibility/finite-forward preflight, checkpoint/tensor hashes, runtime/library/code provenance, log, four result packages (complete probabilities/predictions, class metrics, raw/normalized confusion matrices, PNG/PDF figures), gain/loss identities, comparison figures and verification.
- `checkpoints/short_screening/s75_convnext_checkpoint_average_cpu/averaged.pt`: new local Git-ignored checkpoint; source checkpoints untouched.
- `results/master_experiment_registry.csv`: four S75 rows explicitly recording CPU inference, original controls and candidate variants.
- Runner: `research/short_screening/checkpoint_average_cpu.py`; completed reruns report the saved summary without inference. Full CPU budget10minutes; no automatic GPU fallback.

All source/checkpoint hashes remain unchanged, all 1,503 prediction identities/probabilities and saved confusion-derived metrics verify, and original-test image/lesion overlap is zero. No original-test labels/images were loaded. Reused exploratory validation and prior snapshot selection prevent interpreting this tiny gain as independent generalization evidence. The retained single-image exploratory result remains**93.6128% /0.886857**. The immutable original strict-trained test result remains**86.7598% /0.794473**.

## Next decision

S74 and S75 supply no evidence for promoting these recipes to GPU training. Preserve the retained model and stop checkpoint/weight variants. A further costly run needs a distinct training/data hypothesis and credible evidence; these negative CPU checks do not justify another backbone automatically or any new test evaluation.
