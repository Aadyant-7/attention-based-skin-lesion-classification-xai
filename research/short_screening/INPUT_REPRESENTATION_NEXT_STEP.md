# Input representation screen — 5 October 2026

## Completed CPU evidence

All 7,009 training and 1,503 strict-validation images were measured without test access. Within each diagnosis, low groups use the training 25th percentile, rather than validation-tuned thresholds.

| MEL feature group | Low-group accuracy | Remaining-group accuracy | Counts low/remaining |
|---|---:|---:|---:|
| High-frequency energy (detail proxy) | 45.95% | 65.12% | 37/129 |
| Central dark-region area proxy | 51.22% | 63.93% | 41/122 |
| Contrast | 47.37% | 64.84% | 38/128 |

BKL has no corresponding smaller-dark-region deficit (80.49% vs80.17%). Therefore, the selected montage did not establish a universal small-lesion cause. The intensity component is NOT a verified lesion segmentation; redness, hair, blur and lesion morphology confound it. These descriptive comparisons have no significance claim and do not prove that increased resolution helps.

Data: `results/short_screening/input_representation_audit_v1/`. Code: `research/short_screening/input_representation_audit.py`.

## One bounded S39 diagnostic, not another training queue

Research question: does retaining more image detail improve the preserved ConvNeXt representation sufficiently to justify future resolution-matched training?

- Same immutable S29 epoch33 accuracy checkpoint, with recorded SHA256.
- Full strict DEVELOPMENT validation only; post-test development, not a new held-out estimate.
- Exactly FP32 identity224 control and FP32 identity320; same square bilinear resize and ImageNet normalization. No aspect-ratio change, crop, TTA, weight search, checkpoint selection or training.
- Verify control decisions match the saved224 predictions. Save independent metrics/predictions/probabilities/confusion matrices/PNG/PDF for both resolutions.
- Compare fixed B0(224)+ConvNeXt(224 or320)+V2-S(224), equal thirds, using the already saved unchanged B0/V2-S probabilities. This is one controlled change, not ensemble search.
- Predeclared material gate: ensemble accuracy +>=0.5pp, macro-F1 nondecrease, MEL recall +>=2pp and NV recall decline <=1pp. A failed screen stops this inference route. It does not prove a model trained at320 would fail; no resolution-matched training is automatically authorized by success or failure.
- Runtime estimate2–5minutes; batch8 FP32 on RTX4060, estimated2–4GB VRAM. Hard20minute wall-budget check between batches. No epochs or new training checkpoints; original source is rehashed at completion.
- If a numerical/input/control-consistency check fails, preserve diagnostic evidence and stop, without automatic retry.

Torchvision documents ConvNeXt-Tiny's ImageNet weight preprocessing as short-side236 followed by224 center crop: https://docs.pytorch.org/vision/master/models/generated/torchvision.models.convnext_tiny.html . Our historical square224 pipeline differs. S39 intentionally changes resolution only, so it is NOT an exact reproduction of the pretrained preprocessing and does not combine multiple input changes.

The HAM10000 source describes multiple acquisition sources and several diagnosis-confirmation procedures: https://www.nature.com/articles/sdata2018161 . These differences motivate examining acquisition/label provenance; they do not establish that our labels are wrong. No label edits or exclusions are made.

## Start / monitor / recovery

From project root:

```powershell
.\.venv\Scripts\python.exe -m research.short_screening.resolution_screen --check
.\.venv\Scripts\python.exe -u -m research.short_screening.resolution_screen --run
Get-Content '.\results\short_screening\resolution_screen_v1\run.log' -Tail 15 -Wait
```

GPU worker is launched independently. After the first finite probability batch and source verification are saved, Codex stops interacting. No background monitor or next experiment is scheduled.

Outputs: `results/short_screening/resolution_screen_v1/`; checkpoints: reused original `checkpoints/structured/s29_convnext_tiny_final_strict_seed42/best.pt`, never overwritten. If interrupted, preserve the output folder as an attempt directory before rerunning the same command; incomplete validation passes may be repeated, never test inference. Completed summary makes rerun a no-op. Return for closeout before deciding anything further.
