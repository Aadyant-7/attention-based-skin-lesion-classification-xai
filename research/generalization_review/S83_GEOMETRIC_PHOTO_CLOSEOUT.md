# S83 geometric photo-pooling development screen

12 October 2026. One fixed CPU-only follow-up tested geometric averaging across
all available validation photographs of each lesion. **No improvement was found.**
The candidate was rejected before any old-test scoring. Frozen S83 and the
earlier retrospective multi-image diagnostic are unchanged.

## Question and fixed rule

Arithmetic averaging can let a strongly supported class in one photograph offset
weak support in another. The geometric mean instead rewards agreement across
photographs. This is an unproven pooling ablation, not an established upgrade.
Prior geometric fusion across models was unhelpful; photo pooling is a distinct
operation and was screened once, with no additional alternatives or grids.

For each lesion, compute `normalize(exp(mean(log(max(p, 1e-12)))))` over every
available validation photograph's saved S83 probability vector. Preserve singleton
vectors exactly. No input contains zero probabilities, so the numerical floor
does not change the source values. Grouping uses lesion ID only; labels enter
after predictions exist, solely for consistency checks and scoring.

The existing validation cohort contains 1,503 images representing 1,429 lesions:
1,359 singletons, 66 two-photo lesions and 4 three-photo lesions. Only 70 lesions
(144 photographs) can change through pooling. All observations stay included;
no companions from training or another partition are used.

## Matched development results

| Endpoint | Rule | Correct / total | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Image | Original S83 single image |1,413 / 1,503|94.011976%|0.895201|
| Image | Arithmetic photo mean |1,412 / 1,503|93.945442%|0.895312|
| Image | Geometric photo mean |1,412 / 1,503|93.945442%|0.895312|
| Lesion | First-ID single photo |1,344 / 1,429|94.051784%|0.892304|
| Lesion | Arithmetic photo mean |1,345 / 1,429|94.121763%|0.893674|
| Lesion | Geometric photo mean |1,345 / 1,429|94.121763%|0.893674|

Geometric and arithmetic pooling produced identical predicted classes at both
endpoints: zero corrected or newly incorrect observations. Minority-class recall
and the complete confusion matrices are consequently identical.

The fixed continuation gate required at least 0.5 percentage-point accuracy gain,
nondecreasing macro-F1 and melanoma/akiec recall on both endpoints, and at least
eight net image gains. **The accuracy and net-gain requirements failed.** No
old-test prediction file was opened or scored for this candidate.

## Interpretation and preserved final status

The scores above use two different denominators. The 94.12% lesion-level number
must not replace the 94.01% single-image validation score or become a test claim.
Validation is extensively reused and has natural train/validation lesion overlap;
this is exploratory development evidence, not independent confirmation.

Retain the prior recorded results:

- Frozen S83 single-image exploratory validation: **94.011976% / 0.895201**.
- Recorded S83 single-image test audit: **87.558217% / 0.807031**.
- Supplementary retrospective multi-image lesion assessment: **90.071556% /
  0.809079 on 1,118 previously assessed lesions**. Its earlier material-gain gate
  failed; it remains supplementary and does not replace the frozen method.

No checkpoints, preprocessing, ensemble weights or previous results were changed.
No training, model inference, image decoding or GPU use occurred. Repeatedly
trying aggregation rules on known test labels would not establish fresh held-out
generalization. This screen schedules no further pooling search or training.

## Saved package

[Results](../../results/diagnostics/s83_geometric_photo_screen_2026-10-12/)
contain the frozen plan, full comparison, arithmetic/geometric predictions and
probabilities for both units, precision/recall/F1/support, raw/normalized matrices,
PNG/PDF comparison, verification and hashes.

[Code](s83_multiimage_geometric_screen.py) has a completed-run guard.
[Earlier retrospective closeout](S83_MULTIIMAGE_CLOSEOUT.md) remains separate.
