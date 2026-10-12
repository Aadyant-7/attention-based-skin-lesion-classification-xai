# S83 retrospective multi-image lesion assessment

12 October 2026. One fixed CPU-only diagnostic combined the already-saved S83
probabilities for different held-out photographs of the same lesion. **Accuracy
was 90.071556% across all 1,118 lesions** (1,007 correct / 111 incorrect).
This is a changed input contract and evaluation unit, not a replacement for the
original 87.558217% single-image test audit on 1,503 images.

## Fixed rule and scope

For each lesion, equally average every available held-out image's S83 probability
vector, then apply argmax in the unchanged seven-class order. Lesion IDs alone
determine grouping. No labels, confidence, correctness or appearance select
views. All lesions and images remain included; singletons are unchanged. No
training/validation companions, alternative pooling rules or weight search.

The primary endpoint was predeclared as one prediction per lesion on **all 1,118
lesions**, paired against the lexicographically first image ID of each lesion.
The secondary endpoint broadcasts the pooled prediction to **all 1,503 original
images**, paired against their original single-image predictions. This control
separates averaging benefit from changes in denominator/class prevalence.

The cohort was already evaluated, and its earlier results were known. Label
consistency checks and scoring use its saved ground truth. Thus this is a
**retrospective multi-image lesion-level assessment of saved S83 predictions**,
not a new untouched test, original frozen single-image performance or external
clinical validation. No checkpoint changed, GPU ran, image was decoded, model
was inferred, or new training occurred.

## Full results — retain both units

| Unit / rule | Observations | Correct | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original S83 single-image audit |1,503 images|1,316|87.558217%|0.814039|0.805564|0.807031|
| First-ID single view per lesion |1,118 lesions|1,003|89.713775%|0.809881|0.791473|0.797281|
| Equal mean of available images per lesion |1,118 lesions|1,007|**90.071556%**|0.822710|0.801006|0.809079|
| Pooled predictions broadcast to originals |1,503 images|1,321|87.890885%|0.831398|0.809715|0.818339|

Matched lesion-level gain: **+0.357782 percentage points**, 9 corrected cases
and 5 newly incorrect cases, net +4. Matched image-level gain: **+0.332668
percentage points**, 22 corrected images and 17 newly incorrect images, net +5.
The jump from 87.56% to 90.07% combines a unit/input change with a small pooling
gain; do not describe it as a 2.51-point improvement in single-image accuracy.

## Minority classes and decision

| Endpoint | Melanoma recall | akiec recall |
|---|---:|---:|
| First-ID single-view lesions |55.4348% (51/92)|64.7059% (22/34)|
| Multi-image lesions |54.3478% (50/92)|67.6471% (23/34)|
| Original single images |58.3333% (98/168)|71.4286% (35/49)|
| Broadcast multi-image predictions |55.9524% (94/168)|71.4286% (35/49)|

**The predeclared material-improvement gate failed.** Accuracy gains were below
0.5 percentage points, image net gains below 8, and melanoma recall declined in
both matched comparisons. Macro-F1 improved, but this does not establish a
stronger overall or clinically reliable classifier. Preserve S83's frozen
single-image product and report this only as an additional diagnostic/ablation.

## Cohort and saved artifacts

View counts: 823 singletons, 216 lesions with 2 images, 74 with 3, 2 with 4 and 3
with 6. The 295 multi-image lesions account for 680 of the 1,503 photographs.
Lesion labels were consistent across all groups. No original held-out lesion
overlaps S83 training/validation; grouping uses only photographs in this cohort.

[Result package](../../results/diagnostics/s83_multiimage_lesion_review_2026-10-12/):

- `plan.json`, `summary.json`, `comparison.csv`, verification and inventory.
- `canonical_single_lesion/`, `multiimage_lesion/`, `broadcast_image_endpoint/`:
  complete predictions/probabilities, seven-class precision/recall/F1/support,
  metrics, raw/normalized confusion matrices and publication PNG/PDF figures.
- `comparison_figures/`: separate paired image-unit and lesion-unit comparisons.

Original S83 results/checkpoints and XAI are unchanged. Existing single-image
XAI must not be relabelled as explanations of a pooled lesion prediction.

## Accurate report wording

“In a retrospective multi-image assessment, equal averaging of the S83 ensemble
probabilities across all available held-out photographs achieved **90.07%
lesion-level accuracy and 0.8091 macro-F1 on 1,118 lesions**. The matched first-view
baseline achieved 89.71%; the original single-image audit remained 87.56% on 1,503
images. The small gain and decreased melanoma recall did not meet the study's
material-improvement gate.”

No claim of unused test data or 90% single-image test accuracy is supported.
No further pooling search, training or new inference is scheduled by this study.
