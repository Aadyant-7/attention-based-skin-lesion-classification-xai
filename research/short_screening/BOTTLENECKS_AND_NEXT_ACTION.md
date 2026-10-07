# Accuracy bottlenecks: evidence and next action

## Latest completed checks - 7 October 2026

S70 same-lesion multi-image exploratory pooling reaches93.7458% but gains only two images and lowers melanoma recall; rejected. S71's separate multi-image strict-validation endpoint reaches91.0845% and gains14images, but requires companion views/known grouping and does not change the original single-image validation/test claim. S72's fixed nonlinear frozen-feature classifier ties its linear control at75.8025% grouped accuracy with worse macro-F1/melanoma recall. S73's matched CPU SupCon feature-head regularizer reaches79.1126% versus79.0840% CE control but fails its meaningful/class-consistency gate. No GPU or original-test run followed.

Closeouts: `S70_S72_CLOSEOUT.md`, `S73_CONTRASTIVE_CLOSEOUT.md`. Full predictions, class scores, confusion matrices, histories where training occurred, PNG/PDF figures, checkpoint/source hashes and registry records are preserved. Retain S53 at93.6128% single-image exploratory validation; the original frozen test stays86.7598%. S70's higher number is a different multi-image protocol, not a new accepted single-image best.

Next distinct cheap hypothesis: normalize each training lesion's total influence instead of counting every view equally; compare with the unchanged CE-only feature head on train-only lesion-group folds before any GPU proposal. Derive any view-count/class weights only from the fitting fold. Do not infer that this will work, rescue rejected SupCon/metadata/voting variants, or train another backbone from these negative results.

## What the saved evidence establishes

1. **Errors are concentrated, not uniformly distributed.** S53 has 96 errors. Melanoma, BKL and akiec account for 68 (70.83%) while representing 381/1,503 validation images (25.35%). Their combined error rate is 17.85%, versus 2.50% for the remaining classes. S53 recalls: melanoma82.04%, BKL85.45%, akiec71.43%; nevus97.91%. This identifies the difficult cases to improve, not a diagnosis of their cause. See S62 class/confusion tables.
2. **Diversity exists but cannot be extracted using the true label at inference.** At least one member gets 69 of the96 errors right; all five miss27, including16 unanimous wrong decisions. An oracle selecting the correct member is not a deployable model. Fixed voting, stacking, added models and calibration have repeatedly traded gained predictions for lost predictions. S63 gains5, loses4, lowers melanoma recall and fails the material gate.
3. **More fitting is not automatically more generalization.** S59 achieved100% training accuracy at epochs17–20 and near-zero training loss, but only90.42–90.69% validation accuracy in those epochs. Extending it merely because epoch20 was the numerical best was not justified. This observation concerns S59 and does not prove every run has the same cause.
4. **No discovered plumbing error explains the plateau.** S62 checked image decoding, IDs, class order, member preprocessing and source predictions. No mismatch was found. Both clear and artifact-affected images were misclassified. Blur, label provenance, hair and square distortion are hypotheses or case-level observations, not established universal causes.
5. **The 93% exploratory figure and86.76% original test figure are from different models/protocols.** The original test used the strict-trained three-member S31 methodology; the subsequent S53 candidate uses five exploratory-trained members. Their difference cannot be attributed entirely to one preprocessing or architecture fault. S53 has no new independent test result. Repeated selection on the exploratory cohort also means a small new validation gain is weak evidence of unseen improvement.

## Next bounded screen: S64

One paired **CPU-only** pretrained-feature study, with the same ImageNet1k ConvNeXt-Tiny weights, FP32 features, complete7,009 training/1,503 validation cohorts, training-only weighted StandardScaler/logistic regression C1 and seed42:

- Control: square224 bilinear resize.
- Candidate: documented pretrained transform, short side236 bilinear resize followed by center crop224; same ImageNet normalization.
- Gate before viewing results: candidate accuracy at least+1 percentage point over its frozen-feature control, no macro-F1 decrease and no melanoma-recall decrease.
- No transform search, head tuning, validation fitting, full-network training, new ensemble or test inference.

This jointly changes geometry and visible field/cropping. It is not a pure aspect-ratio ablation, and cropping can discard useful peripheral context. A frozen-feature improvement would justify a training proposal, not establish improved93.61% ensemble accuracy or predict93% test accuracy. A failed gate does not prove matched fine-tuning can never work; it means this screen supplies insufficient evidence to spend GPU time on it.

Command from project root:

```powershell
.\.venv\Scripts\python.exe -u -m research.short_screening.preprocessing_screen_v1
```

Artifacts: `results/short_screening/s64_paired_preprocessing_cpu/`; features and train-only heads: `.cache/s64_paired_preprocessing_cpu/`. CPU hard wall budget30minutes. Completed summaries make reruns a no-op; verified complete feature caches can be reused after interruption. No existing checkpoints/results are overwritten.

Official transform reference: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html

## Direction after the screen

**Metadata follow-up completed:** Fixed S68 age-only and S69 age+sex+location train-only likelihood adjustments both failed the material gate on the retained S53 ensemble:93.4797% and93.0805%, versus93.6128% image-only. Macro-F1 and melanoma recall declined; net correct changes−2 and−8. Metadata fitting used4,923 training lesions, excluded every validation lesion, and never parsed original-test clinical rows. This rejects these two recipes, not all multimodal modeling. No further metadata/weight/bin search or costly multimodal run follows. Report: `research/data_assisted/S68_S69_METADATA_CLOSEOUT.md`.

**External-data follow-up completed:** S66/S67 tested the fixed addition of2,823 cleared BCN lesions with matched frozen features/classifier. HAM exploratory accuracy fell79.3081%→78.5762%; grouped train-OOF accuracy fell75.8025%→74.4471%, all three folds worsened, and melanoma recall declined. External preflight accuracy improved46.4%→58.8%. This supports rejecting simple pooled-data GPU training and investigating source/case-composition differences; it does not prove all external adaptation would fail. No new ensemble or original-test evaluation was performed. Full evidence: `research/data_assisted/S66_S67_CLOSEOUT.md`. The retained93.6128% ensemble stays unchanged; metadata remains a separate deferred follow-up.

**Completed outcome:** S64 passed on frozen-feature exploratory validation (79.3081% ->81.7698%), but S65 failed the fixed train-only grouped consistency gate (75.8025% ->76.4160%, only one positive fold and slightly lower melanoma recall). No crop-based CNN training is justified by these screens. The next substantive action is the external-data identity/label/duplicate clearance stage in `EXTERNAL_DATA_NEXT_DIRECTION.md`, not an automatic GPU launch. Full closeout: `S64_S65_PREPROCESSING_CLOSEOUT.md`.

If the gate passes, first check consistency on train-only lesion-group folds using these cached features, without changing the classifier or candidate transform. Only consistent gains would support proposing a matched15–20-epoch ConvNeXt run, with a saved control and later ensemble complementarity analysis. No GPU launch is implied by this document.

If the gate fails, do not make another automatic transform/temperature/weight tweak. Use the cached features for a fixed train-only group-fold comparison to quantify whether the negative result is consistent before choosing a different substantial training hypothesis. Do not promise an accuracy increase from this diagnostic.

For a new independent performance claim, the final chosen method requires a genuinely untouched, appropriate cohort. The original locked-test report remains immutable; a repartition of previously used images is not automatically a new unseen test. Changing/excluding difficult test cases or selecting methods from test labels would not establish improved generalization.
