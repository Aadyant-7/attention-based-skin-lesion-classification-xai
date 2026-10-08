# S76: use the trained visual representations, not another voting variation

Prepared 7 October 2026 in response to the request to stop the sequence of negligible changes. One bounded study; no new backbone training or original-test evaluation.

## What the audit actually found

The strong models can fit the training data: final recorded training accuracy is99.71% for S06 Tiny,99.61% for S18 Small,99.79% for S15 DenseNet and99.03% for S10 V2-S. Their best exploratory validation accuracies are91.75%,92.22%,89.62% and89.49%. These gaps support a generalization problem; they do not identify one proven causal defect or make more epochs an evidence-based fix.

Recent S72-S74 screens trained heads on **generic ImageNet-only frozen features**, around79% group-OOF accuracy. Their representation and evaluation task differ from our fine-tuned93.61% ensemble. A negative probe does not establish that an end-to-end training intervention would fail; a positive probe also does not promise benefit (S58 passed but S59 did not improve). Treat them as narrow diagnostics. Stop using that proxy as a universal gate for the real model.

The current S53 ensemble exports only35 final probabilities (five models x seven classes). S42 learned log-probability fusion and S49 nonlinear probability fusion used those compressed decisions. They did not jointly classify the fine-tuned visual feature vectors. S62 established69/96 errors with at least one correct member and27 shared misses. This shows useful variation exists, but does not prove a learned feature classifier can exploit it.

No data corruption, class-order mismatch or universally faulty preprocessing was established. S62 found89/96 errors with histology-derived labels. An automatic label-cleaning/relabeling campaign is not supported by that evidence. No source images, labels or historical results are altered here.

## One substantive candidate

Use exactly the S53 checkpoints: ConvNeXt-Tiny, ConvNeXt-Small, DenseNet201 and EfficientNetV2-S macro-F1 winners, plus EfficientNet-B0 accuracy winner. Extract their actual pre-classifier representations (768+768+1920+1280+1280=6016 dimensions) for the7009 training and1503 exploratory validation images. Preserve pooling, native normalization,224-square preprocessing and ImageNet normalization. Identity only, eval mode, FP32, TF32 disabled. Source hashes and finite forward equivalence are verified first.

Fit a StandardScaler for each block on training images only; normalize each transformed block to unit L2 norm; concatenate with equal block scale1/sqrt5. Fit one nonlinear SVM on training labels: C10, RBF gamma1, normalized sqrt inverse-frequency class weights, seed42. Kernel precomputation avoids repeatedly calculating6016-dimensional distances inside the solver. Use argmax of the classifier's probability output as the fixed prediction rule. Internal training calibration is not independent CNN cross-validation; there is no validation-label fitting or C/gamma/head search.

The classifier is a new learned feature-fusion method with a frozen backbone ensemble. It is **not a new CNN architecture, a full end-to-end fusion network, or another average of old predictions**. Its hypothesis is that useful visual distinctions survive in the representations even when the seven-class heads disagree or lose them. It may fail; the claim depends on the observed result.

## Controls, budget and decision

- Fresh equal-five FP32 probability fusion from the same extraction passes controls numerical precision and exact checkpoint/preprocessing identity. Compare also against saved S53 (93.6128% /0.886857).
- Require at least+0.5 percentage points/eight net correct predictions versus both controls, with macro-F1 and melanoma recall nondecreasing. This is an advancement heuristic, not a significance claim after repeated validation selection.
- Exactly one fitted classifier and one control. No member subset, C/gamma grid, new probability mixture or post-result rescue.
- Expected roughly5-10minutes;10-minute extraction limit and15-minute overall process watchdog. GPU performs frozen inference only; CPU fits the classifier. The user's standing permission allows bounded short runs to finish and be analyzed automatically. Long training still needs a concrete proposal and approval.
- Save complete prediction/probability CSVs, class scores, raw/normalized confusion matrices, PNG/PDF figures, gain/loss identities, source/cache hashes, classifier bundle and registry records. An SVM has no epoch learning curve; save convergence/support-vector/fit metadata instead.
- Existing checkpoints/results and original held-out test report remain preserved. All results are reused exploratory development evidence, not a93% test claim.

## Literature and limits

[MedFusionNet](https://www.nature.com/articles/s41598-025-31816-2) studies attention-based ConvNeXt/ViT feature fusion. [Uddin et al.](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1478688/full) distinguish learned feature fusion from decision fusion; their HAM10000 table actually reports92.7% feature fusion and96.1% decision fusion, so feature fusion is not universally superior. Those architectures, augmentation and evaluation protocols differ. These papers support the method category, not our expected score. S76's frozen6016D kernel classifier is an engineering adaptation, not reproduction of their systems. [scikit-learn's SVM documentation](https://scikit-learn.org/stable/modules/svm.html) documents precomputed kernels and training-based probability estimation.

## Run, monitoring and recovery

From the project root:

```powershell
.\.venv\Scripts\python.exe -B -m research.short_screening.feature_fusion --prepare
.\.venv\Scripts\python.exe -B -u -m research.short_screening.feature_fusion --run
Get-Content '.\results\short_screening\s76_trained_feature_fusion\run.log' -Tail 15 -Wait
```

- Results: `results/short_screening/s76_trained_feature_fusion/`.
- Feature caches: `.cache/s76_trained_feature_fusion/` (local, Git-ignored).
- Fitted model: `checkpoints/short_screening/s76_trained_feature_fusion/feature_fusion.joblib` (local, Git-ignored). Bundle includes scalers, normalized training features, SVM, source hashes and class order; future inference also needs the five original checkpoints.
- Completed studies are no-ops. An interrupted extraction can reuse complete verified cache files with the same command. If the fitted classifier already exists without a complete closeout, preserve it and recover reporting explicitly; do not silently overwrite/refit it.

This plan supersedes the proposal to continue small checkpoint/voting changes. Outcome and subsequent decision will be recorded in `S76_FEATURE_FUSION_CLOSEOUT.md`; preparation alone does not establish an accuracy improvement.
