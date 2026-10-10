# Final S83 six-model test audit

10 October 2026. User authorized this final fixed-method audit. The original cohort was already evaluated in earlier S31/S81 stages. This is a **repeated post-development test audit**, not another first untouched independent test. The original first-test report/artifacts and earlier audit remain unchanged. Test results are not used to change any model, checkpoint, preprocessing or ensemble rule afterward.

## Frozen method

S06 Tiny F1 epoch 18 + S18 Small F1 epoch 17 + S15 DenseNet201 F1 epoch 14 + S10 EfficientNetV2-S F1 epoch 15 + S03 B0 accuracy epoch 20 + S79 Tiny-CBAM F1 epoch 35. Equal 1/6 probability averaging; FP32 identity, 224x224 RGB bilinear square/ImageNet normalization; no TF32, AMP, TTA, multi-resolution, calibration, metadata or inference augmentation. Six fresh passes with labels absent from the inference loader. Standalone scores use those same saved probabilities, not additional inference or checkpoint selection.

Exact hashes, epochs, class order, source/config freeze, 1,503-image identities and zero training/validation-to-test image/lesion overlap were verified before inference in `results/final_exploratory_test_audit/v1/pre_inference_verification.json`. The development split itself is image-level and can share lesions between training/validation.

## Final result

- Accuracy: **87.5582%**; macro-F1: **0.807031**.
- Macro precision: 0.814039; macro recall: 0.805564; weighted-F1: 0.872135.
- Correct: 1,316; incorrect: 187; cohort: 1,503.
- Melanoma recall: 58.33%; akiec recall: 71.43%.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| akiec | 0.795455 | 0.714286 | 0.752688 | 49 |
| bcc | 0.772727 | 0.871795 | 0.819277 | 78 |
| bkl | 0.782051 | 0.739394 | 0.760125 | 165 |
| df | 0.700000 | 0.823529 | 0.756757 | 17 |
| mel | 0.725926 | 0.583333 | 0.646865 | 168 |
| nv | 0.922115 | 0.954229 | 0.937897 | 1005 |
| vasc | 1.000000 | 0.952381 | 0.975610 | 21 |


Rows=true, columns=predicted; class order ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']:

```text
[[ 35   5   5   1   1   2   0]
 [  2  68   3   1   0   4   0]
 [  3   4 122   1  14  21   0]
 [  2   0   0  14   1   0   0]
 [  2   2  13   0  98  53   0]
 [  0   9  13   3  21 959   0]
 [  0   0   0   0   0   1  20]]
```

## Constituent descriptions from the same passes

| Frozen component | Accuracy | Macro-F1 |
|---|---:|---:|
| convnext_tiny | 85.2295% | 0.771714 |
| convnext_small | 84.8303% | 0.748929 |
| densenet201 | 83.8323% | 0.728689 |
| efficientnet_v2_s | 83.5662% | 0.756865 |
| efficientnet_b0 | 81.1710% | 0.686064 |
| convnext_tiny + CBAM | 85.9614% | 0.786460 |
| S83 equal-six + CBAM | 87.5582% | 0.807031 |


## Validation versus audit

S83 exploratory validation: 94.0120% / 0.895201; final test audit: 87.5582% / 0.807031. Audit minus validation: -6.4538 accuracy percentage points / -0.088170 macro-F1. This is a descriptive generalization gap across different cohorts. Highest repeatedly selected image-level validation performance is not an independent test estimate and must not be advertised as test accuracy.

Full metrics, complete predictions/probabilities, seven-class scores, raw/normalized confusion matrices and PNG/PDF figures: `results/final_exploratory_test_audit/v1/ensemble/`. Member packages, comparison tables/figures, durable inference receipts and source verification are in the parent folder. No checkpoints were overwritten. Performance experiments are finished; next work is exact-method Grad-CAM/XAI and final report/paper figures and writing.
