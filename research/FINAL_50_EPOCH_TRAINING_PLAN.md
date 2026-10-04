# FINAL MAXIMUM-50-EPOCH TRAINING PLAN (NOT authorized/launched)

| model | epochs | accuracy_peak_epoch | f1_peak_epoch | best_accuracy | final_accuracy |
| --- | --- | --- | --- | --- | --- |
| mobilenet_v3_large | 20 | 16 | 16 | 0.8602794411177644 | 0.8516300731869594 |
| efficientnet_b0 | 20 | 20 | 20 | 0.863606121091151 | 0.863606121091151 |
| efficientnet_b0 | 20 | 19 | 16 | 0.863606121091151 | 0.8569527611443779 |
| efficientnet_v2_s | 20 | 15 | 15 | 0.8948769128409847 | 0.8895542248835662 |
| densenet201 | 17 | 12 | 14 | 0.8962075848303394 | 0.8948769128409847 |
| convnext_tiny | 20 | 17 | 18 | 0.9174983366600132 | 0.9075182967398536 |
| convnext_small | 20 | 17 | 17 | 0.9221556886227544 | 0.9128409846972722 |
| resnet101 | 20 | 19 | 17 | 0.8809048569527611 | 0.8669328010645376 |

Most current models peaked before the cap and often declined;20epochs was not a proven binding limit.50is an allowance for a new strict training phase, not a promised improvement or requirement to chooseepoch50.

1. Fresh strict train/validation from ImageNet; never resume exploratory checkpoints.
2. Frozen architectures and weights; train modelmembers independently with cap50 and earlystopping patience12 after minimum15. Record all member predictions/states each epoch so a coherent fixed-ensemble checkpoint can be selected on strict validation without weight tuning.
3. Save bestaccuracy, independentbestmacroF1, latest/fulloptimizer/scheduler/scaler/RNG and compositeepoch/memberidentities. Epoch23canremainselected if later performance deteriorates. Store every serious run curves, class metrics/matrices, runtime and predictions.
4. FP32validation, finite guards, explicit numerical amendments only; safe atomic resumes.
5. Freeze preprocessing/weights/calibration/selection and all methodology beforeONElockedtest evaluation. Exploratory and strict results must remain separate.
6. Final Grad-CAM/architecture-specific CAM and paper artifacts afterselection. No test loading, strict training or long run executed by current batch.
