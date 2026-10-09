# S83 fixed macro-F1 CBAM addition closeout

One CPU-only comparison uses the already-saved standalone macro-F1-selected S79 checkpoint35 rather than the accuracy-selected33 used by S82. All five S53 sources and equal1/6 fusion are unchanged. These are the two recorded standalone selection rules, not a sweep over training epochs. No training, new inference or test access.

| Method | Exploratory accuracy | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|
| S53 retained equal-five | 93.6128% | 0.886857 | 82.04% |
| S82 added accuracy-selected CBAM | 93.9454% | 0.893190 | 80.84% |
| S83 added macro-F1-selected CBAM | 94.0120% | 0.895201 | 82.04% |

Gained12, lost6, net+6 correct versus S53. Material gate passed:False; decision:reject_no_material_gain. All metrics/predictions/probabilities, class/confusion/comparison PNG/PDF, changed identities, source hashes and verification are saved in `results/short_screening/s83_cbam_f1_addition_cpu/`.

Repeated-validation selection and post-test development remain limitations. S80/S81 and their test/XAI results stay unchanged; neither new six-model method has been evaluated on a new independent test. No further checkpoint/weight search or GPU run is queued.
