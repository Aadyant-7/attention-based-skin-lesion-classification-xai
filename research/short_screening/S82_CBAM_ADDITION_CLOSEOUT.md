# S82 fixed CBAM addition closeout

9October2026. Report writing was paused at the user's request to resume accuracy development. One CPU-only experiment retained the full S53 ensemble and added S79 epoch33 rather than replacing Tiny. Equal1/6 probabilities, unchanged saved FP32 identity predictions, same1503 exploratory validation images. No training, new inference, test access or search.

| Method | Accuracy | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|
| S53 retained equal-five | 93.6128% | 0.886857 | 82.04% |
| S82 equal-six CBAM addition | 93.9454% | 0.893190 | 80.84% |

Gain13, loss8, net+5 correct predictions. CBAM alone fixes39 reference errors; the reference fixes48 CBAM errors. The actual probability fusion, not an oracle union of correct answers, determines the result. Class-wise complementarity and changed identities are saved.

Material gate:>=8netcorrect and>=0.005absolute accuracy, macro-F1/melanoma recall nondecreasing. Passed:False. Decision:reject_no_material_gain. No test result is changed or forecast by this experiment. S80/S81 remain frozen historical evidence. Validation has been reused repeatedly and original test outcomes are already known; this is additional exploratory development, not independent confirmation.

Artifacts:`results/short_screening/s82_cbam_addition_cpu/`: plan/hashes, metrics, predictions/probabilities, class metrics, raw/normalized confusion and comparison PNG/PDF, complementarity and verification. Source checkpoints are unchanged.
