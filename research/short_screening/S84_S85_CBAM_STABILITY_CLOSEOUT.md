# S84 S85 fixed CBAM stability closeout

Exactly two CPU-only interventions were recorded before fusion/scoring. Source checkpoints and input probabilities are unchanged. No training, new inference or test labels/images.

| Method | Accuracy | Macro-F1 | Melanoma recall | Net correct vs S83 |
|---|---:|---:|---:|---:|
| S83 equal-six reference | 94.0120% | 0.895201 | 82.04% | 0 |
| cbam_only_fixed_temperature | 93.8789% | 0.893894 | 82.04% | -2 |
| cbam_two_winner_probability_bag | 93.9454% | 0.894622 | 81.44% | -1 |

S84 uses one fixed temperature2 on the CBAM branch only; this does not change its standalone argmax and is not a fitted calibration. S85 averages the saved standalone accuracy33 and macro-F1 35 probabilities within one CBAM slot; it does not increase the CBAM branch weight or average model parameters. Other five sources stay1/6.

The diagnostic found68/90 S83 errors had at least one correct component,22had no correct component argmax, and CBAM wrong-prediction confidence averaged0.9241. These label-based facts motivate checks, not deployable oracle selection or proof that all these cases can be recovered.

Material gates remain >=0.005 accuracy and 8 net correct with macro-F1/melanoma recall nondecrease; summaries separately show gates against S83 and S53. Both candidates failed. Neither recovered a previously incorrect S83 prediction: S84 lost two correct predictions; S85 lost one. Their small NLL/ECE reductions did not improve classification. **Reject both; retain S83 as the numerical development leader at 94.0120% /0.895201.** S53 remains the conservative policy reference; the gate was not lowered.

The original five components use hash-verified fresh S76 FP32 probability caches. The earlier model CSVs in the source manifest identify the original runs; they are not substituted for the FP32 cache inputs. `input_provenance.json` records the exact caches and checkpoint signatures. S79 CBAM uses its saved FP32 validation probability CSVs. Fusion arithmetic uses float64 to preserve numerical stability; no new model inference occurred.

Reused validation and known earlier test outcomes limit independent performance claims. Both full prediction/probability/class/confusion/comparison PNG/PDF packages and changed identities are saved under `results/short_screening/s84_s85_cbam_stability_cpu/`. No further parameter search or GPU run is authorized by this script. The next meaningful direction requires train-only evidence of a useful specialist or stronger representation, taking account of previous rejected rescue/stacking/feature studies. No new run is prepared or launched.
