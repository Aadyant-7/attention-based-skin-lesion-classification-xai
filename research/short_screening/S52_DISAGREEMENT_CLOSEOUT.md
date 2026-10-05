# S52: ResNet rescue on disagreement

One predeclared CPU-only comparison used preserved validation probabilities. When fewer than three of the four S46 members agreed, probabilities were averaged equally with S19 ResNet101; otherwise S46 was retained. No threshold or weight search was performed.

| Method | Exploratory validation accuracy | Macro-F1 |
|---|---:|---:|
| S46 equal four-model reference | 93.4797% | 0.883680 |
| S52 disagreement-only ResNet rescue | 92.9474% | 0.876792 |

The rule routed 90 of 1,503 images, gained 7 correct predictions and lost 15 (net -8). Melanoma recall was 0.796407. The adoption gate failed: keep S46, whose predictions were independently reproduced by S50 FP32 identity inference. Do not expand this into a threshold or weighting search.

Artifacts: `results/short_screening/s52_disagreement_resnet_rescue/` contains the predeclared plan, source hashes, routing decisions, summary, metrics, predictions, probabilities, class scores, raw/normalized confusion matrices, and PNG/PDF result/comparison figures. The master registry includes S52.

No GPU inference/training or locked-test access occurred. These are post-test exploratory development results, not a new held-out test estimate. Original test accuracy remains 86.7598%.
