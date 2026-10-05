# S40 and one bounded S41 addition check

All results below are post-test STRICT DEVELOPMENT validation, not exploratory or new held-out test scores. Original frozen method/test unchanged.

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| S29 through20, accuracy winner epoch19 | 88.6228% | .808912 |
| S40 Mixup through20, accuracy winner epoch19 | 88.4232% | .800289 |
| S40 separate F1 winner epoch10 | 87.9574% | .820358 |
| Frozen full strict three-model reference | 90.1530% | .843486 |
| Replace original ConvNeXt with S40 | 89.1550% | .819898 |
| S41 add S40, fixed four equal models | 90.4857% | .846707 |

S40 ran exactly20 epochs in19.43minutes. Accuracy did not improve over its matched-window historical control. Raw F1 improved at a separate checkpoint; do not pair that F1 with the accuracy winner. Best/latest/F1 checkpoint tensors are finite;1503-image prediction probabilities, IDs, class mapping and recomputed metrics verified. No training extension.

S41 directly answers whether adding rather than replacing S40 realizes complementarity. Mixup fixes39 original ConvNeXt mistakes, while original ConvNeXt fixes48 Mixup mistakes;126 images are wrong for both. The actual four-model average gains18 correct predictions and loses13, net5 (+.3327pp). It is a real observed improvement, but fails the predeclared >=.5pp material gate. Do not market it as substantial or statistically established. Preserve it as a development candidate without changing the frozen method or performing another test. No weight search, alternative checkpoint or additional training.

Paper comparison checked5October2026:

- https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2026.1847649/full explicitly reports96.37% on a1103-image validation subset used for early stopping/ensemble weight selection, with no independently held-out HAM10000 test. The authors acknowledge optimistic tuned performance.
- https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2025.1699960/full reports98.32+/-0.41% over five-fold cross-validation and five seeds, where each fold is used for testing; it also describes70/15/15 splitting. Do not label it a verified single locked-test result matching our cohort. Its documented balancing/preprocessing and validation arrangement differ.

High test performance does not mathematically require higher validation performance: the cohorts differ and sample variation can reverse the ordering. Best-epoch/method selection can make validation optimistic. Published results do not determine the achievable score on our test.

Decision: retain the proven frozen reference, preserve S41's small development gain, and stop the automatic tweak sequence. CPU analysis can reject bugs and measure existing error complementarity; it cannot certify that a new training recipe will improve. No more speculative GPU training launched here.
