# S04 closeout — fixed 50/50 probability fusion

Completed 3 October 2026 on CPU. ID `s04_s02_s03_equal_probability_exploratory_seed42`. One predeclared candidate; **no weight search, GPU training, test evaluation or new checkpoint**. Parent accuracy winners: MobileNet S02 epoch16 and B0 S03 epoch20. New-image inference requires both parent checkpoints/two model passes.

| Validation metric | Result |
|---|---:|
| Accuracy | **88.55622089155023%** (1,331/1,503) |
| Macro precision | **0.8260743092585557** |
| Macro recall / balanced accuracy | **0.7909651520972928** |
| Macro-F1 | **0.8056543445886161** |
| Training-class-weighted probability NLL | 0.4667166441832347 |
| Melanoma recall | **71.8562874251497%** (120/167) |
| CPU evaluation/artifact time | 3.96 seconds, excluding process import/startup |

## Same-protocol comparison and meaningfulness

| Method | Correct | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| S02 MobileNetV3-Large | 1,293 | 86.03% | 0.7950 | 0.7801 | 0.7861 |
| S03 EfficientNet-B0 | 1,298 | 86.36% | 0.7534 | 0.7916 | 0.7707 |
| **S04 fixed equal fusion** | **1,331** | **88.56%** | **0.8261** | 0.7910 | **0.8057** |

Observed validation gain is practically useful: **+2.5283 accuracy percentage points/+0.019543 F1 over MobileNet**, **+2.1956 points/+0.034994 F1 over B0**. Versus B0:66 errors fixed,33 correct predictions lost,net33 gained. Versus MobileNet:71 fixed,33 lost,net38 gained. This tests the predicted complementarity without another training run.

Not universal dominance: macro recall is slightly below B0 (-0.000668); five of seven class F1 scores improve over both parents, but akiec/vasc F1 remain below MobileNet. Of33 net extra correct images over B0,19 are nv; majority-class gains contribute substantially. This is a meaningful **observed exploratory validation improvement**, not a statistical significance or independent-test/lesion-generalization claim. Both parents were validation-selected; repeated development use remains a limitation.

## Class-wise scores

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| akiec |49 |0.7750 |0.6327 |0.6966 |
| bcc |77 |0.7640 |0.8831 |0.8193 |
| bkl |165 |0.8012 |0.7818 |0.7914 |
| df |17 |0.9286 |0.7647 |0.8387 |
| mel |167 |0.7229 |0.7186 |0.7207 |
| nv |1,007 |0.9408 |0.9464 |0.9436 |
| vasc |21 |0.8500 |0.8095 |0.8293 |

Confusion matrix order: akiec,bcc,bkl,df,mel,nv,vasc; rows=true,columns=predicted:

```text
31  7   8  0   1   2  0
 1 68   0  0   1   7  0
 5  0 129  1  15  15  0
 0  1   1 13   1   1  0
 2  3   7  0 120  34  1
 1  7  16  0  28 953  2
 0  3   0  0   0   1 17
```

Melanoma34→nv,7→bkl; nv28→mel. akiec recall31/49 and vasc17/21 remain trade-offs. df/vasc estimates have very small supports.

## Protocol, preservation and artifacts

Same exploratory manifest `data/splits/exploratory/image_level_dev_v1.csv`, SHA `75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb`:7,009/1,503/1,503 (~70/15/15),563 train/validation shared lesions. Original locked test retained, never loaded. Strict results remain separate; no fresh strict confirmation has occurred.

- `results/structured_experiments/s04_s02_s03_equal_probability_exploratory_seed42/`: config/record,run.log,validation metrics/predictions,parent references,closeout verification. Exact fixed probability average, all1,503 identities/labels, loss, matrix and class/macro scores verified.
- `figures/`: raw/normalized confusion matrices, per-class and support plots in PNG/PDF, three CSVs; normalized matrix and class plot visually checked. No fusion training curve/best epoch exists; parent curves/checkpoints remain referenced in `parent_artifacts.json`.
- `results/model_comparison/structured/s04_fixed_fusion/`: comparison graph/table,class scores and paired gains. `s04_exploratory_context/`: descriptive historical context, all same manifest/support.
- Historical exploratory B0 four-stream result88.16%/.8280 F1: S04 has slightly higher accuracy but lower macro-F1. Historical eight-stream B0+PanDerm90.75%/.8573 remains the overall exploratory maximum, with different recipe/cost/selection. Neither is a matched fusion ablation or a test score.
- Master registry completed/closed out; all other rows unchanged, including473 historical rows. Parent checkpoints/results preserved. S04 requires both original trained models, not a new fused checkpoint.

Next recommendation: [S05 matched B0+CBAM](S05_PROPOSAL.md), to isolate added attention under the common exploratory training policy. Literature attention section motivates the question; historical B0+CBAM recipes do not isolate CBAM. No automatic fusion/weight sweep or next GPU launch. Future GPU sessions follow launch-and-stop.

Current Phase Progress: 75%
Total Project Progress: 65%
