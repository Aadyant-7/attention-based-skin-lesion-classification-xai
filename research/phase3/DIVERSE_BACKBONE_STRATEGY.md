# Diverse backbone and bounded ensemble strategy â€” after S06

**Current evidence:** [S10–S12 closeout](S10_S12_CLOSEOUT.md): S10 weaker standalone but complementary; S12 equal B0/ConvNeXt/V2-S reaches92.42%/.8770 with3 passes. Retain S06 cheap standalone. Next one frozen [S13 TTA inference study](S13_PROPOSAL.md), with FP32 identity control and no retraining/weight search, pending approval. CBAM mixed evidence retained; strict confirmation and original locked test later.

## Preserved plan before S06 evaluation

Evidence: B0/MobileNet give86.36%/86.03% exploratory accuracy; fixed fusion88.56%/.8057 F1. Added B0 CBAM ties accuracy with mixed class/criterion effects. Preserve all results; move screening compute toward a small number of distinct strong transfer models.

1. **Next only: ConvNeXt-Tiny**, no added CBAM, same screening policy. A modern higher-capacity residual CNN complements the lightweight SE family hypothesis. Common-budget comparisons include official pretrained package differences; do not call them topology-only effects.
2. Close out and inspect capacity/generalization,cost,class scores and aligned error overlap. Architecture names alone do not establish diversity. Reserve at most one further distinct family (DenseNet/ResNet or another justified model),only if ConvNeXt evidence or cost requires it; no automatic list.
3. Before ensemble evaluation predeclare a bounded candidate set using accuracy-winner predictions aligned to the identical manifest. If ConvNeXt qualifies, candidates are equalS06+S02,equalS06+S03,and equalS06+S02+S03 (maximum three). Retain/report all evaluated candidates andS04 frozen reference. No exhaustive subset/weight/criterion sweep; class collapse and measured inference cost can reject a nominal accuracy leader.
4. Retain the balanced alternative when accuracy and macro-F1 disagree. S05's two states remain indexed,not automatically admitted because they exist. Additional attention only when a matched question/errors justify it; one-seed CBAM result is not a universal rejection of attention.
5. On the strongest fixed ensemble, consider one separately registered TTA or resolution intervention with fixed views/proportions and its added inference cost. Do not combine preprocessing/balancing/TTA changes into an unidentifiable gain. Grad-CAM/XAI later uses saved representative correct/error examples,not a claimed accuracy intervention.
6. Exploratory image-level screening remains ~70/15/15 with563 shared train/val lesions. Every comparison identifies manifest,recipe,pretraining,selection,class supports and cost. No independent-lesion/test claim; no patient-independence assumption. Never balance validation/test populations.
7. Shortlist only the strongest accuracy/balanced methods for **fresh strict training** from external pretrained weights. Exploratory checkpoints have trained on strict-val images and cannot supply clean strict confirmation. Freeze methodology/member checkpoints/weights/views before the final original locked-test session. Test remains untouched.

Literature foundation: integrated review's transfer-learning,attention and constrained-fusion sections; originalConvNeXt architecture reference; HAM10000/base-paper protocol audits retained. Some papers'95â€“98% use different support/preprocessing/evaluation; those values motivate investigation but are not comparable targets or promises. Report complete candidate/selection history and negative results to limit hidden validation overfitting.

Every serious training run saves best/latest/F1 states,config/environment/source hashes,logs/history,metrics,class scores,probabilities,curves,matrices and comparison plots. CPU ensembles reference parents and do not invent training curves/new checkpoints. Registry separates structured/historical and strict/exploratory records.

GPU policy unchanged: specific proposalâ†’human approvalâ†’independent launchâ†’confirm log/first checkpointsâ†’manual monitor/recovery instructionsâ†’**stop immediately**. No monitoring/epoch updates/completion wait; user returns for analysis. No further run is authorized by this plan.
