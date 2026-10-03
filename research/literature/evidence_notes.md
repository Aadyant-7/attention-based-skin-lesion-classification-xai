# Phase 2 primary-source evidence notes

Checked 3 October 2026. `sources.json` is the curated input; `review.csv`, `review_table.md/.tex` and `references.bib` are exports from `python -m research.phase2.prepare`. Verification means that **named fields were located**, not that every protocol detail or metric was independently reproduced. No score is imported into the project's experiment registry.

## Source scope and access

- HAM10000: [final abstract](https://arxiv.org/abs/1803.10417) and [v1 usage notes](https://arxiv.org/html/1803.10417v1). Final 10,015-image count is corroborated locally. Older manuscript counts are provisional; use saved manifests for project totals.
- Pardede et al.: [journal page](https://www.joig.net/show-111-558-1.html) and [publisher PDF](https://www.joig.net/2026/JOIG-V14N4-551.pdf), III.A/III.C and Table IV p561. Split/balancing descriptions conflict; support is 7,041. Weighted and macro fields remain distinct. Replication leakage is a possibility, not a proven reconstruction of unpublished code.
- Datta et al.: [abstract](https://arxiv.org/abs/2105.03358) and [accessed PDF](https://arxiv.org/pdf/2105.03358), Tables2/3. Record 0.934 accuracy / 0.937 weighted precision from the paper, not the older notebook's approximately 0.938 claim. Support828 differs from the full 1,503-image project cohort. Separate validation/lesion grouping is unresolved here. The current abstract is v4 with changed author spelling; version-specific v3 retrieval was blocked. The PDF was unversioned, so no pin to v3/v4 is asserted.
- Gessert et al.: [author preprint](https://arxiv.org/abs/1910.03910) and [journal metadata](https://pubmed.ncbi.nlm.nih.gov/32292713/). Ensemble/multiresolution/loss-balancing context; ISIC2019 balanced accuracy is a different metric and task, explicitly separated from HAM10000 ordinary accuracy. This motivates limited fusion tests, not replication of a large subset search.
- CBAM, EfficientNet, MobileNetV3, ResNet, DenseNet, ConvNeXt and Grad-CAM: primary abstract/bibliographic checks only, linked per row. No uninspected ImageNet details are used as HAM10000 scores. ResNet/Grad-CAM conference years are cross-referenced from the inspected skin-attention paper references; first arXiv years differ.
- PanDerm: [publisher source](https://www.nature.com/articles/s41591-025-03747-y), abstract, Fig1, pretraining methods and citation. Accessed via indexed publisher text when direct open failed. Main paper ViT-Large differs from our cached Base artifact; no single downstream HAM10000 score is extracted.

Some Nature/CVF/medRxiv direct pages rejected retrieval; author manuscripts/publisher PDF/indexed primary-source text supplied only the fields labelled verified. Existing code audits in `docs/high_accuracy_claims_audit.md` remain historical evidence, not newly verified paper fields. Other older claims remain out of this focused table pending a reason to use them.

## Research gap and proposed contribution

Our repository has useful B0+CBAM and fusion evidence but lacks a matched multi-backbone comparison and an isolated CBAM control. Existing high-score sources use different populations or incompletely established evaluation protocols. This motivates a **repository-specific study gap**, not a claim that nobody has compared these methods before.

The planned contribution is a reproducible, compute-bounded comparison on an unchanged lesion-disjoint seven-class cohort; matched added-CBAM ablations; transparent imbalance/class-wise trade-offs; limited complementary fusion; and labelled explanations of correct and incorrect cases. No novelty, score improvement or clinical effectiveness is claimed before evidence exists. Public scores serve as contextual literature, not a leaderboard against our locked test.
