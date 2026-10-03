# Focused literature review — verified 3 October 2026

12 primary papers, selected fields only; not an exhaustive systematic review. Numeric metrics below are **author-reported**, not independently reproduced. All ratios are 0–1. Unknowns are Not verified; method papers are not HAM10000 result rows.

## Skin-classification evidence

| Year / authors | Model / technique | Dataset / classes | Reported split / test support | Accuracy | Macro-F1 | Comparability |
|---|---|---|---|---:|---:|---|
| 2026 / [A. M. H. Pardede; Solikhun; Juni Ismail](https://www.joig.net/2026/JOIG-V14N4-551.pdf) | EfficientNet-B0 (metric row); MobileNetV3-Large also compared / RandomOverSampler | HAM10000 / 7 | 70/15/15%; support 7041 | 0.9886 | 0.9863 | Not comparable to untouched original-image lesion-disjoint evaluation; code/IDs unverified. |
| 2021 / [Soumyya Kanti Datta; Seyed Mohammad Abuzar Hashemi; Sargur N. Srihari; Mingchen Gao](https://arxiv.org/pdf/2105.03358) | InceptionResNetV2 / soft attention | HAM10000 (selected metric); ISIC2017 also studied / 7 | 85/Not verified/15%; support 828 | 0.934 | Not verified | Selected cohort support828; independent validation/lesion grouping not established here. |

## Other evaluation protocols

[Nils Gessert; Maximilian Nielsen; Mohsin Shaikh; RenÃ© Werner; Alexander Schlaefer, 2020](https://arxiv.org/abs/1910.03910v1): ISIC2019; not standalone HAM10000; **balanced accuracy 0.636**, official task1 test. Different class/task/cohort; no cross-protocol leaderboard comparison.

## Dataset, architecture, attention, domain pretraining and XAI references

| Year | Source | Role / verified basis |
|---|---|---|
| 2018 | [The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions](https://arxiv.org/abs/1803.10417v3) — Philipp Tschandl; Cliff Rosendahl; Harald Kittler | dataset; bibliography;dataset_size;classes;repeated_lesion_views |
| 2020 | [Skin Lesion Classification Using Ensembles of Multi-Resolution EfficientNets with Meta Data](https://arxiv.org/abs/1910.03910v1) — Nils Gessert; Maximilian Nielsen; Mohsin Shaikh; RenÃ© Werner; Alexander Schlaefer | ensemble context; bibliography;ensemble;resolution;balancing;reported_balanced_accuracy;metric_split |
| 2018 | [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521v2) — Sanghyun Woo; Jongchan Park; Joon-Young Lee; In So Kweon | method; bibliography;attention_mechanism;benchmark_names |
| 2019 | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html) — Mingxing Tan; Quoc Le | architecture; bibliography;architecture_family;scaling_method |
| 2019 | [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244v5) — Andrew Howard et al. | architecture; bibliography;architecture_family;design_method |
| 2016 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — Kaiming He; Xiangyu Zhang; Shaoqing Ren; Jian Sun | architecture; authors;title;architecture_method;conference_year |
| 2017 | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5) — Gao Huang; Zhuang Liu; Laurens van der Maaten; Kilian Q. Weinberger | architecture; bibliography;architecture_method |
| 2022 | [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545v2) — Zhuang Liu; Hanzi Mao; Chao-Yuan Wu; Christoph Feichtenhofer; Trevor Darrell; Saining Xie | architecture; bibliography;architecture_family |
| 2017 | [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391v4) — Ramprasaath R. Selvaraju; Michael Cogswell; Abhishek Das; Ramakrishna Vedantam; Devi Parikh; Dhruv Batra | explainability method; bibliography;mechanism;conference_year |
| 2025 | [A multimodal vision foundation model for clinical dermatology](https://www.nature.com/articles/s41591-025-03747-y) — Siyuan Yan; Zhen Yu; Clare Primiero et al. | domain pretraining reference; bibliography;pretraining_size;modalities;main_backbone;benchmark_count |

The source-specific evidence locators, versions, weighted/macro distinctions and unresolved fields are in `review.csv` and `evidence_notes.md`. Do not convert Not verified into assumed defaults.
