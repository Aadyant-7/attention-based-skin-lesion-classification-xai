# Metadata-only ISIC2019 training feasibility

Derived from the official ISIC2019 **training** ground-truth and metadata CSVs linked at https://challenge.isic-archive.com/data/ . This folder contains filtered identities/class counts and a feasibility summary, not images, predictions or an accuracy result. All HAM10000 identities were excluded before diagnosis parsing.

The source release is distributed under **CC-BY-NC** as specified on that page; these derived dataset metadata are not relicensed under the repository's software license.

Source attribution: BCN_20000, Department of Dermatology, Hospital Clínic de Barcelona; HAM10000, ViDIR Group, Department of Dermatology, Medical University of Vienna; MSK dataset, as attributed by the ISIC release. See the official page for the full required publications, including Tschandl et al., *The HAM10000 dataset* (2018), Codella et al., *Skin Lesion Analysis Toward Melanoma Detection* (2017), and Hernández-Pérez et al., *BCN20000: Dermoscopic lesions in the wild* (2024).

Zero known image/lesion-ID overlap does **not** certify absence of aliases, near duplicates or patient overlap. Missing lesion IDs and label harmonization remain unresolved. Nothing in this folder authorizes a GPU run or establishes improved test accuracy.
