# External dermoscopy data preparation — closeout

## Decision

The bounded dataset is prepared for a development screen. **No new accuracy was measured, no GPU training ran, and the original locked test was not evaluated or opened.** Metadata fusion remains deferred.

## Fixed selection and clearance

- Source: official ISIC2019 training release, restricted to known BCN lesion identifiers.
- Every known HAM image identity/canonical downsampled alias and HAM lesion identity was excluded; no SCC/unknown classes, missing lesion IDs or downsampled copies were accepted.
- Published deletion-name lists at revision `b785b75fc1b1440d7be7b17d2cb2348de6c76102` plus confirmed historical duplicate cases flagged 15 lesion groups. All 100 images in those groups were excluded before selection.
- One image per remaining lesion was chosen by fixed seed42 identity hash, before inspecting quality or model predictions. This produced **3,439 selected images/lesions**.
- Exactly **3,439 JPEGs** passed archive CRC, size and decode checks. Local payload: 1,765,988,792 bytes (~1.77GB). Only selected members were downloaded, not the full9.77GB archive.
- Acquisition transport was resumed with higher bounded download concurrency; existing files were CRC-verified and reused. `transport_recovery.json` records the transport-only amendment. The final attempt runtime is not the total preparation runtime.
- Pixel fingerprints compared every selected external image against **8,512 HAM development images** and within the selected external set. Exact byte/decoded RGB hashes and fixed64-bit pHash Hamming distance<=4 were used.
- **84 flagged pairs / 116 external images quarantined**. Hash-near matches are candidates, not confirmed duplicate diagnoses. Conservative exclusions are preserved in manifests and do not relabel or delete source files.
- Flag breakdown: HAM_development/near_hash_candidate_not_confirmed: 17; selected_external/near_hash_candidate_not_confirmed: 67

## Final screened candidate cohort

**3,323 images / distinct lesion groups**: 2,823 external training, 500 external preflight-validation. Original fixed85/15 stratified assignments were preserved through quarantine, without replacing excluded examples.

|Class|Images / lesion groups|External train|External preflight validation|
|---|---:|---:|---:|
|akiec|221|188|33|
|bcc|943|802|141|
|bkl|337|285|52|
|df|38|32|6|
|mel|511|433|78|
|nv|1237|1052|185|
|vasc|36|31|5|

The preflight-validation partition is development data, not a new locked test. Class order: `akiec, bcc, bkl, df, mel, nv, vasc`.

## Label mapping and limits

HAM [akiec definitions](https://www.nature.com/articles/sdata2018161) include actinic keratosis and intraepithelial carcinoma/Bowen disease. ISIC2019 [subdivided the earlier categories](https://forum.isic-archive.com/t/question-re-new-classification-category/1074). External AK is compatible **subset supervision**, not complete akiec coverage; invasive SCC must not be indiscriminately merged. Other six diagnostic categories map directly for this development stage.

[Published duplicate-removal lists](https://github.com/mmu-dermatology-research/isic_duplicate_removal_strategy) are incomplete and depend on removal order. Patient identifiers are unavailable. Unknown aliases of original test images cannot be ruled out without opening the test, which this workflow deliberately does not do. Thus this is **not a certificate of final-test independence**. Source/domain/label shift remains a substantive research risk; adding data is not guaranteed to improve performance.

This workflow changes training-data scope to **external-data-assisted development** and must not be pooled silently with HAM-only comparisons or used to alter the previously reported original test result. Dataset attribution and terms: [official release](https://challenge.isic-archive.com/data/). Raw images remain local and are not redistributed through GitHub.

## Artifacts

- `results/data_assisted/clearance_v1/pixel_screened_candidate_manifest.csv`: final candidate identities, partitions, paths and hashes.
- `source_manifest.json`, `PREDECLARED_PLAN.json`, `PIXEL_GATE_PLAN.json`: sources/version and fixed policies.
- `excluded_candidates.csv`, `published_name_hits.csv`, `duplicate_candidate_flags.csv`, `quarantined_images.csv`: exclusion evidence.
- `acquired_image_manifest.csv`, `acquisition_summary.json`: decoded image/checksum evidence.
- `external_image_fingerprints.csv`, `nearest_HAM_development_hash.csv`, `pixel_clearance_summary.json`: pixel-screen evidence.
- `pixel_screened_class_counts.csv`: final source/class counts.
- `data/raw/ISIC2019_external/clearance_v1/`: local images, excluded from Git.

Final manifest SHA256: `35065c4406b7e50134fb749cea04c403ffd8f83f7c63c8fe43c5619926037ba1`.

## Next action

Follow-up on2026-10-06: S66/S67 executed the plan below and failed both advancement gates. The fixed pooled-data recipe is rejected for GPU training. This report preserves the earlier preparation-stage findings; the completed model evidence is in `research/data_assisted/S66_S67_CLOSEOUT.md`.

Use the predeclared paired CPU frozen-feature data screen in `research/data_assisted/NEXT_DEVELOPMENT_SCREEN_PLAN.md` before spending GPU time. Compare HAM-only versus HAM-plus-external training under the same recipe; require material overall and difficult-class gains plus train-only group consistency. No classifier has yet been trained on this dataset. Metadata evidence and later question are in `METADATA_FOLLOWUP_DEFERRED.md`.
