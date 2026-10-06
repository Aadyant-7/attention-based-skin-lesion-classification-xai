# Substantial next hypothesis: broaden dermoscopy training data

## Why this direction

S62/S63 show difficult-class representation and generalization problems, and no material benefit from confidence calibration. S59 already fits the training examples perfectly. Those observations support investigating additional relevant training examples rather than simply adding epochs or searching more ensemble weights. They do not prove that more data will improve performance.

The official [ISIC2019 release](https://challenge.isic-archive.com/data/) provides25,331 training images from HAM10000, BCN and MSK, with diagnoses and metadata. It is not an independent dataset if blindly added: it contains HAM10000. The metadata-only preflight excluded **all10,015 HAM image identities**, before parsing diagnosis values from the remaining rows, plus any known overlapping HAM lesion IDs. No ISIC test data was downloaded or parsed.

Current downloaded training-CSV hashes and exact version-derived counts are in `results/short_screening/external_isic2019_feasibility/summary.json`. Counts should not be substituted from an older paper or different label revision.

| Class | Current HAM training | Non-HAM candidates | With known lesion IDs |
|---|---:|---:|---:|
| akiec / AK mapping pending review |229|737|737|
| BCC |359|2809|2809|
| BKL |769|1525|1327|
| DF |81|124|124|
| MEL |778|3409|3072|
| NV |4693|6170|4621|
| VASC |100|111|111|

Total:14,885 candidate images, of which12,801 have known lesion IDs spanning4,255 groups (largest31 images).431 unsupported-class rows were excluded. No known lesion group has conflicting candidate class labels in these rows. The relevant known-group counts are725 melanoma and542 BKL, versus478 and541 in our current HAM training partition; the image counts must not be presented as independent cases. These are metadata candidates, not cleared training images. The two source CSVs total only a few MB; the official full image archive is9.1GB and has **not** been downloaded.

## Required cheap work before a training proposal

1. Review current diagnosis definitions, especially project akiec versus external AK/SCC. Do not silently merge SCC into the seven-class label set.
2. Audit source/lesion identifiers, original HAM image aliases and published duplicate mappings. Conservatively exclude2,084 rows lacking lesion IDs until a grouping policy is justified. Known IDs alone do not rule out renamed duplicates or patient overlap.
3. Before downloading images, write a fixed identity manifest, deduplication plan and source-stratified grouping policy. All HAM validation/test images and their known lesion groups must be excluded from external training. Do not inspect original locked-test images/labels to make modeling decisions.
4. If image access proceeds later, check duplicate evidence against permissible development data and published identity mappings. An inability to establish holdout independence must be reported, not treated as clearance.
5. Use existing ImageNet ConvNeXt as the candidate to minimize architecture changes. Define a small, fixed external dermoscopy adaptation stage or source-balanced training pilot, with compute cap, matched control and predeclared difficult-class/overall gates, before any full run.

This would change training-data scope. It must be labeled **external-data-assisted development**, with source counts and comparisons separate from HAM-only experiments. A future fresh holdout must be frozen before any modeling, and cannot be constructed by merely relabeling previously evaluated images as untouched. No93% test result is promised.

## Current decision

S64/S65 are complete. The apparent first preprocessing gain did not pass the train-only group consistency gate, so no crop-based GPU run was justified.

The next external-data preparation stage is now complete: **3,439 selected BCN lesions acquired and decoded; 116 images conservatively quarantined by exact/near-hash screening; 3,323 retained distinct lesions**, with2,823 external-training and500 external-preflight-validation images. This includes433 additional melanoma training lesions. Published-name filtering separately removed100 images from15 flagged lesion groups before image selection. Earlier candidate counts above are historical feasibility counts, not the current training-ready manifest.

Current evidence and limitations: `research/data_assisted/DATA_CLEARANCE_REPORT.md`. Final identities/partitions/hashes: `results/data_assisted/clearance_v1/pixel_screened_candidate_manifest.csv`. Unknown aliases of original test images and patient overlap remain unverified; this is development-data preparation, not a certificate of final-test independence.

Next: the matched bounded CPU frozen-feature data screen in `research/data_assisted/NEXT_DEVELOPMENT_SCREEN_PLAN.md`, before any GPU fine-tuning proposal. No classifier training, new ensemble, new accuracy measurement or locked-test evaluation has been performed by this preparation stage. Metadata fusion remains deferred.
