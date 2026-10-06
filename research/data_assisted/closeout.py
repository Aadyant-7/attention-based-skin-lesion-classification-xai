"""Write data-readiness evidence after acquisition and pixel clearance, not scores."""
import json
import pandas as pd
from research.common import ROOT,sha256,write_json


def main():
    out=ROOT/'results/data_assisted/clearance_v1'
    metadata=json.loads((out/'summary.json').read_text())
    acquisition=json.loads((out/'acquisition_summary.json').read_text())
    pixels=json.loads((out/'pixel_clearance_summary.json').read_text())
    manifest=out/'pixel_screened_candidate_manifest.csv'
    selected=pd.read_csv(out/'selected_one_view_manifest.csv')
    retained=pd.read_csv(manifest)
    assert acquisition['status']=='completed' and acquisition['images']==len(selected)==3439
    assert pixels['candidate_manifest_sha256']==sha256(manifest)
    assert len(retained)==pixels['retained_images'] and retained.image_id.is_unique and retained.lesion_id.is_unique
    assert not pixels['test_loaded'] and not pixels['training_launched']
    assert set(retained.image_id)<=set(selected.image_id)
    before=selected.set_index('image_id').external_partition
    assert all(before[r.image_id]==r.external_partition for r in retained.itertuples(index=False))
    counts=pd.read_csv(out/'pixel_screened_class_counts.csv')
    train=int(counts.external_train.sum());val=int(counts.external_preflight_val.sum())
    table='\n'.join(f'|{r.class_name}|{r.images_and_lesions}|{r.external_train}|{r.external_preflight_val}|' for r in counts.itertuples(index=False))
    flags=pd.read_csv(out/'duplicate_candidate_flags.csv')
    flags=flags.loc[flags.kind!='none']
    kinds=flags.groupby(['scope','kind']).size().to_dict()
    flag_text='; '.join(f'{scope}/{kind}: {n}' for (scope,kind),n in kinds.items()) or 'No exact/near-hash candidate pairs found.'
    report=f'''# External dermoscopy data preparation — closeout

## Decision

The bounded dataset is prepared for a development screen. **No new accuracy was measured, no GPU training ran, and the original locked test was not evaluated or opened.** Metadata fusion remains deferred.

## Fixed selection and clearance

- Source: official ISIC2019 training release, restricted to known BCN lesion identifiers.
- Every known HAM image identity/canonical downsampled alias and HAM lesion identity was excluded; no SCC/unknown classes, missing lesion IDs or downsampled copies were accepted.
- Published deletion-name lists at revision `b785b75fc1b1440d7be7b17d2cb2348de6c76102` plus confirmed historical duplicate cases flagged 15 lesion groups. All 100 images in those groups were excluded before selection.
- One image per remaining lesion was chosen by fixed seed42 identity hash, before inspecting quality or model predictions. This produced **3,439 selected images/lesions**.
- Exactly **{acquisition['images']:,} JPEGs** passed archive CRC, size and decode checks. Local payload: {acquisition['bytes_on_disk']:,} bytes (~1.77GB). Only selected members were downloaded, not the full9.77GB archive.
- Acquisition transport was resumed with higher bounded download concurrency; existing files were CRC-verified and reused. `transport_recovery.json` records the transport-only amendment. The final attempt runtime is not the total preparation runtime.
- Pixel fingerprints compared every selected external image against **8,512 HAM development images** and within the selected external set. Exact byte/decoded RGB hashes and fixed64-bit pHash Hamming distance<=4 were used.
- **{pixels['flagged_pairs']} flagged pairs / {pixels['quarantined_images']} external images quarantined**. Hash-near matches are candidates, not confirmed duplicate diagnoses. Conservative exclusions are preserved in manifests and do not relabel or delete source files.
- Flag breakdown: {flag_text}

## Final screened candidate cohort

**{len(retained):,} images / distinct lesion groups**: {train:,} external training, {val:,} external preflight-validation. Original fixed85/15 stratified assignments were preserved through quarantine, without replacing excluded examples.

|Class|Images / lesion groups|External train|External preflight validation|
|---|---:|---:|---:|
{table}

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

Final manifest SHA256: `{sha256(manifest)}`.

## Next action

Use the predeclared paired CPU frozen-feature data screen in `research/data_assisted/NEXT_DEVELOPMENT_SCREEN_PLAN.md` before spending GPU time. Compare HAM-only versus HAM-plus-external training under the same recipe; require material overall and difficult-class gains plus train-only group consistency. No classifier has yet been trained on this dataset. Metadata evidence and later question are in `METADATA_FOLLOWUP_DEFERRED.md`.
'''
    (ROOT/'research/data_assisted/DATA_CLEARANCE_REPORT.md').write_text(report,encoding='utf-8')
    write_json(out/'closeout_verification.json',dict(status='passed',selected_images=len(selected),retained_images=len(retained),
        external_train=train,external_preflight_val=val,partition_assignments_preserved=True,
        candidate_manifest_sha256=sha256(manifest),test_loaded=False,gpu_used=False,accuracy_measured=False,
        ready_for_bounded_development_screen=True,independence_limitations_preserved=True))
    print(json.dumps(dict(retained=len(retained),train=train,val=val,quarantined=pixels['quarantined_images']),indent=2))


if __name__=='__main__':main()
