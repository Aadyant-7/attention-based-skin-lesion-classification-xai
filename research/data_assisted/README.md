# External dermoscopy development data

This workflow tests the substantive hypothesis that more independent difficult-class lesions may help where extra epochs and small voting changes did not. It changes training-data scope and must remain separate from HAM-only comparisons. Data preparation is not a measured accuracy improvement.

## Commands, in order

Run from the project root with the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m research.short_screening.fetch_isic_duplicate_references
.\.venv\Scripts\python.exe -m research.data_assisted.clearance
.\.venv\Scripts\python.exe -u -m research.data_assisted.acquire
.\.venv\Scripts\python.exe -u -m research.data_assisted.acquire --all
.\.venv\Scripts\python.exe -u -m research.data_assisted.pixel_clearance
```

The default acquisition is a fixed 14-image transport sample. Full acquisition downloads only selected ZIP members, requires exact HTTP byte ranges and a pinned archive ETag, and refuses a full-archive fallback. Each attempt has a 20-minute ceiling; rerunning safely verifies/reuses completed JPEGs and downloads the remainder. Valid files are never overwritten. Acquisition results are written only after all requested files pass CRC and decode checks.

## Completed CPU screen

S66/S67 tested the fixed HAM-only versus HAM+external frozen-feature recipe. Both advancement gates failed; no pooled-data GPU run follows. See `S66_S67_CLOSEOUT.md`. Reproducible commands (a completed screen rerun is a no-op):

```powershell
.\.venv\Scripts\python.exe -u -m research.data_assisted.frozen_screen
.\.venv\Scripts\python.exe -m research.data_assisted.screen_closeout
.\.venv\Scripts\python.exe -m research.data_assisted.source_shift_audit
```

Artifacts: `results/data_assisted/s66_s67_frozen_data_screen/`. Cached external features and eight fitted CPU classifier heads: `.cache/s66_s67_frozen_data_screen/`. These are diagnostic classifiers, not new fine-tuned ensemble results; metadata remains deferred.

## Locations

- `results/data_assisted/clearance_v1/`: fixed plans, source hashes, selection/exclusion manifests, acquisition evidence, duplicate candidates and the final screened manifest.
- `data/raw/ISIC2019_external/clearance_v1/`: local JPEGs; excluded from Git.
- `.cache/isic_duplicate_references/`: pinned published filename lists and source text; excluded from Git.
- `.cache/data_assisted_clearance_v1/`: development-only HAM fingerprints; excluded from Git.
- `research/data_assisted/`: reproducible preparation scripts and decisions.
- `research/data_assisted/METADATA_FOLLOWUP_DEFERRED.md`: earlier metadata evidence and deferred follow-up.

## Interpretation limits

One image is selected per known BCN lesion before viewing pixels or scores. External AK is subset supervision for HAM `akiec`; invasive SCC is not merged. Published filename exclusions and conservative perceptual-hash quarantine reduce known duplication, but do not certify universal patient or test independence. No original locked-test image or diagnosis is opened by these scripts. Unknown renamed copies of original test images remain unverified. A later external-data experiment cannot replace or reinterpret the already reported original test.

The fixed external 85/15 development partition is not a new locked test. Metadata fusion and GPU training are not performed by this package.

## Attribution and use

Use the official [ISIC challenge training release](https://challenge.isic-archive.com/data/) and retain its dataset attribution and terms. Do not redistribute raw images through this repository. Source URLs, checksums and published duplicate-list revision are recorded in `source_manifest.json`; the duplicate-list authors supply their work as-is, and their removal lists depend on their original removal sequence.
