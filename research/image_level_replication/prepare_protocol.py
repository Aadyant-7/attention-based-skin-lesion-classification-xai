"""Freeze/audit the new 70/15/15 protocol using metadata and filenames only.

No torch imports, image decoding, model inference, training, or old-file writes.
"""
import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import sklearn
from sklearn.model_selection import train_test_split

from research.common import CLASSES, ROOT, sha256, write_csv, write_json

STUDY = "image_level_replication_v1"
PROTOCOL = "postdevelopment_image_level_70_15_15_overlap_allowed"
SEED = 42
COUNTS = {"train": 7009, "val": 1503, "test": 1503}
FIELDS = ["image_id", "lesion_id", "diagnosis", "label", "split"]
RAW = ROOT / "data/raw/HAM10000"
SOURCE = RAW / "HAM10000_metadata.csv"
MANIFEST = ROOT / "data/splits/image_level_replication/v1/split_assignments.csv"
POLICY = ROOT / "research/image_level_replication/protocol_v1.json"
OUTPUT = ROOT / "results/image_level_replication/v1/protocol"
REGISTRY = ROOT / "results/image_level_replication/v1/experiment_registry.csv"
PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def metadata_rows():
    rows = read_csv(SOURCE)
    if len(rows) != 10015 or len({r["image_id"] for r in rows}) != 10015:
        raise ValueError("Expected 10,015 unique original image IDs")
    if set(r["dx"] for r in rows) != set(CLASSES):
        raise ValueError("Unexpected diagnoses")
    lesions = {}
    for row in rows:
        if not row["image_id"] or not row["lesion_id"] or not row["dx"]:
            raise ValueError("Null identity/diagnosis")
        if lesions.setdefault(row["lesion_id"], row["dx"]) != row["dx"]:
            raise ValueError("Conflicting diagnosis for one lesion")
    labels = {name: str(index) for index, name in enumerate(CLASSES)}
    saved_labels = json.loads((ROOT / "data/splits/label_mapping.json").read_text())
    if saved_labels != {name: index for index, name in enumerate(CLASSES)}:
        raise ValueError("Existing class order mismatch")
    return [dict(image_id=r["image_id"], lesion_id=r["lesion_id"],
                 diagnosis=r["dx"], label=labels[r["dx"]])
            for r in sorted(rows, key=lambda r: r["image_id"])]


def generate(rows):
    """One fixed split; no seed/overlap/performance search."""
    indexes = list(range(len(rows)))
    train, held = train_test_split(
        indexes, test_size=COUNTS["val"] + COUNTS["test"],
        stratify=[r["diagnosis"] for r in rows], random_state=SEED)
    val, test = train_test_split(
        held, test_size=COUNTS["test"],
        stratify=[rows[i]["diagnosis"] for i in held], random_state=SEED)
    assignments = {i: split for split, ids in (("train", train), ("val", val), ("test", test))
                   for i in ids}
    return [dict(row, split=assignments[i]) for i, row in enumerate(rows)]


def validate(rows, originals):
    if len(rows) != 10015 or len({r["image_id"] for r in rows}) != 10015:
        raise ValueError("Duplicate/missing original images")
    found = {r["image_id"]: {k: r[k] for k in FIELDS[:-1]} for r in rows}
    expected = {r["image_id"]: r for r in originals}
    if found != expected:
        raise ValueError("Manifest identity/diagnosis/label mismatch")
    if dict(Counter(r["split"] for r in rows)) != COUNTS:
        raise ValueError("Wrong 70/15/15 partition sizes")
    image_sets = {s: {r["image_id"] for r in rows if r["split"] == s} for s in COUNTS}
    lesion_sets = {s: {r["lesion_id"] for r in rows if r["split"] == s} for s in COUNTS}
    overlaps = []
    for a, b in PAIRS:
        if image_sets[a] & image_sets[b]:
            raise ValueError("An original image appears in two partitions")
        shared = lesion_sets[a] & lesion_sets[b]
        overlaps.append(dict(partition_a=a, partition_b=b, shared_image_ids=0,
                             shared_lesions=len(shared),
                             images_in_a_with_lesion_in_b=sum(
                                 r["split"] == a and r["lesion_id"] in shared for r in rows),
                             images_in_b_with_lesion_in_a=sum(
                                 r["split"] == b and r["lesion_id"] in shared for r in rows)))
    class_rows = [dict(diagnosis=c, label=i,
                       total=sum(r["diagnosis"] == c for r in rows),
                       **{s: sum(r["diagnosis"] == c and r["split"] == s for r in rows)
                          for s in COUNTS}) for i, c in enumerate(CLASSES)]
    train_lesions = lesion_sets["train"]
    cohorts = {s: dict(images=COUNTS[s], unique_lesions=len(lesion_sets[s]),
                      percent_of_original_images=100 * COUNTS[s] / len(rows),
                      images_with_training_lesion=sum(r["split"] == s and
                          r["lesion_id"] in train_lesions for r in rows) if s != "train" else None)
               for s in COUNTS}
    available = {}
    for folder in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        directory = RAW / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in directory.glob("*.jpg"):
            if path.stem in available:
                raise ValueError("Duplicate JPG image-ID filename")
            available[path.stem] = path
    missing = set(found) - set(available)
    if missing:
        raise ValueError(f"Missing filenames: {len(missing)}")
    return dict(study_id=STUDY, protocol=PROTOCOL, seed=SEED,
                total_images=10015, total_unique_lesions=len({r["lesion_id"] for r in rows}),
                cohorts=cohorts, class_counts=class_rows, pairwise_overlap=overlaps,
                unique_image_ids_disjoint=True, stratification="seven-class diagnosis",
                image_files_available=10015, filenames_checked_only=True,
                labels_read_for_split_and_counts_only=True, images_opened=False,
                gpu_used=False, inference_run=False, training_run=False,
                performance_scores_exist=False, overlap_allowed_not_forced=True,
                original_images_split_before_augmentation_or_sampling=True)


def checkpoint_inventory():
    records = [dict(path=relative(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
               for p in sorted((ROOT / "checkpoints").rglob("*")) if p.is_file()
               and not relative(p).startswith("checkpoints/image_level_replication/")]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return dict(files=len(records), bytes=sum(r["bytes"] for r in records),
                paths_sizes_mtimes_sha256=hashlib.sha256(payload).hexdigest())


def preservation_snapshot():
    paths = ["data/splits/split_assignments.csv",
             "data/splits/exploratory/image_level_dev_v1.csv", "data/splits/label_mapping.json",
             "results/master_experiment_registry.csv",
             "results/final_exploratory_freeze/v1/frozen_method.json",
             "research/FINAL_S83_TEST_AUDIT.md", "research/FINAL_LOCKED_TEST_RESULTS.md"]
    for folder in ("results/final_locked_test/v1", "results/final_cbam_reporting/v1",
                   "results/final_exploratory_test_audit/v1"):
        paths.extend(relative(p) for p in sorted((ROOT / folder).rglob("*")) if p.is_file())
    return dict(files={name: sha256(ROOT / name) for name in sorted(set(paths))},
                master_registry_rows_sha256={r["experiment_id"]: row_hash(r)
                    for r in read_csv(ROOT / "results/master_experiment_registry.csv")},
                checkpoint_inventory=checkpoint_inventory(),
                frozen_method_manifest_sha256=sha256(ROOT / "results/final_exploratory_freeze/v1/frozen_method.json"))


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_preserved(snapshot):
    for name, expected in snapshot["files"].items():
        if sha256(ROOT / name) != expected:
            if name == "results/master_experiment_registry.csv":
                # Later new-study entries are allowed; every historical row stays identical.
                rows = read_csv(ROOT / name)
                current = {r["experiment_id"]: r for r in rows}
                if len(current) != len(rows):
                    raise ValueError("Duplicate master-registry experiment IDs")
                baseline = snapshot["master_registry_rows_sha256"]
                if any(key not in current or row_hash(current[key]) != value
                       for key, value in baseline.items()):
                    raise ValueError("Historical master-registry row changed")
                additions = [r for key, r in current.items() if key not in baseline]
                if any(r["era"] != "structured" or r["protocol"] != PROTOCOL for r in additions):
                    raise ValueError("Unrelated master-registry additions require a separate audit")
                continue
            raise ValueError(f"Historical file changed: {name}")
    if checkpoint_inventory() != snapshot["checkpoint_inventory"]:
        raise ValueError("Historical checkpoint inventory changed")


def load_development():
    """Future runner entry point: return only train/val after frozen-hash checks."""
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if policy["study_id"] != STUDY or policy["protocol"] != PROTOCOL:
        raise ValueError("Wrong study/protocol")
    if policy["manifest"] != relative(MANIFEST) or sha256(MANIFEST) != policy["manifest_sha256"]:
        raise ValueError("Frozen manifest changed; no automatic resplitting")
    if policy["class_order"] != list(CLASSES):
        raise ValueError("Frozen class order mismatch")
    if not policy["initialization"]["fresh_external_pretrained_only"]:
        raise ValueError("Legacy checkpoint initialization is forbidden")
    rows = read_csv(MANIFEST)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if len(train) != COUNTS["train"] or len(val) != COUNTS["val"]:
        raise ValueError("Invalid development partition sizes")
    return train, val


def figures(audit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    destination = OUTPUT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    positions = np.arange(len(CLASSES))
    for offset, split in enumerate(COUNTS):
        values = [r[split] for r in audit["class_counts"]]
        ax.bar(positions + (offset - 1) * .25, values, .25, label=split)
    ax.set_xticks(positions, CLASSES)
    ax.set_ylabel("Original images (log scale)")
    ax.set_yscale("log")
    ax.set_title("New image-level 70/15/15 study: natural class counts")
    ax.legend()
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"class_counts.{suffix}", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    labels = [f"{r['partition_a']} / {r['partition_b']}" for r in audit["pairwise_overlap"]]
    values = [r["shared_lesions"] for r in audit["pairwise_overlap"]]
    bars = ax.bar(labels, values, color=["#4477AA", "#EE6677", "#228833"])
    ax.bar_label(bars, padding=4)
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("Shared lesion IDs")
    ax.set_title("Naturally occurring lesion overlap; original image IDs remain disjoint")
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"lesion_overlap.{suffix}", dpi=220)
    plt.close(fig)


def verify(require_completion=True):
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if policy["seed"] != SEED or policy["unique_original_counts"] != COUNTS:
        raise ValueError("Frozen seed or partition proportions changed")
    if policy["lesion_grouping"] or policy["augmentation_before_split"]:
        raise ValueError("Wrong image-level protocol")
    if sha256(SOURCE) != policy["metadata_sha256"]:
        raise ValueError("Source metadata changed")
    if sha256(MANIFEST) != policy["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    rows = read_csv(MANIFEST)
    originals = metadata_rows()
    audit = validate(rows, originals)
    if rows != generate(originals):
        raise ValueError("Deterministic seed-42 split reproduction failed")
    saved = json.loads((OUTPUT / "split_audit.json").read_text())
    if audit != {k: v for k, v in saved.items() if k not in ("manifest_sha256", "metadata_sha256")}:
        raise ValueError("Saved audit differs from manifest")
    snapshot = json.loads((OUTPUT / "preservation_before.json").read_text())
    verify_preserved(snapshot)
    train, val = load_development()
    test_ids = {r["image_id"] for r in rows if r["split"] == "test"}
    if any(r["image_id"] in test_ids for r in train + val):
        raise ValueError("Test original entered a development loader")
    for name in ("class_counts", "lesion_overlap"):
        for suffix in ("png", "pdf"):
            if not (OUTPUT / "figures" / f"{name}.{suffix}").is_file():
                raise ValueError("Missing protocol figure")
    for filename, expected in (("class_counts.csv", audit["class_counts"]),
                               ("lesion_overlap.csv", audit["pairwise_overlap"])):
        if read_csv(OUTPUT / filename) != [{k: str(v) for k, v in row.items()} for row in expected]:
            raise ValueError(f"Saved table changed: {filename}")
    with REGISTRY.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        with (ROOT / "results/master_experiment_registry.csv").open(encoding="utf-8", newline="") as master:
            if reader.fieldnames != next(csv.reader(master)):
                raise ValueError("New registry schema mismatch")
        registry_rows = list(reader)
    if any(r.get("era") != "structured" or r.get("protocol") != PROTOCOL for r in registry_rows):
        raise ValueError("Wrong-protocol row in the new study registry")
    if require_completion:
        completed = json.loads((OUTPUT / "completion.json").read_text())
        if completed["status"] != "phase1_completed" or completed["training_started"]:
            raise ValueError("Missing valid Phase1 completion receipt")
    return dict(status="passed", study_id=STUDY, manifest_sha256=sha256(MANIFEST),
                deterministic_reproduction=True, original_metadata_consistent=True,
                original_image_ids_disjoint=True, exact_partition_sizes=COUNTS,
                all_10015_filenames_present=True, class_order_verified=True,
                development_loader_excludes_test=True, historical_files_unchanged=True,
                historical_checkpoint_inventory_unchanged=True,
                checkpoints_in_snapshot=snapshot["checkpoint_inventory"]["files"],
                cpu_metadata_only=True, gpu_used=False, images_opened=False,
                model_metrics_not_generated=True)


def prepare():
    if POLICY.exists() or MANIFEST.exists():
        if not POLICY.exists() or not MANIFEST.exists():
            raise RuntimeError("Partial preparation exists; preserve evidence and inspect it")
        return verify()
    if OUTPUT.exists() or REGISTRY.exists():
        raise RuntimeError("Refusing to overwrite an existing study namespace")
    snapshot = preservation_snapshot()
    rows = generate(metadata_rows())
    audit = validate(rows, metadata_rows())
    write_json(OUTPUT / "preservation_before.json", snapshot)
    write_csv(MANIFEST, rows, FIELDS)
    audit.update(manifest_sha256=sha256(MANIFEST), metadata_sha256=sha256(SOURCE))
    write_json(OUTPUT / "split_audit.json", audit)
    write_csv(OUTPUT / "class_counts.csv", audit["class_counts"])
    write_csv(OUTPUT / "lesion_overlap.csv", audit["pairwise_overlap"])
    old_test = {r["image_id"] for r in read_csv(ROOT / "data/splits/split_assignments.csv") if r["split"] == "test"}
    write_json(OUTPUT / "historical_cohort_reassignment.json", dict(
        old_test_original_images=1503,
        old_test_originals_by_new_partition={s: sum(r["image_id"] in old_test and r["split"] == s for r in rows) for s in COUNTS},
        old_test_assignments_modified=False,
        new_study_uses_all_originals=True,
        prior_test_outcomes_known=True,
        new_scores_are_postdevelopment_image_level_not_pristine_external_test=True))
    policy = dict(study_id=STUDY, protocol=PROTOCOL, version=1, seed=SEED,
        created_at_utc=datetime.now(timezone.utc).isoformat(), status="phase1_prepared_no_training",
        manifest=relative(MANIFEST), manifest_sha256=sha256(MANIFEST),
        metadata_source=relative(SOURCE), metadata_sha256=sha256(SOURCE),
        class_order=list(CLASSES), unique_original_counts=COUNTS,
        split_method="canonical image_id sort; stratified holdout3006 then val/test1503; seed42 at both steps",
        python=platform.python_version(), sklearn=sklearn.__version__,
        lesion_grouping=False, lesion_overlap="allowed, measured, not optimized",
        original_image_overlap="forbidden", augmentation_before_split=False,
        validation_test_natural_class_counts=True, test_model_selection=False,
        test_evaluation="one frozen-method session after validation decisions; no test-driven tuning",
        prior_test_outcomes_known=True,
        reporting_label="Post-development image-level HAM10000 70/15/15 evaluation with lesion overlap",
        initialization=dict(fresh_external_pretrained_only=True, legacy_project_checkpoints=False,
                            legacy_fitted_models_embeddings_probabilities=False,
                            later_resume="only the same new run and frozen config"),
        balancing="training-only; exact recipe pending Phase2; original counts unaffected",
        paths=dict(research="research/image_level_replication", results="results/image_level_replication/v1",
                   checkpoints="checkpoints/image_level_replication/v1", experiment_registry=relative(REGISTRY),
                   master_registry="results/master_experiment_registry.csv"),
        next_phase="verify and prepare one paper-guided B3 adaptation; GPU approval required",
        gpu_training_authorized=False)
    write_json(POLICY, policy)
    with (ROOT / "results/master_experiment_registry.csv").open(encoding="utf-8", newline="") as stream:
        fields = next(csv.reader(stream))
    write_csv(REGISTRY, [], fields)
    figures(audit)
    result = verify(require_completion=False)
    write_json(OUTPUT / "verification.json", result)
    write_json(OUTPUT / "completion.json", dict(status="phase1_completed", verification="passed",
        completed_at_utc=datetime.now(timezone.utc).isoformat(), training_started=False,
        experiment_registry_rows=0, next_phase="Phase2 B3 recipe preparation"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = prepare() if args.prepare else verify()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
