"""Prepare one deterministic ten-fold image-level protocol using metadata only."""
import argparse
import csv
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone

import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

from research.common import CLASSES, ROOT, sha256, write_csv, write_json
from research.image_level_replication.prepare_protocol import (
    checkpoint_inventory, metadata_rows, read_csv, relative, row_hash)

STUDY = "image_level_replication_v2_k10"
PROTOCOL = "postdevelopment_image_level_k10_inner_validation_overlap_allowed"
SEED = 42
FOLDS = 10
INNER_FRACTION = .1
FIELDS = ["image_id", "lesion_id", "diagnosis", "label", "split"]
DATA = ROOT / "data/splits/image_level_replication/v2"
POLICY = ROOT / "research/image_level_replication/v2/protocol.json"
RESULTS = ROOT / "results/image_level_replication/v2"
OUTPUT = RESULTS / "protocol"
REGISTRY = RESULTS / "experiment_registry.csv"
SOURCE = ROOT / "data/raw/HAM10000/HAM10000_metadata.csv"
PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def generate(originals):
    labels = [r["diagnosis"] for r in originals]
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    manifests, assignment = {}, {}
    for fold, (development, assessment) in enumerate(splitter.split(range(len(originals)), labels)):
        inner_size = math.ceil(len(development) * INNER_FRACTION)
        train, val = train_test_split(
            development.tolist(), test_size=inner_size,
            stratify=[labels[i] for i in development], random_state=SEED + fold)
        memberships = {i: split for split, ids in
                       (("train", train), ("val", val), ("test", assessment)) for i in ids}
        manifests[fold] = [dict(row, split=memberships[i]) for i, row in enumerate(originals)]
        assignment.update({int(i): fold for i in assessment})
    if len(assignment) != len(originals):
        raise ValueError("Outer folds do not cover every original")
    outer = [dict(row, outer_test_fold=assignment[i]) for i, row in enumerate(originals)]
    return outer, manifests


def audit_manifest(fold, rows, originals):
    if len(rows) != 10015 or len({r["image_id"] for r in rows}) != 10015:
        raise ValueError(f"Duplicate/missing originals in fold {fold}")
    found = {r["image_id"]: {k: str(r[k]) for k in FIELDS[:-1]} for r in rows}
    if found != {r["image_id"]: r for r in originals}:
        raise ValueError("Fold metadata/labels differ from original metadata")
    counts = Counter(r["split"] for r in rows)
    assessment_size = 1002 if fold < 5 else 1001
    expected = dict(train=10015-assessment_size-902, val=902, test=assessment_size)
    if dict(counts) != expected:
        raise ValueError(f"Wrong inner/outer counts in fold {fold}")
    image_sets = {s: {r["image_id"] for r in rows if r["split"] == s} for s in expected}
    lesion_sets = {s: {r["lesion_id"] for r in rows if r["split"] == s} for s in expected}
    overlap = []
    for a, b in PAIRS:
        if image_sets[a] & image_sets[b]:
            raise ValueError("Original-image overlap within one fold")
        shared = lesion_sets[a] & lesion_sets[b]
        overlap.append(dict(fold=fold, partition_a=a, partition_b=b,
                            shared_image_ids=0, shared_lesions=len(shared),
                            images_in_a_with_lesion_in_b=sum(
                                r["split"] == a and r["lesion_id"] in shared for r in rows),
                            images_in_b_with_lesion_in_a=sum(
                                r["split"] == b and r["lesion_id"] in shared for r in rows)))
    classes = [dict(fold=fold, diagnosis=c, label=i,
                    **{s: sum(r["diagnosis"] == c and r["split"] == s for r in rows)
                       for s in expected}) for i, c in enumerate(CLASSES)]
    for row in classes:
        if min(row[s] for s in expected) < 1:
            raise ValueError("A diagnostic class is absent from a fold partition")
    return dict(fold=fold, counts=expected,
                proportions_percent={s: 100*expected[s]/10015 for s in expected},
                outer_development_images=expected["train"]+expected["val"],
                unique_lesions={s: len(lesion_sets[s]) for s in expected},
                images_with_training_lesion={s: sum(r["split"] == s and
                    r["lesion_id"] in lesion_sets["train"] for r in rows) for s in ("val", "test")},
                class_counts=classes, pairwise_overlap=overlap)


def image_inventory():
    seen = set()
    for name in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        folder = SOURCE.parent / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in folder.glob("*.jpg"):
            if path.stem in seen:
                raise ValueError("Duplicate original JPG filenames")
            seen.add(path.stem)
    return seen


def preservation_snapshot():
    paths = set(json.loads((ROOT / "results/image_level_replication/v1/protocol/preservation_before.json").read_text())["files"])
    for folder in ("data/splits/image_level_replication/v1", "results/image_level_replication/v1",
                   "research/image_level_replication"):
        for path in (ROOT / folder).rglob("*"):
            name = relative(path)
            if (path.is_file() and "__pycache__" not in path.parts
                    and not name.startswith("research/image_level_replication/v2/")):
                paths.add(name)
    return dict(files={name: sha256(ROOT/name) for name in sorted(paths)},
                master_registry_rows_sha256={r["experiment_id"]: row_hash(r)
                    for r in read_csv(ROOT/"results/master_experiment_registry.csv")},
                historical_checkpoint_inventory=checkpoint_inventory())


def verify_preserved(snapshot):
    for name, digest in snapshot["files"].items():
        if sha256(ROOT/name) == digest:
            continue
        if name != "results/master_experiment_registry.csv":
            raise ValueError(f"Historical/V1 file changed: {name}")
        rows = read_csv(ROOT/name)
        current = {r["experiment_id"]: r for r in rows}
        baseline = snapshot["master_registry_rows_sha256"]
        if len(current) != len(rows) or any(
                key not in current or row_hash(current[key]) != value for key, value in baseline.items()):
            raise ValueError("Historical master-registry row changed")
        if any(r["era"] != "structured" or r["protocol"] != PROTOCOL
               for key, r in current.items() if key not in baseline):
            raise ValueError("Unrelated registry additions require separate verification")
    if checkpoint_inventory() != snapshot["historical_checkpoint_inventory"]:
        raise ValueError("Historical checkpoint inventory changed")


def check_policy(policy):
    if (policy["study_id"] != STUDY or policy["protocol"] != PROTOCOL or
        policy["seed"] != SEED or policy["outer_folds"] != FOLDS or
        policy["inner_validation_fraction_of_outer_development"] != INNER_FRACTION or
        policy["class_order"] != list(CLASSES)):
        raise ValueError("Frozen protocol identity/seed/folding/class order changed")
    if policy["lesion_grouping"] or policy["augmentation_before_split"]:
        raise ValueError("Wrong image-level split rules")
    if not policy["fresh_external_pretrained_only"] or policy["outer_assessment_used_for_selection"]:
        raise ValueError("Invalid initialization/outer-assessment selection rules")
    if policy["initial_development_fold"] != 0:
        raise ValueError("Cannot choose a more favorable pilot fold")


def frozen_rows(fold):
    if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < FOLDS:
        raise ValueError("Fold must be an integer from 0 to 9")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    check_policy(policy)
    entry = policy["fold_manifests"][str(fold)]
    path = DATA / f"fold_{fold:02d}.csv"
    if entry["path"] != relative(path) or sha256(path) != entry["sha256"]:
        raise ValueError("Frozen fold manifest changed; no automatic resplitting")
    return read_csv(path)


def load_development(fold=0):
    """Return inner train/validation metadata only; never an outer-assessment loader."""
    rows = frozen_rows(fold)
    return ([r for r in rows if r["split"] == "train"],
            [r for r in rows if r["split"] == "val"])


def make_figures(audits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder = OUTPUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    x = np.arange(FOLDS)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    bottom = np.zeros(FOLDS)
    for key, label, color in (("train", "Training", "#4477AA"),
                              ("val", "Inner validation", "#228833"),
                              ("test", "Outer assessment", "#EE6677")):
        values = np.array([r["counts"][key] for r in audits])
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set_xticks(x)
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Original images")
    ax.set_title("K10-inspired protocol: approximately 81/9/10 per fold")
    ax.legend(loc="lower right")
    for suffix in ("png", "pdf"):
        fig.savefig(folder/f"fold_counts.{suffix}", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    values = [r["images_with_training_lesion"]["test"] for r in audits]
    bars = ax.bar(x, values, color="#EE6677")
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(values)*1.16)
    ax.set_xticks(x)
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Outer-assessment images sharing a training lesion")
    ax.set_title("Naturally occurring lesion overlap; original image IDs are disjoint")
    for suffix in ("png", "pdf"):
        fig.savefig(folder/f"assessment_lesion_overlap.{suffix}", dpi=220)
    plt.close(fig)


def verify(require_completion=True):
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    check_policy(policy)
    if policy["metadata_sha256"] != sha256(SOURCE):
        raise ValueError("Source metadata changed")
    original = metadata_rows()
    outer, expected = generate(original)
    outer_path = DATA / "outer_folds.csv"
    if sha256(outer_path) != policy["outer_assignments_sha256"]:
        raise ValueError("Frozen outer assignments changed")
    if read_csv(outer_path) != [{k: str(v) for k, v in r.items()} for r in outer]:
        raise ValueError("Outer assignments fail seed-42 reproduction")
    audits, all_assessment = [], []
    for fold in range(FOLDS):
        rows = frozen_rows(fold)
        if rows != expected[fold]:
            raise ValueError(f"Fold {fold} fails deterministic reconstruction")
        audits.append(audit_manifest(fold, rows, original))
        train, val = load_development(fold)
        assessment_ids = {r["image_id"] for r in rows if r["split"] == "test"}
        if any(r["image_id"] in assessment_ids for r in train+val):
            raise ValueError("Outer-assessment original enters development")
        all_assessment.extend(assessment_ids)
    if len(all_assessment) != 10015 or len(set(all_assessment)) != 10015:
        raise ValueError("Each original must be outer-assessed exactly once")
    if set(r["image_id"] for r in original) - image_inventory():
        raise ValueError("Missing original image files")
    saved = json.loads((OUTPUT/"fold_audit.json").read_text())
    if saved["folds"] != audits:
        raise ValueError("Saved fold audit differs from manifests")
    for filename, rows in (("fold_counts.csv", [dict(fold=r["fold"], **r["counts"],
                outer_development_images=r["outer_development_images"]) for r in audits]),
            ("class_counts.csv", [c for r in audits for c in r["class_counts"]]),
            ("lesion_overlap.csv", [p for r in audits for p in r["pairwise_overlap"]])):
        if read_csv(OUTPUT/filename) != [{k: str(v) for k, v in r.items()} for r in rows]:
            raise ValueError(f"Saved protocol table changed: {filename}")
    with REGISTRY.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        with (ROOT/"results/master_experiment_registry.csv").open(encoding="utf-8", newline="") as master:
            if reader.fieldnames != next(csv.reader(master)):
                raise ValueError("V2 registry header mismatch")
        registry_rows = list(reader)
        if len({r.get("experiment_id") for r in registry_rows}) != len(registry_rows):
            raise ValueError("Duplicate V2 registry experiment IDs")
        if any(not r.get("experiment_id") or r.get("era") != "structured"
               or r.get("protocol") != PROTOCOL for r in registry_rows):
            raise ValueError("Wrong-protocol run in V2 registry")
    for figure in ("fold_counts", "assessment_lesion_overlap"):
        for extension in ("png", "pdf"):
            if not (OUTPUT/"figures"/f"{figure}.{extension}").is_file():
                raise ValueError("Missing protocol figure")
    snapshot = json.loads((OUTPUT/"preservation_before.json").read_text())
    verify_preserved(snapshot)
    if require_completion:
        completed = json.loads((OUTPUT/"completion.json").read_text())
        if completed["status"] != "phase1_v2_completed" or completed["training_started"]:
            raise ValueError("Missing valid V2 preparation receipt")
    return dict(status="passed", study_id=STUDY, folds_verified=10,
                original_images=10015, each_original_in_outer_assessment_exactly_once=True,
                original_image_ids_disjoint_within_folds=True, all_classes_present_in_every_partition=True,
                canonical_seed42_reproduction=True, manifest_hashes_verified=True,
                inner_loaders_exclude_outer_assessment=True, all_original_filenames_present=True,
                historical_and_v1_files_unchanged=True,
                protected_historical_files=len(snapshot["files"]),
                protected_historical_registry_rows=len(snapshot["master_registry_rows_sha256"]),
                historical_checkpoint_inventory_unchanged=True,
                checkpoint_files_in_snapshot=snapshot["historical_checkpoint_inventory"]["files"],
                cpu_metadata_only=True, images_opened=False, gpu_used=False,
                model_training_or_inference=False, accuracy_results_exist=False)


def prepare():
    if POLICY.exists() or DATA.exists() or RESULTS.exists():
        if POLICY.exists() and (DATA/"outer_folds.csv").exists():
            return verify()
        raise RuntimeError("Partial V2 preparation exists; preserve and inspect it")
    snapshot = preservation_snapshot()
    original = metadata_rows()
    outer, manifests = generate(original)
    audits = [audit_manifest(fold, manifests[fold], original) for fold in range(FOLDS)]
    if set(r["image_id"] for r in original) - image_inventory():
        raise ValueError("Missing original image files")
    write_json(OUTPUT/"preservation_before.json", snapshot)
    write_csv(DATA/"outer_folds.csv", outer, FIELDS[:-1]+["outer_test_fold"])
    entries = {}
    for fold, rows in manifests.items():
        path = DATA/f"fold_{fold:02d}.csv"
        write_csv(path, rows, FIELDS)
        entries[str(fold)] = dict(path=relative(path), sha256=sha256(path))
    write_json(OUTPUT/"fold_audit.json", dict(study_id=STUDY, folds=audits,
        labels_read_for_stratification_and_counts_only=True, images_opened=False,
        seed_search=False, lesion_overlap_allowed_not_forced=True,
        model_predictions_or_metrics_generated=False))
    write_csv(OUTPUT/"fold_counts.csv", [dict(fold=r["fold"], **r["counts"],
        outer_development_images=r["outer_development_images"]) for r in audits])
    write_csv(OUTPUT/"class_counts.csv", [c for r in audits for c in r["class_counts"]])
    write_csv(OUTPUT/"lesion_overlap.csv", [p for r in audits for p in r["pairwise_overlap"]])
    with (ROOT/"results/master_experiment_registry.csv").open(encoding="utf-8", newline="") as stream:
        fields = next(csv.reader(stream))
    write_csv(REGISTRY, [], fields)
    policy = dict(study_id=STUDY, protocol=PROTOCOL, version=2, status="phase1_prepared_no_training",
        created_at_utc=datetime.now(timezone.utc).isoformat(), primary_paper="DermAI 1.0 (2023)",
        paper_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10573070/",
        paper_evaluation_basis="K10 in sections3.6/4.5; conflicting80/20 in3.3; exact IDs unavailable",
        exact_paper_reproduction=False, seed=SEED, outer_folds=FOLDS, initial_development_fold=0,
        inner_validation_fraction_of_outer_development=INNER_FRACTION,
        inner_seed_rule="42 + outer_fold", class_order=list(CLASSES),
        metadata_source=relative(SOURCE), metadata_sha256=sha256(SOURCE),
        outer_assignments=relative(DATA/"outer_folds.csv"),
        outer_assignments_sha256=sha256(DATA/"outer_folds.csv"), fold_manifests=entries,
        split_algorithm="Canonical image-ID order; StratifiedKFold10 shuffle seed42; stratified inner10%",
        python=platform.python_version(), sklearn=sklearn.__version__,
        initial_fit_fraction_approx=.81, inner_validation_fraction_approx=.09,
        outer_assessment_fraction_approx=.10, optional_outer90_refit="requires later approval; fresh external weights, inner-selected fixed epoch budget, no outer selection",
        lesion_grouping=False, lesion_overlap="natural, permitted and measured; not optimized",
        original_image_overlap_within_fold=False, augmentation_before_split=False,
        sampling_augmentation_scope="current fold training originals only",
        outer_assessment_natural_class_proportions=True, fresh_external_pretrained_only=True,
        legacy_project_checkpoints_or_fitted_predictions_allowed=False,
        outer_assessment_used_for_selection=False, pilot_uses_inner_validation_only=True,
        automatic_ten_fold_training_authorized=False, gpu_training_authorized=False,
        evaluation=dict(checkpoint_and_ensemble_decisions="inner validation only",
            one_outer_session_per_frozen_fold=True,
            ensemble_members="same outer-fold models only; never average all ten fold checkpoints for OOF",
            original_oof_predictions_required=10015,
            report="pooled OOF accuracy/macro-F1 plus all fold scores and mean/std",
            full_k10_claim_requires_all_ten_completed_folds=True,
            best_fold_selection=False, previous_test_outcomes_known=True,
            label="Post-development internal image-level K10-inspired evaluation with lesion overlap"),
        preserved_v1="research/image_level_replication/protocol_v1.json",
        paths=dict(results=relative(RESULTS), experiment_registry=relative(REGISTRY),
                   future_checkpoints="checkpoints/image_level_replication/v2"),
        next_phase="Prepare one plain-B3 adaptation and an inner-fold0 GPU proposal; no outer scoring")
    write_json(POLICY, policy)
    make_figures(audits)
    result = verify(require_completion=False)
    write_json(OUTPUT/"verification.json", result)
    write_json(OUTPUT/"completion.json", dict(status="phase1_v2_completed", training_started=False,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), model_experiments=0,
        next_phase="Phase2 plain-B3 recipe preparation"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare() if args.prepare else verify(), indent=2))


if __name__ == "__main__":
    main()
