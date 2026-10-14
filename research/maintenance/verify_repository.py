"""Verify portable repository evidence with only the Python standard library.

Run ``python -B -m research.maintenance.verify_repository`` from a checkout or
an extracted source archive. No Git metadata, scientific packages, raw images,
checkpoints or caches are required. This reads source/text and hashes saved
artifacts; it never imports project runtime code, decodes images or scores data.
There are no default writes. ``--output`` may name a new JSON file inside the
repository, in an existing directory; existing files are never overwritten.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = "results/master_experiment_registry.csv"
PATH_FIELDS = (
    "split_manifest", "checkpoint", "config_path", "history_path",
    "metrics_path", "portable_metrics_path", "portable_history_path",
    "portable_config_path", "confusion_matrix_path", "plots_dir",
)
S83_FREEZE = "results/final_exploratory_freeze/v1/frozen_method.json"
S83_STATUS = "results/final_exploratory_freeze/v1/audit_status.json"
S83_ID = "s83_cbam_f1_addition_equal_six_exploratory_seed42"
XAI_DIR = "results/final_s83_xai/v1"
LOCAL_WORKBOOK = "research/literature/incoming/SciSpace Literature Review.xlsx"
WORKBOOK_PROVENANCE = "research/literature/scispace_analysis/provenance.json"
LOCAL_XAI_LOG_RECEIPTS = {
    "results/final_locked_test/v1/evaluation.log",
    "results/final_exploratory_test_audit/v1/audit.log",
    "results/final_exploratory_test_audit/v1/stderr.log",
    "results/final_exploratory_test_audit/v1/stdout.log",
}
EXCLUDED_DIRECTORIES = {
    ".git", ".cache", ".venv", "__pycache__", ".pytest_cache",
    "private_reference", "draft_v1", "draft_v2",
}
ACTIVE_DOCUMENTS = (
    "README.md", "docs/README.md", "docs/SETUP.md", "research/README.md",
    "research/FOLDER_MAP.md", "research/STUDY_INDEX.md",
    "research/phase2/README.md", "research/literature/README.md",
    "research/maintenance/README.md", "reporting/README.md",
    "reporting/report/README.md", "reporting/paper/README.md",
    "reporting/presentation/README.md", "results/README.md",
)


class AuditError(ValueError):
    """An actionable repository validation failure."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def inside(value, base=None):
    """Resolve a relative path, including symlinks, without escaping ROOT."""
    require(isinstance(value, str) and bool(value.strip()), "Empty or non-string path")
    value = value.replace("\\", "/")
    require(not Path(value).is_absolute() and not PureWindowsPath(value).drive,
            "Absolute paths are not portable: " + value)
    candidate = ((ROOT if base is None else base) / value).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise AuditError("Path escapes the repository: " + value) from exc
    return candidate


def local_category(path):
    """Only these scientific-reference roots are intentionally unavailable."""
    parts = path.relative_to(ROOT).parts
    if parts and parts[0] == "checkpoints":
        return "checkpoints"
    if parts and parts[0] == ".cache":
        return "cache"
    if parts[:2] == ("data", "raw"):
        return "raw_data"
    return None


def private_target(path):
    parts = path.relative_to(ROOT).parts
    return (bool(set(parts) & {"private_reference", "draft_v1", "draft_v2"})
            or (parts[:3] == ("research", "literature", "incoming")
                and path.suffix.lower() == ".xlsx"))


def read_json(path):
    require(path.is_file(), "Missing JSON file: " + relative(path))
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise AuditError("Invalid JSON in " + relative(path) + ": " + str(exc)) from exc


def read_csv(path):
    require(path.is_file(), "Missing CSV file: " + relative(path))
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(bool(reader.fieldnames), "Missing CSV header: " + relative(path))
        require(len(reader.fieldnames) == len(set(reader.fieldnames)),
                "Duplicate CSV columns: " + relative(path))
        rows = list(reader)
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            "Malformed CSV row: " + relative(path))
    return rows


def sha256(path):
    require(path.is_file(), "Missing hashed artifact: " + relative(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_matches(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            "Invalid SHA-256 for " + relative(path))
    actual = sha256(path)
    require(actual == expected,
            "SHA-256 mismatch: " + relative(path) + " (expected " + expected + ", got " + actual + ")")


def python_syntax():
    count = 0
    for name in ("src", "scripts", "research"):
        directory = inside(name)
        require(directory.is_dir(), "Missing Python source directory: " + name)
        for current, children, files in os.walk(directory, followlinks=False):
            children[:] = sorted(p for p in children if p not in EXCLUDED_DIRECTORIES)
            for filename in sorted(files):
                if not filename.endswith(".py"):
                    continue
                path = inside((Path(current) / filename).relative_to(ROOT).as_posix())
                try:
                    ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative(path))
                except SyntaxError as exc:
                    raise AuditError("Python syntax error: " + relative(path)
                                     + ":" + str(exc.lineno) + ": " + exc.msg) from exc
                count += 1
    return {"python_files_parsed": count, "runtime_modules_imported": 0}


def registry_references():
    mapping = read_json(inside("legacy/path_map.json"))
    require(isinstance(mapping, dict), "legacy/path_map.json must be a path mapping")
    moves = []
    for old, new in mapping.items():
        old_path, new_path = inside(old), inside(new)
        moves.append((relative(old_path), relative(new_path)))
    moves.sort(key=lambda item: (-len(item[0]), item[0]))
    mirrors = {}
    for entry in read_csv(inside("results/legacy/portable_evidence_index.csv")):
        original, copy = inside(entry["source"]), inside(entry["portable_copy"])
        require(relative(copy).startswith("results/legacy/evidence/"),
                "Portable mirror is outside its recorded evidence directory: " + relative(copy))
        require(not local_category(original) and not private_target(original),
                "Portable mirror source is a local-only scientific asset: " + relative(original))
        key = (relative(original), relative(copy))
        require(key not in mirrors, "Duplicate portable mirror index entry: " + relative(copy))
        hash_matches(copy, entry["sha256"])
        mirrors[key] = entry["sha256"]
    rows = read_csv(inside(REGISTRY))
    require(len(rows) == 600, "Master registry must contain 600 records; found " + str(len(rows)))
    ids = [row.get("experiment_id", "") for row in rows]
    require(all(ids), "Master registry contains an empty experiment_id")
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    require(not duplicates, "Duplicate registry experiment_id: " + ", ".join(duplicates))
    skipped = Counter({"checkpoints": 0, "cache": 0, "raw_data": 0})
    fallbacks = Counter()
    checked = remapped = selectors = 0
    for row in rows:
        for field in PATH_FIELDS:
            require(field in row, "Registry is missing column: " + field)
            value = row[field]
            if not value:
                continue
            values = json.loads(value) if value.startswith("[") else [value]
            require(isinstance(values, list) and all(isinstance(v, str) for v in values),
                    "Invalid registry path list: " + row["experiment_id"] + " / " + field)
            for value in values:
                raw, separator, selector = value.partition("#")
                path = inside(raw)
                category = local_category(path)
                if category:
                    skipped[category] += 1
                    continue
                logical = relative(path)
                for old, new in moves:
                    if logical == old or logical.startswith(old + "/"):
                        path = inside(new + logical[len(old):])
                        remapped += 1
                        break
                require(not private_target(path), "Registry references private content: " + logical)
                used_fallback = False
                if not path.exists():
                    portable_field = {
                        "config_path": "portable_config_path",
                        "history_path": "portable_history_path",
                        "metrics_path": "portable_metrics_path",
                    }.get(field)
                    # Legacy confusion matrices can live inside the same metrics
                    # JSON. Only that exact recorded alias can share its mirror.
                    if field == "confusion_matrix_path" and value == row["metrics_path"]:
                        portable_field = "portable_metrics_path"
                    fallback = row.get(portable_field, "") if portable_field else ""
                    require(bool(fallback) and not fallback.startswith("["),
                            "Missing scientific registry reference without a recorded portable mirror: "
                            + row["experiment_id"] + " / " + field + " -> " + relative(path))
                    fallback_raw, fallback_separator, fallback_selector = fallback.partition("#")
                    portable = inside(fallback_raw)
                    require(not local_category(portable) and not private_target(portable),
                            "Portable fallback targets a local-only asset: " + fallback)
                    mirror_key = (relative(path), relative(portable))
                    require(mirror_key in mirrors,
                            "Portable fallback has no matching authenticated mirror index entry: " + fallback)
                    expected_selector = selector
                    if field in ("metrics_path", "confusion_matrix_path"):
                        expected_selector = expected_selector or row.get("source_selector", "")
                        require(row["source_sha256"] == mirrors[mirror_key],
                                "Portable metrics hash disagrees with the original registry source: " + fallback)
                    require(not fallback_separator or fallback_selector == expected_selector,
                            "Portable fallback selector disagrees with its source: " + fallback)
                    path = portable
                    selector = expected_selector
                    separator = bool(selector)
                    used_fallback = True
                    fallbacks[field] += 1
                require(path.exists(), "Missing scientific registry reference: "
                        + row["experiment_id"] + " / " + field + " -> " + relative(path))
                if separator:
                    if selector == "best_validation":
                        require(selector in read_json(path), "Missing JSON selector: " + value)
                    elif re.fullmatch(r"csv_row:[1-9][0-9]*", selector):
                        require(int(selector.split(":")[1]) <= len(read_csv(path)),
                                "CSV row selector exceeds file length: " + value)
                    else:
                        raise AuditError("Unknown registry selector: " + value)
                    selectors += 1
                if used_fallback and field == "confusion_matrix_path":
                    metric_data = read_json(path)
                    if selector == "best_validation":
                        metric_data = metric_data[selector]
                    require(isinstance(metric_data, dict)
                            and ("confusion_matrix" in metric_data or "confusion_matrix_json" in metric_data),
                            "Portable metrics do not contain the referenced confusion matrix: " + relative(path))
                checked += 1
    return {"records": len(rows), "unique_experiment_ids": len(set(ids)),
            "scientific_references_verified": checked, "historical_references_remapped": remapped,
            "selectors_verified": selectors, "local_references_skipped": dict(skipped),
            "portable_mirror_hashes_verified": len(mirrors),
            "portable_fallback_references_verified": sum(fallbacks.values()),
            "portable_fallbacks_by_field": dict(sorted(fallbacks.items()))}


def reporting_sources():
    rows = read_csv(inside("reporting/evidence/source_manifest.csv"))
    require(bool(rows), "Reporting source manifest is empty")
    hashed = workbook_receipts = 0
    for row in rows:
        path = inside(row["source_path"])
        if relative(path) == LOCAL_WORKBOOK:
            # The original workbook is explicitly private/local-only. Authenticate
            # its recorded digest against tracked provenance without opening it.
            provenance = read_json(inside(WORKBOOK_PROVENANCE))
            require(provenance.get("source_path") == LOCAL_WORKBOOK,
                    "SciSpace provenance references an unexpected workbook")
            require(re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]),
                    "Invalid source-manifest SHA-256 for the local SciSpace workbook")
            require(provenance.get("sha256") == row["source_sha256"],
                    "Local SciSpace workbook digest disagrees with tracked provenance")
            workbook_receipts += 1
            continue
        require(not local_category(path) and not private_target(path),
                "Reporting source manifest references a local-only artifact: " + relative(path))
        hash_matches(path, row["source_sha256"])
        hashed += 1
    require(workbook_receipts == 1,
            "Reporting manifest must contain exactly one original local SciSpace workbook receipt")
    return {"manifest_rows": len(rows), "source_hashes_verified": hashed,
            "local_only_workbook_receipts_verified": workbook_receipts,
            "local_only_workbook_path": LOCAL_WORKBOOK,
            "workbook_provenance_path": WORKBOOK_PROVENANCE,
            "original_workbook_bytes_read": False}


def s83_freeze():
    status = read_json(inside(S83_STATUS))
    require(status.get("status") == "completed_performance_work_finished",
            "S83 audit_status.json does not report completed performance work")
    require(status.get("freeze_manifest_unchanged") is True
            and status.get("no_further_training_or_test_tuning") is True,
            "S83 audit status no longer records the completed frozen method")
    hash_matches(inside(S83_FREEZE), status["freeze_sha256"])
    freeze = read_json(inside(S83_FREEZE))
    require(freeze.get("source_experiment") == S83_ID, "Unexpected S83 source experiment")
    hash_matches(inside(freeze["split_manifest"]), freeze["split_sha256"])
    hash_matches(inside(freeze["validation_predictions"]), freeze["validation_prediction_sha256"])
    require(inside(freeze["validation_metrics"]).is_file(), "Missing S83 validation metrics")
    require(inside(status["audit_results"]).is_dir(), "Missing saved S83 audit results directory")
    sources = freeze.get("sources", [])
    require(len(sources) == 6 and len({s["run"] for s in sources}) == 6,
            "S83 freeze must contain six distinct members")
    for source in sources:
        hash_matches(inside(source["prediction_file"]), source["prediction_sha256"])
        require(local_category(inside(source["checkpoint"])) == "checkpoints",
                "S83 member checkpoint is outside the documented local checkpoint root")
        require(re.fullmatch(r"[0-9a-f]{64}", source["checkpoint_sha256"]),
                "Invalid saved S83 checkpoint hash: " + source["run"])
    caches = freeze.get("fp32_validation_cache_provenance", [])
    for cache in caches:
        require(local_category(inside(cache["path"])) == "cache",
                "S83 cache provenance is outside the documented local cache root")
        require(re.fullmatch(r"[0-9a-f]{64}", cache["sha256"]), "Invalid saved S83 cache hash")
    return {"freeze_status": status["status"], "freeze_and_split_hashes_verified": 2,
            "validation_prediction_hashes_verified": 1 + len(sources),
            "checkpoint_binaries_skipped": len(sources), "cache_assets_skipped": len(caches)}


def xai_artifacts():
    manifest = read_json(inside(XAI_DIR + "/manifest.json"))
    require(manifest.get("status") == "completed", "S83 XAI manifest is not completed")
    require(manifest.get("source_experiment") == S83_ID, "Unexpected S83 XAI source experiment")
    cases = manifest.get("cases", [])
    require(len(cases) == 12 and len({case["image_id"] for case in cases}) == 12,
            "S83 XAI manifest must contain twelve distinct cases")
    hash_matches(inside(XAI_DIR + "/plan.json"), manifest["plan_sha256"])
    plan = read_json(inside(XAI_DIR + "/plan.json"))
    require(plan.get("freeze_path") == S83_FREEZE, "S83 XAI plan references an unexpected freeze")
    hash_matches(inside(plan["freeze_path"]), plan["freeze_sha256"])
    code_hashes = plan.get("code_sha256")
    require(isinstance(code_hashes, dict) and len(code_hashes) == 8,
            "S83 XAI plan must record eight source-code hashes")
    for name, expected in sorted(code_hashes.items()):
        path = inside(name)
        require(not local_category(path) and not private_target(path),
                "XAI source-code ledger references a local-only asset: " + name)
        hash_matches(path, expected)
    preserved = plan.get("preserved_artifact_sha256")
    require(isinstance(preserved, dict) and len(preserved) == 495,
            "S83 XAI plan must record 495 preserved artifact receipts")
    log_receipts = set()
    preserved_hashes = 0
    for name, expected in sorted(preserved.items()):
        path = inside(name)
        if name in LOCAL_XAI_LOG_RECEIPTS:
            require(re.fullmatch(r"[0-9a-f]{64}", expected),
                    "Invalid SHA-256 in local XAI log receipt: " + name)
            log_receipts.add(name)
            continue
        require(not local_category(path) and not private_target(path),
                "XAI preservation ledger references a local-only artifact: " + name)
        hash_matches(path, expected)
        preserved_hashes += 1
    require(log_receipts == LOCAL_XAI_LOG_RECEIPTS,
            "S83 XAI preservation ledger must retain the four original local log receipts")
    artifacts = manifest.get("artifact_sha256")
    require(isinstance(artifacts, dict) and bool(artifacts), "S83 XAI artifact hash ledger is empty")
    for name, expected in sorted(artifacts.items()):
        path = inside(name)
        require(relative(path).startswith(XAI_DIR + "/"), "XAI artifact escapes its package: " + name)
        hash_matches(path, expected)
    verification_path = inside(manifest["verification_path"])
    require(relative(verification_path) == XAI_DIR + "/verification.json",
            "Unexpected S83 XAI verification path")
    verification = read_json(verification_path)
    require(verification.get("status") == "passed" and verification.get("cases") == 12
            and verification.get("branch_gradcams") == 72 and verification.get("label_changes") == 0,
            "S83 XAI verification receipt no longer records its completed 12-case/72-map result")
    return {"completion_status": manifest["status"], "cases": len(cases),
            "artifact_hashes_verified": len(artifacts), "plan_and_freeze_hashes_verified": 2,
            "source_code_hashes_verified": len(code_hashes),
            "preserved_artifact_hashes_verified": preserved_hashes,
            "local_only_log_receipts_verified": len(log_receipts),
            "local_only_log_paths": sorted(log_receipts), "local_log_bytes_read": False,
            "saved_verification_status": verification["status"]}


def unfenced(text):
    lines = []
    fence_character = None
    fence_length = 0
    for line in text.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence_character is None:
                fence_character, fence_length = token[0], len(token)
            elif token[0] == fence_character and len(token) >= fence_length:
                fence_character = None
            continue
        if fence_character is None:
            lines.append(line)
    return "\n".join(lines)


def markdown_targets(text):
    """Read inline and reference destinations without interpreting Markdown."""
    text = re.sub(r"(`+).*?\1", "", unfenced(text))
    for match in re.finditer(r"!?\[[^\]\n]*\]\(\s*", text):
        start = match.end()
        if start < len(text) and text[start] == "<":
            end = text.find(">", start + 1)
            if end >= 0:
                yield text[start + 1:end]
            continue
        value = []
        depth = 0
        cursor = start
        while cursor < len(text):
            char = text[cursor]
            if char == "\\" and cursor + 1 < len(text):
                cursor += 1
                value.append(text[cursor])
            elif char == "(":
                depth += 1
                value.append(char)
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
                value.append(char)
            elif char.isspace() and depth == 0:
                break
            else:
                value.append(char)
            cursor += 1
        if value:
            yield "".join(value)
    for match in re.finditer(r"^\s{0,3}\[[^\]\n]+\]:\s*(<[^>]+>|\S+)", text, re.M):
        yield match.group(1).strip("<>")


def local_link_target(value, source):
    value = value.strip()
    if not value or value.startswith("#") or value.startswith("//"):
        return None
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return None
    if not parsed.path:
        return None
    path = inside(unquote(parsed.path), source.parent)
    require(not local_category(path) and not private_target(path),
            "Active link targets a local-only asset: " + relative(source) + " -> " + value)
    require(path.exists(), "Broken local link: " + relative(source) + " -> " + value)
    return path


def navigation_links():
    paths = {inside(name) for name in ACTIVE_DOCUMENTS}
    paths.update(ROOT.glob("*.md"))
    for name in ("docs/project", "reporting", "reporting/evidence"):
        directory = inside(name)
        require(directory.is_dir(), "Missing active documentation directory: " + name)
        paths.update(directory.glob("*.md"))
    count = external = directories = 0
    for path in sorted(paths):
        path = inside(relative(path))
        require(path.is_file(), "Missing active documentation file: " + relative(path))
        for target in markdown_targets(path.read_text(encoding="utf-8-sig")):
            resolved = local_link_target(target, path)
            if resolved is None:
                external += 1
            else:
                count += 1
                directories += int(resolved.is_dir())
    return {"markdown_files_checked": len(paths), "local_links_verified": count,
            "directory_links_verified": directories, "external_links_and_anchors_skipped": external}


class GalleryResources(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("src", "href") and value is not None:
                self.targets.append(value)

    handle_startendtag = handle_starttag


def gallery_resources():
    path = inside(XAI_DIR + "/gallery.html")
    require(path.is_file(), "Missing S83 gallery HTML")
    parser = GalleryResources()
    parser.feed(path.read_text(encoding="utf-8-sig"))
    parser.close()
    checked = skipped = 0
    for target in parser.targets:
        if local_link_target(target, path) is None:
            skipped += 1
        else:
            checked += 1
    require(checked > 0, "S83 gallery contains no local src/href resources")
    return {"local_src_href_resources_verified": checked,
            "external_links_and_anchors_skipped": skipped, "images_decoded": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", metavar="NEW_RELATIVE_JSON_PATH",
                        help="Write the JSON report to a new file inside the repository (existing parent required)")
    args = parser.parse_args(argv)
    report = {"status": "passed", "scope": "portable_repository_evidence", "checks": {}, "errors": [],
              "scientific_runtime_imported": False, "training": False, "inference": False,
              "image_decoding": False, "test_scoring": False, "default_writes": False}
    if not sys.dont_write_bytecode:
        report["errors"].append("Run Python with -B to avoid writing bytecode")
    checks = (
        ("python_syntax", python_syntax), ("master_registry", registry_references),
        ("reporting_sources", reporting_sources), ("s83_freeze", s83_freeze),
        ("s83_xai", xai_artifacts), ("active_navigation", navigation_links),
        ("gallery_resources", gallery_resources),
    )
    for name, check in checks:
        try:
            report["checks"][name] = check()
        except (AuditError, OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
            message = name + ": " + str(exc)
            report["errors"].append(message)
            report["checks"][name] = {"status": "failed", "error": str(exc)}
    output = None
    if args.output:
        try:
            output = inside(args.output)
            require(output.suffix.lower() == ".json", "--output must have a .json extension")
            require(output.parent.is_dir(), "--output parent directory does not exist")
            require(not output.exists(), "--output refuses to overwrite an existing file: " + relative(output))
            require(not local_category(output) and not private_target(output),
                    "--output must not target a local-only asset directory")
        except (AuditError, OSError, ValueError) as exc:
            report["errors"].append("output: " + str(exc))
            output = None
    if report["errors"]:
        report["status"] = "failed"
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        try:
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
        except OSError as exc:
            report["status"] = "failed"
            report["errors"].append("output: " + str(exc))
            serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(serialized, end="")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
