"""Read-only cleanup gate. No training, inference, image decoding or writes.

Run with the existing project Python and -B; --full-hash authenticates every
preserved non-environment file against the fresh pre-cleanup SHA-256 snapshot.
The unchanged environment is checked against its snapshot metadata; its hash
ledger came from the verified migration. The local baseline is intentionally
not duplicated into Git. --portable skips that receipt; the existing local
evidence, pretrained assets and matching environment are still required.
"""
import argparse
import ast
import concurrent.futures
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / '.cache/cleanup/20261011'
LEDGER = ROOT / 'legacy/organization/cleanup_2026-10-11'
BASELINE_SHA = '1eaf30ad7aa197e23f93912fc14584a5639e3abc235f066181dbeeb9894050be'
NAVIGATION_EDITS = {
    'README.md', 'NEXT_STEPS.md', 'research/README.md', 'research/FOLDER_MAP.md',
    'results/structured_experiments/README.md', 'results/ablations/README.md',
    'results/ensembles/README.md', 'results/final/README.md',
    'results/xai/structured/README.md',
    'docs/private_reference/report_guidance_2026-10-09/REFERENCE_NOTES.md',
}
NEW_FILES = {
    'docs/README.md', 'research/STUDY_INDEX.md', 'results/README.md',
    'research/maintenance/README.md', 'research/maintenance/verify_organization.py',
    'legacy/maintenance/README.md', 'legacy/maintenance/.gitattributes',
}
NEW_PREFIXES = (
    '.cache/cleanup/20261011/', 'reporting/',
    'legacy/navigation/cleanup_2026-10-11/',
    'legacy/organization/cleanup_2026-10-11/',
    'docs/private_reference/report_guidance_2026-10-09/navigation_before_cleanup_2026-10-11/',
)
PATH_FIELDS = ('split_manifest', 'checkpoint', 'config_path', 'history_path',
               'metrics_path', 'portable_metrics_path', 'portable_history_path',
               'portable_config_path', 'confusion_matrix_path', 'plots_dir')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def inside(relative):
    p = (ROOT / relative).resolve()
    p.relative_to(ROOT)
    return p


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def files():
    for folder, subdirs, names in os.walk(ROOT):
        subdirs[:] = [s for s in subdirs if s != '.git']
        for name in names:
            yield (Path(folder) / name).relative_to(ROOT).as_posix()


def preservation(full_hash):
    baseline_file = RECEIPTS / 'before_manifest.json'
    assert sha256(baseline_file) == BASELINE_SHA, 'Local baseline receipt changed'
    rows = json.loads(baseline_file.read_text(encoding='utf-8'))
    baseline = {r['path']: r for r in rows}
    moves = read_csv(LEDGER / 'relocations.csv')
    removed = read_csv(LEDGER / 'deletions.csv')
    archives = json.loads((RECEIPTS / 'archives.json').read_text(encoding='utf-8'))
    assert len(moves) == 10 and len(removed) == 233 and len(archives) == 10
    relocation = {r['original_path']: r['current_path'] for r in moves}
    deleted = {r['path'] for r in removed}
    for r in removed:
        assert not inside(r['path']).exists(), 'Approved disposable file remains'
        assert r['sha256'] == baseline[r['path']]['sha256']
    for r in moves:
        assert not inside(r['original_path']).exists(), 'Original move path remains'
        assert sha256(inside(r['current_path'])) == baseline[r['original_path']]['sha256'] == r['sha256']
    assert {r['original_path'] for r in archives} == NAVIGATION_EDITS
    for r in archives:
        assert sha256(inside(r['archived_path'])) == baseline[r['original_path']]['sha256'] == r['sha256']

    preserved = [r for r in rows if r['path'] not in deleted | NAVIGATION_EDITS | relocation.keys()]

    def verify(row):
        path = inside(row['path'])
        assert path.is_file(), 'Missing preserved file: ' + row['path']
        st = path.stat()
        assert (st.st_size, st.st_mtime_ns) == (row['bytes'], row['mtime_ns']), 'Preserved metadata changed: ' + row['path']
        content_check = not row['path'].startswith('.venv/') and (full_hash or row['bytes'] <= 1024 * 1024)
        if content_check:
            assert sha256(path) == row['sha256'], 'Preserved bytes changed: ' + row['path']
        return content_check

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hashed = sum(pool.map(verify, preserved))
    additions = set(files()) - baseline.keys()
    unexpected = [p for p in additions if p not in NEW_FILES and p not in relocation.values() and not p.startswith(NEW_PREFIXES)]
    assert not unexpected, 'Unplanned new files: ' + repr(unexpected[:10])
    return dict(baseline_files=len(rows), preserved_files=len(preserved), fresh_content_checks=hashed,
                preserved_checkpoints=sum(r['path'].startswith('checkpoints/') for r in preserved),
                environment_metadata_checks=sum(r['path'].startswith('.venv/') for r in preserved),
                archives_verified=len(archives), moves_verified=len(moves), approved_deletions_verified=len(removed),
                baseline_sha256=BASELINE_SHA, full_hash=full_hash)


def registry_references():
    from research.common import historical_path
    rows = read_csv(ROOT / 'results/master_experiment_registry.csv')
    count = 0
    for row in rows:
        for field in PATH_FIELDS:
            value = row[field]
            if not value:
                continue
            values = json.loads(value) if value.startswith('[') else [value]
            for value in values:
                path, _, selector = value.partition('#')
                target = historical_path(path)
                assert target.exists(), (row['experiment_id'], field, value)
                if selector == 'best_validation':
                    assert selector in json.loads(target.read_text(encoding='utf-8'))
                elif selector.startswith('csv_row:'):
                    assert 1 <= int(selector.split(':')[1]) <= len(read_csv(target))
                elif selector:
                    raise ValueError('Unknown registry selector: ' + selector)
                count += 1
    return dict(rows=len(rows), resolved_references=count)


def syntax_and_imports():
    parsed = 0
    for folder in ('src', 'scripts', 'research'):
        for path in (ROOT / folder).rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            parsed += 1
    ast.parse((ROOT / 'legacy/maintenance/prepare_strict_source_generator.py').read_text(encoding='utf-8-sig'))
    import torch
    assert not torch.cuda.is_initialized()
    modules = ['src.cbam', 'src.models', 'src.data', 'src.gradcam',
               'research.common', 'research.models', 'research.registry',
               'research.train', 'research.strict_train', 'research.aggressive.core',
               'research.phase3.close_s02', 'research.final_cbam_development.runtime',
               'research.final_cbam_development.artifacts',
               'research.convnext_base_original.core',
               'research.image_level_replication.v2.convnext_core']
    for name in modules:
        importlib.import_module(name)
    signatures = {}
    for key, name in [('R202', modules[-1]), ('R203', modules[-2])]:
        mod = importlib.import_module(name)
        _, sig = mod.config_signature()
        assert sig == json.loads(mod.FREEZE.read_text(encoding='utf-8'))['signature'], key + ' freeze changed'
        signatures[key] = 'unchanged'
    from research.common import ROOT as runtime_root
    assert runtime_root == ROOT
    assert not torch.cuda.is_initialized()
    return dict(python_files_parsed=parsed, archived_generator_parsed_not_executed=True,
                cpu_imports=len(modules), frozen_signatures=signatures, cuda_initialized=False)


def evidence_manifest():
    rows = read_csv(ROOT / 'reporting/evidence/source_manifest.csv')
    assert rows
    for row in rows:
        path = inside(row['source_path'])
        assert path.is_file(), 'Manifest source missing: ' + row['source_path']
        assert sha256(path) == row['source_sha256'], 'Manifest source hash mismatch: ' + row['source_path']
    return dict(source_files=len(rows), hashes_verified=True)


def navigation_links():
    paths = [ROOT / p for p in NAVIGATION_EDITS | NEW_FILES if p.endswith('.md')]
    paths += list((ROOT / 'reporting').rglob('*.md'))
    paths += [LEDGER / 'README.md', LEDGER / 'VERIFICATION.md']
    checked = 0
    for path in paths:
        if not path.exists():
            continue
        content = re.sub(r'```.*?```', '', path.read_text(encoding='utf-8'), flags=re.S)
        for match in re.finditer(r'!?\[[^\]]*\]\((<[^>]+>|[^)]+)\)', content):
            target = match.group(1).strip()
            target = target[1:-1] if target.startswith('<') else target.split(' "')[0]
            if re.match(r'^(https?://|mailto:|#|codex:)', target):
                continue
            target = unquote(target.split('#')[0])
            assert (path.parent / target).resolve().exists(), 'Broken navigation link: ' + str(path.relative_to(ROOT)) + ' -> ' + target
            checked += 1
    return dict(local_links_checked=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-hash', action='store_true')
    parser.add_argument('--portable', action='store_true', help='Skip the local-only cleanup baseline; check links, registry, manifest, imports and freezes')
    args = parser.parse_args()
    assert sys.dont_write_bytecode, 'Run Python with -B to keep bytecode cleanup intact'
    sys.path.insert(0, str(ROOT))
    results = dict(status='passed', scope='organization_only', training=False, inference=False)
    if not args.portable:
        results['preservation'] = preservation(args.full_hash)
    results['runtime'] = syntax_and_imports()
    results['registry'] = registry_references()
    results['reporting'] = evidence_manifest()
    results['navigation'] = navigation_links()
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
