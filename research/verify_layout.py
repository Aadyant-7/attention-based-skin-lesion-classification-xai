"""Preservation/navigation checks; phase-specific reports, no model/image execution."""
import argparse
import ast
import csv
import io
import json
import re
import subprocess
from pathlib import Path
from .common import ROOT, sha256, historical_path, write_json

# These are the only existing files intentionally edited for organization.
ORGANIZATION_EDITS = {
    '.gitattributes','.gitignore','README.md','research/FOLDER_MAP.md',
    'research/README.md','research/build.py','research/common.py',
    'research/phase2/prepare.py','research/phase2/README.md',
    'legacy/README.md','legacy/navigation/README.md','results/legacy/README.md',
}
# Explicitly authorized navigation updates for the subsequent literature/runner
# preparation. Original inventories and the cleanup audit are never rewritten.
PHASE3_EDITS = {'NEXT_STEPS.md','research/literature/README.md',
                'research/literature/incoming/README.md',
                'research/experiment.py','research/plots.py','results/master_experiment_registry.csv'}


def rows(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def verify(phase3=False,require_unlaunched=True):
    moves={r['original_path']:r for r in rows(ROOT/'legacy/organization/relocations.csv')}
    mapping=json.loads((ROOT/'legacy/path_map.json').read_text())
    for old,new in mapping.items():
        assert not (ROOT/old).exists(), f'Old root folder still exists: {old}'
        assert (ROOT/new).is_dir(), f'Archive folder missing: {new}'
    before=rows(ROOT/'legacy/organization/before_inventory.csv')
    edits=[]
    unchanged=0
    for r in before:
        original=r['path']
        path=ROOT/moves[original]['current_path'] if original in moves else ROOT/original
        assert path.is_file(), f'Missing pre-cleanup asset: {original}'
        actual=sha256(path)
        if original in ORGANIZATION_EDITS or (phase3 and original in PHASE3_EDITS):
            if actual!=r['sha256']:
                edits.append(original)
        else:
            assert actual==r['sha256'], f'Unexpected byte change: {original}'
            unchanged+=1
        if original in moves:
            assert actual==moves[original]['sha256'], f'Relocation hash changed: {original}'
    manifest=ROOT/'results/legacy/artifact_manifest.csv'
    baseline=json.loads((ROOT/'results/legacy/baseline.json').read_text())
    assert sha256(manifest)==baseline['manifest_sha256'], 'Immutable manifest changed'
    old=rows(manifest)
    for r in old:
        assert sha256(historical_path(r['path']))==r['sha256'], f'Historical bytes changed: {r["path"]}'
    registry=rows(ROOT/'results/master_experiment_registry.csv')
    if phase3 and not require_unlaunched:
        # Compare actual historical row values, not a now-changing whole-file hash.
        # Pin the clean organization commit and authenticate its bytes against
        # the existing pre-cleanup inventory (Git may normalize CSV line endings).
        original=subprocess.check_output(['git','show','028586a:results/master_experiment_registry.csv'],cwd=ROOT)
        import hashlib
        expected=next(r['sha256'] for r in before if r['path']=='results/master_experiment_registry.csv')
        restored=original.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
        assert expected in (hashlib.sha256(original).hexdigest(),hashlib.sha256(restored).hexdigest())
        original_rows=list(csv.DictReader(io.StringIO(original.decode('utf-8-sig'))))
        assert [r for r in registry if r['era']=='legacy']==[r for r in original_rows if r['era']=='legacy'], 'Historical registry rows changed'
    references=set()
    fields=('metrics_path','config_path','history_path','checkpoint','plots_dir',
            'confusion_matrix_path','split_manifest','portable_metrics_path',
            'portable_history_path','portable_config_path')
    for r in registry:
        for field in fields:
            value=r.get(field,'')
            if not value:
                continue
            values=json.loads(value) if value.startswith('[') else [value]
            for target in values:
                # Historical selectors point inside JSON/CSV, not to a new file.
                target=target.split('#',1)[0]
                assert historical_path(target).exists(), f'Broken registry reference: {target}'
                references.add(target)
    figures=rows(ROOT/'results/figures/figure_index.csv')
    # Every pre-existing figure/provenance byte was already checked above.
    for folder in ('research','src','scripts'):
        for p in (ROOT/folder).rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    docs=[ROOT/'README.md',ROOT/'NEXT_STEPS.md',ROOT/'legacy/README.md',
          ROOT/'legacy/navigation/README.md',ROOT/'legacy/organization/README.md',
          ROOT/'research/README.md',ROOT/'research/FOLDER_MAP.md']
    docs+=list((ROOT/'research/phase2').glob('*.md'))
    docs+=list((ROOT/'research/literature').glob('*.md'))
    if phase3:
        docs+=list((ROOT/'research/phase3').glob('*.md'))
        docs+=list((ROOT/'research/literature/scispace_analysis').glob('*.md'))
        docs+=list((ROOT/'research/literature/incoming').glob('*.md'))
    docs+=list((ROOT/'docs').glob('*.md'))
    for p in docs:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            link=link.strip('<>').split('#')[0]
            if not link or re.match(r'[a-z]+:',link):
                continue
            assert (p.parent/link).exists(), f'Broken Markdown link: {p.name} -> {link}'
    workbook=ROOT/'research/literature/incoming/SciSpace Literature Review.xlsx'
    assert workbook.is_file() and not (ROOT/'SciSpace Literature Review.xlsx').exists()
    if phase3:
        provenance=json.loads((ROOT/'research/literature/scispace_analysis/provenance.json').read_text())
        assert sha256(workbook)==provenance['sha256'], 'Analyzed workbook changed'
        if require_unlaunched:
            assert not any(r['era']=='structured' for r in registry), 'Preparation check expects no real run yet'
            assert not (ROOT/'checkpoints/structured/s01_efficientnet_b0_none_strict_seed42').exists()
            assert not (ROOT/'results/structured_experiments/s01_efficientnet_b0_none_strict_seed42').exists()
    report={'status':'passed','pre_cleanup_files':len(before),'unchanged_files':unchanged,
            'documented_code_navigation_edits':edits,'moved_files_byte_verified':len(moves),
            'immutable_historical_files_verified':len(old),'registry_rows':len(registry),
            'registry_references_verified':len(references),'saved_figure_packages':len(figures),
            'current_markdown_files_checked':len(docs),'python_syntax':'passed',
            'scispace_path':workbook.relative_to(ROOT).as_posix(),
            'workbook_parsed_by_this_check':False,'model_instantiated_by_this_check':False,'gpu_work':False,
            'test_images_opened':False,'historical_results_modified':False}
    scope=('phase3_preparation' if require_unlaunched else 'phase3_current') if phase3 else 'organization_cleanup'
    report['verification_scope']=scope
    filename={'phase3_preparation':'phase3_preservation_verification.json',
              'phase3_current':'phase3_current_verification.json','organization_cleanup':'organization_verification.json'}[scope]
    write_json(ROOT/'results/audit'/filename,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase3-preparation',action='store_true')
    parser.add_argument('--phase3-current',action='store_true')
    args=parser.parse_args()
    verify(args.phase3_preparation or args.phase3_current,require_unlaunched=not args.phase3_current)
