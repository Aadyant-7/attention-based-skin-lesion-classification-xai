"""Read-only organization checks plus an audit report; no models/images/XLSX parsing."""
import ast
import csv
import json
import re
from pathlib import Path
from .common import ROOT, sha256, historical_path, write_json

# These are the only existing files intentionally edited for organization.
ORGANIZATION_EDITS = {
    '.gitattributes','.gitignore','README.md','research/FOLDER_MAP.md',
    'research/README.md','research/build.py','research/common.py',
    'research/phase2/prepare.py','research/phase2/README.md',
    'legacy/README.md','legacy/navigation/README.md','results/legacy/README.md',
}


def rows(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def verify():
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
        if original in ORGANIZATION_EDITS:
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
    docs+=list((ROOT/'docs').glob('*.md'))
    for p in docs:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            link=link.strip('<>').split('#')[0]
            if not link or re.match(r'[a-z]+:',link):
                continue
            assert (p.parent/link).exists(), f'Broken Markdown link: {p.name} -> {link}'
    workbook=ROOT/'research/literature/incoming/SciSpace Literature Review.xlsx'
    assert workbook.is_file() and not (ROOT/'SciSpace Literature Review.xlsx').exists()
    report={'status':'passed','pre_cleanup_files':len(before),'unchanged_files':unchanged,
            'documented_code_navigation_edits':edits,'moved_files_byte_verified':len(moves),
            'immutable_historical_files_verified':len(old),'registry_rows':len(registry),
            'registry_references_verified':len(references),'saved_figure_packages':len(figures),
            'current_markdown_files_checked':len(docs),'python_syntax':'passed',
            'scispace_path':workbook.relative_to(ROOT).as_posix(),
            'workbook_parsed':False,'model_instantiated':False,'gpu_work':False,
            'test_images_opened':False,'historical_results_modified':False}
    write_json(ROOT/'results/audit/organization_verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    verify()
