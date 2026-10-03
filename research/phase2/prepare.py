"""Validate Phase 2 and export documentation/planned configs; never executes models."""
import csv
import json
import math
from pathlib import Path
from ..common import ROOT, CLASSES, sha256, write_csv, write_json, atomic_text, historical_path


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def preserve_check():
    checked, unavailable = 0, []
    for row in read_csv(ROOT/'results/legacy/artifact_manifest.csv'):
        path = historical_path(row['path'])
        if not path.is_file():
            if row['tracked'].lower() == 'true':
                raise ValueError(f"Tracked historical file missing: {row['path']}")
            unavailable.append(row['path'])
            continue
        if sha256(path) != row['sha256']:
            raise ValueError(f"Historical evidence changed: {row['path']}")
        checked += 1
    index = ROOT/'legacy/navigation/relocation_index.csv'
    if index.exists():
        for row in read_csv(index):
            if sha256(ROOT/row['preserved_path']) != row['sha256']:
                raise ValueError('Navigation snapshot changed')
    else:
        entries = [('PROJECT_HANDOFF.md','PROJECT_HANDOFF.md','moved'),
                   ('Presentation_Handoff_Guide.md','Presentation_Handoff_Guide.md','moved'),
                   ('README.md','README_phase1.md','copied_before_replacement')]
        write_csv(index,[{'original_root_path':old,'preserved_path':f'legacy/navigation/{new}',
                         'sha256':sha256(ROOT/'legacy/navigation'/new),'action':action}
                        for old,new,action in entries])
    return {'historical_files_verified':checked,'unavailable_ignored_assets':unavailable,
            'historical_experiment_paths_moved':False,'archival_report_notebook_moves':'legacy/path_map.json',
            'navigation_snapshots_verified':3}


def verify_protocol(recipe):
    manifest=ROOT/recipe['split_manifest']
    if sha256(manifest)!=recipe['split_sha256']:
        raise ValueError('Primary manifest changed')
    rows=read_csv(manifest)
    labels=json.loads((ROOT/'data/splits/label_mapping.json').read_text())
    if labels!={c:i for i,c in enumerate(CLASSES)} or recipe['class_order']!=list(CLASSES):
        raise ValueError('Class order changed')
    if len(rows)!=10015 or len({r['image_id'] for r in rows})!=len(rows):
        raise ValueError('Wrong primary image cohort')
    if any(int(r['label'])!=labels[r['diagnosis']] for r in rows):
        raise ValueError('Labels inconsistent')
    groups={s:{r['lesion_id'] for r in rows if r['split']==s} for s in ('train','val','test')}
    counts={s:sum(r['split']==s for r in rows) for s in groups}
    if counts!={'train':7009,'val':1503,'test':1503} or any(groups[a]&groups[b] for a,b in (('train','val'),('train','test'),('val','test'))):
        raise ValueError('Wrong primary partitions or lesion overlap')
    exp=read_csv(ROOT/'data/splits/exploratory/image_level_dev_v1.csv')
    audit=json.loads((ROOT/'results/datasets/split_audit.json').read_text())
    if sha256(ROOT/audit['exploratory_image_level']['manifest'])!=audit['exploratory_image_level']['sha256']:
        raise ValueError('Exploratory manifest changed')
    if {r['image_id'] for r in exp if r['split']=='test'}!={r['image_id'] for r in rows if r['split']=='test'}:
        raise ValueError('Locked test assignments differ')
    raw=ROOT/'data/raw/HAM10000/HAM10000_metadata.csv'
    raw_verified=False
    if raw.is_file():
        raw_rows=read_csv(raw)
        expected={r['image_id']:(r['lesion_id'],r['diagnosis']) for r in rows}
        actual={r['image_id']:(r['lesion_id'],r['dx']) for r in raw_rows}
        if len(raw_rows)!=len(rows) or actual!=expected:
            raise ValueError('Raw metadata does not match primary cohort')
        raw_verified=True
    class_counts={c:sum(r['split']=='train' and r['diagnosis']==c for r in rows) for c in CLASSES}
    weights={c:math.sqrt(7009/(7*n)) for c,n in class_counts.items()}
    average=sum(weights.values())/7
    write_csv(ROOT/'results/datasets/strict_lesion_disjoint/training_weights_recipe_v1.csv',
              [{'class_index':i,'class':c,'training_images':class_counts[c],
                'weight':weights[c]/average,'formula':'sqrt(N/(K*n_c)); normalize mean1'} for i,c in enumerate(CLASSES)])
    return {'split_sha256':sha256(manifest),'images':len(rows),'lesions':len(set.union(*groups.values())),
            'partition_counts':counts,'lesion_counts':{s:len(v) for s,v in groups.items()},
            'lesion_overlap':0,'raw_metadata_verified':raw_verified,
            'test_images_opened':False,'test_metrics_computed':False}


def planned_configs(recipe):
    shortlist=read_csv(ROOT/'research/backbone_shortlist.csv')
    plans=[]
    allowed_changes={'experiment_id','question','model','weights','optional'}
    common_keys=set(recipe)
    for item in shortlist:
        model=item['model']
        config=dict(recipe,experiment_id=f"s{int(item['order']):02}_{model}_none_strict_seed42",
                    question='Which ImageNet CNN maximizes strict validation macro-F1 under recipe_v1?',
                    model=model,weights=item['weights_enum'],optional=model=='efficientnet_b2')
        if not set(config)-common_keys<=allowed_changes or any(config[k]!=recipe[k] for k in common_keys):
            raise ValueError('Unregistered backbone setting difference')
        if config['batch_size']*config['gradient_accumulation']!=config['effective_batch_size']:
            raise ValueError('Effective batch mismatch')
        path=ROOT/'research/configs/backbone_comparison'/f'{model}.json'
        if path.exists() and json.loads(path.read_text())!=config:
            raise ValueError(f'Existing planned config differs: {path}; record a recipe amendment')
        if not path.exists():
            write_json(path,config)
        plans.append({'experiment_id':config['experiment_id'],'model':model,'weights':config['weights'],
                      'added_attention':'none','recipe_version':recipe['recipe_version'],'seed':42,
                      'optional':config['optional'],'config_path':path.relative_to(ROOT).as_posix(),
                      'config_sha256':sha256(path),'status':'planned_not_launched'})
    write_csv(ROOT/'research/configs/backbone_comparison/comparison_plan.csv',plans)
    return plans


def latex(value):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','_':r'\_',
                    '#':r'\#','{':r'\{','}':r'\}'}.get(c,c) for c in str(value))


def literature():
    base=ROOT/'research/literature'
    sources=json.loads((base/'sources.json').read_text())
    if len({r['paper_id'] for r in sources})!=len(sources):
        raise ValueError('Duplicate paper ID')
    fields=next(csv.reader((base/'review_template.csv').read_text().splitlines()))
    fields+=['paper_id','study_role','metric_averaging','balanced_accuracy']
    fields=list(dict.fromkeys(fields))
    rows=[]
    for source in sources:
        if not source['source_url'].startswith('https://') or not source['verified_fields']:
            raise ValueError('Missing evidence locator')
        for k in ('accuracy','precision','recall','f1','macro_f1','balanced_accuracy'):
            if isinstance(source.get(k),(int,float)) and not 0<=source[k]<=1:
                raise ValueError('Literature metric units invalid')
        row={k:source.get(k,'Not verified') for k in fields}
        row['date_checked']='2026-10-03'
        if source['study_role'] not in ('skin classification comparison','skin attention comparison'):
            for k in ('accuracy','precision','recall','f1','macro_f1'):
                row[k]='Not applicable to HAM10000 comparison'
        rows.append(row)
    write_csv(base/'review.csv',rows,fields)
    text='# Focused literature review — verified 3 October 2026\n\n'
    text+=f'{len(sources)} primary papers, selected fields only; not an exhaustive systematic review. Numeric metrics below are **author-reported**, not independently reproduced. All ratios are 0–1. Unknowns are Not verified; method papers are not HAM10000 result rows.\n\n'
    text+='## Skin-classification evidence\n\n| Year / authors | Model / technique | Dataset / classes | Reported split / test support | Accuracy | Macro-F1 | Comparability |\n|---|---|---|---|---:|---:|---|\n'
    for r in rows:
        if r['study_role'] in ('skin classification comparison','skin attention comparison'):
            text+=f"| {r['year']} / [{r['authors']}]({r['source_url']}) | {r['backbone']} / {r['attention'] if r['attention']!='Not verified' else r['class_balancing']} | {r['dataset']} / {r['classes']} | {r['train_pct']}/{r['validation_pct']}/{r['test_pct']}%; support {r['test_count']} | {r['accuracy']} | {r['macro_f1']} | {r['limitations']} |\n"
    text+='\n## Other evaluation protocols\n\n'
    for r in rows:
        if isinstance(r['balanced_accuracy'],(int,float)):
            text+=f"[{r['authors']}, {r['year']}]({r['source_url']}): {r['dataset']}; **balanced accuracy {r['balanced_accuracy']}**, {r['metric_split']}. {r['limitations']}\n\n"
    text+='## Dataset, architecture, attention, domain pretraining and XAI references\n\n| Year | Source | Role / verified basis |\n|---|---|---|\n'
    for r in rows:
        if r['study_role'] not in ('skin classification comparison','skin attention comparison'):
            text+=f"| {r['year']} | [{r['paper_title']}]({r['source_url']}) — {r['authors']} | {r['study_role']}; {r['verified_fields']} |\n"
    text+='\nThe source-specific evidence locators, versions, weighted/macro distinctions and unresolved fields are in `review.csv` and `evidence_notes.md`. Do not convert Not verified into assumed defaults.\n'
    atomic_text(base/'review_table.md',text)
    fragment='% Requires tabularx; source-reported ratios, not project results.\n\\begin{tabularx}{\\linewidth}{lXrrX}\n\\hline\nStudy & Model & Accuracy & Macro F1 & Protocol limit \\\\\n\\hline\n'
    for r in rows:
        if isinstance(r['accuracy'],(int,float)):
            f1=r['macro_f1'] if isinstance(r['macro_f1'],(int,float)) else 'NV'
            fragment+=f"{latex(r['paper_id'])} & {latex(r['backbone'])} & {r['accuracy']} & {f1} & {latex(r['limitations'])} \\\\\n"
    fragment+='\\hline\n\\end{tabularx}\n'
    atomic_text(base/'review_table.tex',fragment)
    bib=[]
    for r in rows:
        authors=' and '.join(r['authors'].replace(' et al.',' and others').split('; '))
        doi=('  doi={'+r['doi']+'},\n') if r['doi']!='Not verified' else ''
        bib.append('@misc{'+r['paper_id']+',\n  title={'+r['paper_title']+'},\n  author={'+authors+'},\n  year={'+str(r['year'])+'},\n'+doi+'  url={'+r['source_url']+'},\n  note={Selected fields checked 2026-10-03; '+r['evidence_page_table']+'}\n}\n')
    atomic_text(base/'references.bib','\n'.join(bib))
    return len(rows)


def main():
    recipe=json.loads((ROOT/'research/phase2/recipe_v1.json').read_text())
    preservation=preserve_check()
    protocol=verify_protocol(recipe)
    plans=planned_configs(recipe)
    papers=literature()
    registry=read_csv(ROOT/'results/master_experiment_registry.csv')
    summary={'phase':2,'preservation':preservation,'protocol':protocol,
             'literature_papers_selected_fields_verified':papers,'planned_configs':len(plans),
             'initial_models':sum(r['optional'] is False for r in plans),
             'structured_result_rows':sum(r['era']=='structured' for r in registry),
             'recipe_sha256':sha256(ROOT/'research/phase2/recipe_v1.json'),
             'registry_sha256':sha256(ROOT/'results/master_experiment_registry.csv'),
             'gpu_run_launched':False,'training_launched':False,'next_gate':'CPU runner implementation and review; then announced GPU preflight'}
    write_json(ROOT/'results/audit/phase2_validation.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
