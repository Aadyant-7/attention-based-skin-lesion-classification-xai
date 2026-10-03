"""Rebuild Phase-1 indexes/tables/figures from saved evidence; never runs models."""
import argparse
import csv
import json
import re
import subprocess
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT, CLASSES, SPLITS, relative, sha256, write_csv, write_json, atomic_text
from .registry import rebuild
from .plots import validate_metrics, metric_figures, training_figures, comparison_figures, save


def dataset_tables():
    strict = pd.read_csv(ROOT/SPLITS['strict_lesion_disjoint'])
    exploratory = pd.read_csv(ROOT/SPLITS['exploratory_image_level'])
    label_map = json.loads((ROOT/'data/splits/label_mapping.json').read_text())
    frames = {'strict_lesion_disjoint':strict, 'exploratory_image_level':exploratory}
    audits = {}
    for protocol, frame in frames.items():
        if len(frame)!=10015 or frame.image_id.duplicated().any() or frame.isna().any().any():
            raise ValueError(f'Invalid manifest: {protocol}')
        if set(frame.split)!={'train','val','test'} or not (frame.label==frame.diagnosis.map(label_map)).all():
            raise ValueError(f'Invalid labels/partitions: {protocol}')
        if not strict.set_index('image_id')[['lesion_id','diagnosis','label']].sort_index().equals(frame.set_index('image_id')[['lesion_id','diagnosis','label']].sort_index()):
            raise ValueError('Manifest identities/labels changed')
        groups={s:set(frame.loc[frame.split==s,'lesion_id']) for s in ('train','val','test')}
        overlaps={f'{a}_{b}':len(groups[a]&groups[b]) for a,b in (('train','val'),('train','test'),('val','test'))}
        if protocol=='strict_lesion_disjoint' and any(overlaps.values()):
            raise ValueError('Strict lesion overlap')
        if overlaps['train_test'] or overlaps['val_test']:
            raise ValueError('Test lesion entered development')
        out=ROOT/'results/datasets'/protocol
        order=('nv','mel','bkl','bcc','akiec','vasc','df')
        rows=[]
        for c in order:
            counts={s:int(((frame.diagnosis==c)&(frame.split==s)).sum()) for s in ('train','val','test')}
            total=sum(counts.values())
            rows.append({'class':c,'total':total,'train':counts['train'],'validation':counts['val'],'test':counts['test'],
                         'share_of_dataset_pct':round(total/len(frame)*100,3)})
        rows.append({'class':'TOTAL','total':len(frame),'train':int((frame.split=='train').sum()),
                     'validation':int((frame.split=='val').sum()),'test':int((frame.split=='test').sum()),'share_of_dataset_pct':100})
        write_csv(out/'class_counts.csv',rows)
        markdown='| Class | Total | Train | Validation | Test | Dataset % |\n|---|---:|---:|---:|---:|---:|\n'
        markdown+=''.join(f"| {r['class']} | {r['total']} | {r['train']} | {r['validation']} | {r['test']} | {r['share_of_dataset_pct']} |\n" for r in rows)
        markdown+=f'\nProtocol: `{protocol}`. Source: `{SPLITS[protocol]}`. Lesion overlaps: `{overlaps}`.\n'
        markdown+='Counts use saved manifest labels only; no test images, predictions, or test metrics were loaded.\n'
        atomic_text(out/'class_counts.md',markdown)
        latex='\\begin{tabular}{lrrrr}\n\\hline\nClass & Total & Train & Validation & Test \\\\\n\\hline\n'
        latex+=''.join(f"{r['class']} & {r['total']} & {r['train']} & {r['validation']} & {r['test']} \\\\\n" for r in rows)
        latex+='\\hline\n\\end{tabular}\n'
        atomic_text(out/'class_counts_table.tex',latex)
        fig,ax=plt.subplots(figsize=(8,4))
        bottom=[0]*7
        for key in ('train','validation','test'):
            values=[r[key] for r in rows[:-1]]
            ax.bar(order,values,bottom=bottom,label=key)
            bottom=[a+b for a,b in zip(bottom,values)]
        ax.set(title=f'HAM10000 | {protocol}',ylabel='Image count')
        ax.legend()
        save(fig,out,'class_distribution')
        audits[protocol]={'manifest':SPLITS[protocol],'sha256':sha256(ROOT/SPLITS[protocol]),
                          'images':len(frame),'lesions':frame.lesion_id.nunique(), 'counts':rows[-1],
                          'lesion_overlaps':overlaps,'validation_images_with_training_lesion':int(frame.loc[frame.split=='val','lesion_id'].isin(groups['train']).sum())}
    strict_test=strict.loc[strict.split=='test'].set_index('image_id').sort_index()
    exp_test=exploratory.loc[exploratory.split=='test'].set_index('image_id').sort_index()
    if not strict_test.equals(exp_test):
        raise ValueError('Exploratory test manifest differs from locked test')
    raw=ROOT/'data/raw/HAM10000/HAM10000_metadata.csv'
    if raw.exists():
        metadata=pd.read_csv(raw)
        a=metadata.set_index('image_id')[['lesion_id','dx']].sort_index().rename(columns={'dx':'diagnosis'})
        b=strict.set_index('image_id')[['lesion_id','diagnosis']].sort_index()
        if not a.equals(b):
            raise ValueError('Saved manifest differs from original metadata')
        audits['raw_metadata_verified']=True
        audits['raw_metadata_sha256']=sha256(raw)
    else:
        audits['raw_metadata_verified']=False
    write_json(ROOT/'results/datasets/split_audit.json',audits)
    return audits


def legacy_inventory(require_local=True):
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    files=[]
    for folder in ('src','scripts','configs','results','checkpoints','docs','reports','data/splits','notebooks/legacy'):
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:
                continue
            name=relative(p)
            if name.startswith(('results/legacy/','results/structured_experiments/','results/figures/','results/datasets/','results/audit/','results/model_comparison/','results/final/','results/ablations/','results/ensembles/')) or p.name=='master_experiment_registry.csv':
                continue
            files.append({'path':name,'bytes':p.stat().st_size,'sha256':sha256(p),'tracked':name in tracked,
                          'category':folder,'preservation':'in_place'})
    manifest=ROOT/'results/legacy/artifact_manifest.csv'
    if manifest.exists():
        previous=pd.read_csv(manifest,keep_default_na=False)
        lookup={r['path']:r for r in files}
        missing=[]
        for r in previous.to_dict('records'):
            if r['path'] not in lookup:
                if require_local or str(r['tracked']).lower()=='true':
                    raise ValueError(f"Historical artifact missing: {r['path']}")
                missing.append(r['path'])
            elif r['sha256']!=lookup[r['path']]['sha256']:
                raise ValueError(f"Historical artifact changed or vanished: {r['path']}")
    else:
        missing=[]
        write_csv(manifest,files)
    baseline=ROOT/'results/legacy/baseline.json'
    if not baseline.exists():
        write_json(baseline,{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                            'manifest_sha256':sha256(manifest),'note':'Original historical paths preserved in place'})
    elif json.loads(baseline.read_text())['manifest_sha256']!=sha256(manifest):
        raise ValueError('Immutable legacy manifest changed')
    duplicates=defaultdict(list)
    for item in files:
        duplicates[item['sha256']].append(item['path'])
    duplicates={k:v for k,v in duplicates.items() if len(v)>1}
    write_json(ROOT/'results/audit/duplicate_content.json',duplicates)
    broken=[]
    for p in (ROOT/'docs').glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if target.startswith(('https:','http:','#')):
                continue
            target=target.split('#',1)[0]
            if target and not (p.parent/target).exists():
                broken.append({'document':relative(p),'target':target})
    raw_names=[]
    for part in ('HAM10000_images_part_1','HAM10000_images_part_2'):
        raw_names.extend(p.stem for p in (ROOT/'data/raw/HAM10000'/part).glob('*.jpg'))
    counts=Counter(raw_names)
    expected=set(pd.read_csv(ROOT/SPLITS['strict_lesion_disjoint']).image_id)
    audit={'snapshot_git_commit':json.loads(baseline.read_text())['git_commit'],
           'legacy_files_indexed':len(files),'legacy_bytes':sum(r['bytes'] for r in files),
           'ignored_assets_unavailable_on_this_machine':missing,
           'duplicate_content_groups':len(duplicates),'broken_markdown_links':broken,
           'raw_image_filename_count':len(raw_names),'duplicate_raw_image_ids':sum(n>1 for n in counts.values()),
           'missing_raw_image_ids':sorted(expected-set(raw_names)),
           'extra_raw_image_ids':sorted(set(raw_names)-expected),
           'raw_pixels_opened':False,'historical_paths_moved':False}
    write_json(ROOT/'results/audit/repository_audit.json',audit)
    return audit


def portable_evidence():
    """Copy small ignored run records, never checkpoints or raw imagery."""
    exports=[]
    for folder in ('results/runs','results/exploratory/runs'):
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or p.name not in ('config.json','history.csv','validation_metrics.json'):
                continue
            target=ROOT/'results/legacy/evidence'/relative(p)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha256(target)!=sha256(p):
                raise ValueError('Portable historical evidence differs: '+str(target))
            if not target.exists():
                shutil.copy2(p,target)
            exports.append({'source':relative(p),'portable_copy':relative(target),'sha256':sha256(p)})
    index=ROOT/'results/legacy/portable_evidence_index.csv'
    if index.exists():
        with index.open(encoding='utf-8',newline='') as f:
            prior=list(csv.DictReader(f))
        present={r['source'] for r in exports}
        for r in prior:
            if r['source'] not in present:
                p=ROOT/r['portable_copy']
                if not p.is_file() or sha256(p)!=r['sha256']:
                    raise ValueError('Portable historical evidence missing/changed')
                exports.append(r)
    write_csv(index,sorted(exports,key=lambda r:r['source']))


def local_assets():
    items=[]
    for folder in ((ROOT/'.cache').iterdir() if (ROOT/'.cache').is_dir() else ()):
        if folder.is_dir():
            files=[p for p in folder.rglob('*') if p.is_file()]
            items.append({'path':relative(folder),'files':len(files),'bytes':sum(p.stat().st_size for p in files),
                          'policy':'local cache/source only; no redistribution'})
    weights=[p for p in (ROOT/'.cache').glob('*.pth')]
    weights+=list((ROOT/'.cache/torch/hub/checkpoints').glob('*.pth'))
    for p in weights:
        items.append({'path':relative(p),'files':1,'bytes':p.stat().st_size,'sha256':sha256(p),
                      'policy':'local pretrained weight; no redistribution'})
    panderm=ROOT/'.cache/panderm_bb_data6_checkpoint-499.pth'
    if panderm.is_file():
        expected=json.loads((ROOT/'results/accuracy_exploration/panderm_base_strict_lp_v1/protocol.json').read_text())['weights_sha256']
        actual=next(r['sha256'] for r in items if r['path']==relative(panderm))
        if actual!=expected:
            raise ValueError('PanDerm source weight hash differs from recorded protocol')
    write_json(ROOT/'results/audit/local_only_assets.json',items)
    docs=[]
    for p in (ROOT/'results/legacy/documentation').glob('*.md'):
        docs.append({'original_root_path':p.name,'preserved_snapshot':relative(p),'sha256':sha256(p)})
    if docs:
        write_csv(ROOT/'results/legacy/documentation/snapshot_index.csv',docs)


def evidence_figures(rows):
    selected=[
        ('Weighted B0 + CBAM','results/runs/efficientnet_b0_cbam_weighted_v1/validation_metrics.json','strict_lesion_disjoint','results/runs/efficientnet_b0_cbam_weighted_v1/history.csv'),
        ('Oversampled B0 + CBAM','results/runs/efficientnet_b0_cbam_oversampled_v1/validation_metrics.json','strict_lesion_disjoint','results/runs/efficientnet_b0_cbam_oversampled_v1/history.csv'),
        ('Focal B0 + CBAM','results/runs/efficientnet_b0_cbam_focal_v1/validation_metrics.json','strict_lesion_disjoint','results/runs/efficientnet_b0_cbam_focal_v1/history.csv'),
        ('PanDerm two-epoch pilot','results/accuracy_exploration/panderm_base_last2_pilot_v1/best_validation_metrics.json','strict_lesion_disjoint','results/accuracy_exploration/panderm_base_last2_pilot_v1/history.csv'),
        ('PanDerm/B0 accuracy leader','results/accuracy_exploration/panderm_base_strict_tta_v1/prior_four_with_svc_tta_equal.json','strict_lesion_disjoint',''),
        ('PanDerm/B0 macro F1 leader','results/accuracy_exploration/panderm_base_strict_tta_v1/weighted_b0_plus_svc_tta_two_equal.json','strict_lesion_disjoint',''),
        ('Image-level weighted B0','results/exploratory/image_level_weighted_b0_cbam_v1_summary.json','exploratory_image_level','results/exploratory/image_level_weighted_b0_cbam_v1_history.csv'),
        ('Image-level stronger augmentation','results/exploratory/image_level_strong_aug_b0_cbam_v1_summary.json','exploratory_image_level','results/exploratory/image_level_strong_aug_b0_cbam_v1_history.csv'),
        ('Four B0 resolution streams','results/exploratory/multires_ensemble/four_equal.json','exploratory_image_level',''),
        ('Eight B0/PanDerm streams','results/exploratory/panderm_base_image_level_tta_v1/b0_four_plus_four_svc_views_eight_equal.json','exploratory_image_level','')]
    tables=defaultdict(list)
    index=[]
    for name, source, protocol, history in selected:
        path=ROOT/source
        portable_source=''
        if not path.exists() and (ROOT/'results/legacy/evidence'/source).exists():
            path=ROOT/'results/legacy/evidence'/source
            portable_source=relative(path)
        if not path.exists():
            index.append({'source':source,'status':'unavailable_local','protocol':protocol})
            continue
        obj=json.loads(path.read_text(encoding='utf-8'))
        metrics=obj.get('best_validation',obj)
        safe=Path(source).parent.name if 'validation_metrics' in source else Path(source).stem
        out=ROOT/'results/figures/legacy'/protocol/safe
        title=f'{name}\nHistorical {protocol} validation'
        metric_figures(metrics,out,title)
        history_path=ROOT/history if history else None
        if history_path and not history_path.exists():
            history_path=ROOT/'results/legacy/evidence'/history
        if history_path and history_path.exists():
            training_figures(pd.read_csv(history_path),out,title)
        write_json(out/'provenance.json',{'source':source,'source_sha256':sha256(path),
                   'selector':'best_validation' if 'best_validation' in obj else '', 'protocol':protocol,
                   'class_order':CLASSES,'evaluation_split':'validation','history':history,
                   'portable_source':portable_source,
                   'note':'Generated from saved evidence; no model inference or training. Not held-out test results.'})
        item={'display_name':name,'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1'],
              'macro_precision':metrics['macro_precision'],'macro_recall':metrics['macro_recall'],
              'protocol':protocol,'metrics_path':source,'plots_dir':relative(out)}
        tables[protocol].append(item)
        index.append({'source':source,'protocol':protocol,'plots_dir':relative(out),'status':'generated'})
        for row in rows:
            if row['metrics_path']==source:
                row['plots_dir']=relative(out)
    for protocol, values in tables.items():
        comparison_figures(values,ROOT/'results/model_comparison/legacy'/protocol,
                           f'Historical {protocol} validation\nMethods differ; not a controlled backbone comparison')
    write_csv(ROOT/'results/figures/figure_index.csv',index)
    return rows


def environment():
    import torch, torchvision, sklearn, matplotlib
    from torchvision.models import get_model_weights
    info={'python':__import__('sys').version,'torch':torch.__version__,'torchvision':torchvision.__version__,
          'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__,
          'cuda_available':torch.cuda.is_available(), 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          'vram_gib':round(torch.cuda.get_device_properties(0).total_memory/2**30,2) if torch.cuda.is_available() else None}
    write_json(ROOT/'results/audit/environment.json',info)
    atomic_text(ROOT/'results/audit/environment_freeze.txt',subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True))
    names=('efficientnet_b0','mobilenet_v3_large','resnet50','densenet121','convnext_tiny','efficientnet_b2')
    rows=[]
    for i,name in enumerate(names):
        weight=get_model_weights(name).DEFAULT
        rows.append({'order':i+1,'model':name,'weights_enum':str(weight),'original_imagenet_parameters':weight.meta.get('num_params',''),
                     'weight_url':weight.url,'status':'proposed_not_trained_in_structured_phase',
                     'note':'Original 1000-class model metadata, not seven-class head parameters or HAM10000 accuracy'})
    write_csv(ROOT/'research/backbone_shortlist.csv',rows)
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-figures',action='store_true')
    parser.add_argument('--portable',action='store_true',help='On a clone, allow missing ignored assets; retain historical rows and use verified small evidence copies')
    args=parser.parse_args()
    for folder in ('results/structured_experiments','results/ablations','results/ensembles','results/final','checkpoints/structured','research/configs'):
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    audit=legacy_inventory(require_local=not args.portable)
    portable_evidence()
    local_assets()
    data=dataset_tables()
    env=environment()
    rows=rebuild()
    # Verify every imported full matrix before generating highlighted artifacts.
    full=0
    for row in rows:
        if row['era']!='legacy' or not row['metrics_path'].endswith('.json'):
            continue
        source=ROOT/row['metrics_path']
        if not source.exists() and row.get('portable_metrics_path'):
            source=ROOT/row['portable_metrics_path']
        if not source.exists():
            continue
        obj=json.loads(source.read_text())
        if row['source_selector']=='best_validation':
            obj=obj['best_validation']
        if 'per_class' in obj and 'confusion_matrix' in obj:
            validate_metrics(obj)
            full+=1
    if not args.skip_figures:
        rows=evidence_figures(rows)
        from .registry import locked, read_registry, FIELDS
        with locked():
            rows=[r for r in rows if r['era']=='legacy']+[r for r in read_registry() if r['era']!='legacy']
            write_csv(ROOT/'results/master_experiment_registry.csv',sorted(rows,key=lambda r:r['experiment_id']),FIELDS)
    summary={'legacy_registry_rows':sum(r['era']=='legacy' for r in rows),
             'structured_registry_rows':sum(r['era']=='structured' for r in rows),
             'verified_full_metric_records':full,'evidence_build_complete':True,
             'historical_assets_unchanged':True,'test_images_opened':False,'training_launched':False,
             'protocol_counts':dict(Counter(r['protocol'] for r in rows)),
             'missing_runtime_training_rows':[r['experiment_id'] for r in rows if r['record_kind']=='training_run' and not r['runtime_seconds']]}
    write_json(ROOT/'results/audit/phase1_build_summary.json',summary)
    print(json.dumps({'audit':audit,'dataset':data,'environment':env,'registry':summary},indent=2))


if __name__=='__main__':
    main()
