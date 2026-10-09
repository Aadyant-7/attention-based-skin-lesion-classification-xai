"""One fixed post-development S80 test audit plus validation-only explainability."""
import argparse
import gc
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from research.common import ROOT, CLASSES, relative, sha256, write_json, write_csv, atomic_text
from research.models import ResearchClassifier
from research.plots import validate_metrics, metric_figures, comparison_figures, save
from research.registry import upsert
from research.strict_protocol import SPLIT, DIGEST
from research.evaluate_final_locked_test import LockedImages
from research.final_cbam_development.protocol import CONFIG, Images
from research.final_cbam_development.runtime import require_launch_freeze, deterministic_cuda
from research.short_screening.feature_fusion import development
from research.short_screening.lesion_bag_screen import report

OUT = ROOT/'results/final_cbam_reporting/v1'
OLD = ROOT/'results/final_locked_test/v1'
S79 = ROOT/'results/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42'
COLS = [f'p_{c}' for c in CLASSES]
RID = 's81_s80_fixed_cbam_five_postdevelopment_test_audit_seed42'
CODE = [relative(__file__), 'research/final_cbam_reporting/PLAN.md', 'research/models.py',
        'src/cbam.py', 'research/plots.py', 'research/evaluate_final_locked_test.py',
        'research/short_screening/lesion_bag_screen.py', 'research/final_cbam_development/protocol.py']


def tree_hashes(folder):
    return {relative(p): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def cpu_model(member):
    saved = torch.load(ROOT/member['checkpoint'], map_location='cpu', weights_only=False)
    c = saved['config']
    if list(saved['class_order']) != list(CLASSES) or saved['best_epoch'] != member['epoch']:
        raise ValueError('Checkpoint selection/class order changed')
    model = ResearchClassifier(c['model'], weights=None, attention=c['attention'], dropout=c['head_dropout'])
    model.load_state_dict(saved['model'], strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
        raise FloatingPointError('Nonfinite frozen model')
    return model.eval()


def prepare():
    torch.set_num_threads(2)
    c, sig = require_launch_freeze()
    train, val = development()
    if sha256(ROOT/SPLIT) != DIGEST:
        raise ValueError('Original split changed')
    ids = pd.read_csv(ROOT/SPLIT, usecols=['image_id', 'lesion_id', 'split'])
    test = ids.loc[ids.split == 'test'].reset_index(drop=True)
    assert len(test) == 1503 and test.image_id.is_unique
    assert not set(test.image_id) & set(pd.concat([train, val]).image_id)
    assert not set(test.lesion_id) & set(pd.concat([train, val]).lesion_id)
    assert json.loads((OLD/'completion.json').read_text())['status'] == 'completed'
    sources = [dict(run=c['experiment_id'], checkpoint=c['checkpoints']+'/best.pt')]+sig['phase1']['ensemble']['sources'][1:]
    members = []
    for source in sources:
        path = ROOT/source['checkpoint']; z = torch.load(path, map_location='cpu', weights_only=False); cfg = z['config']
        assert cfg['split_manifest'] == c['split_manifest'] and cfg['split_sha256'] == c['split_sha256']
        assert cfg['image_size'] == 224 and cfg['normalization_mean'] == c['normalization_mean'] and cfg['normalization_std'] == c['normalization_std']
        assert list(z['class_order']) == list(CLASSES)
        m = dict(run=source['run'], model=cfg['model'], attention=cfg['attention'], pretrained_weights=cfg['weights'],
                 checkpoint=source['checkpoint'], sha256=sha256(path), epoch=z['best_epoch'], selector=z['selection_metric'], weight=.2)
        if source.get('checkpoint_sha256'): assert m['sha256'] == source['checkpoint_sha256']
        members.append(m); del z
    expected = json.loads((S79/'fixed_ensemble/summary.json').read_text())['candidate_checkpoint_sha256']
    assert members[0]['sha256'] == expected and members[0]['epoch'] == 33
    # Synthetic CPU probe only; never reads test images/labels.
    for m in members:
        model = cpu_model(m)
        with torch.inference_mode(): z = model(torch.zeros(1, 3, 224, 224))
        assert z.shape == (1, 7) and torch.isfinite(z).all()
        del model; gc.collect()
    freeze = dict(status='frozen', date='2026-10-09', method='S80 unchanged equal-five CBAM-inclusive fusion', members=members,
        class_order=list(CLASSES), image_size=224, normalization_mean=c['normalization_mean'], normalization_std=c['normalization_std'],
        preprocessing='RGB; PIL bilinear square224 with antialias; ToTensor; ImageNet normalization',
        precision='FP32', tf32=False, views=['identity'], batch_size=16, seed=42,
        test_manifest=SPLIT, test_manifest_sha256=DIGEST, development_manifest=c['split_manifest'], development_manifest_sha256=c['split_sha256'],
        test_image_ids=test.image_id.tolist(), test_lesion_count=int(test.lesion_id.nunique()),
        test_training_image_overlap=0, test_development_lesion_overlap=0,
        code_sha256={p: sha256(ROOT/p) for p in CODE}, original_test_artifacts=tree_hashes(OLD),
        original_test_report_sha256=sha256(ROOT/'research/FINAL_LOCKED_TEST_RESULTS.md'),
        validation_predictions_sha256=sha256(S79/'fixed_ensemble/validation_predictions.csv'),
        protocol='exploratory_trained_postdevelopment_test_audit', pristine_first_test=False,
        amendment='User explicitly authorized one fixed S80 audit on the previously evaluated cohort; original S31 receipt/results preserved',
        labels_read_during_preparation=False, future_test_tuning=False, xai_cohort='exploratory validation only')
    path=OUT/'frozen_audit.json'
    if path.exists():
        assert json.loads(path.read_text()) == freeze, 'Existing audit freeze changed'
    else: write_json(path, freeze)
    print('CPU preparation passed: five frozen models,1503 identities,zero training/test lesion overlap; no test labels/images loaded', flush=True)


def frozen():
    f = json.loads((OUT/'frozen_audit.json').read_text())
    assert f['code_sha256'] == {p: sha256(ROOT/p) for p in CODE}
    assert tree_hashes(OLD) == f['original_test_artifacts']
    assert sha256(ROOT/'research/FINAL_LOCKED_TEST_RESULTS.md') == f['original_test_report_sha256']
    assert sha256(ROOT/SPLIT) == f['test_manifest_sha256']
    assert sha256(ROOT/f['development_manifest']) == f['development_manifest_sha256']
    assert all(sha256(ROOT/m['checkpoint']) == m['sha256'] for m in f['members'])
    assert sha256(S79/'fixed_ensemble/validation_predictions.csv') == f['validation_predictions_sha256']
    return f


def log_setup():
    OUT.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('final_cbam_audit'); log.setLevel(logging.INFO)
    if not log.handlers:
        for h in [logging.FileHandler(OUT/'audit.log'), logging.StreamHandler()]:
            h.setFormatter(logging.Formatter('%(asctime)s %(message)s')); log.addHandler(h)
    return log


def evaluate():
    f=frozen(); log=log_setup(); tick=time.perf_counter()
    # Atomic exclusive receipt: never bypass original S31's execution protection.
    with (OUT/'inference_started.json').open('x', encoding='utf-8') as receipt:
        json.dump(dict(pid=os.getpid(), freeze_sha256=sha256(OUT/'frozen_audit.json'),
                       kind='one post-development S80 audit; not first original-test evaluation'), receipt)
    try:
        deterministic_cuda(42); assert torch.cuda.is_available()
        ids=pd.read_csv(ROOT/SPLIT, usecols=['image_id','lesion_id','split']); test=ids.loc[ids.split=='test'].reset_index(drop=True)
        assert test.image_id.tolist()==f['test_image_ids']
        dataset=LockedImages(test, f); outputs=[]
        log.info('START fixed S80 audit; FP32 identity,1503 images/member,one model resident; no training/test tuning')
        for m in f['members']:
            path=OUT/m['run']/'inference_probabilities_unscored.csv'
            if path.exists(): raise FileExistsError('Refuse repeated member inference')
            model=cpu_model(m).cuda(); rows=[]; torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for i,(x,image_ids) in enumerate(DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0,pin_memory=True)):
                    x=x.cuda(non_blocking=True); z=model(x.float()); p=z.softmax(1)
                    if not torch.isfinite(z).all() or not torch.isfinite(p).all(): raise FloatingPointError('Nonfinite test forward')
                    rows.extend(dict(image_id=image_id,**dict(zip(COLS,a.tolist()))) for image_id,a in zip(image_ids,p.cpu()))
                    if i==0 or (i+1)%30==0: log.info('%s inference batch %d/94',m['model'],i+1)
            assert [r['image_id'] for r in rows]==f['test_image_ids']
            write_csv(path,rows); outputs.append(relative(path))
            log.info('SAVED %s single pass;1503 probability vectors; peak memory %.0fMiB',m['model'],torch.cuda.max_memory_allocated()/1024**2)
            del model,x,z,p; gc.collect(); torch.cuda.empty_cache()
        write_json(OUT/'inference_completed.json',dict(status='completed',probability_files=outputs,labels_used_during_inference=False,
                   model_passes=5,images_per_model=1503,inference_seconds=time.perf_counter()-tick))
        summarize()
    except BaseException as exc:
        write_json(OUT/'failure.json',dict(error=repr(exc),repeated_inference_allowed=False)); log.exception('Preserved audit evidence'); raise


def scored_package(folder, y, p, cohort, title):
    assert p.shape==(1503,7) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    m=report(y,p); m.update(correct=int((p.argmax(1)==y).sum()), incorrect=int((p.argmax(1)!=y).sum()),
        cohort_size=len(y), evaluation_split='previously_evaluated_heldout_test', protocol='exploratory_trained_postdevelopment_test_audit',
        pristine_first_test=False, loss_definition='unweighted negative log likelihood with probability floor1e-12')
    m['weighted_f1']=sum(a['support']*a['f1'] for a in m['per_class'].values())/len(y)
    validate_metrics(m)
    frame=cohort[['image_id','lesion_id']].copy();frame['true_class']=[CLASSES[v] for v in y]
    frame['predicted_class']=[CLASSES[v] for v in p.argmax(1)];frame['correct']=p.argmax(1)==y;frame[COLS]=p
    write_json(folder/'test_metrics.json',m);write_csv(folder/'test_predictions.csv',frame.to_dict('records'))
    write_csv(folder/'test_probabilities.csv',frame[['image_id']+COLS].to_dict('records'))
    write_csv(folder/'per_class_metrics.csv',[dict(class_name=k,**v) for k,v in m['per_class'].items()])
    metric_figures(m,folder/'figures',title+' | post-development test audit')
    return m


def summarize():
    f=frozen(); done=json.loads((OUT/'inference_completed.json').read_text()); assert done['model_passes']==5
    # Parse test labels only after all five fixed probability files have been saved.
    ids=pd.read_csv(ROOT/SPLIT,usecols=['image_id','split']); excluded=set((ids.index[ids.split!='test']+1).tolist())
    cohort=pd.read_csv(ROOT/SPLIT,skiprows=lambda row:row in excluded).reset_index(drop=True)
    assert cohort.image_id.tolist()==f['test_image_ids'] and len(cohort)==1503
    assert cohort.label.tolist()==cohort.diagnosis.map(dict(zip(CLASSES,range(7)))).tolist()
    arrays=[]; rows=[]
    for path,m in zip(done['probability_files'],f['members']):
        a=pd.read_csv(ROOT/path); assert a.image_id.tolist()==f['test_image_ids']; p=a[COLS].to_numpy(dtype=np.float32);arrays.append(p)
        score=scored_package(OUT/m['run'],cohort.label.to_numpy(),p,cohort,m['model']+(' + CBAM' if m['attention']=='cbam' else ''))
        rows.append(dict(display_name=m['model']+(' + CBAM' if m['attention']=='cbam' else ''),**{k:score[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']}))
    p=np.mean(arrays,axis=0);m=scored_package(OUT/'ensemble',cohort.label.to_numpy(),p,cohort,'Fixed S80 equal-five')
    rows.append(dict(display_name='S80 equal-five + CBAM',**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']}))
    write_csv(OUT/'constituent_test_comparison.csv',rows);comparison_figures(rows,OUT/'comparison_figures','Frozen S80 constituents | post-development test audit')
    val=json.loads((S79/'fixed_ensemble/validation_metrics.json').read_text());old=json.loads((OLD/'ensemble/test_metrics.json').read_text())
    summary=dict(status='completed',accuracy=m['accuracy'],macro_f1=m['macro_f1'],macro_precision=m['macro_precision'],macro_recall=m['macro_recall'],
        weighted_f1=m['weighted_f1'],correct=m['correct'],incorrect=m['incorrect'],melanoma_recall=m['per_class']['mel']['recall'],akiec_recall=m['per_class']['akiec']['recall'],
        validation_accuracy=val['accuracy'],validation_macro_f1=val['macro_f1'],test_minus_validation_pp=100*(m['accuracy']-val['accuracy']),
        original_s31_test_accuracy=old['accuracy'],test_minus_original_s31_pp=100*(m['accuracy']-old['accuracy']),
        caveat='Post-development audit of previously evaluated held-out cohort; different models/training from S31; not pristine first-and-only test',
        method_changed_after_audit=False,original_test_artifacts_unchanged=True,checkpoint_hashes_unchanged=True)
    comparison_figures([dict(display_name='S80 exploratory validation',accuracy=val['accuracy'],macro_f1=val['macro_f1']),
        dict(display_name='S80 post-development test audit',accuracy=m['accuracy'],macro_f1=m['macro_f1']),
        dict(display_name='Original S31 strict test',accuracy=old['accuracy'],macro_f1=old['macro_f1'])],OUT/'protocol_comparison','Different cohorts/protocols | descriptive comparison')
    write_json(OUT/'summary.json',summary)
    upsert(dict(experiment_id=RID,era='structured',record_kind='fixed_ensemble_postdevelopment_test_audit',phase='final_cbam_reporting',
        protocol=f['protocol'],evaluation_split='previously_evaluated_heldout_test',split_manifest=SPLIT,split_sha256=DIGEST,
        method='Unchanged S80 equal probability fusion;FP32 identity;no fitting/test tuning',model='+'.join(a['model'] for a in f['members']),
        attention='CBAM in S79 ConvNeXt-Tiny only',ensemble_members=json.dumps([a['run'] for a in f['members']]),ensemble_weights='[0.2,0.2,0.2,0.2,0.2]',
        image_size=224,seed=42,epochs=0,status='completed',decision='descriptive_record_only_no_method_change',
        config_path=relative(OUT/'frozen_audit.json'),metrics_path=relative(OUT/'ensemble/test_metrics.json'),plots_dir=relative(OUT/'ensemble/figures'),
        notes=summary['caveat'],**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    class_table='| Class | Precision | Recall | F1 | Support |\n|---|---:|---:|---:|---:|\n'
    for c,a in m['per_class'].items(): class_table+=f"| {c} | {a['precision']:.4f} | {a['recall']:.4f} | {a['f1']:.4f} | {a['support']} |\n"
    text=f"""# S80 fixed CBAM ensemble: post-development test audit

User-authorized on9October2026, with the exact S80 method frozen before inference. This is the first S80 evaluation on this cohort, **not the first evaluation of the cohort**. Original S31 artifacts/receipt/report are preserved unchanged. Earlier test performance was already known before later development, so this score is descriptive post-development evidence, not a pristine independent final-test claim. No test-derived tuning or method change follows it.

Five frozen constituent passes,FP32 identity224RGB/ImageNet normalization; labels withheld during inference; equal0.2 probability averaging. Grad-CAM does not change inference. Exact checkpoint epochs/hashes and zero training/development-to-test image/lesion overlap are recorded in `results/final_cbam_reporting/v1/frozen_audit.json`.

- Accuracy: **{100*m['accuracy']:.4f}%**, macro precision:{m['macro_precision']:.6f}, macro recall:{m['macro_recall']:.6f}, macro-F1:**{m['macro_f1']:.6f}**, weighted-F1:{m['weighted_f1']:.6f}.
- Correct:{m['correct']}; incorrect:{m['incorrect']}; total1503. Melanoma recall:{100*m['per_class']['mel']['recall']:.2f}%; akiec recall:{100*m['per_class']['akiec']['recall']:.2f}%.
- S80 exploratory validation93.3466% /0.877017; audit minus validation:{summary['test_minus_validation_pp']:+.4f} percentage points.
- Original S31 strict test86.7598% /0.794473; different models and training protocol, so this is not a controlled treatment-effect comparison.

{class_table}
## Saved evidence

`results/final_cbam_reporting/v1/ensemble/`: metrics, predictions/probabilities, seven-class scores and raw/normalized confusion PNG/PDF. Each member's folder contains the same package from its single original pass. Comparison tables/figures, freeze, execution receipts and original-artifact hashes are in the parent folder.

## Paper wording

The retained non-CBAM S53 achieved93.6128% best exploratory validation accuracy /0.886857 macro-F1. The separately reported CBAM-inclusive S80 achieved93.3466% /0.877017 on exploratory validation. Both development scores use a repeatedly reused image-level split with within-development lesion overlap; neither is a strict independent-test claim. Do not attribute S53's score to S80 or label a highest observed validation variant as test performance.

No further training or alternative test method is authorized by this audit. Next work: reviewed XAI examples, report/paper figures and writing.
"""
    atomic_text(ROOT/'research/final_cbam_reporting/S80_POSTDEVELOPMENT_TEST_AUDIT.md',text)
    log_setup().info('COMPLETED S80 audit accuracy=%.6f macro-F1=%.6f; no method change',m['accuracy'],m['macro_f1'])


def select_cases(frame):
    selected={}; missing=[]
    groups=[]
    for c in ['mel','akiec']:
        groups.extend([(f'correct_{c}',(frame.true_class==c)&(frame.predicted_class==c)),
                       (f'incorrect_{c}',(frame.true_class==c)&(frame.predicted_class!=c))])
    groups.append(('melanoma_as_nv',(frame.true_class=='mel')&(frame.predicted_class=='nv')))
    for c in ['bcc','bkl','df','nv','vasc']: groups.append(('representative_'+c,frame.true_class==c))
    for category,mask in groups:
        candidates=frame.loc[mask].sort_values('image_id')
        if candidates.empty: missing.append(category); continue
        selected.setdefault(candidates.iloc[0].image_id,[]).append(category)
    return list(selected.items()),missing


def target_layer(model):
    if model.backbone_name=='densenet201': return model.features[0].denseblock4,'features.0.denseblock4'
    return model.features[-1],'features.'+str(len(model.features)-1)


def cam_of_probability(model,x,target,weight=.2):
    layer,name=target_layer(model); state={}; handles=[]
    def capture(_m,_i,o):
        state['activation']=o
        o.register_hook(lambda g:state.__setitem__('gradient',g))
    handles.append(layer.register_forward_hook(capture))
    if isinstance(model.attention,torch.nn.Sequential):
        for i,key in [(0,'channel'),(1,'spatial')]:
            handles.append(model.attention[i].sigmoid.register_forward_hook(lambda _m,_i,o,k=key:state.__setitem__(k,o.detach())))
    try:
        model.zero_grad(set_to_none=True);p=model(x).softmax(1);(weight*p[0,target]).backward()
        a,g=state['activation'],state['gradient']
        assert a.ndim==g.ndim==4 and a.shape==g.shape and torch.isfinite(a).all() and torch.isfinite(g).all()
        raw=torch.relu((g.mean((2,3),keepdim=True)*a).sum(1,keepdim=True))
        raw=torch.nn.functional.interpolate(raw,(224,224),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy()
        assert np.isfinite(raw).all();maximum=float(raw.max()); heat=raw/maximum if maximum>0 else raw
        attention={}
        if 'spatial' in state:
            attention['spatial']=torch.nn.functional.interpolate(state['spatial'],(224,224),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            attention['channel']=state['channel'][0,:,0,0].cpu().numpy()
            assert all(np.isfinite(v).all() and (v>=0).all() and (v<=1).all() for v in attention.values())
        return p.detach().cpu().numpy()[0],raw,heat,attention,dict(target_layer=name,activation_shape=list(a.shape),gradient_shape=list(g.shape),degenerate_map=maximum<=0)
    finally:
        for h in handles:h.remove()


def xai():
    import matplotlib.pyplot as plt
    f=frozen();log=log_setup();deterministic_cuda(42)
    if (OUT/'xai/manifest.json').exists():
        print('Completed XAI preserved; no repeated generation');return
    _,val=development();c=json.loads(CONFIG.read_text());dataset=Images(val,c,False);indices=dict(zip(val.image_id,range(len(val))))
    frame=pd.read_csv(S79/'fixed_ensemble/validation_predictions.csv');cases,missing=select_cases(frame)
    assert len(cases)<=11;rows={};data={}
    for image_id,categories in cases:
        row=frame.loc[frame.image_id==image_id].iloc[0];target=CLASSES.index(row.predicted_class)
        rows[image_id]=dict(image_id=image_id,categories=categories,true_class=row.true_class,predicted_class=row.predicted_class,
            cohort='exploratory_validation',target_class=row.predicted_class,saved_confidence=float(row[COLS[target]]),branches=[])
        data[image_id]=dict(probabilities=[],heats=[])
    for m in f['members']:
        model=cpu_model(m).cuda()
        for image_id,_ in cases:
            x,_,_=dataset[indices[image_id]];x=x.unsqueeze(0).cuda().requires_grad_(True);target=CLASSES.index(rows[image_id]['target_class'])
            p,raw,heat,attention,metadata=cam_of_probability(model,x,target)
            folder=OUT/'xai'/image_id/m['model'];folder.mkdir(parents=True,exist_ok=True)
            np.save(folder/'raw_gradcam.npy',raw);np.save(folder/'normalized_gradcam.npy',heat)
            base=np.asarray(Image.open(val.iloc[indices[image_id]].path).convert('RGB').resize((224,224),Image.Resampling.BILINEAR))/255.
            color=plt.get_cmap('magma')(heat)[...,:3];overlay=np.clip(.65*base+.35*color,0,1)
            Image.fromarray((color*255).astype('uint8')).save(folder/'heatmap.png');Image.fromarray((overlay*255).astype('uint8')).save(folder/'overlay.png')
            if attention:
                np.save(folder/'cbam_spatial_weights.npy',attention['spatial'])
                write_csv(folder/'cbam_channel_weights.csv',[dict(channel=i,weight=float(v)) for i,v in enumerate(attention['channel'])])
                fig,axes=plt.subplots(1,2,figsize=(8,3));axes[0].imshow(attention['spatial'],cmap='viridis',vmin=0,vmax=1);axes[0].set_title('Actual CBAM spatial weights');axes[0].axis('off')
                axes[1].plot(attention['channel']);axes[1].set_ylim(0,1);axes[1].set_title('Actual CBAM channel weights');save(fig,folder,'cbam_attention')
            rows[image_id]['branches'].append(dict(model=m['model'],checkpoint=m['checkpoint'],sha256=m['sha256'],weight=.2,
                target_probability=float(p[target]),**metadata))
            data[image_id]['probabilities'].append(p);data[image_id]['heats'].append(heat)
            del x; model.zero_grad(set_to_none=True)
        del model;gc.collect();torch.cuda.empty_cache();log.info('XAI branch completed %s, cases=%d',m['model'],len(cases))
    for image_id,_ in cases:
        meta=rows[image_id];p=np.mean(data[image_id]['probabilities'],axis=0);saved=frame.loc[frame.image_id==image_id,COLS].to_numpy(dtype=np.float32)[0]
        assert np.allclose(p,saved,atol=1e-5,rtol=1e-5) and CLASSES[int(p.argmax())]==meta['predicted_class']
        meta.update(recomputed_confidence=float(p.max()),max_probability_difference=float(abs(p-saved).max()),
            gradient_target='actual 0.2-weighted ensemble predicted-class probability, branch derivative',
            composite_rule='equal mean of per-branch normalized positive Grad-CAM maps; display summary only, not exact causal attribution',
            attention_note='CBAM sigmoid weights are internal attention, not Grad-CAM or segmentation; maps do not prove clinical reasoning')
        folder=OUT/'xai'/image_id;heat=np.mean(data[image_id]['heats'],axis=0);np.save(folder/'display_composite.npy',heat)
        base=np.asarray(Image.open(val.iloc[indices[image_id]].path).convert('RGB').resize((224,224),Image.Resampling.BILINEAR))/255.
        Image.fromarray((base*255).astype('uint8')).save(folder/'input_224.png')
        fig,axes=plt.subplots(1,7,figsize=(20,3));axes[0].imshow(base);axes[0].set_title('Input224')
        for ax,m in zip(axes[1:6],f['members']):
            ax.imshow(Image.open(folder/m['model']/'overlay.png'));ax.set_title(m['model']+('\n+ CBAM' if m['attention']=='cbam' else ''),fontsize=9)
        axes[-1].imshow(np.clip(.65*base+.35*plt.get_cmap('magma')(heat)[...,:3],0,1));axes[-1].set_title('Display composite',fontsize=9)
        for ax in axes:ax.axis('off')
        fig.suptitle(f"{image_id} | true={meta['true_class']} predicted={meta['predicted_class']} | S80 confidence={meta['recomputed_confidence']:.3f}",fontsize=12)
        save(fig,folder,'gradcam_panel');write_json(folder/'metadata.json',meta)
    write_json(OUT/'xai/manifest.json',dict(status='completed',cases=list(rows.values()),unavailable_categories=missing,
        selection='first lexicographic image ID per declared category; categories can share an image; no favorable-case selection',
        test_images_used=False,predictions_changed=False,member_checkpoint_hashes=[m['sha256'] for m in f['members']]))
    log.info('COMPLETED validation XAI; cases=%d; predictions unchanged',len(rows))


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    for flag in ['prepare','evaluate','report','xai']:group.add_argument('--'+flag,action='store_true')
    a=parser.parse_args()
    if a.prepare:prepare()
    elif a.evaluate:evaluate()
    elif a.report:summarize()
    else:xai()


if __name__=='__main__':main()
