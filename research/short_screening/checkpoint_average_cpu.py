"""S75: one same-run ConvNeXt-Tiny weight average; bounded CPU inference only."""
import argparse
import hashlib
import json
import logging
import os
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from research.common import ROOT, CLASSES, relative, sha256, write_json, write_csv
from research.models import ResearchClassifier
from research.plots import metric_figures, comparison_figures, validate_metrics
from research.registry import upsert
from research.strict_train import DevelopmentImages
from research.short_screening.current_ensemble_tta import development
from research.short_screening.lesion_bag_screen import report

OUT = ROOT/'results/short_screening/s75_convnext_checkpoint_average_cpu'
CKPT = ROOT/'checkpoints/short_screening/s75_convnext_checkpoint_average_cpu/averaged.pt'
SOURCE = ROOT/'checkpoints/structured/s06_convnext_tiny_none_exploratory_seed42'
S53 = ROOT/'results/short_screening/s53_equal_five_b0_addition'
COLS = [f'p_{c}' for c in CLASSES]
PLAN = dict(
    question='Does one fixed average of the preserved late same-run ConvNeXt-Tiny states improve its predictions and the unchanged S53 fusion?',
    states=['best.pt', 'best_macro_f1.pt', 'latest.pt'], expected_epochs=[17,18,20],
    averaging='Equal weights over unique model-tensor states; float64 accumulation then original FP32 dtype',
    scope='Post-hoc same-run checkpoint averaging, NOT a reproduction of constant/cyclic-SGD SWA training',
    source_reference='https://arxiv.org/abs/1803.05407',
    limitation='Snapshots were previously selected on reused development validation; not independent evidence of generalization',
    precision='FP32 CPU; original epoch18 checkpoint rerun as matched control',
    preprocessing='Original square224 RGB bilinear antialias resize and ImageNet normalization; identity only',
    ensemble='Replace only Tiny probabilities in S53; other four cached probabilities unchanged; all five weights0.20',
    gate='Ensemble >=8 net additional correct predictions and >=0.005 accuracy vs matched control; nondecreasing macro-F1/MEL recall vs both control and saved S53; also >=0.005 accuracy vs S53',
    search='Exactly one average and one original-checkpoint control; no subset/weight/checkpoint/view tuning',
    cpu_threads=4, batch_size=8, max_inference_seconds=600,
    training=False, gpu_used=False, test_loaded=False)


def tensor_hash(state):
    h=hashlib.sha256()
    for k,t in sorted(state.items()):
        h.update(k.encode()); h.update(str((t.dtype,tuple(t.shape))).encode())
        h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def model_for(state,c):
    net=ResearchClassifier(c['model'],None,c['attention'],c['head_dropout']).eval().float()
    net.load_state_dict(state,strict=True)
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in net.modules())
    return net


def prepare():
    torch.set_num_threads(4); OUT.mkdir(parents=True,exist_ok=True)
    plan=OUT/'PREDECLARED_PLAN.json'
    if plan.exists(): assert json.loads(plan.read_text())==PLAN
    else: write_json(plan,PLAN)
    if (OUT/'preflight.json').exists():
        saved=json.loads((OUT/'preflight.json').read_text())
        assert all(sha256(ROOT/p)==h for p,h in saved['source_hashes'].items())
        assert sha256(CKPT)==saved['averaged_checkpoint_sha256']
        assert sha256(__file__)==saved['runner_sha256']
        return saved
    source_hashes={};states=[];epochs=[];configs=[];digests=[]
    for name,epoch in zip(PLAN['states'],PLAN['expected_epochs']):
        path=SOURCE/name; source_hashes[relative(path)]=sha256(path)
        # Trusted local training checkpoints include optimizer/RNG objects.
        raw=torch.load(path,map_location='cpu',weights_only=False)
        committed=int(raw['history'][-1]['epoch']) if name=='latest.pt' else raw['best_epoch']
        assert committed==epoch
        c=raw['config']; assert c['class_order']==list(CLASSES)
        assert c['model']=='convnext_tiny' and c['attention']=='none' and c['image_size']==224
        state=raw['model']; assert all(torch.isfinite(t).all() for t in state.values())
        digest=tensor_hash(state)
        if digest not in digests: states.append(state);digests.append(digest);epochs.append(epoch)
        configs.append(c);del raw
    assert all(c==configs[0] for c in configs) and len(states)==3
    c=configs[0]; assert all(set(s)==set(states[0]) for s in states)
    averaged={}; delta_square=norm_square=0.
    for k,t in states[0].items():
        assert all(s[k].dtype==t.dtype and s[k].shape==t.shape for s in states)
        if t.is_floating_point():
            averaged[k]=sum(s[k].double() for s in states).div(len(states)).to(t.dtype)
            delta_square+=float((averaged[k].double()-states[1][k].double()).square().sum())
            norm_square+=float(states[1][k].double().square().sum())
        else:
            assert all(torch.equal(s[k],t) for s in states); averaged[k]=t.clone()
    net=model_for(averaged,c)
    with torch.inference_mode():
        z=net(torch.zeros(2,3,224,224)); assert z.shape==(2,7) and torch.isfinite(z).all()
    CKPT.parent.mkdir(parents=True,exist_ok=True)
    assert not CKPT.exists()
    fd,temp=tempfile.mkstemp(dir=CKPT.parent,suffix='.tmp');os.close(fd)
    try:
        torch.save(dict(model=averaged,config=c,class_order=list(CLASSES),source_epochs=epochs,
                        source_hashes=source_hashes,averaging=PLAN['averaging']),temp)
        os.replace(temp,CKPT)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    saved=dict(status='passed',source_hashes=source_hashes,epochs=epochs,unique_model_states=3,
               model_tensor_sha256=digests,all_configs_equal=True,batchnorm_layers=0,finite_cpu_probe=True,
               relative_l2_change_from_epoch18=(delta_square/norm_square)**.5,
               averaged_checkpoint=relative(CKPT),averaged_checkpoint_sha256=sha256(CKPT),runner_sha256=sha256(__file__))
    write_json(OUT/'preflight.json',saved);return saved


def save_scores(val,p,name):
    assert p.shape==(1503,7) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    y=val.label.to_numpy();m=report(y,p);validate_metrics(m);folder=OUT/name
    write_json(folder/'validation_metrics.json',m)
    f=val[['image_id','lesion_id']].copy();f['true_class']=val.diagnosis.to_numpy()
    f['predicted_class']=[CLASSES[i] for i in p.argmax(1)];f[COLS]=p
    write_csv(folder/'validation_predictions.csv',f.to_dict('records'))
    write_csv(folder/'validation_probabilities.csv',f[['image_id']+COLS].to_dict('records'))
    metric_figures(m,folder/'figures',f'S75 {name} | exploratory validation')
    saved=pd.read_csv(folder/'validation_predictions.csv');assert saved.image_id.tolist()==val.image_id.tolist()
    rem=report(y,saved[COLS].to_numpy());assert rem['confusion_matrix']==m['confusion_matrix']
    return m


def run():
    pre=prepare()
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    lock=OUT/'run.lock';handle=lock.open('x')
    try:
        torch.use_deterministic_algorithms(True);torch.manual_seed(42)
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
        val,manifest=development();y=val.label.to_numpy();start=time.perf_counter()
        saved=json.loads((S53/'summary.json').read_text());source_hashes=dict(pre['source_hashes']);parts=[]
        for path,expected in saved['source_prediction_sha256'].items():
            assert sha256(ROOT/path)==expected;source_hashes[path]=expected
            f=pd.read_csv(ROOT/path);assert f.image_id.is_unique and len(f)==1503
            f=f.set_index('image_id').loc[val.image_id];assert f.true_class.tolist()==val.diagnosis.tolist()
            parts.append(f[COLS].to_numpy())
        cached=pd.read_csv(S53/'validation_predictions.csv').set_index('image_id').loc[val.image_id]
        np.testing.assert_allclose(np.mean(parts,axis=0),cached[COLS].to_numpy(),atol=1e-12)
        source_hashes[relative(S53/'validation_predictions.csv')]=sha256(S53/'validation_predictions.csv')
        source_hashes[relative(S53/'summary.json')]=sha256(S53/'summary.json')
        source_hashes[relative(ROOT/'data/splits/exploratory/image_level_dev_v1.csv')]=sha256(ROOT/'data/splits/exploratory/image_level_dev_v1.csv')
        source_hashes[relative(ROOT/'results/short_screening/s46_all_f1_checkpoint_fusion/candidate_manifest.json')]=sha256(ROOT/'results/short_screening/s46_all_f1_checkpoint_fusion/candidate_manifest.json')
        assert manifest['sources'][0]['checkpoint']==relative(SOURCE/'best_macro_f1.pt')
        source_hashes[relative(CKPT)]=pre['averaged_checkpoint_sha256'];write_json(OUT/'source_manifest.json',source_hashes)
        arrays={};logging.info('START two ConvNeXt-Tiny CPU FP32 identity passes; no training/GPU/test')
        for name,path in [('original_tiny_cpu_control',SOURCE/'best_macro_f1.pt'),('averaged_tiny_cpu',CKPT)]:
            dest=OUT/name/'validation_predictions.csv'
            if dest.exists():
                f=pd.read_csv(dest);assert f.image_id.tolist()==val.image_id.tolist() and f.true_class.tolist()==val.diagnosis.tolist()
                p=f[COLS].to_numpy()
            else:
                raw=torch.load(path,map_location='cpu',weights_only=False);net=model_for(raw['model'],raw['config'])
                loader=DataLoader(DevelopmentImages(val,raw['config']),batch_size=8,shuffle=False,num_workers=0)
                del raw;values=[];ids=[]
                with torch.inference_mode():
                    for x,_,images in loader:
                        if time.perf_counter()-start>600:raise TimeoutError('CPU budget exhausted; no automatic GPU retry')
                        z=net(x.float());assert torch.isfinite(z).all()
                        values.append(z.softmax(1).numpy());ids.extend(images)
                assert ids==val.image_id.tolist();p=np.concatenate(values);del net
            arrays[name]=p;save_scores(val,p,name);logging.info('SAVED %s',name)
        other=sum(parts[1:]);arrays['equal_five_cpu_tiny_control']=(other+arrays['original_tiny_cpu_control'])/5
        arrays['equal_five_averaged_tiny']=(other+arrays['averaged_tiny_cpu'])/5
        metrics={name:save_scores(val,p,name) for name,p in arrays.items()};rows=[]
        reference=report(y,cached[COLS].to_numpy());a=metrics['equal_five_cpu_tiny_control'];b=metrics['equal_five_averaged_tiny']
        old=arrays['equal_five_cpu_tiny_control'].argmax(1)==y;new=arrays['equal_five_averaged_tiny'].argmax(1)==y
        net_gain=int(new.sum()-old.sum())
        gate=net_gain>=8 and b['accuracy']>=a['accuracy']+.005-1e-12 and b['accuracy']>=reference['accuracy']+.005-1e-12 and all(b[field]>=r[field]-1e-12 for r in [a,reference] for field in ['macro_f1']) and all(b['per_class']['mel']['recall']>=r['per_class']['mel']['recall']-1e-12 for r in [a,reference])
        for name,m in metrics.items():
            folder=OUT/name;rows.append(dict(display_name=name,accuracy=m['accuracy'],macro_f1=m['macro_f1']))
            upsert(dict(experiment_id=f's75_{name}_exploratory_seed42',era='structured',record_kind='fixed_cpu_checkpoint_average_inference',
                        phase='post_test_exploratory_development',protocol='exploratory_image_level',evaluation_split='validation',model='convnext_tiny' if 'five' not in name else 'S53_equal_five',
                        method=name,seed=42,epochs=0,image_size=224,status='completed',decision='control' if 'control' in name else ('gate_passed' if gate else 'gate_failed'),
                        split_manifest='data/splits/exploratory/image_level_dev_v1.csv',split_sha256=pre_source_split(),
                        checkpoint=relative(SOURCE/'best_macro_f1.pt') if 'control' in name else relative(CKPT),
                        metrics_path=relative(folder/'validation_metrics.json'),plots_dir=relative(folder/'figures'),
                        notes='One fixed average epochs17/18/20; fresh CPU FP32 Tiny control; other four cached members unchanged. Reused exploratory validation, not test. No training/GPU.',
                        **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
        rows.insert(0,dict(display_name='S53 saved reference',accuracy=reference['accuracy'],macro_f1=reference['macro_f1']))
        comparison_figures(rows,OUT/'comparison_figures','S75 fixed late-checkpoint average | exploratory')
        gains=val[['image_id','lesion_id','diagnosis']].copy();gains['control_correct']=old;gains['candidate_correct']=new
        write_csv(OUT/'gain_loss.csv',gains.to_dict('records'))
        assert all(sha256(ROOT/p)==h for p,h in source_hashes.items())
        summary=dict(status='completed',metrics=metrics,saved_s53_metrics=reference,gained=int((~old&new).sum()),lost=int((old&~new).sum()),net_correct=net_gain,
                     gate_passed=bool(gate),runtime_seconds=time.perf_counter()-start,training=False,gpu_used=False,test_loaded=False,
                     decision='Promising exploratory candidate; independent confirmation still required' if gate else 'Reject this fixed average; retain S53, no subset/weight rescue or GPU training')
        write_json(OUT/'verification.json',dict(status='passed',source_hashes_unchanged=True,matched_cpu_precision=True,
                   all_1503_predictions_verified=True,metrics_recomputed_from_saved_predictions=True,locked_test_image_overlap=0,locked_test_lesion_overlap=0,test_labels_read=False,test_images_loaded=False,gpu_used=False,training=False))
        write_json(OUT/'summary.json',summary);logging.info('COMPLETE material_gate=%s net=%+d',gate,net_gain)
        print(json.dumps({k:v for k,v in summary.items() if k not in ['metrics','saved_s53_metrics']},indent=2))
    finally:handle.close();lock.unlink(missing_ok=True)


def pre_source_split():
    return sha256(ROOT/'data/splits/exploratory/image_level_dev_v1.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    print(json.dumps(prepare(),indent=2)) if args.prepare else run()
