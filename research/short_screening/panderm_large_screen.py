"""S77: published PanDerm-Large vs Base, matched frozen inference/CPU heads.

S78 is one conditional fixed B0 replacement. No backbone training or test use.
"""
import argparse
import gc
import importlib.util
import json
import logging
import os
import sys
import tempfile
import threading
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader
from torchvision import transforms

from research.common import ROOT, CLASSES, relative, sha256, write_json, write_csv
from research.plots import metric_figures, comparison_figures, validate_metrics
from research.strict_train import DevelopmentImages, run_lock, runtime_versions
from research.registry import upsert
from research.short_screening.feature_fusion import development, atomic_npz, kernel
from research.short_screening.lesion_bag_screen import report

OUT=ROOT/'results/short_screening/s77_panderm_large_transfer'
CACHE=ROOT/'.cache/s77_panderm_large_transfer'
CKPT=ROOT/'checkpoints/short_screening/s77_panderm_large_transfer'
MODEL_SOURCE=ROOT/'.cache/research_panderm/classification/models/modeling_finetune.py'
FILES={'base':ROOT/'.cache/panderm_bb_data6_checkpoint-499.pth','large':ROOT/'.cache/panderm_ll_data6_checkpoint-499.pth'}
S53=ROOT/'results/short_screening/s53_equal_five_b0_addition'
SPLIT=ROOT/'data/splits/exploratory/image_level_dev_v1.csv'
COLS=[f'p_{c}' for c in CLASSES]
PLAN=dict(
    question='Does the published dermatology-pretrained ViT-L encoder add materially stronger task-relevant features than the previously used smaller Base?',
    control='PanDerm Base ViT-B16,86M-class encoder, newly extracted FP32 features and same fixed head',
    candidate='Authors PanDerm Large ViT-L16,24blocks,1024D CLS features; official2million dermatology pretraining model',
    sources={'repository':'https://github.com/SiyuanYan1/PanDerm','paper':'https://www.nature.com/articles/s41591-025-03747-y',
             'large_weights':'https://drive.google.com/file/d/1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE/view',
             'base_weights':'https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB/view'},
    feature_extraction='Both frozen/eval FP32 identity only; original author resize-short-side256/center224; mean(.485,.456,.406),std(.228,.224,.225); native CLS embedding',
    head='Same training-only StandardScaler and RBF SVM C10 gamma=scale, no class weights, probability=True seed42; argmax predict_proba',
    head_rationale='Preserve the previously used exploratory Base SVM recipe; no head/hyperparameter search',
    implementation='Equivalent precomputed RBF kernel on standardized features, gamma computed from train only; probability calibration inside training',
    cohort='Existing7009 exploratory train and1503 validation images; original locked-test images/labels excluded',
    s78_condition='Only if Large exceeds matched Base by>=0.01 accuracy with nondecreasing macro-F1/MEL recall and Large accuracy>=0.90',
    s78_rule='Exactly one fixed replacement of B0 in S53 with Large; Tiny/Small/Dense201/V2-S unchanged, all five probabilities weighted0.20',
    advancement='S78 accuracy>=S53+0.005 (>=8 net gains), macro-F1 and melanoma recall nondecreasing; no weight/member rescue',
    caveats='Published results use different evaluation/data. Complete pretraining overlap with our HAM images cannot be independently ruled out. No claim of untouched independent test accuracy.',
    local_prior='There was a two-epoch Base last-two-block STRICT pilot84.43%; no previous Large run or complete exploratory PanDerm fine-tuning',
    limits='One paired Base/Large frozen study, one conditional fusion, no training epochs or further automatic experiment',
    gpu_seconds_limit=600,wall_seconds_limit=900,batch_size=16,cpu_threads=4,
    backbone_training=False,test_loaded=False)


def load_model(kind):
    spec=importlib.util.spec_from_file_location('panderm_s77_author_model',MODEL_SOURCE)
    module=sys.modules.get(spec.name)
    if module is None:
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    model=getattr(module,f'panderm_{kind}_patch16_224')();model.head=torch.nn.Identity()
    raw=torch.load(FILES[kind],map_location='cpu',weights_only=True)
    if kind=='large':
        state={k.removeprefix('encoder.'):v for k,v in raw.items() if k.startswith('encoder.')}
        excluded=[k for k in raw if not k.startswith('encoder.')]
    else:state=raw;excluded=[]
    # Require every encoder tensor; ignore only documented SSL non-encoder heads.
    model.load_state_dict(state,strict=True)
    assert all(torch.isfinite(t).all() for t in state.values() if t.is_floating_point())
    model.eval().float();model.requires_grad_(False)
    meta=dict(encoder_parameters=sum(p.numel() for p in model.parameters()),dimensions=model.num_features,
              loaded_encoder_tensors=len(state),excluded_pretraining_keys=excluded,strict_encoder_load=True)
    del raw,state;gc.collect();return model,meta


def dataset(frame):
    c=dict(image_size=224,normalization_mean=[.485,.456,.406],normalization_std=[.228,.224,.225])
    ds=DevelopmentImages(frame,c,training=set(frame.split)=={'train'})
    ds.transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                                    transforms.Normalize(c['normalization_mean'],c['normalization_std'])])
    return ds


def prepare():
    torch.set_num_threads(4);torch.manual_seed(42);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREDECLARED_PLAN.json').exists():assert json.loads((OUT/'PREDECLARED_PLAN.json').read_text())==PLAN
    else:write_json(OUT/'PREDECLARED_PLAN.json',PLAN)
    train,val=development()
    signature=dict(weights_sha256={k:sha256(p) for k,p in FILES.items()},source_sha256=sha256(MODEL_SOURCE),
                   runner_sha256=sha256(__file__),shared_runner_sha256=sha256(ROOT/'research/short_screening/feature_fusion.py'),split_sha256=sha256(SPLIT))
    assert signature['weights_sha256']['base']=='be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1'
    if (OUT/'signature.json').exists():assert json.loads((OUT/'signature.json').read_text())==signature
    else:write_json(OUT/'signature.json',signature)
    if not (OUT/'preflight.json').exists():
        x=dataset(train.iloc[:1])[0][0].unsqueeze(0);rows={}
        for kind in FILES:
            model,meta=load_model(kind)
            with torch.inference_mode():
                z=model.forward_features(x,is_train=False)
                assert z.shape==(1,768 if kind=='base' else 1024) and torch.isfinite(z).all()
            rows[kind]=meta;del model;gc.collect()
        # Exact equivalence of the precomputed gamma=scale kernel.
        from sklearn.metrics.pairwise import rbf_kernel
        x=StandardScaler().fit_transform(np.array([[1.,0.,2.],[0.,1.,4.],[2.,1.,0.]]))
        gamma=1/(x.shape[1]*x.var());np.testing.assert_allclose(kernel(x*np.sqrt(gamma),x*np.sqrt(gamma)),rbf_kernel(x,gamma=gamma),atol=1e-12)
        write_json(OUT/'preflight.json',dict(status='passed',models=rows,train_images=len(train),validation_images=len(val),
                   locked_test_image_overlap=0,locked_test_lesion_overlap=0,finite_cpu_forward=True,kernel_equivalence=True,
                   test_images_loaded=False,test_labels_read=False,runtime=runtime_versions()))
    return train,val,signature


def extract(train,val,sig):
    assert torch.cuda.is_available();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter();items=[]
    for kind in FILES:
        model=None
        for part,frame in [('train',train),('val',val)]:
            path=CACHE/f'{kind}_{part}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as s:
                    assert s['ids'].tolist()==frame.image_id.tolist() and s['y'].tolist()==frame.label.tolist()
                    assert str(s['checkpoint_sha256'])==sig['weights_sha256'][kind] and str(s['signature_sha256'])==sha256(OUT/'signature.json')
                    assert np.isfinite(s['features']).all()
                logging.info('REUSE %s %s',kind,part)
            else:
                if model is None:model,_=load_model(kind);model=model.cuda()
                loader=DataLoader(dataset(frame),batch_size=16,shuffle=False,num_workers=2,pin_memory=True)
                arrays=[];ids=[];ys=[]
                with torch.inference_mode():
                    for x,y,image_ids in loader:
                        if time.perf_counter()-start>600:raise TimeoutError('GPU extraction limit reached; preserve caches')
                        z=model.forward_features(x.cuda(non_blocking=True),is_train=False);assert torch.isfinite(z).all()
                        arrays.append(z.cpu().numpy());ids.extend(image_ids);ys.extend(y.tolist())
                assert ids==frame.image_id.tolist() and ys==frame.label.tolist()
                atomic_npz(path,ids=np.array(ids),y=np.array(ys),features=np.concatenate(arrays),
                           checkpoint_sha256=np.array(sig['weights_sha256'][kind]),signature_sha256=np.array(sha256(OUT/'signature.json')))
                logging.info('SAVED %s %s %d frozen features',kind,part,len(ids))
            items.append(dict(path=relative(path),sha256=sha256(path),model=kind,split=part))
            write_json(OUT/'progress.json',dict(status='extracting',model=kind,split=part,pid=os.getpid()))
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(OUT/'feature_cache_manifest.json',dict(files=items,seconds=time.perf_counter()-start,
               peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,gpu=torch.cuda.get_device_name(0),precision='FP32',optimizer_updates=0))


def scores(val,p,name):
    assert p.shape==(1503,7) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    m=report(val.label.to_numpy(),p);validate_metrics(m);folder=OUT/name
    f=val[['image_id','lesion_id']].copy();f['true_class']=val.diagnosis;f['predicted_class']=[CLASSES[i] for i in p.argmax(1)];f[COLS]=p
    write_json(folder/'validation_metrics.json',m);write_csv(folder/'validation_predictions.csv',f.to_dict('records'))
    write_csv(folder/'validation_probabilities.csv',f[['image_id']+COLS].to_dict('records'))
    metric_figures(m,folder/'figures',f'{name} | exploratory development')
    check=pd.read_csv(folder/'validation_predictions.csv');assert check.image_id.tolist()==val.image_id.tolist()
    assert report(val.label.to_numpy(),check[COLS].to_numpy())['confusion_matrix']==m['confusion_matrix']
    return m


def fit_head(kind,train,val):
    arr={}
    for part,frame in [('train',train),('val',val)]:
        with np.load(CACHE/f'{kind}_{part}.npz',allow_pickle=False) as s:
            assert s['ids'].tolist()==frame.image_id.tolist() and s['y'].tolist()==frame.label.tolist()
            arr[part]=s['features'].astype(np.float64)
    scaler=StandardScaler().fit(arr['train']);x=scaler.transform(arr['train']);vx=scaler.transform(arr['val'])
    gamma=1/(x.shape[1]*x.var());x*=np.sqrt(gamma);vx*=np.sqrt(gamma)
    classifier=SVC(C=10,kernel='precomputed',probability=True,random_state=42,tol=.001,max_iter=50000,cache_size=512)
    start=time.perf_counter()
    with threadpool_limits(limits=4):
        k=kernel(x,x)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning);classifier.fit(k,train.label.to_numpy())
        assert classifier.fit_status_==0 and classifier.classes_.tolist()==list(range(7));del k;gc.collect()
        kv=kernel(vx,x);p=classifier.predict_proba(kv)
    folder=CKPT/kind;folder.mkdir(parents=True,exist_ok=True);path=folder/'svm.joblib'
    if path.exists():raise FileExistsError('Preserve saved classifier; recover reporting explicitly')
    fd,temp=tempfile.mkstemp(dir=folder,suffix='.tmp');os.close(fd)
    try:
        joblib.dump(dict(classifier=classifier,scaler=scaler,gamma=gamma,training_features=x,
                        training_ids=train.image_id.tolist(),class_order=list(CLASSES),model=kind,plan=PLAN),temp,compress=3)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    restored=joblib.load(path)
    with threadpool_limits(limits=4):
        vr=restored['scaler'].transform(arr['val'])*np.sqrt(restored['gamma'])
        np.testing.assert_allclose(restored['classifier'].predict_proba(kernel(vr,restored['training_features'])),p,atol=1e-12,rtol=1e-12)
    meta=dict(training_images=len(train),features=x.shape[1],gamma=gamma,C=10,class_weight=None,
              support_vectors=classifier.n_support_.tolist(),iterations=classifier.n_iter_.tolist(),seconds=time.perf_counter()-start,
              checkpoint=relative(path),checkpoint_sha256=sha256(path),converged=True,reloaded_predictions_reproduced=True)
    write_json(OUT/kind/'fit_metadata.json',meta);return p,meta


def close(train,val,sig):
    probabilities={};metrics={};heads={};rows=[]
    reference=pd.read_csv(S53/'validation_predictions.csv').set_index('image_id').loc[val.image_id]
    assert reference.true_class.tolist()==val.diagnosis.tolist();y=val.label.to_numpy();refp=reference[COLS].to_numpy();refm=report(y,refp)
    reference_hash=sha256(S53/'validation_predictions.csv')
    for kind in FILES:
        logging.info('CPU FIT matched %s SVM C10, train only',kind)
        p,meta=fit_head(kind,train,val);probabilities[kind]=p;heads[kind]=meta;metrics[kind]=scores(val,p,kind)
    a,b=metrics['base'],metrics['large']
    eligible=b['accuracy']>=a['accuracy']+.01-1e-12 and b['macro_f1']>=a['macro_f1'] and b['per_class']['mel']['recall']>=a['per_class']['mel']['recall'] and b['accuracy']>=.90
    fusion_gate=False;complement={};correct=refp.argmax(1)==y;large_correct=probabilities['large'].argmax(1)==y
    complement=dict(s53_errors_large_fixes=int((~correct&large_correct).sum()),s53_correct_large_misses=int((correct&~large_correct).sum()))
    if eligible:
        summary=json.loads((S53/'summary.json').read_text());parts=[];source_hashes={relative(S53/'validation_predictions.csv'):reference_hash}
        for path,h in summary['source_prediction_sha256'].items():
            assert sha256(ROOT/path)==h;source_hashes[path]=h
            frame=pd.read_csv(ROOT/path).set_index('image_id').loc[val.image_id];assert frame.true_class.tolist()==val.diagnosis.tolist()
            parts.append(frame[COLS].to_numpy())
        np.testing.assert_allclose(np.mean(parts,axis=0),refp,atol=1e-12)
        # Source order is independently asserted; B0 is the fifth source.
        assert list(summary['source_prediction_sha256'])[-1].endswith('s03_efficientnet_b0_none_exploratory_seed42/validation_predictions.csv')
        fusion=(sum(parts[:4])+probabilities['large'])/5
        probabilities['s78_large_replaces_b0']=fusion;metrics['s78_large_replaces_b0']=scores(val,fusion,'s78_large_replaces_b0')
        m=metrics['s78_large_replaces_b0'];new=fusion.argmax(1)==y
        fusion_gate=m['accuracy']>=refm['accuracy']+.005-1e-12 and int(new.sum()-correct.sum())>=8 and m['macro_f1']>=refm['macro_f1'] and m['per_class']['mel']['recall']>=refm['per_class']['mel']['recall']
        complement.update(fusion_gained=int((~correct&new).sum()),fusion_lost=int((correct&~new).sum()),fusion_net=int(new.sum()-correct.sum()))
        write_json(OUT/'s78_source_manifest.json',source_hashes)
    for name,m in metrics.items():
        folder=OUT/name;rows.append(dict(display_name=name,accuracy=m['accuracy'],macro_f1=m['macro_f1']))
        rid=f's77_panderm_{name}_fp32_svm_exploratory_seed42' if name in FILES else 's78_panderm_large_replaces_b0_equal_five_exploratory_seed42'
        upsert(dict(experiment_id=rid,era='structured',record_kind='domain_pretrained_feature_classifier' if name in FILES else 'fixed_probability_fusion',
                    phase='post_test_exploratory_development',protocol='exploratory_image_level',evaluation_split='validation',split_manifest=relative(SPLIT),split_sha256=sha256(SPLIT),
                    model='PanDerm_'+name,method='Frozen author encoder plus fixed training-only RBF SVM' if name in FILES else PLAN['s78_rule'],
                    seed=42,epochs=0,image_size=224,status='completed',decision='control' if name=='base' else ('material_gate_passed' if fusion_gate else 'no_material_ensemble_gate'),
                    checkpoint=heads[name]['checkpoint'] if name in FILES else heads['large']['checkpoint'],metrics_path=relative(folder/'validation_metrics.json'),
                    plots_dir=relative(folder/'figures'),config_path=relative(OUT/'PREDECLARED_PLAN.json'),
                    notes='Published Large versus smaller Base, matched FP32/head. No encoder updates/test. Pretraining overlap unresolved; reused exploratory validation.',
                    **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    rows.insert(0,dict(display_name='Retained S53',accuracy=refm['accuracy'],macro_f1=refm['macro_f1']))
    comparison_figures(rows,OUT/'comparison_figures','Published PanDerm Large: matched transfer and conditional fusion')
    audit=val[['image_id','lesion_id','diagnosis']].copy();audit['s53_correct']=correct;audit['large_correct']=large_correct
    if eligible:audit['fusion_correct']=probabilities['s78_large_replaces_b0'].argmax(1)==y
    write_csv(OUT/'complementarity.csv',audit.to_dict('records'))
    assert sha256(S53/'validation_predictions.csv')==reference_hash
    assert all(sha256(FILES[k])==v for k,v in sig['weights_sha256'].items())
    assert all(sha256(ROOT/s['path'])==s['sha256'] for s in json.loads((OUT/'feature_cache_manifest.json').read_text())['files'])
    summary=dict(status='completed',metrics=metrics,reference_metrics=refm,s78_condition_passed=bool(eligible),material_fusion_gate=bool(fusion_gate),
                 complementarity=complement,backbone_training=False,test_loaded=False,
                 decision='Promising exploratory replacement; no automatic training/test' if fusion_gate else 'No material fusion improvement; preserve evidence and retained S53; no search rescue')
    write_json(OUT/'summary.json',summary)
    write_json(OUT/'verification.json',dict(status='passed',source_and_cache_hashes_unchanged=True,head_fitting_train_only=True,
               saved_heads_reproduce_predictions=True,metrics_recomputed=True,test_labels_read=False,test_images_loaded=False,backbone_updates=0))
    write_json(OUT/'progress.json',dict(status='completed',active_process=False,material_fusion_gate=bool(fusion_gate)))
    logging.info('COMPLETE Base=%.6f Large=%.6f fusion_gate=%s',a['accuracy'],b['accuracy'],fusion_gate)


def run():
    if (OUT/'summary.json').exists():print('Completed study preserved');return
    train,val,sig=prepare()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
    def timeout():
        write_json(OUT/'failure.json',dict(reason='15minute hard wall limit; preserved complete artifacts'));os._exit(124)
    timer=threading.Timer(900,timeout);timer.daemon=True;timer.start()
    try:
        logging.info('START matched PanDerm Base/Large frozen FP32 inference; no backbone training/test')
        extract(train,val,sig);close(train,val,sig)
    except Exception as exc:
        write_json(OUT/'failure.json',dict(error=repr(exc)));logging.exception('Stopped; preserve artifacts');raise
    finally:timer.cancel()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.prepare:prepare();print('CPU preflight passed: strict author encoder loads, finite Base/Large features and safe development cohort')
    elif a.run:
        with run_lock('s77_panderm_large_transfer'):run()
    else:p.error('Use --prepare or --run')
