"""S76: learn one nonlinear classifier on actual fine-tuned ensemble features.

One bounded inference cache pass; all classifier fitting uses training labels.
No backbone training, parameter search, held-out test loader or automatic sequel.
"""
import argparse
import gc
import json
import logging
import os
import tempfile
import threading
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader
from torchvision import transforms

from research.common import ROOT, CLASSES, relative, sha256, write_csv, write_json
from research.models import ResearchClassifier
from research.plots import metric_figures, comparison_figures, validate_metrics
from research.registry import upsert
from research.strict_train import DevelopmentImages, run_lock, runtime_versions
from research.short_screening.lesion_bag_screen import report

OUT=ROOT/'results/short_screening/s76_trained_feature_fusion'
CACHE=ROOT/'.cache/s76_trained_feature_fusion'
CKPT=ROOT/'checkpoints/short_screening/s76_trained_feature_fusion'
S53=ROOT/'results/short_screening/s53_equal_five_b0_addition'
MANIFEST=ROOT/'results/short_screening/s46_all_f1_checkpoint_fusion/candidate_manifest.json'
SPLIT=ROOT/'data/splits/exploratory/image_level_dev_v1.csv'
SPLIT_SHA='75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb'
COLS=[f'p_{c}' for c in CLASSES]
PLAN=dict(
    question='Do fine-tuned visual representations contain useful joint information lost by fixed probability fusion?',
    members='Exact S53 Tiny/Small/Dense201/V2-S macro-F1 checkpoints and B0 accuracy checkpoint; all five, no subset search',
    representation='Post-pooling/native-normalization/pre-linear feature vectors:768+768+1920+1280+1280=6016 dimensions',
    preprocessing='Original RGB square224 bilinear antialias/ImageNet normalization; identity only in train and validation extraction',
    extraction_precision='FP32 with TF32 disabled; no AMP; eval mode and inference_mode; no weight updates',
    classifier='One SVC C10, RBF gamma1, sqrt inverse-frequency class weights; probability=True seed42, tol0.001, max_iter50000',
    feature_normalization='Per-member StandardScaler fitted on7009 training images only, then per-image L2 normalize each block; concatenate and divide by sqrt5',
    kernel='exp(-squared Euclidean distance) on normalized6016D representation; precomputed for efficient CPU fitting',
    fitting='Training partition only; libsvm probability calibration internal to training, not independent CNN cross-validation',
    prediction='Argmax of predict_proba; raw SVC decision accuracy not used to choose a rule',
    control='Fresh all-five FP32 equal soft vote from the exact same passes; also compare saved S53',
    gate='>=0.005 accuracy and>=8 net correct gains vs fresh control AND saved S53; macro-F1 and melanoma recall nondecreasing vs both',
    protocol='Existing exploratory development; prior checkpoint/recipe selection and lesion overlap remain; not new independent test evidence',
    scope='One actual feature-fusion candidate, no C/gamma/head/member/weight search or post-result rescue',
    gpu_extraction_limit_seconds=600,total_wall_limit_seconds=900,cpu_threads=4,batch_size=16,workers=2,
    backbone_training=False,test_loaded=False,
    literature=['https://www.nature.com/articles/s41598-025-31816-2',
                'https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1478688/full',
                'https://scikit-learn.org/stable/modules/svm.html'],
    literature_scope='Feature fusion is supported as a distinct method; these architectures/classifier/protocol are an adaptation, not paper replication or a promised accuracy')


def development():
    assert sha256(SPLIT)==SPLIT_SHA
    ids=pd.read_csv(SPLIT,usecols=['image_id','lesion_id','split'])
    excluded=set((ids.index[ids.split=='test']+1).tolist())
    dev=pd.read_csv(SPLIT,skiprows=lambda row:row in excluded)
    assert set(dev.split)=={'train','val'} and dev.image_id.is_unique
    assert dev.label.tolist()==dev.diagnosis.map(dict(zip(CLASSES,range(7)))).tolist()
    strict=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','lesion_id','split'])
    test=strict.loc[strict.split=='test']
    assert not set(dev.image_id)&set(test.image_id) and not set(dev.lesion_id)&set(test.lesion_id)
    files={}
    for part in ['HAM10000_images_part_1','HAM10000_images_part_2']:
        for p in (ROOT/'data/raw/HAM10000'/part).glob('*.jpg'):
            assert p.stem not in files;files[p.stem]=p
    dev['path']=[str(files[i]) for i in dev.image_id]
    train=dev.loc[dev.split=='train'].reset_index(drop=True);val=dev.loc[dev.split=='val'].reset_index(drop=True)
    assert (len(train),len(val))==(7009,1503)
    return train,val


def sources():
    src=json.loads(MANIFEST.read_text())['sources']
    s=json.loads((S53/'summary.json').read_text())
    path='results/structured_experiments/s03_efficientnet_b0_none_exploratory_seed42/validation_predictions.csv'
    src.append(dict(run='s03_efficientnet_b0_none_exploratory_seed42',prediction_file=path,
                    prediction_sha256=s['source_prediction_sha256'][path],checkpoint=s['rescue_checkpoint'],checkpoint_sha256=s['rescue_checkpoint_sha256']))
    assert len(src)==5
    for entry in src:
        assert sha256(ROOT/entry['checkpoint'])==entry['checkpoint_sha256']
        assert sha256(ROOT/entry['prediction_file'])==entry['prediction_sha256']
    return src


def model_for(src):
    raw=torch.load(ROOT/src['checkpoint'],map_location='cpu',weights_only=False)
    c=raw['config'];assert c['class_order']==list(CLASSES) and c['split_sha256']==SPLIT_SHA
    assert c['image_size']==224 and c['attention']=='none'
    assert c['normalization_mean']==[.485,.456,.406] and c['normalization_std']==[.229,.224,.225]
    assert all(torch.isfinite(v).all() for v in raw['model'].values())
    net=ResearchClassifier(c['model'],None,c['attention'],c['head_dropout']).float().eval()
    net.load_state_dict(raw['model'],strict=True)
    for p in net.parameters():p.requires_grad_(False)
    return net,c


def identity_dataset(frame,c):
    ds=DevelopmentImages(frame,c,training=set(frame.split)=={'train'})
    ds.transform=transforms.Compose([t for t in ds.transform.transforms if not isinstance(t,transforms.RandomHorizontalFlip)])
    return ds


def features(net,x):
    # Preserve native ConvNeXt post-pooling normalization and all original heads.
    h=net.head[:-1](net.attention(net.features(x)))
    return h,net.head[-1](h)


def prepare():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREDECLARED_PLAN.json').exists():assert json.loads((OUT/'PREDECLARED_PLAN.json').read_text())==PLAN
    else:write_json(OUT/'PREDECLARED_PLAN.json',PLAN)
    train,val=development();src=sources()
    signature=dict(runner_sha256=sha256(__file__),sources=src,split_sha256=SPLIT_SHA,
                   code_hashes={f:sha256(ROOT/f) for f in ['research/models.py','research/strict_train.py','research/short_screening/lesion_bag_screen.py','research/plots.py']})
    if (OUT/'signature.json').exists():assert json.loads((OUT/'signature.json').read_text())==signature
    else:write_json(OUT/'signature.json',signature)
    if not (OUT/'preflight.json').exists():
        rows=[]
        for s in src:
            net,c=model_for(s);ds=identity_dataset(train.iloc[:2],c);x=torch.stack([ds[i][0] for i in range(2)])
            with torch.inference_mode():
                h,z=features(net,x);direct=net(x);torch.testing.assert_close(z,direct,atol=1e-6,rtol=1e-5)
                assert torch.isfinite(h).all() and torch.isfinite(z).all() and z.shape==(2,7)
            rows.append(dict(run=s['run'],dimensions=h.shape[1],original_logits_reconstructed=True))
            del net;gc.collect()
        assert sum(r['dimensions'] for r in rows)==6016
        # Verify our precomputed kernel against sklearn's direct RBF convention.
        from sklearn.metrics.pairwise import rbf_kernel
        toy=np.array([[1.,0.],[0.,1.],[-1.,0.]])
        np.testing.assert_allclose(kernel(toy,toy),rbf_kernel(toy,toy,gamma=1),atol=1e-12)
        write_json(OUT/'preflight.json',dict(status='passed',members=rows,train_images=len(train),validation_images=len(val),
                   locked_test_image_overlap=0,locked_test_lesion_overlap=0,training_classifier_labels_only=True,
                   kernel_matches_sklearn=True,test_labels_read=False,test_images_loaded=False,runtime=runtime_versions()))
    return train,val,src,signature


def atomic_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,suffix='.npz');os.close(fd)
    try:np.savez_compressed(name,**arrays);os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)


def extraction(train,val,src,signature):
    assert torch.cuda.is_available(),'CUDA needed for bounded feature extraction; no training is performed'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter();manifest=[]
    for s in src:
        net=None
        for part,frame in [('train',train),('val',val)]:
            path=CACHE/f"{s['run']}_{part}.npz"
            if path.exists():
                with np.load(path,allow_pickle=False) as z:
                    assert z['ids'].tolist()==frame.image_id.tolist() and z['y'].tolist()==frame.label.tolist()
                    assert str(z['checkpoint_sha256'])==s['checkpoint_sha256'] and str(z['signature_sha256'])==sha256(OUT/'signature.json')
                    assert np.isfinite(z['features']).all() and np.isfinite(z['probabilities']).all()
                logging.info('REUSE %s %s verified cache',s['run'],part)
            else:
                if net is None:net,c=model_for(s);net=net.cuda()
                loader=DataLoader(identity_dataset(frame,c),batch_size=16,shuffle=False,num_workers=2,pin_memory=True)
                vectors=[];probabilities=[];ids=[];labels=[]
                with torch.inference_mode():
                    for x,y,image_ids in loader:
                        if time.perf_counter()-start>600:raise TimeoutError('Ten-minute GPU extraction budget; preserve completed caches')
                        h,z=features(net,x.cuda(non_blocking=True));p=z.softmax(1)
                        assert torch.isfinite(h).all() and torch.isfinite(p).all()
                        vectors.append(h.cpu().numpy());probabilities.append(p.cpu().numpy());ids.extend(image_ids);labels.extend(y.tolist())
                assert ids==frame.image_id.tolist() and labels==frame.label.tolist()
                atomic_npz(path,features=np.concatenate(vectors),probabilities=np.concatenate(probabilities),ids=np.array(ids),y=np.array(labels),
                           checkpoint_sha256=np.array(s['checkpoint_sha256']),signature_sha256=np.array(sha256(OUT/'signature.json')))
                logging.info('SAVED %s %s features %d',s['run'],part,len(ids))
            manifest.append(dict(path=relative(path),sha256=sha256(path),run=s['run'],split=part))
            write_json(OUT/'progress.json',dict(stage='feature_extraction',run=s['run'],split=part,cache_complete=True,pid=os.getpid()))
        del net;gc.collect();torch.cuda.empty_cache()
    write_json(OUT/'feature_cache_manifest.json',dict(files=manifest,seconds=time.perf_counter()-start,
               gpu_name=torch.cuda.get_device_name(0),peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,precision='FP32',optimizer_updates=0))


def kernel(a,b):
    out=a@b.T
    out*= -2
    out+=np.sum(a*a,axis=1)[:,None]
    out+=np.sum(b*b,axis=1)[None,:]
    np.maximum(out,0,out=out);out*= -1;np.exp(out,out=out)
    return np.asarray(out,dtype=np.float64,order='C')


def save_result(val,p,name):
    assert p.shape==(1503,7) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    m=report(val.label.to_numpy(),p);validate_metrics(m);folder=OUT/name
    f=val[['image_id','lesion_id']].copy();f['true_class']=val.diagnosis
    f['predicted_class']=[CLASSES[i] for i in p.argmax(1)];f[COLS]=p
    write_csv(folder/'validation_predictions.csv',f.to_dict('records'))
    write_csv(folder/'validation_probabilities.csv',f[['image_id']+COLS].to_dict('records'))
    write_json(folder/'validation_metrics.json',m);metric_figures(m,folder/'figures',f'S76 {name} | exploratory validation')
    saved=pd.read_csv(folder/'validation_predictions.csv');assert saved.image_id.tolist()==val.image_id.tolist()
    assert report(val.label.to_numpy(),saved[COLS].to_numpy())['confusion_matrix']==m['confusion_matrix']
    return m


def fit_and_close(train,val,src):
    start=time.perf_counter();train_blocks=[];val_blocks=[];fresh=[];scalers=[];cache_hashes={}
    for s in src:
        pair=[]
        for part,frame in [('train',train),('val',val)]:
            path=CACHE/f"{s['run']}_{part}.npz";cache_hashes[relative(path)]=sha256(path)
            with np.load(path,allow_pickle=False) as z:
                assert z['ids'].tolist()==frame.image_id.tolist() and z['y'].tolist()==frame.label.tolist()
                pair.append(z['features'].astype(np.float64))
                if part=='val':fresh.append(z['probabilities'].astype(np.float64))
        scaler=StandardScaler().fit(pair[0]);scalers.append(scaler)
        train_blocks.append(normalize(scaler.transform(pair[0])));val_blocks.append(normalize(scaler.transform(pair[1])))
    x=np.concatenate(train_blocks,axis=1)/np.sqrt(5);vx=np.concatenate(val_blocks,axis=1)/np.sqrt(5)
    del train_blocks,val_blocks,pair;gc.collect()
    assert x.shape==(7009,6016) and vx.shape==(1503,6016) and np.isfinite(x).all() and np.isfinite(vx).all()
    y=train.label.to_numpy();counts=np.bincount(y,minlength=7);cw=np.sqrt(len(y)/(7*counts));cw/=cw.mean()
    logging.info('CPU FIT one nonlinear6016D feature-fusion SVM;7009 training labels only')
    with threadpool_limits(limits=4):
        k=kernel(x,x);assert np.allclose(k,k.T,atol=1e-12) and np.allclose(k.diagonal(),1,atol=1e-12)
        classifier=SVC(C=10,kernel='precomputed',probability=True,random_state=42,
                       class_weight={i:float(w) for i,w in enumerate(cw)},tol=.001,max_iter=50000,cache_size=512)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning);classifier.fit(k,y)
        assert classifier.fit_status_==0 and classifier.classes_.tolist()==list(range(7));del k;gc.collect()
        kv=kernel(vx,x);p=classifier.predict_proba(kv)
    CKPT.mkdir(parents=True,exist_ok=True);dest=CKPT/'feature_fusion.joblib'
    if dest.exists():raise FileExistsError('Classifier already saved; preserve it for explicit closeout recovery')
    fd,tmp=tempfile.mkstemp(dir=CKPT,suffix='.tmp');os.close(fd)
    try:
        joblib.dump(dict(classifier=classifier,scalers=scalers,training_features=x,training_ids=train.image_id.tolist(),
                        class_order=list(CLASSES),sources=src,plan=PLAN),tmp,compress=3);os.replace(tmp,dest)
    finally:Path(tmp).unlink(missing_ok=True)
    # Re-load the deployed bundle and reproduce predictions, including scalers.
    restored=joblib.load(dest);blocks=[]
    for s,scaler in zip(src,restored['scalers']):
        with np.load(CACHE/f"{s['run']}_val.npz",allow_pickle=False) as z:blocks.append(normalize(scaler.transform(z['features'].astype(np.float64))))
    rx=np.concatenate(blocks,axis=1)/np.sqrt(5)
    with threadpool_limits(limits=4):rp=restored['classifier'].predict_proba(kernel(rx,restored['training_features']))
    np.testing.assert_allclose(rp,p,atol=1e-12,rtol=1e-12);del restored,blocks,rx;gc.collect()
    control=np.mean(fresh,axis=0);results={'fresh_equal_five_fp32':save_result(val,control,'fresh_equal_five_fp32'),
                                      'trained_feature_fusion':save_result(val,p,'trained_feature_fusion')}
    cached=pd.read_csv(S53/'validation_predictions.csv').set_index('image_id').loc[val.image_id]
    assert cached.true_class.tolist()==val.diagnosis.tolist()
    reference=report(val.label.to_numpy(),cached[COLS].to_numpy());a=results['fresh_equal_five_fp32'];b=results['trained_feature_fusion']
    yy=val.label.to_numpy();old=control.argmax(1)==yy;new=p.argmax(1)==yy;old_saved=cached[COLS].to_numpy().argmax(1)==yy
    gate=all(b['accuracy']>=r['accuracy']+.005-1e-12 and b['macro_f1']>=r['macro_f1']-1e-12 and b['per_class']['mel']['recall']>=r['per_class']['mel']['recall']-1e-12 for r in [a,reference]) and int(new.sum()-old.sum())>=8 and int(new.sum()-old_saved.sum())>=8
    gain=val[['image_id','lesion_id','diagnosis']].copy();gain['control_correct']=old;gain['feature_fusion_correct']=new
    write_csv(OUT/'gain_loss.csv',gain.to_dict('records'))
    rows=[dict(display_name='S53 saved reference',accuracy=reference['accuracy'],macro_f1=reference['macro_f1'])]
    for name,m in results.items():
        rows.append(dict(display_name=name,accuracy=m['accuracy'],macro_f1=m['macro_f1']))
        folder=OUT/name
        upsert(dict(experiment_id=f's76_{name}_exploratory_seed42',era='structured',record_kind='trained_feature_fusion' if name=='trained_feature_fusion' else 'fp32_reproduction_control',
                    phase='post_test_exploratory_development',protocol='exploratory_image_level',evaluation_split='validation',
                    split_manifest=relative(SPLIT),split_sha256=SPLIT_SHA,model='S53_five_frozen_finetuned_backbones',method=name,
                    seed=42,epochs=0,image_size=224,status='completed',decision='control' if 'fresh' in name else ('material_gate_passed' if gate else 'material_gate_failed'),
                    checkpoint=relative(dest) if name=='trained_feature_fusion' else '',checkpoint_sha256=sha256(dest) if name=='trained_feature_fusion' else '',
                    metrics_path=relative(folder/'validation_metrics.json'),plots_dir=relative(folder/'figures'),config_path=relative(OUT/'PREDECLARED_PLAN.json'),
                    notes='Classifier fit only on7009 training labels; exact S53 trained features, not generic ImageNet probes. No backbone update/test. Reused exploratory validation.',
                    **{key:m[key] for key in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    comparison_figures(rows,OUT/'comparison_figures','S76 learned visual feature fusion vs equal probabilities')
    assert all(sha256(ROOT/path)==h for path,h in cache_hashes.items());sources()
    fit_info=dict(fitting_images=len(y),dimensions=x.shape[1],C=10,gamma=1,class_weights=cw.tolist(),
                  support_vectors=classifier.n_support_.tolist(),iterations=classifier.n_iter_.tolist(),converged=True,
                  probability_rule='Argmax predict_proba; libsvm internal training calibration',seconds=time.perf_counter()-start,
                  classifier_checkpoint=relative(dest),classifier_sha256=sha256(dest))
    write_json(OUT/'fit_metadata.json',fit_info)
    write_json(OUT/'verification.json',dict(status='passed',original_checkpoint_hashes_unchanged=True,all_feature_cache_hashes_unchanged=True,
               classifier_fit_uses_train_only=True,normalizers_fit_uses_train_only=True,saved_classifier_reproduces_predictions=True,
               probability_metrics_and_confusions_recomputed=True,train_images=7009,val_images=1503,test_images_loaded=False,test_labels_read=False,
               no_backbone_updates=True,no_hyperparameter_search=True,gpu_inference_only=True))
    summary=dict(status='completed',metrics=results,saved_s53_metrics=reference,material_gate_passed=bool(gate),
                 gained=int((~old&new).sum()),lost=int((old&~new).sum()),net_correct=int(new.sum()-old.sum()),
                 fresh_control_label_changes_vs_saved=int((control.argmax(1)!=cached[COLS].to_numpy().argmax(1)).sum()),
                 decision='Promising feature-fusion development candidate; no automatic test/training' if gate else 'Feature-fusion hypothesis not supported by this fixed classifier; preserve all artifacts, retain S53; no search rescue',
                 cnn_training=False,gpu_training=False,test_loaded=False)
    write_json(OUT/'summary.json',summary);logging.info('COMPLETE feature_fusion=%.6f control=%.6f gate=%s',b['accuracy'],a['accuracy'],gate)
    return summary


def run():
    train,val,src,signature=prepare()
    if (OUT/'summary.json').exists():print('Completed; preserved results at',relative(OUT));return
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
    def timeout():
        write_json(OUT/'failure.json',dict(reason='Fifteen-minute hard wall budget exceeded; caches preserved; no auto-retry'))
        os._exit(124)
    timer=threading.Timer(900,timeout);timer.daemon=True;timer.start()
    try:
        logging.info('START fixed trained-feature fusion; short FP32 extraction then one CPU fit; no test/backbone training')
        extraction(train,val,src,signature);fit_and_close(train,val,src)
    except Exception as exc:
        write_json(OUT/'failure.json',dict(error=repr(exc)));logging.exception('Stopped; completed artifacts preserved');raise
    finally:timer.cancel()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.prepare:prepare();print('CPU preflight passed; exact five trained checkpoints and6016D feature path verified')
    elif args.run:
        with run_lock('s76_trained_feature_fusion'):run()
    else:p.error('Use --prepare or --run')
