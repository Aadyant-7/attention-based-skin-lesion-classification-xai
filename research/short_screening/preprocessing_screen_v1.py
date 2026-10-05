"""Fixed CPU-only paired pretrained-feature preprocessing screen; no test loader."""
import json, logging, time
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from research.common import ROOT, CLASSES, sha256, relative, write_json, write_csv
from research.models import ResearchClassifier
from research.strict_train import DevelopmentImages, metric_report, run_lock
from research.short_screening.convnextv2_transfer import config, development
from research.aggressive.core import predictions, retry_registry_upsert
from research.plots import metric_figures, comparison_figures

OUT = ROOT/'results/short_screening/s64_paired_preprocessing_cpu'
CACHE = ROOT/'.cache/s64_paired_preprocessing_cpu'
WEIGHTS = ROOT/'.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth'


def main():
    if (OUT/'summary.json').exists():
        print((OUT/'summary.json').read_text()); return
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
    plan=dict(question='Does official aspect-preserving resize/center-crop improve frozen ConvNeXt representation over square resizing?',
        control='224x224 bilinear square resize',candidate='Short side236 bilinear resize then center crop224',
        weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1',precision='CPU FP32 for both',
        head='StandardScaler + LogisticRegression C1 max_iter1000 seed42; train-only normalized sqrt inverse-frequency sample weights',
        cohorts='Same complete exploratory training7009 and validation1503; no validation fit/tuning',
        gate='Accuracy >= control+0.01 AND macro-F1 >= control AND melanoma recall >= control',
        limit='Exactly two predefined transforms; no further transform/head search or automatic GPU training',
        caveat='Changes geometry and cropping/visible field jointly, not a pure geometry ablation. Frozen-feature results do not predict fine-tuning or test accuracy.',
        source='https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html')
    write_json(OUT/'PREDECLARED_PLAN.json',plan)
    c=config(); train,val,w=development(c)
    assert WEIGHTS.is_file(), 'Pretrained weights must already exist; no download'
    weight_hash=sha256(WEIGHTS)
    write_json(OUT/'source_manifest.json',dict(weights_path=relative(WEIGHTS),weights_sha256=weight_hash,
        script_sha256=sha256(ROOT/'research/short_screening/preprocessing_screen_v1.py'),
        split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],class_order=list(CLASSES),
        test_loaded=False,gpu_used=False,train_images=len(train),validation_images=len(val)))
    torch.set_num_threads(4); torch.manual_seed(42)
    net=ResearchClassifier('convnext_tiny',weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1').eval()
    for p in net.parameters(): p.requires_grad_(False)
    variants={
        'square224': transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
        'official236_crop224': transforms.Compose([
            transforms.Resize(236,interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
            transforms.CenterCrop(224)])}
    matrices={k:[] for k in variants}; start=time.perf_counter()
    for part,frame in [('train',train),('val',val)]:
        for name,resize in variants.items():
            cache=CACHE/f'{name}_{part}.npz'
            if cache.exists():
                with np.load(cache,allow_pickle=False) as saved:
                    assert saved['ids'].tolist()==frame.image_id.tolist()
                    assert saved['y'].tolist()==frame.label.tolist()
                    assert str(saved['weights_sha256'])==weight_hash
                    a=saved['features'].copy()
                logging.info('Verified existing CPU feature cache %s %s',name,part)
            else:
                ds=DevelopmentImages(frame,c,training=(part=='train'))
                ds.transform=transforms.Compose([resize,transforms.ToTensor(),
                    transforms.Normalize(c['normalization_mean'],c['normalization_std'])])
                loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0)
                vectors=[]; seen=[]; labels=[]
                with torch.inference_mode():
                    for step,(x,y,ids) in enumerate(loader):
                        assert x.device.type=='cpu' and torch.isfinite(x).all()
                        z=net.head[:3](net.features(x))
                        assert torch.isfinite(z).all()
                        vectors.append(z.numpy());seen.extend(ids);labels.extend(y.tolist())
                        if step%100==0: logging.info('%s %s %d/%d CPU features',name,part,len(seen),len(frame))
                        if time.perf_counter()-start>1800:
                            raise TimeoutError('Hard30-minute CPU screen budget exceeded; no automatic retry/training')
                a=np.concatenate(vectors)
                assert seen==frame.image_id.tolist() and labels==frame.label.tolist()
                np.savez_compressed(cache,features=a,ids=np.array(seen),y=np.array(labels),weights_sha256=np.array(weight_hash))
            assert a.shape==(len(frame),768) and np.isfinite(a).all()
            matrices[name].append(a)
    y=val.label.to_numpy(); probs={}; metrics={}
    import joblib
    for name,(xtrain,xval) in matrices.items():
        head=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
        with threadpool_limits(limits=4):
            head.fit(xtrain,train.label,logisticregression__sample_weight=w.numpy()[train.label.to_numpy()])
            assert head[-1].classes_.tolist()==list(range(7)) and head[-1].n_iter_.max()<1000
            p=head.predict_proba(xval)
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
        m=metric_report(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()))
        target=OUT/name
        write_json(target/'validation_metrics.json',m)
        rows=predictions(val.image_id,y,p)
        write_csv(target/'validation_predictions.csv',rows)
        write_csv(target/'validation_probabilities.csv',pd.DataFrame(rows)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
        joblib.dump(head,CACHE/f'{name}_head.joblib')
        metric_figures(m,target/'figures',f'S64 {name}: frozen features, exploratory validation')
        saved=pd.read_csv(target/'validation_predictions.csv')
        assert saved.image_id.tolist()==val.image_id.tolist()
        recalculated=metric_report(y,saved[[f'p_{cl}' for cl in CLASSES]].to_numpy(),m['loss'])
        assert abs(recalculated['accuracy']-m['accuracy'])<1e-12
        assert abs(recalculated['macro_f1']-m['macro_f1'])<1e-12
        assert recalculated['confusion_matrix']==m['confusion_matrix']
        metrics[name]=m;probs[name]=p
        logging.info('%s accuracy %.6f macroF1 %.6f',name,m['accuracy'],m['macro_f1'])
    a=metrics['square224'];b=metrics['official236_crop224']
    class_rows=[]
    for i,cl in enumerate(CLASSES):
        def stats(m):
            cm=np.array(m['confusion_matrix']); tp=cm[i,i]
            return dict(precision=float(tp/max(cm[:,i].sum(),1)),recall=float(tp/max(cm[i].sum(),1)),
                f1=float(2*tp/max(cm[:,i].sum()+cm[i].sum(),1)),support=int(cm[i].sum()))
        ca=stats(a);cb=stats(b)
        class_rows.append(dict(class_name=cl,**{f'control_{k}':v for k,v in ca.items()},**{f'candidate_{k}':v for k,v in cb.items()}))
    mel=next(r for r in class_rows if r['class_name']=='mel')
    passed=b['accuracy']>=a['accuracy']+.01-1e-12 and b['macro_f1']>=a['macro_f1']-1e-12 and mel['candidate_recall']>=mel['control_recall']-1e-12
    ac=probs['square224'].argmax(1)==y;bc=probs['official236_crop224'].argmax(1)==y
    gained=int((~ac&bc).sum());lost=int((ac&~bc).sum())
    write_csv(OUT/'class_comparison.csv',class_rows)
    comparison_figures([dict(display_name=k,accuracy=m['accuracy'],macro_f1=m['macro_f1']) for k,m in metrics.items()],OUT/'comparison_figures','S64 paired CPU pretrained-feature preprocessing screen')
    assert sha256(WEIGHTS)==weight_hash
    write_json(OUT/'verification.json',dict(status='passed',same_weights=True,paired_cpu_fp32=True,
        train_only_head_fit=True,all_images_complete=True,source_weights_unchanged=True,
        metrics_and_confusion_matrix_independently_recomputed=True,test_loaded=False,gpu_used=False))
    summary=dict(status='completed',gate_passed=bool(passed),metrics=metrics,gained=gained,lost=lost,net_correct_change=gained-lost,
        runtime_seconds=time.perf_counter()-start,test_loaded=False,gpu_used=False,training_epochs=0,
        decision='prepare_training_proposal_only' if passed else 'do_not_launch_training_from_this_screen',
        ensemble_reference_unchanged_accuracy=.936127744510978)
    for name,m in metrics.items():
        retry_registry_upsert(dict(experiment_id=f's64_{name}_frozen_cpu_exploratory_seed42',era='structured',
            record_kind='frozen_feature_probe',phase='post_test_exploratory_development',protocol=c['protocol'],
            evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
            model='convnext_tiny',method='Paired CPU FP32 frozen pretrained features; fixed train-only weighted logistic regression C1; '+name,
            epochs=0,seed=42,status='completed',decision='gate_passed' if passed else 'gate_failed',
            metrics_path=relative(OUT/name/'validation_metrics.json'),plots_dir=relative(OUT/name/'figures'),
            notes='Representation diagnostic, not fine-tuned ensemble/test performance. Same full cohorts and pretrained weights.',
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    write_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    with run_lock('s64_paired_preprocessing_cpu'): main()
