"""One bounded two-resolution development inference screen, no training/test loader."""
import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from research.common import ROOT, CLASSES, sha256, write_json, write_csv, relative
from research.models import ResearchClassifier
from research.strict_protocol import development_data, partition_metadata, DIGEST
from research.strict_train import DevelopmentImages, metric_report, run_lock
from research.aggressive.core import predictions, retry_registry_upsert
from research.plots import metric_figures, comparison_figures

CFG=ROOT/'research/short_screening/resolution_screen_v1.json'
OUT=ROOT/'results/short_screening/resolution_screen_v1'


def setup():
    c=json.loads(CFG.read_text())
    assert c['split_sha256']==DIGEST and c['resolutions']==[224,320]
    assert c['training_epochs']==0 and c['test_evaluation'] is False
    assert sha256(ROOT/c['source_checkpoint'])==c['source_sha256']
    _,val,_=development_data(c)
    return c,val


def report(val,p):
    return metric_report(val.label.to_numpy(),p,float(-np.log(np.maximum(p[np.arange(len(val)),val.label],1e-12)).mean()))


def run(check=False):
    c,val=setup()
    torch.set_num_threads(4)
    if check:
        m=ResearchClassifier('convnext_tiny',weights=None)
        cp=torch.load(ROOT/c['source_checkpoint'],map_location='cpu',weights_only=False)
        m.load_state_dict(cp['model']);m.eval()
        with torch.inference_mode():
            for size in c['resolutions']:
                dataset=DevelopmentImages(val,{**c,'image_size':size})
                x,_,_=dataset[0];z=m(x.unsqueeze(0))
                assert z.shape==(1,7) and torch.isfinite(z).all()
        print('CPU preflight passed: source hash, split, 224/320 finite logits; no GPU/test used',flush=True)
        return
    if (OUT/'summary.json').exists():
        print('Already completed; preserved.');return
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(c['seed'])
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
    write_json(OUT/'verification.json',dict(config=c,config_sha256=sha256(CFG),
        source_sha256=sha256(ROOT/c['source_checkpoint']),split=partition_metadata()[1],
        class_order=CLASSES,training=False,test_loaded=False))
    write_json(OUT/'progress.json',dict(status='initializing',pid=os.getpid()))
    cp=torch.load(ROOT/c['source_checkpoint'],map_location='cpu',weights_only=False)
    model=ResearchClassifier('convnext_tiny',weights=None);model.load_state_dict(cp['model']);del cp
    model=model.cuda().eval()
    start=time.monotonic();arrays=[];comparisons=[]
    logging.info('START PID=%s; preserved epoch33 checkpoint verified; FP32 validation only; resolutions224/320; no training/test',os.getpid())
    for size in c['resolutions']:
        folder=OUT/str(size);folder.mkdir(exist_ok=True)
        dataset=DevelopmentImages(val,{**c,'image_size':size})
        loader=DataLoader(dataset,batch_size=c['batch_size'],shuffle=False,num_workers=c['workers'])
        parts=[];ids=[]
        with torch.inference_mode():
            for i,(images,labels,image_ids) in enumerate(loader):
                if time.monotonic()-start>1200:raise TimeoutError('Hard 20-minute screen budget exceeded')
                logits=model(images.cuda())
                if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite FP32 inference')
                parts.append(logits.softmax(1).cpu().numpy());ids.extend(image_ids)
                if i==0:
                    np.save(folder/'first_batch_probabilities.npy',parts[0])
                    write_json(OUT/'progress.json',dict(status='inference_active',pid=os.getpid(),resolution=size,first_batch_saved=True))
                    logging.info('FIRST_BATCH %s: finite probabilities saved; original checkpoint reused unchanged',size)
        assert ids==val.image_id.tolist()
        p=np.concatenate(parts);assert np.allclose(p.sum(1),1,atol=1e-5)
        metrics=report(val,p);arrays.append(p)
        write_json(folder/'validation_metrics.json',metrics)
        write_csv(folder/'validation_predictions.csv',predictions(ids,val.label.to_numpy(),p))
        write_csv(folder/'validation_probabilities.csv',pd.DataFrame(p,columns=[f'p_{cl}' for cl in CLASSES]).assign(image_id=ids).to_dict('records'))
        metric_figures(metrics,folder/'figures',f'S39 ConvNeXt FP32 {size}px | strict development')
        comparisons.append(dict(experiment_id=f"{c['experiment_id']}_{size}",display_name=f'ConvNeXt {size}px',**{k:metrics[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
        logging.info('COMPLETED %spx accuracy=%.6f F1=%.6f',size,metrics['accuracy'],metrics['macro_f1'])
    old=pd.read_csv(ROOT/'results/structured_experiments/s29_convnext_tiny_final_strict_seed42/validation_predictions.csv').set_index('image_id').loc[val.image_id]
    assert np.array_equal(arrays[0].argmax(1),old[[f'p_{cl}' for cl in CLASSES]].to_numpy().argmax(1)), '224 control predictions changed; stop comparison'
    fixed=[]
    for rid in ['s28_efficientnet_b0_final_strict_seed42','s30_efficientnet_v2_s_final_strict_seed42']:
        frame=pd.read_csv(ROOT/'results/structured_experiments'/rid/'validation_predictions.csv').set_index('image_id').loc[val.image_id]
        assert frame.true_class.tolist()==val.diagnosis.tolist()
        fixed.append(frame[[f'p_{cl}' for cl in CLASSES]].to_numpy())
    ensembles=[]
    for size,p in zip(c['resolutions'],arrays):
        folder=OUT/f'ensemble_{size}';folder.mkdir(exist_ok=True)
        fused=(fixed[0]+p+fixed[1])/3;metrics=report(val,fused);ensembles.append(metrics)
        write_json(folder/'validation_metrics.json',metrics)
        write_csv(folder/'validation_predictions.csv',predictions(val.image_id,val.label.to_numpy(),fused))
        write_csv(folder/'validation_probabilities.csv',pd.DataFrame(fused,columns=[f'p_{cl}' for cl in CLASSES]).assign(image_id=val.image_id.tolist()).to_dict('records'))
        metric_figures(metrics,folder/'figures',f'S39 fixed equal ensemble | ConvNeXt {size}px')
        row=dict(experiment_id=f"{c['experiment_id']}_ensemble_{size}",display_name=f'Fixed ensemble ConvNeXt {size}px',**{k:metrics[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']})
        comparisons.append(row)
        retry_registry_upsert(dict(**{k:v for k,v in row.items() if k!='display_name'},era='structured',record_kind='inference_screen',phase='post_test_development',protocol='strict_lesion_disjoint',evaluation_split='val',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],method='equal fusion; only ConvNeXt resolution differs',image_size=f'224/{size}/224',seed=42,epochs=0,checkpoint=c['source_checkpoint'],checkpoint_sha256=c['source_sha256'],config_path=relative(CFG),metrics_path=relative(folder/'validation_metrics.json'),plots_dir=relative(folder/'figures'),status='completed',notes='Post-test development; fixed weights; no training/test inference; 320 checkpoint trained at224.'))
    b,m=ensembles
    passed=(m['accuracy']-b['accuracy']>=.005-1e-12 and m['macro_f1']>=b['macro_f1'] and
            m['per_class']['mel']['recall']-b['per_class']['mel']['recall']>=.02-1e-12 and
            m['per_class']['nv']['recall']>=b['per_class']['nv']['recall']-.01)
    assert sha256(ROOT/c['source_checkpoint'])==c['source_sha256']
    write_csv(OUT/'comparison.csv',comparisons);comparison_figures(comparisons,OUT/'figures_comparison','S39 fixed resolution screen')
    write_json(OUT/'summary.json',dict(status='completed',material_gate_passed=bool(passed),runtime_seconds=time.monotonic()-start,comparisons=comparisons,no_training=True,test_loaded=False,decision='stop; user returns for closeout; no automatic training'))
    write_json(OUT/'progress.json',dict(status='completed',pid=os.getpid()))
    logging.info('SCREEN COMPLETE; material_gate=%s; STOP; no automatic next experiment',passed)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.check and not args.run:parser.error('Specify --check or --run')
    if args.check:
        run(check=True)
    else:
        OUT.mkdir(parents=True,exist_ok=True)
        with run_lock(OUT/'process.lock'):
            try:
                run()
            except Exception as exc:
                write_json(OUT/'progress.json',dict(status='failed',pid=os.getpid(),error=repr(exc)))
                logging.exception('Screen failed; evidence preserved; no automatic retry/training')
                raise
