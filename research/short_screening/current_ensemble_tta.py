"""Single fixed four-view TTA study on S46, plus FP32 identity control."""
import argparse,json,logging,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from research.common import ROOT,CLASSES,sha256,relative,write_json,write_csv
from research.models import ResearchClassifier
from research.strict_train import DevelopmentImages,metric_report,run_lock
from research.aggressive.core import predictions,retry_registry_upsert
from research.plots import metric_figures,comparison_figures

OUT=ROOT/'results/short_screening/s50_s51_current_ensemble_tta'
PARENT=ROOT/'results/short_screening/s46_all_f1_checkpoint_fusion'
SPLIT='data/splits/exploratory/image_level_dev_v1.csv'
VIEWS=['identity','horizontal','vertical','both']
PLAN=dict(parent_manifest=relative(PARENT/'candidate_manifest.json'),views=VIEWS,weights=[.25]*4,
    view_weights=[.25]*4,precision='fp32',image_size=224,batch_size=16,
    preprocessing='RGB; square224 bilinear antialias; original per-model ImageNet normalization',
    question='Does fixed four-flip TTA improve the current S46 four-model F1-checkpoint candidate?',
    control='Fresh FP32 identity-only four-member mean; separates precision from views',
    protocol='exploratory_image_level',scope='Already selected exploratory development validation, not independent test',
    material_gate='>=+.005 accuracy over S46 with no macro-F1/MEL recall decline, and accuracy above FP32 identity',
    study_limit='Exactly identity control and one four-view average; no view/model/weight/checkpoint search',
    max_runtime_seconds=900,training=False,test_loaded=False)


def development():
    assert sha256(ROOT/SPLIT)=='75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb'
    ids=pd.read_csv(ROOT/SPLIT,usecols=['image_id','lesion_id','split'])
    excluded=set((ids.index[ids.split!='val']+1).tolist())
    val=pd.read_csv(ROOT/SPLIT,skiprows=lambda line:line in excluded)
    assert len(val)==1503 and val.image_id.is_unique and set(val.split)=={'val'}
    assert val.label.tolist()==val.diagnosis.map(dict(zip(CLASSES,range(7)))).tolist()
    strict=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','lesion_id','split'])
    test=strict.loc[strict.split=='test'];assert not set(val.image_id)&set(test.image_id)
    assert not set(val.lesion_id)&set(test.lesion_id)
    manifest=json.loads((PARENT/'candidate_manifest.json').read_text())
    cached=pd.read_csv(PARENT/'validation_predictions.csv').set_index('image_id').loc[val.image_id]
    assert cached.true_class.tolist()==val.diagnosis.tolist()
    for source in manifest['sources']:
        assert sha256(ROOT/source['checkpoint'])==source['checkpoint_sha256']
    files={p.stem:p for part in ['HAM10000_images_part_1','HAM10000_images_part_2'] for p in (ROOT/'data/raw/HAM10000'/part).glob('*.jpg')}
    val['path']=[str(files[i]) for i in val.image_id]
    return val,manifest


def flip(x,view):
    dims={'identity':[],'horizontal':[-1],'vertical':[-2],'both':[-2,-1]}[view]
    return x.flip(dims) if dims else x


def model_from(source):
    c=json.loads((ROOT/'results/structured_experiments'/source['run']/'config.json').read_text())
    cp=torch.load(ROOT/source['checkpoint'],map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in cp['model'].values() if torch.is_tensor(v))
    model=ResearchClassifier(c['model'],None,c['attention'],c['head_dropout']);model.load_state_dict(cp['model'])
    return model.eval().float(),c


def preflight():
    torch.set_num_threads(4);val,manifest=development()
    for source in manifest['sources']:
        model,c=model_from(source);dataset=DevelopmentImages(val,c)
        x,_,_=dataset[0];assert torch.equal(flip(flip(x,'both'),'both'),x)
        with torch.inference_mode():
            z=model(x.unsqueeze(0));assert z.shape==(1,7) and torch.isfinite(z).all()
        del model
    print('CPU preflight passed: four preserved F1 checkpoints, split, transforms, finite logits; no test/GPU',flush=True)


def report(y,p):return metric_report(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()))


def run():
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    val,manifest=development();OUT.mkdir(parents=True,exist_ok=True)
    signature=dict(plan=PLAN,source_manifest_sha256=sha256(PARENT/'candidate_manifest.json'),runner_sha256=sha256(Path(__file__)),split_sha256=sha256(ROOT/SPLIT),supporting_hashes={p:sha256(ROOT/p) for p in ['research/models.py','research/strict_train.py','research/common.py','research/plots.py']})
    if (OUT/'signature.json').exists():assert json.loads((OUT/'signature.json').read_text())==signature
    else:write_json(OUT/'signature.json',signature)
    write_json(OUT/'PREDECLARED_PLAN.json',PLAN)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
    start=time.perf_counter();identities=[];tta=[]
    write_json(OUT/'verification.json',dict(sources=manifest['sources'],class_order=CLASSES,validation_images=1503,locked_test_image_overlap=0,locked_test_lesion_overlap=0,test_images_loaded=False,test_labels_loaded=False,training=False,precision='fp32',signature=signature))
    logging.info('START:4preserved checkpoints x4fixed views, FP32, exploratory validation only; no training')
    for source in manifest['sources']:
        model,c=model_from(source);model=model.cuda();model_arrays=[]
        loader=DataLoader(DevelopmentImages(val,c),batch_size=16,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
        for view in VIEWS:
            dest=OUT/'source_views'/f"{source['run']}_{view}.csv"
            if dest.exists():
                saved=pd.read_csv(dest).set_index('image_id').loc[val.image_id]
                assert saved.true_class.tolist()==val.diagnosis.tolist()
                p=saved[[f'p_{cl}' for cl in CLASSES]].to_numpy()
            else:
                values=[];image_ids=[]
                with torch.inference_mode():
                    for images,targets,ids in loader:
                        if time.perf_counter()-start>900:raise TimeoutError('Hard15minute inference budget exceeded')
                        z=model(flip(images.cuda(),view));p=z.softmax(1)
                        if not torch.isfinite(z).all() or not torch.isfinite(p).all():raise FloatingPointError('Nonfinite FP32 view inference')
                        values.append(p.cpu().numpy());image_ids.extend(ids)
                assert image_ids==val.image_id.tolist();p=np.concatenate(values)
                write_csv(dest,predictions(val.image_id,val.label.to_numpy(),p))
            assert len(p)==1503 and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
            model_arrays.append(p);write_json(OUT/'progress.json',dict(status='view_saved',parent=source['run'],view=view,pid=os.getpid()))
            logging.info('SAVED %s %s',source['run'],view)
        identities.append(model_arrays[0]);tta.append(np.mean(model_arrays,axis=0));del model;torch.cuda.empty_cache()
    y=val.label.to_numpy();results={};arrays={};rows=[]
    cached=json.loads((PARENT/'validation_metrics.json').read_text())
    rows.append(dict(display_name='S46 saved-output reference',accuracy=cached['accuracy'],macro_f1=cached['macro_f1']))
    for index,(name,parts) in enumerate([('identity_fp32',identities),('four_view_tta',tta)],start=50):
        p=np.mean(parts,axis=0);m=report(y,p);folder=OUT/name
        write_json(folder/'validation_metrics.json',m);pred=predictions(val.image_id,y,p);write_csv(folder/'validation_predictions.csv',pred)
        write_csv(folder/'validation_probabilities.csv',pd.DataFrame(pred)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
        metric_figures(m,folder/'figures',f'S{index} current four-model {name} | exploratory validation')
        results[name]=m;arrays[name]=p;rows.append(dict(display_name=name,accuracy=m['accuracy'],macro_f1=m['macro_f1']))
        retry_registry_upsert(dict(experiment_id=f's{index}_current_four_{name}_exploratory_seed42',era='structured',record_kind='fixed_inference_screen',phase='post_test_exploratory_development',protocol='exploratory_image_level',evaluation_split='validation',split_manifest=SPLIT,split_sha256=sha256(ROOT/SPLIT),method=name,ensemble_members=json.dumps([s['run'] for s in manifest['sources']]),ensemble_weights='[0.25,0.25,0.25,0.25]',image_size=224,epochs=0,seed=42,metrics_path=relative(folder/'validation_metrics.json'),plots_dir=relative(folder/'figures'),status='completed',notes='Preserved S46 F1 checkpoints; fixed views/weights; FP32; no training/test; repeated validation selection.',**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    comparison_figures(rows,OUT/'comparison_figures','Current four-model precision control and fixed TTA')
    a,b=results['identity_fp32'],results['four_view_tta'];ac=arrays['identity_fp32'].argmax(1)==y;bc=arrays['four_view_tta'].argmax(1)==y
    passed=b['accuracy']-cached['accuracy']>=.005-1e-12 and b['accuracy']>a['accuracy'] and b['macro_f1']>=cached['macro_f1'] and b['per_class']['mel']['recall']>=cached['per_class']['mel']['recall']
    for source in manifest['sources']:assert sha256(ROOT/source['checkpoint'])==source['checkpoint_sha256']
    summary=dict(status='completed',comparisons=rows,runtime_seconds=time.perf_counter()-start,gained_vs_identity=int((~ac&bc).sum()),lost_vs_identity=int((ac&~bc).sum()),material_gate_passed=bool(passed),results=results,checkpoint_hashes_unchanged=True,test_loaded=False,training=False,gpu_inference=True,decision='Fixed study complete; no additional TTA patterns or automatic training')
    write_json(OUT/'summary.json',summary);write_json(OUT/'progress.json',dict(status='completed',pid=os.getpid()))
    logging.info('COMPLETE identity=%.6f TTA=%.6f; material_gate=%s',a['accuracy'],b['accuracy'],passed)
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.check:preflight()
    elif args.run:
        with run_lock('s50_s51_current_four_tta'):
            try:run()
            except Exception as exc:
                OUT.mkdir(parents=True,exist_ok=True);write_json(OUT/'failure.json',dict(error=repr(exc)));logging.exception('Stopped; saved views preserved');raise
    else:parser.error('Use --check or --run')
