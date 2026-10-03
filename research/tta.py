"""Fixed S12 four-view validation inference; approval required, no training/test loader.
--check is CPU-only metadata/provenance verification. Completed views are resumable.
"""
import argparse,json,logging,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from research.common import ROOT,CLASSES,sha256,write_json,write_csv
from research.fuse import preflight,aligned_probabilities
from research.models import ResearchClassifier
from research.train import development_data,DevelopmentImages,metric_report,tensor_nonfinite_names
from research.plots import metric_figures,comparison_figures
from research.registry import FIELDS,upsert
from research.phase3.close_s02 import load,verify_predictions,verify_figures

ID='s13_s12_four_flip_tta_exploratory_seed42'

def check(config):
    if (config['experiment_id']!=ID or config['views']!=['identity','horizontal','vertical','both']
        or config['precision']!='fp32' or config['image_size']!=224 or config['batch_size']!=16):
        raise ValueError('Only the fixed reviewed S13 candidate is supported')
    parent_path=ROOT/config['parent_fusion_config']
    assert sha256(parent_path)==config['parent_fusion_sha256']
    parent=load(parent_path)
    frame,val,parents,_=preflight(parent)
    for k in ('parent_run_ids','weights','split_sha256','checkpoint_sha256','protocol'):
        assert config[k]==parent[k]
    for r,m in parents:
        assert sha256(ROOT/r['checkpoint'])==config['checkpoint_sha256'][r['experiment_id']]
    return frame,val,parents

def flip(images,view):
    dims={'identity':[], 'horizontal':[-1], 'vertical':[-2], 'both':[-2,-1]}[view]
    return images.flip(dims) if dims else images

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--config',type=Path,required=True);a.add_argument('--check',action='store_true');a.add_argument('--resume',action='store_true')
    args=a.parse_args();config=load(args.config);frame,val,parents=check(config)
    out=ROOT/'results/structured_experiments'/ID
    if args.check:
        print(json.dumps(dict(status='prepared_inference_only',validation_images=len(val),views=4,model_passes=12,precision='fp32',gpu_used=False,test_loader=False,output=str(out))));return
    if out.exists() and not args.resume: raise ValueError('Existing output; inspect then --resume')
    out.mkdir(parents=True,exist_ok=True)
    signature=dict(config=config,runner_sha256=sha256(Path(__file__)),supporting_source_hashes={f:sha256(ROOT/f) for f in ('research/models.py','research/train.py','research/fuse.py','research/common.py','research/plots.py','research/registry.py')})
    if (out/'inference_signature.json').exists(): assert load(out/'inference_signature.json')==signature
    else: write_json(out/'inference_signature.json',signature)
    write_json(out/'config.json',config)
    if (out/'record.json').exists() and load(out/'record.json')['status']=='completed': print('Already completed; preserved.');return
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log',encoding='utf-8'),logging.StreamHandler()])
    record={k:'' for k in FIELDS}
    record.update(experiment_id=ID,era='structured',record_kind='fixed_tta_inference',phase='inference_screening',protocol=config['protocol'],evaluation_split='validation',split_manifest='data/splits/exploratory/image_level_dev_v1.csv',split_sha256=config['split_sha256'],method=config['question'],model='+'.join(r['model'] for r,m in parents),attention='none',image_size=224,seed=42,status='running',config_path=(out/'config.json').relative_to(ROOT).as_posix(),ensemble_members=json.dumps(config['parent_run_ids']),ensemble_weights=json.dumps(config['weights']),checkpoint=json.dumps([r['checkpoint'] for r,m in parents]),checkpoint_available_local=True)
    upsert(record);write_json(out/'record.json',record)
    start=time.perf_counter();all_views=[];identities=[]
    try:
        if not torch.cuda.is_available(): raise RuntimeError('Approved inference requires CUDA')
        write_json(out/'environment.json',dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(0),precision='fp32',training=False))
        logging.info('START fixed 4-view FP32 inference, 3 saved checkpoints, validation only; no training')
        for r,m in parents:
            rid=r['experiment_id'];saved=load(ROOT/'results/structured_experiments'/rid/'config.json')
            _,validation,_=development_data(saved)
            assert validation.image_id.tolist()==val.index.tolist()
            loader=DataLoader(DevelopmentImages(validation,saved,False),batch_size=16,shuffle=False,num_workers=0)
            checkpoint=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False)
            assert checkpoint['config']==saved
            assert not tensor_nonfinite_names(checkpoint['model'])
            model=ResearchClassifier(saved['model'],None,saved['attention'],saved['head_dropout'])
            model.load_state_dict(checkpoint['model']);model=model.eval().float().cuda()
            arrays=[]
            for view in config['views']:
                dest=out/f'{rid}_{view}.csv'
                if dest.exists():
                    probabilities=aligned_probabilities(pd.read_csv(dest),val)
                else:
                    values=[]
                    with torch.inference_mode():
                        for images,targets,ids in loader:
                            logits=model(flip(images.cuda(),view))
                            probs=logits.softmax(1)
                            if not torch.isfinite(logits).all() or not torch.isfinite(probs).all(): raise FloatingPointError(f'Nonfinite FP32 inference {rid}/{view}: {ids}')
                            values.append(probs.cpu().numpy())
                    probabilities=np.concatenate(values)
                    write_csv(dest,[dict(image_id=i,true_class=t,predicted_class=CLASSES[int(p.argmax())],**{f'p_{c}':float(p[j]) for j,c in enumerate(CLASSES)}) for i,t,p in zip(val.index,val.diagnosis,probabilities)])
                    probabilities=aligned_probabilities(pd.read_csv(dest),val)
                arrays.append(probabilities)
                write_json(out/'progress.json',dict(parent=rid,view=view,status='view_saved',precision='fp32'))
                logging.info('SAVED %s %s',rid,view)
            identities.append(arrays[0]);all_views.append(np.mean(arrays,axis=0));del model
        counts=frame.query("split=='train'").diagnosis.value_counts();weights=np.sqrt(7009/np.array([counts[c] for c in CLASSES]));weights/=weights.mean()
        results={}
        for name,values in (('identity_fp32',identities),('tta',all_views)):
            probabilities=sum(w*p for w,p in zip(config['weights'],values));labels=val.label.to_numpy()
            loss=float(np.average(-np.log(np.clip(probabilities[np.arange(len(val)),labels],1e-12,1)),weights=weights[labels]))
            metrics=metric_report(labels,probabilities,loss);suffix='' if name=='tta' else '_identity_fp32'
            write_json(out/f'validation_metrics{suffix}.json',metrics)
            predictions=out/f'validation_predictions{suffix}.csv'
            write_csv(predictions,[dict(image_id=i,true_class=t,predicted_class=CLASSES[int(p.argmax())],**{f'p_{c}':float(p[j]) for j,c in enumerate(CLASSES)}) for i,t,p in zip(val.index,val.diagnosis,probabilities)])
            cm=verify_predictions(predictions,metrics,val);folder=out/('figures' if name=='tta' else 'figures_identity_fp32')
            metric_figures(metrics,folder,f'S13 {name} | exploratory validation');verify_figures(folder,cm,metrics);results[name]=metrics
        old=load(ROOT/'results/structured_experiments/s12_s03_s06_s10_equal_probability_exploratory_seed42/validation_metrics.json')
        comparison_figures([dict(display_name=n,accuracy=m['accuracy'],macro_f1=m['macro_f1']) for n,m in [('S12 saved AMP',old),('S13 FP32 identity control',results['identity_fp32']),('S13 FP32 four-view TTA',results['tta'])]],ROOT/'results/model_comparison/structured/s13_tta','Exploratory validation | precision control and fixed TTA | 3 vs 12 passes')
        m=results['tta'];record.update(status='completed',runtime_seconds=time.perf_counter()-start,**{k:m[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1')},val_loss=m['loss'],metrics_path=(out/'validation_metrics.json').relative_to(ROOT).as_posix(),source_sha256=sha256(out/'validation_metrics.json'),source_availability='original',plots_dir=(out/'figures').relative_to(ROOT).as_posix(),confusion_matrix_path=(out/'figures/confusion_matrix.csv').relative_to(ROOT).as_posix(),decision='fixed_tta_completed_pending_review',notes='FP32 inference for all models/views with matched FP32 identity control; no training/new checkpoints/test images; 12 model passes; no view/weight sweep. Parent S10 recovery disclosed in parent closeout.')
        write_json(out/'closeout_verification.json',dict(metrics=results,gpu_inference=True,training=False,test_images_loaded=False,parent_checkpoints=config['checkpoint_sha256'],signature=signature))
        logging.info('COMPLETED accuracy=%.6f macro_f1=%.6f',m['accuracy'],m['macro_f1'])
    except BaseException as exc:
        record.update(status='failed',notes=repr(exc));logging.exception('FAILED; saved views and parent checkpoints preserved');raise
    finally:
        upsert(record);write_json(out/'record.json',record)

if __name__=='__main__': main()
