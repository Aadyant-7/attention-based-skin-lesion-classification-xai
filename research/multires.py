"""Fixed224/320 ConvNeXt inference in S12. --check is metadata-only, no CUDA/images.
Approval required for actual GPU inference. Other224px parent outputs are reused.
"""
import argparse,json,logging,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from research.common import ROOT,CLASSES,sha256,write_json,write_csv
from research.tta import check as check_source
from research.fuse import aligned_probabilities
from research.train import DevelopmentImages,development_data,metric_report,tensor_nonfinite_names
from research.models import ResearchClassifier
from research.plots import metric_figures,comparison_figures
from research.registry import FIELDS,upsert
from research.phase3.close_s02 import load,verify_predictions,verify_figures
ID='s14_s12_convnext_224_320_exploratory_seed42'
S06='s06_convnext_tiny_none_exploratory_seed42'

def check(config):
    if (config['experiment_id']!=ID or config['resolutions']!=[224,320] or config['resolution_weights']!=[.5,.5]
        or config['multires_parent']!=S06 or config['precision']!='fp32' or config['batch_size']!=8 or config['views']!=['identity']):
        raise ValueError('Only one fixed reviewed resolution intervention is supported')
    source=ROOT/config['source_config'];assert sha256(source)==config['source_config_sha256']
    original=load(source);frame,val,parents=check_source(original)
    for k in ('protocol','parent_run_ids','checkpoint_sha256','weights','split_sha256'):assert config[k]==original[k]
    folder=ROOT/'results/structured_experiments'/original['experiment_id'];probabilities={}
    for rid in config['parent_run_ids']:
        f=folder/f'{rid}_identity.csv';assert sha256(f)==config['identity_prediction_sha256'][rid]
        probabilities[rid]=aligned_probabilities(pd.read_csv(f),val)
    assert sha256(folder/'validation_predictions_identity_fp32.csv')==config['baseline_prediction_sha256']
    assert sha256(folder/'validation_metrics_identity_fp32.json')==config['baseline_metrics_sha256']
    baseline=aligned_probabilities(pd.read_csv(folder/'validation_predictions_identity_fp32.csv'),val)
    assert np.allclose(sum(w*probabilities[r] for w,r in zip(config['weights'],config['parent_run_ids'])),baseline,rtol=0,atol=1e-12)
    return frame,val,parents,probabilities,folder

def fixed_fusion(config,identity,higher):
    values=dict(identity)
    values[S06]=config['resolution_weights'][0]*identity[S06]+config['resolution_weights'][1]*higher
    return sum(w*values[r] for w,r in zip(config['weights'],config['parent_run_ids']))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--check',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=load(args.config);frame,val,parents,identity,source=check(config)
    out=ROOT/'results/structured_experiments'/ID
    if args.check:
        print(json.dumps(dict(status='prepared_inference_only',validation_images=1503,new_model_passes=1,deployment_model_passes=4,gpu_used=False,test_loader=False,output=str(out))));return
    if out.exists() and not args.resume:raise ValueError('Existing output; inspect then --resume')
    out.mkdir(parents=True,exist_ok=True)
    signature=dict(config=config,source_hashes={f:sha256(ROOT/f) for f in ('research/multires.py','research/tta.py','research/train.py','research/models.py','research/fuse.py','research/common.py','research/plots.py','research/registry.py')})
    if (out/'inference_signature.json').exists():assert load(out/'inference_signature.json')==signature
    else:write_json(out/'inference_signature.json',signature)
    write_json(out/'config.json',config)
    if (out/'record.json').exists() and load(out/'record.json')['status']=='completed':print('Already completed; preserved.');return
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log',encoding='utf-8'),logging.StreamHandler()])
    record={k:'' for k in FIELDS}
    record.update(experiment_id=ID,era='structured',record_kind='fixed_multires_inference',phase='inference_screening',protocol=config['protocol'],evaluation_split='validation',split_manifest='data/splits/exploratory/image_level_dev_v1.csv',split_sha256=config['split_sha256'],method=config['question'],model='+'.join(r['model'] for r,m in parents),attention='none',image_size='224;ConvNeXt224/320',batch_size=8,seed=42,ensemble_members=json.dumps(config['parent_run_ids']),ensemble_weights=json.dumps(config['weights']),checkpoint=json.dumps([r['checkpoint'] for r,m in parents]),checkpoint_available_local=True,config_path=(out/'config.json').relative_to(ROOT).as_posix(),status='running')
    upsert(record);write_json(out/'record.json',record);start=time.perf_counter()
    try:
        if not torch.cuda.is_available():raise RuntimeError('Approved inference requires CUDA')
        write_json(out/'environment.json',dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(0),precision='fp32',training=False,batch_size=8))
        logging.info('START fixed224/320 ConvNeXt intervention; saved FP32 identity control; validation only')
        f=out/'convnext_320_predictions.csv'
        if f.exists():higher=aligned_probabilities(pd.read_csv(f),val)
        else:
            parent=next(r for r,m in parents if r['experiment_id']==S06);saved=load(ROOT/'results/structured_experiments'/S06/'config.json')
            _,validation,_=development_data(saved);assert validation.image_id.tolist()==val.index.tolist()
            inference_config=dict(saved,image_size=320)
            loader=DataLoader(DevelopmentImages(validation,inference_config,False),batch_size=8,shuffle=False,num_workers=0)
            checkpoint=torch.load(ROOT/parent['checkpoint'],map_location='cpu',weights_only=False)
            assert checkpoint['config']==saved and not tensor_nonfinite_names(checkpoint['model'])
            model=ResearchClassifier(saved['model'],None,saved['attention'],saved['head_dropout'])
            model.load_state_dict(checkpoint['model']);model=model.eval().float().cuda()
            logging.info('CHECKPOINT loaded;320px output batches begin')
            values=[]
            with torch.inference_mode():
                for batch,(images,targets,ids) in enumerate(loader):
                    logits=model(images.cuda());probabilities=logits.softmax(1)
                    if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():raise FloatingPointError(f'Nonfinite FP32 outputs: {ids}')
                    values.append(probabilities.cpu().numpy())
                    if batch==0:
                        np.save(out/'startup_probabilities.npy',values[0]);write_json(out/'progress.json',dict(status='first_batch_saved',images=len(values[0]),image_size=320))
            higher=np.concatenate(values)
            write_csv(f,[dict(image_id=i,true_class=t,predicted_class=CLASSES[int(p.argmax())],**{f'p_{c}':float(p[j]) for j,c in enumerate(CLASSES)}) for i,t,p in zip(val.index,val.diagnosis,higher)])
            higher=aligned_probabilities(pd.read_csv(f),val);del model
        logging.info('SAVED full1503-image ConvNeXt320 probabilities')
        counts=frame.query("split=='train'").diagnosis.value_counts();weights=np.sqrt(7009/np.array([counts[c] for c in CLASSES]));weights/=weights.mean();labels=val.label.to_numpy()
        metrics={};fused=fixed_fusion(config,identity,higher)
        for name,probabilities in [('convnext_320',higher),('multires',fused)]:
            loss=float(np.average(-np.log(np.clip(probabilities[np.arange(len(val)),labels],1e-12,1)),weights=weights[labels]));m=metric_report(labels,probabilities,loss);metrics[name]=m
            suffix='' if name=='multires' else '_convnext_320';write_json(out/f'validation_metrics{suffix}.json',m)
            prediction=out/f'validation_predictions{suffix}.csv'
            write_csv(prediction,[dict(image_id=i,true_class=t,predicted_class=CLASSES[int(p.argmax())],**{f'p_{c}':float(p[j]) for j,c in enumerate(CLASSES)}) for i,t,p in zip(val.index,val.diagnosis,probabilities)])
            cm=verify_predictions(prediction,m,val);folder=out/('figures' if name=='multires' else 'figures_convnext_320');metric_figures(m,folder,f'S14 {name} | exploratory validation');verify_figures(folder,cm,m)
        baseline=load(source/'validation_metrics_identity_fp32.json');amp=load(ROOT/'results/structured_experiments/s12_s03_s06_s10_equal_probability_exploratory_seed42/validation_metrics.json')
        comparison_figures([dict(display_name=n,**{k:m[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1')}) for n,m in [('S12 saved AMP',amp),('S12 FP32 identity',baseline),('S14 fixed224/320',metrics['multires'])]],ROOT/'results/model_comparison/structured/s14_multires','Same exploratory validation | one fixed resolution intervention |3 vs4 model passes')
        ok=np.asarray(CLASSES)[fused.argmax(1)]==val.diagnosis.to_numpy();original=pd.read_csv(source/'validation_predictions_identity_fp32.csv').set_index('image_id').loc[val.index];other=original.predicted_class.to_numpy()==val.diagnosis.to_numpy()
        m=metrics['multires'];record.update(status='completed',**{k:m[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1')},val_loss=m['loss'],runtime_seconds=time.perf_counter()-start,metrics_path=(out/'validation_metrics.json').relative_to(ROOT).as_posix(),source_sha256=sha256(out/'validation_metrics.json'),plots_dir=(out/'figures').relative_to(ROOT).as_posix(),confusion_matrix_path=(out/'figures/confusion_matrix.csv').relative_to(ROOT).as_posix(),source_availability='original',decision='fixed_resolution_completed_pending_review',notes='FP32;unchanged S03/S06/S10 best weights;equal224/320 ConvNeXt only;equal three-model weights;other identity probabilities reused;no TTA/train/test/new checkpoints;4 deployment passes.320 batch8 vs cached224 batch16 disclosed.')
        write_json(out/'closeout_verification.json',dict(metrics=metrics,paired_control=dict(fixed=int((ok&~other).sum()),broken=int((~ok&other).sum()),net_correct=int(ok.sum()-other.sum())),signature=signature,gpu_inference=True,training=False,test_images_loaded=False))
        write_json(out/'progress.json',dict(status='completed',validation_images=1503));logging.info('COMPLETED accuracy=%.6f macro_f1=%.6f',m['accuracy'],m['macro_f1'])
    except BaseException as exc:
        record.update(status='failed',notes=repr(exc));logging.exception('FAILED;parent weights/evidence preserved');raise
    finally:upsert(record);write_json(out/'record.json',record)
if __name__=='__main__':main()
