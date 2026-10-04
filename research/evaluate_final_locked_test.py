"""Single authorized held-out inference. Exclusive receipt prevents a second pass."""
import gc,json,logging,os,time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from .common import ROOT,CLASSES,write_json,write_csv,sha256,atomic_text,relative
from .strict_protocol import partition_metadata,SPLIT
from .strict_train import code_hashes,metric_report,tensor_nonfinite_names,run_lock
from .models import ResearchClassifier
from .plots import metric_figures,comparison_figures,validate_metrics
from .registry import FIELDS,upsert

OUT=ROOT/'results/final_locked_test/v1'
FREEZE=ROOT/'results/final_strict/v1/ensemble/frozen_ensemble.json'

class LockedImages(Dataset):
    def __init__(self,identities,config):
        # Deliberately no labels in the inference dataset.
        self.ids=identities.image_id.tolist()
        files={}
        for folder in ('HAM10000_images_part_1','HAM10000_images_part_2'):
            for path in (ROOT/'data/raw/HAM10000'/folder).glob('*.jpg'):
                if path.stem in files:raise ValueError('Duplicate filename')
                files[path.stem]=path
        self.paths=[files[i] for i in self.ids]
        self.transform=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
            transforms.ToTensor(),transforms.Normalize(config['normalization_mean'],config['normalization_std'])])
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as image:x=self.transform(image.convert('RGB'))
        return x,self.ids[i]

def report(probability_files,freeze):
    # Labels become available for scoring only after all frozen predictions exist.
    identities=pd.read_csv(ROOT/SPLIT,usecols=['image_id','split'])
    excluded=set((identities.index[identities.split!='test']+1).tolist())
    cohort=pd.read_csv(ROOT/SPLIT,skiprows=lambda line:line in excluded)
    if len(cohort)!=1503 or not (cohort.label==cohort.diagnosis.map(dict(zip(CLASSES,range(7))))).all():raise ValueError('Locked labels invalid')
    ids=cohort.image_id.to_numpy();y=cohort.label.to_numpy();arrays=[];rows=[]
    for path,member in zip(probability_files,freeze['members']):
        frame=pd.read_csv(path)
        if not np.array_equal(frame.image_id,ids):raise ValueError('Locked prediction alignment mismatch')
        p=frame[[f'p_{c}' for c in CLASSES]].to_numpy(float)
        if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid frozen probabilities')
        arrays.append(p)
    final=np.mean(np.stack(arrays),axis=0)
    for name,p in [(m['model'],a) for m,a in zip(freeze['members'],arrays)]+[('S31 equal ensemble',final)]:
        true_p=p[np.arange(len(y)),y]
        # Test loss is unweighted NLL; no test-derived class weights.
        loss=float(-np.log(true_p).mean()) if (true_p>0).all() else None
        metrics=metric_report(y,p,loss if loss is not None else 0.)
        metrics['loss']=loss;metrics['loss_definition']='unweighted negative log likelihood; null if true-class probability is zero'
        support=np.array([metrics['per_class'][c]['support'] for c in CLASSES])
        metrics['weighted_f1']=float(np.dot(support,[metrics['per_class'][c]['f1'] for c in CLASSES])/support.sum())
        correct=int((p.argmax(1)==y).sum());metrics.update(correct=correct,incorrect=len(y)-correct,cohort_size=len(y),evaluation_split='locked_test',protocol='strict_lesion_disjoint')
        validate_metrics(metrics)
        folder=OUT/('ensemble' if name=='S31 equal ensemble' else name)
        write_json(folder/'test_metrics.json',metrics)
        predictions=[dict(image_id=i,true_class=CLASSES[t],predicted_class=CLASSES[int(a.argmax())],correct=bool(a.argmax()==t),**{f'p_{c}':float(a[j]) for j,c in enumerate(CLASSES)}) for i,t,a in zip(ids,y,p)]
        write_csv(folder/'test_predictions.csv',predictions)
        write_csv(folder/'test_probabilities.csv',[dict(image_id=i,**{f'p_{c}':float(a[j]) for j,c in enumerate(CLASSES)}) for i,a in zip(ids,p)])
        metric_figures(metrics,folder/'figures',name+' | final locked test')
        rows.append(dict(display_name=name,**{k:metrics[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1','weighted_f1')},correct=correct,incorrect=len(y)-correct))
    comparison_figures(rows,OUT/'figures_constituents','Frozen constituents and ensemble | one locked-test pass')
    strict=json.loads((ROOT/'results/final_strict/v1/ensemble/validation_metrics.json').read_text())
    exploratory=json.loads((ROOT/'results/structured_experiments/s13_s12_four_flip_tta_exploratory_seed42/validation_metrics_identity_fp32.json').read_text())
    final_metrics=json.loads((OUT/'ensemble/test_metrics.json').read_text())
    comparison_figures([dict(display_name='Exploratory validation',accuracy=exploratory['accuracy'],macro_f1=exploratory['macro_f1']),dict(display_name='Strict validation',accuracy=strict['accuracy'],macro_f1=strict['macro_f1']),dict(display_name='Locked test',accuracy=final_metrics['accuracy'],macro_f1=final_metrics['macro_f1'])],OUT/'figures_protocols','Frozen methodology | descriptive protocol comparison')
    delta=dict(test_vs_strict_accuracy_pp=100*(final_metrics['accuracy']-strict['accuracy']),test_vs_strict_macro_f1=final_metrics['macro_f1']-strict['macro_f1'],
        test_vs_exploratory_accuracy_pp=100*(final_metrics['accuracy']-exploratory['accuracy']),test_vs_exploratory_macro_f1=final_metrics['macro_f1']-exploratory['macro_f1'])
    write_json(OUT/'comparison_summary.json',delta);write_csv(OUT/'model_test_comparison.csv',rows)
    table='| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in rows:table+=f"| {r['display_name']} | {100*r['accuracy']:.4f}% | {r['macro_precision']:.6f} | {r['macro_recall']:.6f} | {r['macro_f1']:.6f} | {r['weighted_f1']:.6f} |\n"
    classes='| Class | Precision | Recall | F1 | Support |\n|---|---:|---:|---:|---:|\n'
    for c in CLASSES:
        a=final_metrics['per_class'][c];classes+=f"| {c} | {a['precision']:.6f} | {a['recall']:.6f} | {a['f1']:.6f} | {a['support']} |\n"
    text=f"""# Final locked-test results — first and only held-out evaluation

This is the first and only locked-test evaluation in the documented structured workflow. Test labels/images played no role in architecture selection, hyperparameter selection, ensemble weighting or checkpoint selection. Predictions from each frozen constituent were generated exactly once, with labels withheld from inference; standalone scores reuse those same probabilities. No alternative test method was evaluated and no methodology was modified after the result.

Frozen S31: S28 B0 epoch19 + S29 ConvNeXt-Tiny epoch33 + S30 EfficientNetV2-S epoch24. Equal probability averaging (1/3 each), FP32, identity only,224x224 bilinear antialias,RGB/ImageNet normalization. No inference augmentation,TTA,multi-resolution or tuned weights. Exact paths/hashes, unchanged config/source and zero lesion overlap verification are recorded in `results/final_locked_test/v1/pre_inference_verification.json`.

{table}
## S31 class performance

{classes}
Correct {final_metrics['correct']}; incorrect {final_metrics['incorrect']}; cohort1503. Melanoma recall {final_metrics['per_class']['mel']['recall']:.6f}; akiec recall {final_metrics['per_class']['akiec']['recall']:.6f}.

## Confusion matrix

Rows=true,columns=predicted; order {list(CLASSES)}.

```text
{np.array(final_metrics['confusion_matrix'])}
```

## Descriptive comparison

Exploratory validation {100*exploratory['accuracy']:.4f}% / {exploratory['macro_f1']:.6f}; strict validation {100*strict['accuracy']:.4f}% / {strict['macro_f1']:.6f}; locked test {100*final_metrics['accuracy']:.4f}% / {final_metrics['macro_f1']:.6f}.

Test minus strict: {delta['test_vs_strict_accuracy_pp']:+.4f} accuracy percentage points; {delta['test_vs_strict_macro_f1']:+.6f} macro-F1. Test minus exploratory: {delta['test_vs_exploratory_accuracy_pp']:+.4f} percentage points; {delta['test_vs_exploratory_macro_f1']:+.6f} macro-F1. Different cohorts/protocols are not interchangeable. Single split evidence does not establish robustness or statistical superiority; interpret melanoma/akiec recall separately from majority-class accuracy.

Outputs: `results/final_locked_test/v1/`: full probabilities/predictions, raw and normalized matrices, per-class scores, and publication PNG/PDF figures. No training checkpoints were overwritten.

Performance work is FINISHED. No additional test inference, architecture selection or optimization follows this report. Next phase: Grad-CAM/XAI, final figures/tables and paper writing.
"""
    atomic_text(ROOT/'research/FINAL_LOCKED_TEST_RESULTS.md',text)
    record={k:'' for k in FIELDS}
    record.update(experiment_id='s31_final_equal_fp32_strict_locked_test_seed42',era='structured',record_kind='final_locked_test',phase='final_held_out_evaluation',
        protocol='strict_lesion_disjoint',evaluation_split='locked_test',split_manifest=SPLIT,split_sha256=freeze['split_verification']['split_sha256'],
        method='First and only frozen S31 held-out evaluation',model='efficientnet_b0 + convnext_tiny + efficientnet_v2_s',ensemble_weights=json.dumps([1/3]*3),
        ensemble_members=json.dumps(freeze['members']),image_size=224,seed=42,status='completed',decision='performance_work_finished',
        metrics_path=relative(OUT/'ensemble/test_metrics.json'),source_sha256=sha256(OUT/'ensemble/test_metrics.json'),confusion_matrix_path=relative(OUT/'ensemble/figures/confusion_matrix.csv'),
        plots_dir=relative(OUT/'ensemble/figures'),config_path=relative(OUT/'pre_inference_verification.json'),notes='Test never used for selection; method unchanged; no further test pass.',
        **{k:final_metrics[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1')})
    upsert(record)
    return final_metrics,delta

def main():
    with run_lock('final_locked_test_once_v1'):
        if (OUT/'inference_started.json').exists():raise RuntimeError('Locked test already started; second inference prohibited. Preserve saved probabilities.')
        freeze=json.loads(FREEZE.read_text());state=json.loads((ROOT/'results/final_strict/v1/status.json').read_text())
        if state['status']!='completed' or state['source_hashes']!=code_hashes():raise ValueError('Strict freeze/source changed')
        if freeze['weights']!=[1/3]*3 or freeze['precision']!='fp32' or freeze['views']!=['identity'] or freeze['input_size']!=224:raise ValueError('Frozen inference changed')
        expected=['efficientnet_b0','convnext_tiny','efficientnet_v2_s']
        if [m['model'] for m in freeze['members']]!=expected:raise ValueError('Members changed')
        _,split_report=partition_metadata();configs=[];verified=[]
        for member in freeze['members']:
            path=ROOT/member['checkpoint']
            if sha256(path)!=member['sha256']:raise ValueError('Checkpoint hash changed')
            rid=path.parent.name;configpath=f'research/configs/final_strict/{rid}.json'
            if sha256(ROOT/configpath)!=state['config_hashes'][configpath]:raise ValueError('Config since strict freeze changed')
            config=json.loads((ROOT/configpath).read_text())
            if config!=json.loads((ROOT/'results/structured_experiments'/rid/'config.json').read_text()):raise ValueError('Run configuration mismatch')
            if config['class_order']!=list(CLASSES) or config['selection_metric']!='accuracy':raise ValueError('Class/checkpoint selection changed')
            if config['image_size']!=224 or config['normalization_mean']!=[.485,.456,.406] or config['normalization_std']!=[.229,.224,.225]:raise ValueError('Preprocessing changed')
            configs.append(config);verified.append(dict(member,absolute_checkpoint=str(path)))
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no inference started')
        identities=pd.read_csv(ROOT/SPLIT,usecols=['image_id','lesion_id','split']);test=identities.loc[identities.split=='test'].reset_index(drop=True)
        if len(test)!=1503:raise ValueError('Locked cohort changed')
        verification=dict(members=verified,freeze_manifest_sha256=sha256(FREEZE),split_verification=split_report,class_order=list(CLASSES),
            cohort_size=len(test),test_identity_sha256=__import__('hashlib').sha256('\n'.join(test.image_id).encode()).hexdigest(),
            preprocessing=dict(mode='RGB',size=[224,224],resize='bilinear antialias',mean=configs[0]['normalization_mean'],std=configs[0]['normalization_std'],augmentation=False),
            precision='fp32',autocast=False,views=['identity'],ensemble_weights=[1/3]*3,configuration_unchanged=True,test_labels_available_to_inference=False,
            source_hashes=code_hashes(),evaluation_script_sha256=sha256(__file__),device=torch.cuda.get_device_name(0))
        write_json(OUT/'pre_inference_verification.json',verification)
        # Exclusive durable receipt: even a crash cannot authorize another test pass.
        receipt=OUT/'inference_started.json'
        with receipt.open('x',encoding='utf-8') as file:
            json.dump(dict(pid=os.getpid(),started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),single_evaluation=True),file);file.flush();os.fsync(file.fileno())
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'evaluation.log',encoding='utf-8'),logging.StreamHandler()])
        log=logging.getLogger('locked_test');probability_files=[]
        try:
            dataset=LockedImages(test,configs[0]);loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
            for member,config in zip(freeze['members'],configs):
                checkpoint=torch.load(ROOT/member['checkpoint'],map_location='cpu',weights_only=False)
                if checkpoint['config']!=config or checkpoint['best_epoch']!=member['best_epoch'] or checkpoint['code_hashes']!=code_hashes() or tensor_nonfinite_names(checkpoint['model']):raise ValueError('Checkpoint metadata invalid')
                model=ResearchClassifier(config['model'],weights=None,attention='none',dropout=config['head_dropout'])
                model.load_state_dict(checkpoint['model'],strict=True);del checkpoint;model=model.cuda().eval();predictions=[];ids=[]
                log.info('ONE FP32 identity pass START %s',member['model'])
                with torch.inference_mode():
                    for images,image_ids in loader:
                        images=images.cuda(non_blocking=True)
                        if not torch.isfinite(images).all():raise FloatingPointError('Nonfinite test input')
                        logits=model(images);p=logits.softmax(1)
                        if not torch.isfinite(logits).all() or not torch.isfinite(p).all():raise FloatingPointError('Nonfinite FP32 test outputs; no retry/clamp')
                        predictions.extend(p.cpu().tolist());ids.extend(image_ids)
                if ids!=test.image_id.tolist():raise ValueError('Inference cohort incomplete')
                path=OUT/member['model']/'inference_probabilities_unscored.csv'
                write_csv(path,[dict(image_id=i,**{f'p_{c}':float(a[j]) for j,c in enumerate(CLASSES)}) for i,a in zip(ids,predictions)])
                probability_files.append(path);log.info('ONE pass SAVED %s count=%d',member['model'],len(ids))
                del model,images,logits,p;gc.collect();torch.cuda.empty_cache()
            write_json(OUT/'inference_completed.json',dict(model_passes=3,images_per_model=1503,labels_used_during_inference=False,probability_files=[relative(p) for p in probability_files]))
            metrics,delta=report(probability_files,freeze)
            write_json(OUT/'completion.json',dict(status='completed',first_and_only_evaluation=True,performance_work_finished=True,method_changed_after_test=False,
                checkpoint_hashes_unchanged=all(sha256(ROOT/m['checkpoint'])==m['sha256'] for m in freeze['members']),accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],comparison=delta))
            log.info('COMPLETED final accuracy=%.6f macro_f1=%.6f; STOP performance work',metrics['accuracy'],metrics['macro_f1'])
        except BaseException as exc:
            write_json(OUT/'failure.json',dict(error=repr(exc),no_automatic_retry=True));log.exception('Preserved evidence; no second inference');raise

if __name__=='__main__':main()
