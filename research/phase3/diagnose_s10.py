"""Read-only CPU checkpoint/validation diagnosis; never trains or creates a test loader."""
import json
import shutil
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from research.common import ROOT, sha256, write_json
from research.models import ResearchClassifier
from research.train import development_data, DevelopmentImages, weighted_numerator, metric_report, code_hashes, runtime_versions
from research.phase3.close_s02 import load, verify_launch_code, verify_predictions

ID = 's10_efficientnet_v2_s_none_exploratory_seed42'
AUDIT = ROOT/'results/audit/s10_nonfinite_20261003'


def bad_tensors(obj, prefix=''):
    if torch.is_tensor(obj):
        return [prefix] if obj.is_floating_point() and not torch.isfinite(obj).all() else []
    if isinstance(obj,dict):
        return sum((bad_tensors(v,prefix+'.'+str(k)) for k,v in obj.items()),[])
    if isinstance(obj,(list,tuple)):
        return sum((bad_tensors(v,prefix+'.'+str(k)) for k,v in enumerate(obj)),[])
    return []


def main():
    torch.set_num_threads(2)
    path=ROOT/'results/structured_experiments'/ID
    checkpoints=ROOT/'checkpoints/structured'/ID
    config=load(path/'config.json')
    AUDIT.mkdir(parents=True,exist_ok=True)
    preservation={}
    # Original failed run remains untouched; frozen copies protect later resumption.
    for source in list(path.iterdir())+list(checkpoints.glob('*.pt')):
        if not source.is_file(): continue
        kind='checkpoints' if source.suffix=='.pt' else 'run_artifacts'
        target=AUDIT/'preserved'/kind/source.name
        target.parent.mkdir(parents=True,exist_ok=True)
        digest=sha256(source)
        if target.exists(): assert sha256(target)==digest, 'Do not overwrite preserved failure evidence'
        else: shutil.copy2(source,target)
        assert sha256(target)==digest
        preservation[source.relative_to(ROOT).as_posix()]=dict(sha256=digest,copy=target.relative_to(ROOT).as_posix())
    write_json(AUDIT/'preservation.json',preservation)
    checkpoint=torch.load(checkpoints/'latest.pt',map_location='cpu',weights_only=False)
    assert checkpoint['config']==config
    assert checkpoint['runtime_versions']==runtime_versions()
    history=pd.read_csv(path/'history.csv')
    pd.testing.assert_frame_equal(pd.DataFrame(checkpoint['history']),history,check_exact=False,rtol=1e-12,atol=1e-12)
    sources=verify_launch_code(checkpoint,load(path/'environment.json'))
    train,val,weights=development_data(config)
    model=ResearchClassifier(config['model'],weights=None,attention=config['attention'],dropout=config['head_dropout'])
    model.load_state_dict(checkpoint['model']);model.eval()
    assert not bad_tensors(checkpoint['model']) and not bad_tensors(checkpoint['optimizer'])
    for name,key,criterion in [('best.pt','best','accuracy'),('best_macro_f1.pt','secondary_best','macro_f1')]:
        winner=torch.load(checkpoints/name,map_location='cpu',weights_only=False)
        state=checkpoint[key]
        winner_epoch=winner['best']['epoch'] if name=='best.pt' else winner['best_epoch']
        assert winner['config']==config and winner_epoch==state['epoch']==13
        assert all(torch.equal(winner['model'][k],v) for k,v in state['model'].items())
        assert winner.get('selection_metric',criterion)==criterion
        found=pd.DataFrame(state['predictions'])
        assert found.image_id.is_unique and set(found.image_id)==set(val.image_id)
        probabilities=found.filter(regex='^p_').to_numpy()
        assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1),1,atol=1e-6)
    maxima={};handles=[]
    def hook(name):
        def record(_module,_input,output):
            maxima[name]=max(maxima.get(name,0),float(output.abs().max()))
        return record
    for name,module in model.features.named_children():
        handles.append(module.register_forward_hook(hook('features.'+name)))
    loader=DataLoader(DevelopmentImages(val,config),batch_size=config['validation_batch_size'],shuffle=False,num_workers=0)
    p=[];ys=[];ids=[];vn=vd=0.;largest=[];start=time.perf_counter()
    with torch.inference_mode():
        for images,targets,image_ids in loader:
            assert torch.isfinite(images).all() and 0<=targets.min() and targets.max()<7
            logits=model(images)
            assert torch.isfinite(logits).all(), image_ids
            num=weighted_numerator(logits,targets,weights)
            assert torch.isfinite(num)
            probs=logits.softmax(1);assert torch.isfinite(probs).all()
            vn+=float(num);vd+=float(weights[targets].sum())
            p.extend(probs.tolist());ys.extend(targets.tolist());ids.extend(image_ids)
            largest.extend((float(z.abs().max()),str(i)) for z,i in zip(logits,image_ids))
    for h in handles:h.remove()
    # Narrow CPU FP16 probes of the largest FP32-logit batches. These do not
    # reproduce CUDA kernels or the lost epoch14 state; report that distinction.
    dataset=DevelopmentImages(val,config)
    ordered=sorted(largest,reverse=True)
    batch_starts=[]
    for _,image_id in ordered:
        start_index=int(val.index[val.image_id==image_id][0])//config['validation_batch_size']*config['validation_batch_size']
        if start_index not in batch_starts:batch_starts.append(start_index)
        if len(batch_starts)==3:break
    probes=[]
    for start_index in batch_starts:
        samples=[dataset[i] for i in range(start_index,min(start_index+config['validation_batch_size'],len(dataset)))]
        images=torch.stack([x[0] for x in samples]);targets=torch.tensor([x[1] for x in samples])
        first_bad=[];hook_handles=[]
        def capture(name):
            def record(_module,_input,output):
                if torch.is_tensor(output) and not first_bad and not torch.isfinite(output).all():
                    first_bad.append(dict(module=name,dtype=str(output.dtype),nan=int(torch.isnan(output).sum()),inf=int(torch.isinf(output).sum())))
            return record
        for name,module in model.named_modules():
            if not list(module.children()):hook_handles.append(module.register_forward_hook(capture(name)))
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.float16):logits=model(images)
        loss=weighted_numerator(logits,targets,weights)
        finite=logits[torch.isfinite(logits)]
        probes.append(dict(batch_start=start_index,image_ids=[x[2] for x in samples],first_nonfinite_module=first_bad,
            logits_nonfinite=int((~torch.isfinite(logits)).sum()),max_abs_finite_logit=float(finite.abs().max()) if finite.numel() else None,
            loss_finite=bool(torch.isfinite(loss)),loss=float(loss) if torch.isfinite(loss) else None))
        for h in hook_handles:h.remove()
    report=dict(status='saved_epoch13_cpu_fp32_verified',created_utc=datetime.now(timezone.utc).isoformat(),
        committed_epochs=len(history), failed_epoch=14, failed_epoch_state_saved=False,
        checkpoint_hashes={n:sha256(checkpoints/n) for n in ('latest.pt','best.pt','best_macro_f1.pt')},
        checkpoint_model_nonfinite=[],checkpoint_optimizer_nonfinite=[],source_verification=sources,
        saved_epoch13_metrics=checkpoint['best']['metrics'],cpu_fp32_metrics=metric_report(np.asarray(ys),p,vn/vd),
        cpu_validation_images=len(ids),cpu_fp32_seconds=time.perf_counter()-start,largest_abs_logits=sorted(largest,reverse=True)[:10],
        feature_stage_abs_maxima=maxima,cpu_fp16_probes=probes,scaler=checkpoint['scaler'],
        loss_implementation='FP32 cross_entropy(logits.float(), weighted sum) / positive target weight sum; not log of softmax probabilities',
        failure_log_limit='Epoch14 failed before checkpoint publication. Log only records aggregate nonfinite loss; no failed logits, sample IDs or epoch14 weights survive. Exact original offending operation/sample cannot be recovered without approved deterministic replay.',
        gpu_used=False,training_updates=False,test_images_loaded=False)
    write_json(AUDIT/'diagnosis.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('checkpoint_hashes','source_verification','saved_epoch13_metrics','cpu_fp32_metrics')},indent=2))


if __name__=='__main__':main()
