"""Bounded disposable GPU probes for S79; no data training, epochs or scores."""
import gc
import json
import logging
import tempfile
import time
from pathlib import Path
import torch
from research.common import ROOT, sha256, write_json
from research.strict_train import atomic_checkpoint, runtime_versions
from .protocol import make_model, make_optimizer, make_scheduler, stage, stopping, check_resume
from .runtime import PHASE2, signature, deterministic_cuda, NUMERICS, amp_update, capture_rng, load_rng, model_cpu


def main():
    c,sig=signature();PHASE2.mkdir(parents=True,exist_ok=True)
    if (PHASE2/'launch_manifest.json').exists():
        assert json.loads((PHASE2/'launch_manifest.json').read_text())==sig
        print('Completed GPU preflight preserved; no probe repeated');return
    for k in ['results','checkpoints']:assert not (ROOT/c[k]).exists()
    registry_before=sha256(ROOT/'results/master_experiment_registry.csv')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(PHASE2/'gpu_preflight.log'),logging.StreamHandler()])
    deterministic_cuda(c['seed']);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    logging.info('Disposable synthetic AMP warmup/full-stage probes; no training images/test/accuracy')
    model=make_model(c).cuda();opt=make_optimizer(model,c);scheduler=make_scheduler(opt,c)
    scaler=torch.amp.GradScaler('cuda',init_scale=NUMERICS['gradscaler_initial_scale'])
    x=torch.randn(32,3,224,224,device='cuda');y=torch.arange(32,device='cuda')%7;weights=torch.ones(7,device='cuda')
    batches=[(x[:16],y[:16]),(x[16:],y[16:])]
    stage(model,1,c);amp_update(model,opt,scaler,batches,weights,c)
    stage(model,3,c);timings=[]
    for _ in range(3):
        torch.cuda.synchronize();tick=time.perf_counter();amp_update(model,opt,scaler,batches,weights,c);torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
    hist=[dict(epoch=i,val_accuracy=.9,val_macro_f1=.85,scheduler_lr_reduced=False) for i in range(1,4)]
    payload=dict(config=c,signature=sig,epoch=3,model=model_cpu(model),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(),rng=capture_rng(),history=hist,meaningful_stopping=stopping(hist,c),
        optimizer_updates=4,best_accuracy=None,best_macro_f1=None,disposable_gpu_probe=True)
    parent=ROOT/'.cache/final_cbam_development/v1';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gpu-probe-',dir=parent) as temp:
        path=Path(temp)/'latest.pt';atomic_checkpoint(path,payload);saved=torch.load(path,map_location='cpu',weights_only=False);check_resume(saved,c,sig)
        a=amp_update(model,opt,scaler,batches,weights,c)
        restored=make_model(c,pretrained=False).cuda();restored.load_state_dict(saved['model'],strict=True)
        ropt=make_optimizer(restored,c);ropt.load_state_dict(saved['optimizer'])
        rscheduler=make_scheduler(ropt,c);rscheduler.load_state_dict(saved['scheduler'])
        rscaler=torch.amp.GradScaler('cuda',init_scale=1024.);rscaler.load_state_dict(saved['scaler'])
        stage(restored,3,c);load_rng(saved['rng']);b=amp_update(restored,ropt,rscaler,batches,weights,c)
        assert a==b and scaler.state_dict()==rscaler.state_dict()
        assert all(torch.equal(t,restored.state_dict()[k]) for k,t in model.state_dict().items())
        restored.eval();validation_times=[]
        for _ in range(3):
            torch.cuda.synchronize();tick=time.perf_counter()
            with torch.inference_mode():
                z=restored(x[:16]);assert z.dtype==torch.float32 and torch.isfinite(z).all()
                p=z.softmax(1);assert torch.isfinite(p).all() and torch.allclose(p.sum(1),torch.ones(16,device='cuda'),atol=1e-5)
            torch.cuda.synchronize();validation_times.append(time.perf_counter()-tick)
        probe_peak=torch.cuda.max_memory_allocated()/1024**2
        del restored,ropt,rscheduler,rscaler,saved,payload;gc.collect();torch.cuda.empty_cache()
    # Measure normal single-model peak after disposing the recovery comparison.
    torch.cuda.reset_peak_memory_stats();stage(model,3,c);amp_update(model,opt,scaler,batches,weights,c)
    peak=torch.cuda.max_memory_allocated()/1024**2
    assert sha256(ROOT/'results/master_experiment_registry.csv')==registry_before
    train_seconds=float(torch.tensor(timings).median());val_seconds=float(torch.tensor(validation_times).median())
    lower_epoch=219*train_seconds+94*val_seconds
    result=dict(status='passed',gpu=torch.cuda.get_device_name(0),runtime=runtime_versions(),synthetic_only=True,
        optimizer_probe_updates=7,training_epochs_completed=0,accuracy_measured=False,training_started=False,
        batch_size=16,effective_batch_size=32,finite_amp_warmup=True,finite_amp_full_stage=True,fp32_validation_finite=True,
        resumed_next_update_bitwise_equal=True,cuda_rng_optimizer_scheduler_scaler_restored=True,
        peak_single_model_allocated_mb=peak,peak_two_model_recovery_probe_mb=probe_peak,
        full_effective_batch_seconds=timings,fp32_validation_batch_seconds=validation_times,
        synthetic_epoch_compute_lower_bound_seconds=lower_epoch,
        caveat='Synthetic timings exclude loading/augmentation/checkpoint/reporting; not a promised wall runtime',
        gpu_preflight_seconds=time.perf_counter()-start,original_registry_unchanged=True,test_images_loaded=False,test_labels_read=False,
        temporary_probe_checkpoint_deleted=True,numerics=NUMERICS,
        remaining_gate='Human approval for full S79 training; no training launched')
    write_json(PHASE2/'gpu_preflight.json',result);write_json(PHASE2/'launch_manifest.json',sig)
    logging.info('GPU PREFLIGHT PASSED single-model_peak=%.1fMB synthetic_epoch_lower_bound=%.1fs; no training launched',peak,lower_epoch)
    print(json.dumps({k:result[k] for k in ['status','peak_single_model_allocated_mb','resumed_next_update_bitwise_equal','synthetic_epoch_compute_lower_bound_seconds','gpu_preflight_seconds']}))


if __name__=='__main__':main()
