"""Brief fit/finite-value check on real training images; no checkpoint retained."""
import json,os,time,gc
import psutil,torch
from torch.utils.data import DataLoader
from .common import ROOT,write_json
from .strict_protocol import OUT
from .strict_train import development_data,DevelopmentImages,optimizer_groups,checked_validation_batch
from .models import ResearchClassifier

def main():
    queue=json.loads((ROOT/'research/configs/final_strict_queue_v1.json').read_text());reports=[]
    for path in queue['configs']:
        c=json.loads((ROOT/path).read_text());os.environ['CUBLAS_WORKSPACE_CONFIG']=c['cuda_workspace_config']
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        train,_,w=development_data(c)
        # Zero workers here isolates GPU fit; prior measured worker/host costs justify sequential execution.
        x,y,_=next(iter(DataLoader(DevelopmentImages(train,c,True),batch_size=c['batch_size'],num_workers=0)))
        x=x.cuda();y=y.cuda();w=w.cuda()
        m=ResearchClassifier(c['model'],c['weights'],c['attention'],c['head_dropout']).cuda().train()
        opt=torch.optim.AdamW(optimizer_groups(m,c),betas=tuple(c['optimizer_betas']),eps=c['optimizer_eps'],weight_decay=c['weight_decay'])
        scaler=torch.amp.GradScaler('cuda');torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        skipped=0
        for _ in range(8):
            opt.zero_grad(set_to_none=True)
            for _ in range(c['gradient_accumulation']):
                with torch.autocast('cuda',dtype=torch.float16):logits=m(x)
                loss=torch.nn.functional.cross_entropy(logits.float(),y,weight=w)/c['gradient_accumulation']
                assert torch.isfinite(logits).all() and torch.isfinite(loss)
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gradients_finite=bool(torch.stack([torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None]).all())
            if gradients_finite:torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
            else:skipped+=1
            scaler.step(opt);scaler.update()
        if skipped==8:raise FloatingPointError('AMP scaler did not reach finite gradients during brief fit check')
        m.eval()
        with torch.inference_mode():checked_validation_batch(m,x,y,w,fp32=True)
        torch.cuda.synchronize()
        r=dict(model=c['model'],weights=c['weights'],peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,total_gpu_mb=torch.cuda.get_device_properties(0).total_memory/2**20,
            host_available_gib=psutil.virtual_memory().available/2**30,seconds=time.perf_counter()-tick,
            amp_training_finite=True,amp_overflow_rejected_updates=skipped,fp32_forward_finite=True,scope='8 repeated real-training-batch attempted optimizer steps; no validation/test images or saved model; discarded probe state',
            execution_mode='sequential',reason='Prior worker memory plus current host headroom; avoid concurrency contention on one 8GB GPU')
        reports.append(r);write_json(OUT/'resource_preflight.json',reports);print(json.dumps(r),flush=True)
        del m,opt,scaler,x,y,w,logits,loss;gc.collect();torch.cuda.empty_cache()
    write_json(OUT/'resource_preflight_complete.json',dict(status='passed',execution_mode='sequential',models=len(reports),test_loader=False))

if __name__=='__main__':main()
