"""Brief resource probe only, no experiment checkpoints or test images."""
import argparse,json,os,time
from pathlib import Path
import psutil,torch
from .common import ROOT,write_json
from .models import ResearchClassifier
from .train import development_data,DevelopmentImages,optimizer_groups,seed_worker
from torch.utils.data import DataLoader

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 c=json.loads(Path(args.config).read_text());os.environ['CUBLAS_WORKSPACE_CONFIG']=c['cuda_workspace_config']
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 train,val,w=development_data(c);ds=DevelopmentImages(train,c,True)
 tick=time.perf_counter();loader=DataLoader(ds,batch_size=c['batch_size'],num_workers=c['workers'],pin_memory=True,worker_init_fn=seed_worker)
 iterator=iter(loader);x,y,_=next(iterator);load_seconds=time.perf_counter()-tick
 parent=psutil.Process();host_rss=parent.memory_info().rss+sum(p.memory_info().rss for p in parent.children(recursive=True))
 del iterator,loader
 x=x.cuda();y=y.cuda();w=w.cuda();m=ResearchClassifier(c['model'],c['weights'],c['attention'],c['head_dropout']).cuda().train()
 opt=torch.optim.AdamW(optimizer_groups(m,c),weight_decay=c['weight_decay']);scaler=torch.amp.GradScaler('cuda');torch.cuda.reset_peak_memory_stats()
 def step():
  opt.zero_grad(set_to_none=True)
  for _ in range(c['gradient_accumulation']):
   with torch.autocast('cuda',dtype=torch.float16):
    out=m(x);loss=torch.nn.functional.cross_entropy(out.float(),y,weight=w)/c['gradient_accumulation']
   assert torch.isfinite(out).all() and torch.isfinite(loss)
   scaler.scale(loss).backward()
  scaler.unscale_(opt);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
  scaler.step(opt);scaler.update()
 for _ in range(2):step()
 torch.cuda.synchronize();tick=time.perf_counter()
 for _ in range(6):step()
 torch.cuda.synchronize();seconds=time.perf_counter()-tick
 r=dict(model=c['model'],pretrained_weights=c['weights'],microbatch=c['batch_size'],effective_batch=c['effective_batch_size'],peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,seconds_per_effective_step=seconds/6,images_per_second=6*c['effective_batch_size']/seconds,dataloader_first_batch_seconds=load_seconds,dataloader_host_rss_gib=host_rss/2**30,host_available_gib=psutil.virtual_memory().available/2**30,total_gpu_mb=torch.cuda.get_device_properties(0).total_memory/2**20,amp_logits_loss_gradients_finite=True,scope='8 optimizer steps on repeated real training batch, not accuracy training; exact batch/AMP/groups; no saved model or test loader',test_images_loaded=False)
 write_json(ROOT/args.output,r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
