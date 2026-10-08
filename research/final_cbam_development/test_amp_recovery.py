"""Disposable tiny GPU test for explicit AMP overflow skip/backoff, no data/model run."""
import json
import torch
from research.common import write_json
from .runtime import PHASE2, deterministic_cuda, amp_update


def main():
    deterministic_cuda(42);m=torch.nn.Linear(4,7).cuda();opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=4096.)
    x=torch.randn(32,4,device='cuda');y=torch.arange(32,device='cuda')%7;batches=[(x[:16],y[:16]),(x[16:],y[16:])]
    old={k:v.detach().clone() for k,v in m.state_dict().items()}
    hook=m.weight.register_hook(lambda grad:grad*float('inf'))
    overflow=amp_update(m,opt,scaler,batches,torch.ones(7,device='cuda'),dict(gradient_clip_norm=1.))
    assert overflow['skipped'] and overflow['scale_before']==4096 and overflow['scale_after']==2048
    assert not opt.state and all(torch.equal(old[k],v) for k,v in m.state_dict().items())
    hook.remove();normal=amp_update(m,opt,scaler,batches,torch.ones(7,device='cuda'),dict(gradient_clip_norm=1.))
    assert not normal['skipped'] and bool(opt.state)
    result=dict(status='passed',synthetic_gradient_overflow=True,optimizer_step_skipped=True,model_unchanged_on_skip=True,
        scale_before=4096,scale_after=2048,next_finite_update_succeeded=True,skin_images_loaded=False,
        accuracy_measured=False,training_epochs=0)
    write_json(PHASE2/'amp_recovery_probe.json',result);print(json.dumps(result))


if __name__=='__main__':main()
