"""S79 runtime primitives; frozen Phase 1 policy is imported unchanged."""
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from research.common import ROOT, sha256
from research.strict_train import seed_worker
from .protocol import CONFIG, Images, rng_state, restore_rng

PHASE1=ROOT/'results/final_cbam_development/v1/phase1'
PHASE2=ROOT/'results/final_cbam_development/v1/phase2'
CODE=['research/final_cbam_development/runtime.py','research/final_cbam_development/train.py',
      'research/final_cbam_development/artifacts.py','research/final_cbam_development/prepare_gpu.py',
      'research/final_cbam_development/test_runner.py',
      'research/registry.py','research/plots.py','research/short_screening/lesion_bag_screen.py']
NUMERICS=dict(gradscaler_initial_scale=1024.0,gradient_overflow='abort_preserve_valid_boundary',
              fp32_validation=True,tf32=False,deterministic=True)


def signature():
    c=json.loads(CONFIG.read_text());p=json.loads((PHASE1/'frozen_manifest.json').read_text())
    assert sha256(CONFIG)==p['config_sha256']
    assert all(sha256(ROOT/f)==h for f,h in p['code_sha256'].items())
    assert sha256(ROOT/c['split_manifest'])==p['split_sha256']
    assert sha256(ROOT/'.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth')==p['pretrained_weights_sha256']
    for s in p['ensemble']['sources']:
        assert sha256(ROOT/s['checkpoint'])==s['checkpoint_sha256']
        assert sha256(ROOT/s['prediction_file'])==s['prediction_sha256']
    manifest=json.loads((ROOT/'results/short_screening/s76_trained_feature_fusion/feature_cache_manifest.json').read_text())
    caches=[]
    for s in p['ensemble']['sources']:
        item=next(x for x in manifest['files'] if x['run']==s['run'] and x['split']=='val')
        assert sha256(ROOT/item['path'])==item['sha256'];caches.append(item)
    return c,dict(phase1_manifest_sha256=sha256(PHASE1/'frozen_manifest.json'),phase1=p,
        code_sha256={f:sha256(ROOT/f) for f in CODE},numerics=NUMERICS,
        fp32_reference_caches=caches,s76_signature_sha256=sha256(ROOT/'results/short_screening/s76_trained_feature_fusion/signature.json'))


def require_launch_freeze():
    c,s=signature();f=json.loads((PHASE2/'launch_manifest.json').read_text())
    if f!=s:raise ValueError('Runner/config/source changed after GPU preflight; review and refreeze explicitly')
    report=json.loads((PHASE2/'gpu_preflight.json').read_text())
    if report['status']!='passed':raise ValueError('GPU preflight has not passed')
    return c,s


def deterministic_cuda(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False


def capture_rng():
    return dict(**rng_state(),cuda=torch.cuda.get_rng_state_all())


def load_rng(state):
    restore_rng(state);torch.cuda.set_rng_state_all(state['cuda'])


def loader(frame,c,epoch,training):
    seed=c['seed']+epoch if training else c['seed']
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(Images(frame,c,training),batch_size=c['batch_size'],shuffle=training,
        num_workers=c['workers'],drop_last=training,pin_memory=True,worker_init_fn=seed_worker,generator=generator)


def finite(obj):
    flags={};scalar_ok=True
    def visit(x):
        nonlocal scalar_ok
        if torch.is_tensor(x) and x.is_floating_point():flags.setdefault(x.device,[]).append(torch.isfinite(x).all())
        elif isinstance(x,dict):
            for v in x.values():visit(v)
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        elif isinstance(x,float):scalar_ok &= bool(np.isfinite(x))
    visit(obj)
    return scalar_ok and all(bool(torch.stack(v).all()) for v in flags.values())


def model_cpu(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def amp_update(model,opt,scaler,batches,weights,c):
    """Exact weighted effective-batch loss; any nonfinite gradient stops safely."""
    opt.zero_grad(set_to_none=True);den=sum(weights[y].sum() for _,y in batches)
    num_sum=0.;correct=0;used=0
    for x,y in batches:
        if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite training input')
        with torch.autocast('cuda',dtype=torch.float16):z=model(x)
        if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite training logits')
        num=torch.nn.functional.cross_entropy(z.float(),y,weight=weights,reduction='sum')
        if not torch.isfinite(num):raise FloatingPointError('Nonfinite training loss')
        scaler.scale(num/den).backward();num_sum+=float(num.detach());correct+=int((z.argmax(1)==y).sum());used+=len(y)
    scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True)
    scaler.step(opt);scaler.update()
    if not finite(model.state_dict()) or not finite(opt.state_dict()):raise FloatingPointError('Nonfinite updated state')
    return dict(numerator=num_sum,denominator=float(den),correct=correct,used=used,gradient_norm=float(norm))
