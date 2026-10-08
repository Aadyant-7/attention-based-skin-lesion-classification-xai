"""Phase 1 CPU-only preparation. Disposable optimizer probes; no training launch."""
import copy
import gc
import json
import logging
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torchvision.models import ConvNeXt_Tiny_Weights
from research.common import ROOT, CLASSES, sha256, relative, write_json
from research.strict_train import atomic_checkpoint, runtime_versions
from research.final_cbam_development.protocol import CONFIG, Images, verified_development, weights_from_training, make_model, stage, make_optimizer, make_scheduler, stopping, rng_state, restore_rng, check_resume

OUT=ROOT/'results/final_cbam_development/v1/phase1'
CACHE=ROOT/'.cache/final_cbam_development/v1'
CODE=['research/final_cbam_development/protocol.py','research/final_cbam_development/prepare_phase1.py',
      'research/models.py','src/cbam.py','research/common.py','research/strict_train.py',
      'research/short_screening/feature_fusion.py']


def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Expected unsafe operation to be rejected')


def stopping_checks(c):
    def rows(n,drift=0):
        return [dict(epoch=i,val_accuracy=.90+drift*(i-1),val_macro_f1=.85,scheduler_lr_reduced=i==8) for i in range(1,n+1)]
    assert not stopping(rows(19),c)['stop']
    tiny=stopping(rows(20,.00005),c)
    assert tiny['stale']==19 and tiny['stop'] # Raw records do not reset patience.
    cumulative=stopping(rows(20,.00011),c)
    assert cumulative['stale']==0 and not cumulative['stop']
    f1=rows(20);f1[-1]['val_macro_f1']=.853
    assert stopping(f1,c)['stale']==0 and not stopping(f1,c)['stop']
    no_lr=rows(30)
    for row in no_lr:row['scheduler_lr_reduced']=False
    assert not stopping(no_lr,c)['stop'] and stopping(rows(30),c)['reason']=='late_meaningful_plateau'
    assert stopping(no_lr+rows(40)[30:],c)['reason']=='epoch_cap'
    bad=rows(20);bad[-1]['val_accuracy']=float('nan');reject(lambda:stopping(bad,c))
    return dict(minimum_window=True,tiny_raw_records_do_not_reset=True,cumulative_gain_resets=True,
                f1_gain_resets=True,lr_opportunity_required=True,late_plateau=True,hard_cap=True,nonfinite_metric_rejected=True)


def source_manifest():
    folder=ROOT/'results/short_screening/s53_equal_five_b0_addition'
    summary=json.loads((folder/'summary.json').read_text())
    four=json.loads((ROOT/'results/short_screening/s46_all_f1_checkpoint_fusion/candidate_manifest.json').read_text())['sources']
    sources=copy.deepcopy(four)
    path=list(summary['source_prediction_sha256'])[-1]
    sources.append(dict(run='s03_efficientnet_b0_none_exploratory_seed42',checkpoint=summary['rescue_checkpoint'],
        checkpoint_sha256=summary['rescue_checkpoint_sha256'],prediction_file=path,prediction_sha256=summary['source_prediction_sha256'][path]))
    for s in sources:
        for key,hash_key in [('checkpoint','checkpoint_sha256'),('prediction_file','prediction_sha256')]:
            assert sha256(ROOT/s[key])==s[hash_key]
    assert len(sources)==5 and sources[0]['run']=='s06_convnext_tiny_none_exploratory_seed42'
    return dict(reference='S53',accuracy=.936127744510978,macro_f1=.8868574111567237,
        sources=sources,reference_prediction_sha256=sha256(folder/'validation_predictions.csv'),
        fusion='replace source0 with S79 accuracy-selected checkpoint; all other sources unchanged; weights0.2',
        selection_difference='S53 Tiny uses macro-F1 selection; new Tiny+CBAM uses predeclared accuracy selection. Comparison is a training-package comparison, not a pure CBAM ablation.')


def optimizer_probe(c,weights,val,signature):
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=make_model(c);optimizer=make_optimizer(model,c);scheduler=make_scheduler(optimizer,c)
    # Confirm original cached pretrained tensors loaded unchanged before any probe.
    pretrained=torch.load(ROOT/'.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth',map_location='cpu',weights_only=True)
    assert torch.equal(model.features[0][0].weight,pretrained['features.0.0.weight'])
    assert torch.equal(model.head[1].weight,pretrained['classifier.0.weight']);del pretrained;gc.collect()
    shapes=[];handle=model.attention.register_forward_pre_hook(lambda _m,args:shapes.append(list(args[0].shape)))
    x=torch.randn(1,3,224,224);y=torch.tensor([4])
    def step(m,opt,epoch):
        stage(m,epoch,c);opt.zero_grad(set_to_none=True);z=m(x)
        loss=F.cross_entropy(z.float(),y,weight=weights,reduction='sum')/weights[y].sum()
        assert z.shape==(1,7) and torch.isfinite(z).all() and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.attention.parameters())
        torch.nn.utils.clip_grad_norm_(m.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True)
        opt.step();return float(loss.detach())
    original=model.features[0][0].weight.detach().clone()
    warm_loss=step(model,optimizer,1)
    assert torch.equal(original,model.features[0][0].weight) and model.features[0][0].weight.grad is None
    # Freeze-to-full transition carries optimizer state; no optimizer reset.
    full_loss=step(model,optimizer,3)
    assert not torch.equal(original,model.features[0][0].weight) and model.features[0][0].weight.grad is not None
    assert shapes[0]==[1,768,7,7];handle.remove()
    history=[dict(epoch=i,val_accuracy=.9,val_macro_f1=.85,scheduler_lr_reduced=False) for i in range(1,4)]
    # Probe values are fabricated inputs to a recovery test, never research metrics.
    payload=dict(diagnostic_only=False,config=c,signature=signature,epoch=3,model=model.state_dict(),
        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler={},rng=rng_state(),history=history,
        meaningful_stopping=stopping(history,c),optimizer_updates=2,best_accuracy=None,best_macro_f1=None,
        disposable_cpu_probe=True)
    with tempfile.TemporaryDirectory(prefix='cpu-probe-',dir=CACHE) as temp:
        path=Path(temp)/'latest.pt';atomic_checkpoint(path,payload)
        saved=torch.load(path,map_location='cpu',weights_only=False);check_resume(saved,c,signature)
        loss_a=step(model,optimizer,4)
        restored=make_model(c,pretrained=False);restored.load_state_dict(saved['model'],strict=True)
        restored_opt=make_optimizer(restored,c);restored_opt.load_state_dict(saved['optimizer'])
        restored_scheduler=make_scheduler(restored_opt,c);restored_scheduler.load_state_dict(saved['scheduler'])
        assert restored_scheduler.state_dict()==scheduler.state_dict()
        restore_rng(saved['rng']);loss_b=step(restored,restored_opt,4)
        assert loss_a==loss_b
        assert all(torch.equal(t,restored.state_dict()[k]) for k,t in model.state_dict().items())
        reject(lambda:check_resume(dict(saved,diagnostic_only=True),c,signature))
        reject(lambda:check_resume(saved,dict(c,seed=43),signature))
        reject(lambda:check_resume(saved,c,dict(signature,config_sha256='wrong')))
        reject(lambda:check_resume(dict(saved,meaningful_stopping={}),c,signature))
        reject(lambda:check_resume(dict(saved,epoch=4),c,signature))
        assert not Images(val.iloc[:1],c)[0][0].requires_grad
        model.eval();im=Images(val.iloc[:1],c)[0][0].unsqueeze(0)
        with torch.inference_mode():
            p=model(im).float().softmax(1)
            assert torch.isfinite(p).all() and torch.allclose(p.sum(1),torch.ones(1))
        # Weighted accumulation uses the complete effective-batch denominator.
        logits=torch.randn(4,7);targets=torch.tensor([0,4,5,6]);den=weights[targets].sum()
        a=F.cross_entropy(logits,targets,weight=weights,reduction='sum')/den
        b=sum(F.cross_entropy(logits[i:i+2],targets[i:i+2],weight=weights,reduction='sum')/den for i in [0,2])
        torch.testing.assert_close(a,b)
        count=sum(p.numel() for p in model.parameters());attention=sum(p.numel() for p in model.attention.parameters())
    return dict(parameters=count,attention_parameters=attention,attention_input_shape=shapes[0],
        pretrained_tensors_verified=True,finite_cpu_forward=True,finite_attention_gradients=True,
        warmup_preserves_backbone=True,full_finetuning_updates_backbone=True,optimizer_preserved_at_transition=True,
        atomic_checkpoint_roundtrip=True,resumed_next_update_bitwise_equal=True,
        recovery_mismatch_guards=True,deterministic_validation=True,weighted_effective_batch_loss=True,
        synthetic_warmup_loss=warm_loss,synthetic_full_loss=full_loss,
        accuracy_measured=False,temporary_checkpoint_deleted=True)


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'prepare.log'),logging.StreamHandler()])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_initialized():raise RuntimeError('CPU phase must not initialize CUDA')
    c=json.loads(CONFIG.read_text());registry=ROOT/'results/master_experiment_registry.csv';registry_before=sha256(registry)
    assert not (ROOT/c['checkpoints']).exists() and not (ROOT/c['results']).exists(), 'Actual run already exists; do not prepare over it'
    logging.info('Verifying development identities and existing ensemble sources; CPU only')
    train,val=verified_development(c);weights,counts=weights_from_training(train)
    reject(lambda:Images(val,c,True));reject(lambda:Images(train,c,False));reject(lambda:Images(val.assign(split='test'),c))
    reject(lambda:weights_from_training(val))
    ds=Images(val.iloc[:1],c);assert torch.equal(ds[0][0],ds[0][0])
    td=Images(train.iloc[:1],c,True);torch.manual_seed(42);a=td[0][0];torch.manual_seed(42);assert torch.equal(a,td[0][0])
    pretrained=ROOT/'.cache/torch/hub/checkpoints'/Path(ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url).name
    assert pretrained.exists(), 'Acquire weights separately before CPU preparation; no automatic download'
    pretrained_sha=sha256(pretrained);assert pretrained_sha.startswith('983f1562')
    signature=dict(config_sha256=sha256(CONFIG),code_sha256={p:sha256(ROOT/p) for p in CODE},
        split_sha256=c['split_sha256'],pretrained_weights_sha256=pretrained_sha,ensemble=source_manifest())
    if (OUT/'frozen_manifest.json').exists():assert json.loads((OUT/'frozen_manifest.json').read_text())==signature
    else:write_json(OUT/'frozen_manifest.json',signature)
    if (OUT/'phase1_report.json').exists():print('Completed Phase1 preserved; no probe repeated');return
    logging.info('Checking meaningful stopping and disposable CPU optimizer/recovery probes')
    checks=stopping_checks(c);probe=optimizer_probe(c,weights,val,signature)
    assert sha256(registry)==registry_before and not torch.cuda.is_initialized()
    report=dict(status='phase1_complete_pending_phase2_implementation',cpu_only=True,gpu_training_started=False,
        training_epochs_completed=0,accuracy_measured=False,train_images=len(train),validation_images=len(val),
        training_class_counts=dict(zip(CLASSES,counts)),class_weights=dict(zip(CLASSES,weights.tolist())),
        locked_test_image_overlap=0,locked_test_lesion_overlap=0,test_images_loaded=False,test_labels_read=False,
        exploratory_train_validation_shared_lesions=563,exploratory_validation_images_with_train_lesion=596,
        stopping_checks=checks,model_and_recovery_checks=probe,registry_unchanged=True,
        original_sources_and_checkpoints_verified=True,runtime=runtime_versions(),seconds=time.perf_counter()-start,
        pending=['GPU memory/timing/AMP preflight','full GPU training runner and artifact closeout','explicit training approval'],
        paths=dict(config=relative(CONFIG),frozen_manifest=relative(OUT/'frozen_manifest.json'),
                   future_results=c['results'],future_checkpoints=c['checkpoints']))
    assert all(sha256(ROOT/s[k])==s[h] for s in signature['ensemble']['sources']
               for k,h in [('checkpoint','checkpoint_sha256'),('prediction_file','prediction_sha256')])
    write_json(OUT/'phase1_report.json',report)
    logging.info('PHASE1 COMPLETE: no GPU use or training; original registry/results unchanged')
    print(json.dumps(dict(status=report['status'],parameters=probe['parameters'],attention_parameters=probe['attention_parameters'],
                         resume_bitwise_equal=True,cuda_used=False,training_started=False,seconds=report['seconds'])))


if __name__=='__main__':main()
