"""CPU protocol freeze and short disposable numerical/resource preflight."""
import argparse,gc,json,os,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch,psutil
from PIL import Image
from torch.utils.data import DataLoader
from research.common import ROOT,write_json,atomic_text,sha256,CLASSES
from research.strict_protocol import SPLIT,DIGEST,partition_metadata
from research.registry import read_registry
from .core import CONFIG,OUT,config,data,Images,EnhancedModel,mixed_focal_sum

MODELS=[dict(id='s32_efficientnet_b3_cbam_enhanced_seed42',model='efficientnet_b3',attention='cbam',weights='EfficientNet_B3_Weights.IMAGENET1K_V1',lr=1e-4),
        dict(id='s33_densenet201_enhanced_seed42',model='densenet201',attention='none',weights='DenseNet201_Weights.IMAGENET1K_V1',lr=7.5e-5),
        dict(id='s34_resnet101_enhanced_seed42',model='resnet101',attention='none',weights='ResNet101_Weights.IMAGENET1K_V2',lr=5e-5)]

def freeze():
    if CONFIG.exists():raise FileExistsError('Enhanced recipe already predeclared; do not replace after results')
    c=dict(version='aggressive_enhanced_v1',protocol='enhanced_lesion_disjoint_development',split_manifest=SPLIT,split_sha256=DIGEST,
        seed=42,image_size=224,class_order=list(CLASSES),models=MODELS,microbatch=32,effective_batch=32,workers=2,
        normalization_mean=[.485,.456,.406],normalization_std=[.229,.224,.225],training_precision='bf16',validation_precision='fp32',
        focal_gamma=2.2,weight_rule='clip(N_train/(7*n_class),1,20); normalize arithmetic mean1; no manual multipliers',mixup_alpha=.2,mixup_probability=.3,
        weight_decay=.0015,gradient_clip=.5,max_epochs=50,minimum_epochs=25,patience=12,min_delta=.0003,scheduler='StepLR gamma.7 every10epochs',
        freeze_head_epochs=2,last_stage_epochs=3,selection='earliest true maximum standalone validation accuracy; independent macroF1 winner retained',
        primary_weights=[.4,.4,.2],secondary_weights=[1/3]*3,old_test_excluded=True,target_accuracy=.93,
        numerical_policy='BF16 where supported; FP32 loss/BN/validation. One safe automatic full-FP32 retry from last valid committed boundary; no clamping/skipping. FP32 micro<=16 preserves effective batch.',
        terminal_batch='All training images consumed; singleton terminal batch merges preceding batch; terminal33 split safely for BN',
        xai_cases='deterministic validation category coverage; Grad-CAM of weighted probability target per branch plus composite; original/spatial/channel attention artifacts; max14cases; no test')
    train,val,w=data(c)
    def moments(path):
        with Image.open(path) as image:a=np.asarray(image.convert('RGB').resize((224,224),resample=Image.Resampling.BILINEAR),dtype=np.float64)/255.
        return a.sum((0,1)),(a*a).sum((0,1)),a.shape[0]*a.shape[1]
    sums=np.zeros(3);squares=np.zeros(3);pixels=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for s,q,n in pool.map(moments,train.path):sums+=s;squares+=q;pixels+=n
    mean=sums/pixels;std=np.sqrt(squares/pixels-mean*mean)
    if not np.isfinite(std).all() or (std<=0).any():raise ValueError('Invalid train normalization')
    c['normalization_mean']=mean.tolist();c['normalization_std']=std.tolist()
    write_json(CONFIG,c);_,verification=partition_metadata();write_json(OUT/'split_verification.json',verification)
    write_json(OUT/'normalization_provenance.json',dict(training_images=len(train),pixels=pixels,mean=mean.tolist(),std=std.tolist(),method='Every training RGB image, resized224 bilinear; population channel moments; validation/test excluded'))
    # Protect every pre-existing saved result/report and every registry row, including original test.
    protected={}
    for directory in ['results/structured_experiments','results/final_strict','results/final_locked_test','results/model_comparison']:
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and path.suffix not in ['.log','.tmp','.lock']:protected[path.relative_to(ROOT).as_posix()]=sha256(path)
    for name in ['research/FINAL_LOCKED_TEST_RESULTS.md','research/FINAL_STRICT_VALIDATION_RESULTS.md','research/FINAL_ARCHITECTURE_SELECTION.md']:
        protected[name]=sha256(ROOT/name)
    write_json(OUT/'historical_preservation.json',dict(files=protected,baseline_registry=read_registry()))
    write_json(OUT/'class_distribution.json',dict(training_counts={x:int((train.label==i).sum()) for i,x in enumerate(CLASSES)},
        validation_counts={x:int((val.label==i).sum()) for i,x in enumerate(CLASSES)},class_weights=dict(zip(CLASSES,w.tolist())),
        after_augmentation_counts='Unchanged: on-the-fly class-specific augmentation adds diversity, not fictitious stored images',resampling=False,test_labels_read=False))
    atomic_text(ROOT/'research/aggressive/ENHANCED_PROTOCOL_FREEZE.md',f'''# Enhanced protocol freeze — post-test development study

Option B selected BEFORE any enhanced outcome: reuse existing strict lesion-disjoint development split,7009train/1503validation,seed42. The original1503test images/labels are excluded from all enhanced training, selection and XAI. Full HAM10000 is NOT used. All lesion overlaps zero; immutable SHA {DIGEST}. No new split or subset cherry-picking. Development validation has been used previously; it is NOT a new independent test. The original rigorous held-out result86.7598% / macroF1 .794473 is permanently preserved separately.

Cost rationale: three models × up to50epochs instead of nine CV jobs. Prior local epoch costs roughly45–115s imply2–5hours for one queue versus6–15hours for3folds, before richer augmentation/FP32 BN/XAI overhead. Disposable preflight will refine an estimate, not select models or protocols. One fold cannot reproduce the paper's3fold mean/SD. Sequential execution selected due approximately4GB available host RAM and one8GBGPU.

Frozen architecture: B3+existing channel/spatial CBAM after final convolution, DenseNet201,ResNet101; fresh explicit ImageNet weight versions in config.json. Primary voting .40/.40/.20, equal1/3 secondary baseline only. No weight/member search. Finish the three branches and report the real result; stop whether above or below93%.

One combined recipe: FP32 stable log-softmax weighted focal gamma2.2; training-only bounded inverse-frequency weights (no manually tuned source multipliers); Mixup alpha.2/p.3 with a convex blend of correct class-specific focal objectives. Natural class counts retained; targeted spatial/color/erasing augmentation only in training; strongerAKIEC/DF/VASC and moderateBCC/MEL. Hue capped.05 and erasing1–5% to limit morphology damage. Norm mean {mean.tolist()},std {std.tolist()} computed on ALL training pixels only, not copied source statistics.

Custom GAP/d→512→256→7 head,BN/ReLU,dropout.65/.55/.45 across layers (NOT across epochs). AdamW/model LRs1e-4/7.5e-5/5e-5; decay.0015; clip.5; StepLR.7/10. Head/attention only epochs1–2,last backbone stage3–5,full fine-tune6+. Frozen backbone BN stats remain frozen; active BN isFP32. Max50; patience12 begins at25,delta.0003 for stopping reference; always preserve true earliest maximum accuracy even if smaller gains don't reset patience. Independent bestF1/latest retained. Effectivebatch32; terminal33 preserves ALL images.

Numerical safety: oldB3 overflow atfeatures.7.0.block.0.0; oldFP32 probe was not authorized (false flag means not performed, not failure). BF16 uses FP32-range exponents; loss/BN/validationFP32. BF16 requires GPU support and finite forward/backward preflight. Predeclared one fullFP32 recovery from latest finite boundary; micro<=16 if needed, sameeffectivebatch. Every failure and precision amendment retained; no invalid sample skip or NaN/Inf clamp. Disposable fit-probe weights are discarded; actual runs start fresh.

Accuracy winner rule is independent per branch; source trained simultaneously/validated epoch ensemble, while our sequential queue cannot reproduce that selection trajectory. This is an adapted pipeline, not exact replication. Optional sourceLIME is omitted by default; complete Grad-CAM, actualCBAM maps and bounded validation-only occlusion examples are included. These visualizations are descriptive and do not prove clinical reasoning.

Original results and registry rows are hash-protected. New files only under research/aggressive,results/aggressive_enhanced,checkpoints/aggressive_enhanced. Post-test enhanced scores are labeled explicitly in every table/report. No further performance round or test evaluation is scheduled.
''')
    print(json.dumps(dict(status='protocol_frozen',train=len(train),val=len(val),test_excluded=1503,mean=mean.tolist(),std=std.tolist())),flush=True)

def preflight():
    c=config();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    train,_,w=data(c);x,y,_=next(iter(DataLoader(Images(train,c,True),batch_size=32,num_workers=0)))
    reports=[];use_bf16=torch.cuda.is_bf16_supported()
    if not use_bf16:c['training_precision']='fp32';c['microbatch']=16
    for spec in c['models']:
        m=EnhancedModel(spec).cuda().train();m.stage(6);opt=torch.optim.AdamW(m.parameters(),lr=spec['lr'],weight_decay=.0015)
        xx=x.cuda();yy=y.cuda();ww=w.cuda();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        micro=c['microbatch'];precision=c['training_precision'];attempts=[]
        while True:
            try:
                for _ in range(3):
                    opt.zero_grad(set_to_none=True);order=torch.arange(31,-1,-1,device='cuda');lam=.35
                    mixed=lam*xx+(1-lam)*xx[order];denom=float((lam*ww[yy]+(1-lam)*ww[yy[order]]).sum())
                    for i in range(0,32,micro):
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):out=m(mixed[i:i+micro])
                        loss=mixed_focal_sum(out,yy[i:i+micro],yy[order][i:i+micro],lam,ww,2.2)/denom
                        if not torch.isfinite(out).all() or not torch.isfinite(loss):raise FloatingPointError('Nonfinite preflight forward/loss')
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(),.5,error_if_nonfinite=True);opt.step()
                m.eval()
                with torch.inference_mode():
                    fp=m(xx[:micro].float());assert torch.isfinite(fp).all()
                torch.cuda.synchronize()
                allocated=torch.cuda.max_memory_allocated()/2**20
                reserved=torch.cuda.max_memory_reserved()/2**20
                total=torch.cuda.get_device_properties(0).total_memory/2**20
                if micro==32 and (reserved>total*.82 or allocated>total*.8):raise torch.OutOfMemoryError('Preflight safety margin: insufficient physical VRAM headroom at micro32')
                break
            except (torch.OutOfMemoryError,FloatingPointError,RuntimeError) as exc:
                attempts.append(dict(precision=precision,micro=micro,error=repr(exc)))
                write_json(OUT/(spec['id']+'_preflight_attempts.json'),attempts)
                if isinstance(exc,torch.OutOfMemoryError) and micro==32:micro=16
                elif precision=='bf16':precision='fp32';micro=16
                else:raise
                # Discard failed disposable probe state, reinitialize pretrained weights.
                out=loss=fp=mixed=None
                del m,opt;gc.collect();torch.cuda.empty_cache()
                m=EnhancedModel(spec).cuda().train();m.stage(6);opt=torch.optim.AdamW(m.parameters(),lr=spec['lr'],weight_decay=.0015)
                torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        report=dict(model=spec['model'],attention=spec['attention'],precision=precision,microbatch=micro,effective_batch=32,
            peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,
            host_available_gib=psutil.virtual_memory().available/2**30,three_step_seconds=time.perf_counter()-tick,
            scope='3 disposable full-fine-tuning optimizer steps on real augmented training batch with fixed Mixup; FP32 eval forward; no validation accuracy measured',attempts=attempts,
            safe=True,probe_weights_discarded=True)
        reports.append(report);write_json(OUT/'resource_preflight.json',reports);print(json.dumps(report),flush=True)
        c['microbatch']=min(c['microbatch'],micro)
        if precision=='fp32':spec['initial_precision']='fp32'
        del m,opt,xx,yy,ww,out,loss,fp,mixed;gc.collect();torch.cuda.empty_cache()
    write_json(CONFIG,c) # Only resource choices before ANY enhanced training outcome.
    estimate=sum(max(60, r['three_step_seconds']/3*219)*50 for r in reports)/3600
    write_json(OUT/'preflight_complete.json',dict(status='passed',execution_mode='sequential',config_sha256=sha256(CONFIG),estimated_max_hours=estimate,
        expectation='Conservative synthetic optimizer timing extrapolation; augment/I/O/validation/checkpoints/XAI may add substantial overhead; early stop may shorten'))
    with (ROOT/'research/aggressive/ENHANCED_PROTOCOL_FREEZE.md').open('a',encoding='utf-8') as file:
        file.write(f'\nPre-launch resource amendment: microbatch{c["microbatch"]},effective32; defaultprecision{c["training_precision"]}; model-specific FP32 initial overrides if present inconfig. Disposable fit-probe conservative maximum estimate {estimate:.1f}hours plus I/O/validation/XAI; no outcome seen.\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--preflight',action='store_true');ap.add_argument('--fallback-fit',action='store_true');a=ap.parse_args()
    if a.freeze:freeze()
    if a.preflight:preflight()
    if a.fallback_fit:fallback_fit()

def fallback_fit():
    c=config();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    train,_,w=data(c);x,y,_=next(iter(DataLoader(Images(train,c,True),batch_size=32,num_workers=0)))
    reports=[]
    for spec in c['models']:
        m=EnhancedModel(spec).cuda().train();m.stage(6);opt=torch.optim.AdamW(m.parameters(),lr=spec['lr'],weight_decay=.0015)
        xx=x.cuda();yy=y.cuda();ww=w.cuda();torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
        denom=float(ww[yy].sum())
        for i in range(0,32,16):
            logits=m(xx[i:i+16].float());loss=mixed_focal_sum(logits,yy[i:i+16],yy[i:i+16],1.,ww,2.2)/denom
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):raise FloatingPointError('Nonfinite FP32 fallback preflight')
            loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),.5,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated()/2**20;total=torch.cuda.get_device_properties(0).total_memory/2**20
        if peak>total*.85:raise RuntimeError('Full FP32 fallback lacks safe headroom; must fix before launch')
        reports.append(dict(model=spec['model'],precision='full_fp32',microbatch=16,effective_batch=32,peak_allocated_mb=peak,finite=True,scope='One disposable training-only optimizer update; no validation accuracy'))
        print(json.dumps(reports[-1]),flush=True);write_json(OUT/'fp32_fallback_fit.json',reports)
        del m,opt,xx,yy,ww,logits,loss;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
