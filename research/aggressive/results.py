"""Bounded fixed weighted/equal closeout, XAI, and honest post-test reporting."""
import json
import numpy as np
import pandas as pd
from research.common import ROOT,CLASSES,write_csv,write_json,atomic_text,relative,sha256
from research.plots import metric_figures,comparison_figures,save
from research.registry import FIELDS,upsert
from .core import OUT,CKPT,CONFIG,config,data,verify_saved,weighted_metrics,predictions

def fusion(arrays,weights):
    if len(arrays)!=3 or weights not in [[.4,.4,.2],[1/3]*3] or len({a.shape for a in arrays})!=1:raise ValueError('Only predeclared aligned fusion permitted')
    p=sum(w*a for w,a in zip(weights,arrays))
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid fusion')
    return p

def report_loss(y,p,w,gamma):
    true=p[np.arange(len(y)),y]
    if (true<=0).any():raise FloatingPointError('Zero true-class probability; no clamping')
    return float((-w[y]*(1-true)**gamma*np.log(true)).sum()/w[y].sum())

def auc_figures(y,p,m,out,title):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig,ax=plt.subplots(figsize=(7,5))
    curves=[]
    for j,c in enumerate(CLASSES):
        fpr,tpr,threshold=roc_curve(y==j,p[:,j]);ax.plot(fpr,tpr,label=f'{c} AUC={m["per_class_auc"][c]:.3f}')
        curves.extend(dict(class_name=c,false_positive_rate=float(a),true_positive_rate=float(b)) for a,b in zip(fpr,tpr))
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate',title=title);ax.legend();save(fig,out,'roc_auc');write_csv(out/'roc_curves.csv',curves)

def closeout(c):
    import matplotlib.pyplot as plt
    train,val,w=data(c);y=val.label.to_numpy();arrays=[];rows=[];summaries=[]
    for spec in c['models']:
        record,m,frame=verify_saved(spec['id'],c)
        p=frame[[f'p_{x}' for x in CLASSES]].to_numpy(float);arrays.append(p)
        write_csv(OUT/spec['id']/'validation_probabilities.csv',frame[['image_id']+[f'p_{x}' for x in CLASSES]].to_dict('records'))
        auc_figures(y,p,m,OUT/spec['id']/'figures',spec['model']+' | enhanced validation ROC')
        summary=json.loads((OUT/spec['id']/'training_summary.json').read_text());summaries.append(summary)
        history=pd.read_csv(OUT/spec['id']/'history.csv');fig,ax=plt.subplots(figsize=(7,4));ax.plot(history.epoch,history.backbone_lr);ax.set(xlabel='Epoch',ylabel='Learning rate',title=spec['model']+' | StepLR',yscale='log');save(fig,OUT/spec['id']/'figures','lr_history')
        rows.append(dict(display_name=spec['model']+(' + CBAM' if spec['attention']=='cbam' else ''),experiment_id=spec['id'],best_epoch=record['best_epoch'],
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','macro_auc']}))
    primary=None;primary_frame=None
    for name,weights in [('primary_weighted',[.4,.4,.2]),('secondary_equal',[1/3]*3)]:
        p=fusion(arrays,weights);m=weighted_metrics(y,p,report_loss(y,p,w.numpy(),2.2));folder=OUT/name
        frame=pd.DataFrame(predictions(val.image_id,y,p));frame['disagreement']=np.any(np.stack([a.argmax(1) for a in arrays])!=arrays[0].argmax(1),axis=0)
        write_json(folder/'validation_metrics.json',m);write_csv(folder/'validation_predictions.csv',frame.to_dict('records'))
        write_csv(folder/'validation_probabilities.csv',frame[['image_id']+[f'p_{x}' for x in CLASSES]].to_dict('records'))
        metric_figures(m,folder/'figures','Enhanced lesion-disjoint DEVELOPMENT validation | '+name);auc_figures(y,p,m,folder/'figures','Enhanced validation | '+name)
        write_json(folder/'frozen_members.json',dict(models=[dict(spec,checkpoint=relative(CKPT/spec['id']/'best.pt'),checkpoint_sha256=sha256(CKPT/spec['id']/'best.pt'),best_epoch=r['best_epoch']) for spec,r in zip(c['models'],rows)],weights=weights,
            selected_by='Independent earliest maximum validation accuracy; no voting search',protocol=c['protocol'],post_test_development=True,old_test_evaluated=False))
        rows.append(dict(display_name=name,experiment_id='s35_enhanced_weighted_seed42' if name=='primary_weighted' else 's36_enhanced_equal_baseline_seed42',best_epoch='independent member winners',
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','macro_auc']}))
        record={k:'' for k in FIELDS};record.update(experiment_id=rows[-1]['experiment_id'],era='structured',record_kind='fixed_ensemble',phase='aggressive_enhanced_post_test',protocol=c['protocol'],evaluation_split='validation',
            model='efficientnet_b3+cbam + densenet201 + resnet101',ensemble_members=json.dumps([s['id'] for s in c['models']]),ensemble_weights=json.dumps(weights),
            image_size=224,seed=42,split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],status='completed',decision='predeclared_primary' if name=='primary_weighted' else 'secondary_baseline_only',
            metrics_path=relative(folder/'validation_metrics.json'),source_sha256=sha256(folder/'validation_metrics.json'),config_path=relative(folder/'frozen_members.json'),
            plots_dir=relative(folder/'figures'),confusion_matrix_path=relative(folder/'figures/confusion_matrix.csv'),notes='Post-test development/validation; NOT untouched test; original rigorous result preserved.',
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']})
        upsert(record);write_json(folder/'record.json',record)
        if name=='primary_weighted':primary=m;primary_frame=frame
    comparison_figures(rows[:3],OUT/'figures_standalone','Enhanced lesion-disjoint development | standalone');comparison_figures(rows,OUT/'figures_comparison','Enhanced lesion-disjoint development | fixed ensembles')
    distribution=json.loads((OUT/'class_distribution.json').read_text());counts=[distribution['training_counts'][c] for c in CLASSES]
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(7);ax.bar(x-.2,counts,.4,label='Original training images');ax.bar(x+.2,counts,.4,label='On-the-fly augmented training: same image counts');ax.set(xticks=x,xticklabels=CLASSES,ylabel='Unique training images',title='No offline duplication/resampling; diversity + focal weights');ax.legend();save(fig,OUT/'figures_distribution','class_distribution')
    write_csv(OUT/'class_balancing_table.csv',[dict(class_name=c,unique_before=counts[i],unique_after=counts[i],focal_weight=distribution['class_weights'][c]) for i,c in enumerate(CLASSES)])
    original=json.loads((ROOT/'results/final_locked_test/v1/ensemble/test_metrics.json').read_text());strict=json.loads((ROOT/'results/final_strict/v1/ensemble/validation_metrics.json').read_text());exploratory=json.loads((ROOT/'results/structured_experiments/s13_s12_four_flip_tta_exploratory_seed42/validation_metrics_identity_fp32.json').read_text())
    protocol_rows=[dict(display_name=name,accuracy=m['accuracy'],macro_f1=m['macro_f1'],protocol=label) for name,m,label in [
        ('Historical exploratory validation',exploratory,'image-level shared lesions'),('Historical strict validation',strict,'lesion-disjoint validation'),
        ('Original first/only untouched test',original,'rigorous held-out test'),('Enhanced post-test development validation',primary,c['protocol'])]]
    comparison_figures(protocol_rows,OUT/'figures_protocols','Different protocol/status — NOT a common leaderboard')
    write_csv(OUT/'model_vs_accuracy.csv',rows);write_csv(OUT/'method_comparison.csv',protocol_rows)
    # GPU XAI only on validation, bounded; absent categories explicitly recorded.
    from .xai import generate
    xai_manifest=OUT/'xai/manifest.json'
    xai=json.loads(xai_manifest.read_text()) if xai_manifest.exists() else generate(c,primary_frame)
    table='| Model/method | Best epoch | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | Macro AUC |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:table+=f"| {r['display_name']} | {r['best_epoch']} | {100*r['accuracy']:.4f}% | {r['macro_precision']:.6f} | {r['macro_recall']:.6f} | {r['macro_f1']:.6f} | {r['weighted_f1']:.6f} | {r['macro_auc']:.6f} |\n"
    classes='| Class | Precision | Recall | F1 | Support | AUC |\n|---|---:|---:|---:|---:|---:|\n'
    for cl in CLASSES:
        a=primary['per_class'][cl];classes+=f"| {cl} | {a['precision']:.6f} | {a['recall']:.6f} | {a['f1']:.6f} | {a['support']} | {primary['per_class_auc'][cl]:.6f} |\n"
    cm=np.array(primary['confusion_matrix']);achieved=primary['accuracy']>=.93
    write_csv(OUT/'literature_comparison.csv',[dict(reference='Frontiers2026 DOI10.3389/fpubh.2026.1847649',reported_accuracy=.9637,protocol='Nominal8012/2003 lesion partition;1103single-image-lesion primary subset;3fold95.56±.32%',comparability='Different cohort, timm B3, manually selected multipliers and weights, simultaneous training, no CBAM'),
        dict(reference='EnsembleSkinNet2025 DOI10.3389/fonc.2025.1699960',reported_accuracy=.9832,protocol='5fold and5seeds; modified VGG/Res50/Inception/Dense, duplicate filtering and rebalancing',comparability='Different architecture/cohort/balancing/evaluation; not a matched comparison'),
        dict(reference='Our enhanced pipeline',reported_accuracy=primary['accuracy'],protocol=c['protocol']+';7009/1503;post-test;testexcluded',comparability='Own true observed enhanced validation result; not original test')])
    text=f'''# Aggressive enhanced final results — post-test development study

## Original rigorous result — permanently preserved

First/only untouched S31 test: accuracy86.7598%,macro-F1 .794473,macroprecision .814010,macrorecall .778724. Exact original files remain inresults/final_locked_test/v1 and research/FINAL_LOCKED_TEST_RESULTS.md. No old test image/label was loaded by this enhanced study. Enhanced work began after the score was revealed, and its design was motivated by prior outcomes; it is not independent confirmation.

## Protocol and recipe

One predeclared existing lesion-ID-disjoint development split:7009training/1503validation,seed42,zero train/validation lesion overlap. Original1503test excluded. No full-dataset CV; no post-result split selection. Frozen protocol/config inresearch/aggressive/ENHANCED_PROTOCOL_FREEZE.md andconfig.json. Three fresh ImageNet branches:B3+CBAM,DenseNet201,ResNet101. Primary weights .40/.40/.20; equal baseline is secondary regardless of which score is larger. No tuned fusion.

Stable FP32 weighted focal gamma2.2,training-only bounded inverse-frequency weights,controlled class-specific augmentation,Mixup alpha.2/p.3 via mathematically valid blended focal objectives. Train-only normalization; GAP512/256/7 BN/ReLU/dropout.65/.55/.45. Head2epochs,laststage3epochs,alllayers thereafter. AdamW/modelLR1e-4/7.5e-5/5e-5,decay.0015,StepLR.7/10,clip.5,effectivebatch32 (all images retained; terminal33),max50/min25/patience12/delta.0003. True accuracy winners and independent bestF1/latest retained. Active BN/loss/validationFP32; BF16 training or explicit predeclared FP32 amendment documented per run.

Training summaries:
```json
{json.dumps(summaries,indent=2)}
```

## Actual enhanced validation results

{table}
{classes}
Primary target >=93% {'met' if achieved else 'not met'}. STOP either way; no follow-up optimization or old-test evaluation.

Melanoma recall {primary['per_class']['mel']['recall']:.6f}; MEL→NV {int(cm[4,5])},NV→MEL {int(cm[5,4])}. Akiec recall {primary['per_class']['akiec']['recall']:.6f}. Correct/incorrect {primary['correct']}/{primary['incorrect']}.

Raw matrix rows=true,columns=predicted;order{list(CLASSES)}:
```text
{cm}
```

## Attention and XAI

Existing CBAM's channel and spatial attention operate after B3's final convolution before pooling. No claim that CBAM alone causes a gain: historical B0+CBAM remained mixed/neutral. Enhanced effect belongs to the combined framework; no new costly CBAM ablation.

Grad-CAM complete on deterministically selected validation categories, actual ensemble target probability gradients across each architecture's final layer; original/heatmap/overlay,confidence,truth,prediction inresults/aggressive_enhanced/v1/xai. Actual CBAM sigmoid spatial/channel weights saved separately. Composite CAM display averages normalized branch maps, not a unique causal attribution. Bounded occlusion on first2cases; LIME not required/run. Missing categories and per-case layer mapping inxai/manifest.json. Heatmaps do not prove clinical reasoning or diagnostic safety.

## Literature comparison and limits

[2026 primary](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2026.1847649/full): authors report96.37% on a1103-image subset,95.56±.32% in3fold CV. [2025 primary](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2025.1699960/full):98.32% under its own CV/seed/balancing workflow. Neither score is reproduced or directly comparable here. Our method differs in torchvisionB3/CBAM,train-derived normalization and nonmanual focal weights,progressive schedule,sequential accuracy-winner fusion,cohort and development history. Protocol-labelled literature/method/class tables,ROC,matrices,curves,figures saved. AUC is binary one-v-rest per class with all classes present; no thresholds fitted. Natural unique class counts unchanged; augmentation is not fictitious sample multiplication.

All historical evidence is preserved and indexed; master registry adds enhanced entries with explicit protocol labels. Performance-optimization work now stops permanently. Next work is presentation/paper/XAI interpretation, not another model or tuning round.
'''
    atomic_text(ROOT/'research/aggressive/AGGRESSIVE_ENHANCED_FINAL_RESULTS.md',text)
    write_json(OUT/'completion.json',dict(status='completed',primary_accuracy=primary['accuracy'],primary_macro_f1=primary['macro_f1'],target_met=achieved,
        post_test_development=True,original_test_preserved=True,old_test_used=False,no_more_performance_runs=True,xai=xai))
    return primary
