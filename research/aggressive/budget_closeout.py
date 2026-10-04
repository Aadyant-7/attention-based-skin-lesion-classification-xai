"""CPU-only budget closeout using saved validation probabilities; no image inference."""
import json
import numpy as np
import pandas as pd
import torch
from research.common import ROOT,CLASSES,write_json,write_csv,atomic_text,relative,sha256
from research.plots import metric_figures,training_figures,comparison_figures
from .core import OUT,CKPT,config,weighted_metrics,retry_registry_upsert
from .train import registry_row
from .stopping import stopping_status

def register_screen():
    from research.registry import FIELDS
    analysis=OUT/'cpu_budget_screen'
    for result in pd.read_csv(analysis/'comparison.csv').to_dict('records'):
        name=result['display_name']
        if name=='strict_reference':continue
        folder=analysis/name;m=json.loads((folder/'validation_metrics.json').read_text())
        row={key:'' for key in FIELDS}
        row.update(experiment_id='budget_screen_'+name,era='structured',record_kind='fixed_ensemble',
            phase='post_test_cpu_budget_screen',protocol='enhanced_lesion_disjoint_development',evaluation_split='validation',
            model=name,method='Fixed equal averaging of saved strict-reference ensemble and enhanced model probabilities',
            status='completed',metrics_path=relative(folder/'validation_metrics.json'),plots_dir=relative(folder/'figures'),
            confusion_matrix_path=relative(folder/'figures/confusion_matrix.csv'),seed=42,
            decision='rejected_no_gain',notes='Same 1503 development images; post-test CPU study; no inference or weight search; DenseNet incomplete.',
            **{key:m[key] for key in ['accuracy','macro_precision','macro_recall','macro_f1']})
        retry_registry_upsert(row);write_json(folder/'record.json',row)

def run():
    c=config();spec=c['models'][1];rid=spec['id'];folder=OUT/rid
    assert json.loads((OUT/'budget_control.json').read_text())['automatic_training_disabled']
    k=torch.load(CKPT/rid/'latest.pt',map_location='cpu',weights_only=False)
    history=k['history'];best=k['best'];f1best=k['f1best']
    for suffix,chosen in [('',best),('_macro_f1',f1best)]:
        m=chosen['metrics'];frame=pd.DataFrame(chosen['predictions'])
        write_json(folder/f'validation_metrics{suffix}.json',m)
        write_csv(folder/f'validation_predictions{suffix}.csv',frame.to_dict('records'))
        write_csv(folder/f'validation_probabilities{suffix}.csv',frame[['image_id']+[f'p_{x}' for x in CLASSES]].to_dict('records'))
        metric_figures(m,folder/('figures'+suffix),'DenseNet201 | budget-stopped development validation')
        saved=torch.load(CKPT/rid/('best.pt' if not suffix else 'best_macro_f1.pt'),map_location='cpu',weights_only=False)
        assert saved['best_epoch']==chosen['epoch']
    training_figures(pd.DataFrame(history),folder/'figures','DenseNet201 | incomplete budget-stopped run',selection_metric='accuracy')
    decision=stopping_status(history)
    write_json(folder/'early_stopping_state.json',decision)
    resets=[];last=0
    for i,row in enumerate(history):
        d=stopping_status(history[:i+1])
        if d['last_meaningful_epoch']!=last:
            previous=history[i-1] if i else row
            resets.append(dict(epoch=int(row['epoch']),accuracy_change_from_previous=float(row['val_accuracy']-previous['val_accuracy']),
                macro_f1_change_from_previous=float(row['val_macro_f1']-previous['val_macro_f1']),
                accuracy_reference=d['accuracy_reference'],macro_f1_reference=d['macro_f1_reference']))
            last=d['last_meaningful_epoch']
    write_json(folder/'patience_audit.json',dict(policy=decision['policy'],resets=resets,
        explanation='Either metric crossing its last meaningful reference resets; cumulative small gains can cross a threshold. Raw-best changes alone do not reset.'))
    summary=dict(status='budget_stopped',completed=False,committed_epochs=len(history),best_epoch=best['epoch'],best_macro_f1_epoch=f1best['epoch'],
        best_accuracy=best['metrics']['accuracy'],best_macro_f1=f1best['metrics']['macro_f1'],stop_reason='user_authorized_compute_budget',meaningful_stopping=decision)
    write_json(folder/'budget_closeout.json',summary)
    row=registry_row(spec,c,history,'budget_stopped')
    row.update(best_epoch=best['epoch'],checkpoint=relative(CKPT/rid/'best.pt'),checkpoint_available_local=True,
        checkpoint_sha256=sha256(CKPT/rid/'best.pt'),metrics_path=relative(folder/'validation_metrics.json'),plots_dir=relative(folder/'figures'),
        decision='incomplete_budget_stop',notes='Stopped for compute budget; not a completed 50-epoch failure. Saved validation only; locked test excluded.',
        **{name:best['metrics'][name] for name in ['accuracy','macro_precision','macro_recall','macro_f1']})
    retry_registry_upsert(row);write_json(folder/'record.json',row)
    reference=pd.read_csv(ROOT/'results/final_strict/v1/ensemble/validation_predictions.csv')
    b3=pd.read_csv(OUT/c['models'][0]['id']/'validation_predictions.csv')
    dense=pd.DataFrame(best['predictions'])
    arrays=[]
    for frame in [reference,b3,dense]:
        assert not frame.image_id.duplicated().any()
        assert set(frame.image_id)==set(reference.image_id)
        ordered=frame.set_index('image_id').loc[reference.image_id].reset_index()
        assert ordered.true_class.tolist()==reference.true_class.tolist()
        p=ordered[[f'p_{x}' for x in CLASSES]].to_numpy()
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        arrays.append(p)
    y=np.array([CLASSES.index(label) for label in reference.true_class]);base=arrays[0].argmax(1)
    analysis=OUT/'cpu_budget_screen';rows=[]
    candidates=[('strict_reference',arrays[0]),('reference_plus_b3_equal',(arrays[0]+arrays[1])/2),
                ('reference_plus_dense_equal',(arrays[0]+arrays[2])/2),
                ('reference_plus_b3_dense_equal',sum(arrays)/3)]
    from .core import predictions
    from .results import report_loss
    distribution=json.loads((OUT/'class_distribution.json').read_text())
    weights=np.array([distribution['class_weights'][cl] for cl in CLASSES])
    for name,p in candidates:
        m=weighted_metrics(y,p,report_loss(y,p,weights,c['focal_gamma']));out=analysis/name
        m['loss_definition']='Weighted focal on saved probabilities with enhanced train-only class weights'
        write_json(out/'validation_metrics.json',m);write_csv(out/'validation_predictions.csv',predictions(reference.image_id,y,p))
        metric_figures(m,out/'figures',name+' | post-test strict development screening')
        fixed=int(((base!=y)&(p.argmax(1)==y)).sum());lost=int(((base==y)&(p.argmax(1)!=y)).sum())
        rows.append(dict(display_name=name,accuracy=m['accuracy'],macro_f1=m['macro_f1'],fixed_reference_errors=fixed,lost_reference_correct=lost))
    comparison_figures(rows,analysis/'figures','CPU-only fixed screening | same development cohort')
    write_csv(analysis/'comparison.csv',rows)
    text='# Budget decision and CPU-only screening\n\nLong queue stopped; ResNet101 cancelled before launch. No test inference, raw-image loading, weight search or new training. DenseNet is incomplete, not a full-run failure.\n\n'
    text+=f'DenseNet best saved accuracy: {100*summary["best_accuracy"]:.4f}% at epoch {best["epoch"]}; best macro-F1 {summary["best_macro_f1"]:.6f} at epoch {f1best["epoch"]}; committed epochs {len(history)}.\n\n'
    text+='| Method | Accuracy | Macro-F1 | Reference errors fixed | Correct lost |\n|---|---:|---:|---:|---:|\n'
    for r in rows:text+=f'| {r["display_name"]} | {100*r["accuracy"]:.4f}% | {r["macro_f1"]:.6f} | {r["fixed_reference_errors"]} | {r["lost_reference_correct"]} |\n'
    text+='\nThese are post-test development comparisons, not new held-out test results. The reference is itself the equal B0/ConvNeXt-Tiny/EfficientNetV2-S ensemble; added-model comparisons average at the ensemble level.\n'
    atomic_text(ROOT/'research/aggressive/BUDGET_SCREENING_RESULTS.md',text)
    register_screen()
    print(json.dumps(dict(dense=summary,comparisons=rows),indent=2))

if __name__=='__main__':run()
