"""CPU-only S05 closeout and criterion-matched S03/S05 CBAM comparison."""
import json
import numpy as np
import pandas as pd
import torch
from research.common import ROOT, CLASSES, sha256, write_json, write_csv
from research.plots import metric_figures, comparison_figures
from research.registry import read_registry, upsert
from research.phase3.close_s02 import load, verify_predictions, verify_figures, verify_launch_code

ID='s05_efficientnet_b0_cbam_exploratory_seed42'
CONTROL='s03_efficientnet_b0_none_exploratory_seed42'


def main():
    path=ROOT/'results/structured_experiments'/ID
    config,record=load(path/'config.json'),load(path/'record.json')
    assert record['status']=='completed' and config['protocol']=='exploratory_image_level'
    assert sha256(ROOT/config['split_manifest'])==config['split_sha256']
    val=pd.read_csv(ROOT/config['split_manifest']).query("split=='val'").set_index('image_id')
    history=pd.read_csv(path/'history.csv');assert history.epoch.tolist()==list(range(1,21)) and record['epochs']==20
    checkpoints=ROOT/'checkpoints/structured'/ID
    latest=torch.load(checkpoints/'latest.pt',map_location='cpu',weights_only=False)
    assert latest['config']==config
    sources=verify_launch_code(latest,load(path/'environment.json'))
    pd.testing.assert_frame_equal(pd.DataFrame(latest['history']),history,check_exact=False,rtol=1e-12,atol=1e-12)
    other=ROOT/'results/structured_experiments'/CONTROL
    control_config=load(other/'config.json')
    differences={k:[control_config.get(k),config.get(k)] for k in sorted(set(config)|set(control_config)) if control_config.get(k)!=config.get(k)}
    assert set(differences)=={'recipe_version','phase','attention','experiment_id','question'}
    out=ROOT/'results/model_comparison/structured/s03_s05_cbam_ablation'
    selected={};comparisons=[];class_rows=[]
    for criterion,suffix,name,key in (('accuracy','','best.pt','best'),('macro_f1','_macro_f1','best_macro_f1.pt','secondary_best')):
        metrics=load(path/f'validation_metrics{suffix}.json')
        cm=verify_predictions(path/f'validation_predictions{suffix}.csv',metrics,val)
        epoch=int(history.loc[history['val_'+criterion].idxmax(),'epoch'])
        winner=torch.load(checkpoints/name,map_location='cpu',weights_only=False);state=latest[key]
        assert winner['config']==config and winner['best_epoch']==state['epoch']==epoch
        assert winner['selection_metric']==criterion and winner['metrics']==state['metrics']==metrics
        assert all(torch.equal(winner['model'][k],v) for k,v in state['model'].items())
        pd.testing.assert_frame_equal(pd.DataFrame(state['predictions']),pd.read_csv(path/f'validation_predictions{suffix}.csv'),check_exact=False,rtol=1e-12,atol=1e-12)
        row=history.loc[history.epoch==epoch].iloc[0]
        for field in ('accuracy','macro_precision','macro_recall','macro_f1','loss'):
            assert np.isclose(metrics[field],row['val_'+field],rtol=0,atol=1e-10)
        if not suffix:
            assert record['best_epoch']==epoch and sha256(checkpoints/name)==record['checkpoint_sha256']
            for field in ('accuracy','macro_precision','macro_recall','macro_f1'):
                assert np.isclose(record[field],metrics[field],rtol=0,atol=1e-10)
        folder=path/('figures_macro_f1' if suffix else 'figures')
        if suffix:metric_figures(metrics,folder,'S05 B0+CBAM | exploratory validation | macro-F1 winner')
        verify_figures(folder,cm,metrics,curves=not suffix)
        control=load(other/f'validation_metrics{suffix}.json')
        verify_predictions(other/f'validation_predictions{suffix}.csv',control,val)
        selected[criterion]=dict(epoch=epoch,metrics=metrics,control_metrics=control,
            deltas={k:metrics[k]-control[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1','loss')})
        for label,rid,m,ep in [('S03 B0 | no added CBAM',CONTROL,control,20),('S05 B0 | added CBAM',ID,metrics,epoch)]:
            comparisons.append(dict(display_name=label+' | '+criterion+' selected',experiment_id=rid,selection_metric=criterion,
                best_epoch=ep,accuracy=m['accuracy'],macro_precision=m['macro_precision'],macro_recall=m['macro_recall'],macro_f1=m['macro_f1'],
                melanoma_recall=m['per_class']['mel']['recall'],scope='same exploratory training policy; one seed; criteria separate'))
        for c in CLASSES:
            class_rows.append(dict(selection_metric=criterion,class_name=c,support=metrics['per_class'][c]['support'],
                **{f'control_{k}':control['per_class'][c][k] for k in ('precision','recall','f1')},
                **{f'cbam_{k}':metrics['per_class'][c][k] for k in ('precision','recall','f1')},
                f1_delta=metrics['per_class'][c]['f1']-control['per_class'][c]['f1']))
    comparison_figures(comparisons,out,'Matched B0 CBAM study | exploratory validation | seed42\nAccuracy and macro-F1 selection shown separately')
    write_csv(out/'per_class_comparison.csv',class_rows)
    baseline=read_registry();untouched=[r for r in baseline if r['experiment_id']!=ID]
    record.update(source_availability='original',decision='cbam_no_primary_accuracy_gain_preserve_secondary_candidate_screen_diverse_backbone',
        notes=record['notes'].split(' Closed out on CPU:')[0]+' Closed out on CPU: both checkpoint/prediction winners, history, class scores, matrices and figures verified. Matched S03: accuracy tied; primary F1 slightly lower; secondary F1 slightly higher. No broad CBAM advantage established; screen distinct backbone next.')
    upsert(record);write_json(path/'record.json',record)
    assert untouched==[r for r in read_registry() if r['experiment_id']!=ID]
    report=dict(status='verified_completed',experiment_id=ID,protocol=config['protocol'],epochs=20,selected=selected,
        final_epoch_metrics={k:float(history.iloc[-1]['val_'+k]) for k in ('accuracy','macro_precision','macro_recall','macro_f1','loss')},
        final_train_accuracy=float(history.iloc[-1].train_accuracy),minimum_val_loss_epoch=int(history.loc[history.val_loss.idxmin(),'epoch']),
        runtime_seconds=record['runtime_seconds'],matched_config_differences=differences,verified_launch_sources=sources,
        checkpoint_hashes={p.name:sha256(p) for p in checkpoints.glob('*.pt')},
        source_hashes={n:sha256(path/n) for n in ('config.json','history.csv','validation_metrics.json','validation_metrics_macro_f1.json','validation_predictions.csv','validation_predictions_macro_f1.csv')},
        historical_rows_preserved=sum(r['era']=='legacy' for r in untouched),all_other_registry_rows_unchanged=True,gpu_used=False,test_images_loaded=False)
    write_json(path/'closeout_verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('selected','source_hashes','verified_launch_sources')},indent=2))


if __name__=='__main__':main()
