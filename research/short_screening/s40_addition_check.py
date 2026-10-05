"""One equal four-member saved-probability check. No training/inference/test."""
import json
import numpy as np
import pandas as pd
from research.common import ROOT, CLASSES, write_json, write_csv, relative
from research.strict_protocol import partition_metadata, SPLIT, DIGEST
from research.strict_train import metric_report
from research.aggressive.core import predictions, retry_registry_upsert
from research.plots import metric_figures, comparison_figures


def main():
    out=ROOT/'results/short_screening/s41_add_mixup_fixed_four'
    if (out/'summary.json').exists():
        print((out/'summary.json').read_text());return
    val=partition_metadata()[0].query("split=='val'").reset_index(drop=True)
    ids=['s28_efficientnet_b0_final_strict_seed42','s29_convnext_tiny_final_strict_seed42','s30_efficientnet_v2_s_final_strict_seed42','s40_convnext_tiny_mixup_strict_dev_seed42']
    arrays=[]
    for i,rid in enumerate(ids):
        folder=(ROOT/'results/structured_experiments'/rid if i<3 else ROOT/'results/short_screening/mixup_v1'/rid)
        frame=pd.read_csv(folder/'validation_predictions.csv')
        assert frame.image_id.is_unique and set(frame.image_id)==set(val.image_id)
        frame=frame.set_index('image_id').loc[val.image_id]
        assert frame.true_class.tolist()==val.diagnosis.tolist()
        p=frame[[f'p_{cl}' for cl in CLASSES]].to_numpy()
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        arrays.append(p)
    old=np.mean(arrays[:3],axis=0);new=np.mean(arrays,axis=0)
    y=val.label.to_numpy();a=old.argmax(1)==y;b=new.argmax(1)==y
    reference=metric_report(y,old,float(-np.log(np.maximum(old[np.arange(len(y)),y],1e-12)).mean()))
    metrics=metric_report(y,new,float(-np.log(np.maximum(new[np.arange(len(y)),y],1e-12)).mean()))
    original=arrays[1].argmax(1)==y;mixup=arrays[3].argmax(1)==y
    overlap=dict(mixup_fixes_original_convnext=int((~original&mixup).sum()),
                 original_convnext_fixes_mixup=int((original&~mixup).sum()),
                 both_convnext_variants_wrong=int((~original&~mixup).sum()))
    gained=int((~a&b).sum());lost=int((a&~b).sum())
    passed=metrics['accuracy']-reference['accuracy']>=.005-1e-12 and metrics['macro_f1']>=reference['macro_f1']
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'validation_metrics.json',metrics)
    pred=predictions(val.image_id,y,new);write_csv(out/'validation_predictions.csv',pred)
    write_csv(out/'validation_probabilities.csv',pd.DataFrame(pred)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
    metric_figures(metrics,out/'figures','S41 fixed four-model equal fusion | strict development')
    rows=[dict(display_name='Frozen three-model reference',accuracy=reference['accuracy'],macro_f1=reference['macro_f1']),dict(display_name='Add S40, four equal members',accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'])]
    comparison_figures(rows,out/'comparison_figures','Does adding Mixup help without replacing original ConvNeXt?')
    summary=dict(status='completed',protocol='post_test_strict_development',reference=rows[0],candidate=rows[1],
                 gained=gained,lost=lost,net=gained-lost,error_overlap=overlap,material_gate_passed=bool(passed),
                 weights=[.25]*4,members=ids,checkpoint_selection='existing raw accuracy winners only',
                 test_loaded=False,gpu_used=False,decision='No weight/checkpoint search; stop after this combination')
    write_json(out/'summary.json',summary)
    retry_registry_upsert(dict(experiment_id='s41_add_mixup_fixed_four_strict_dev_seed42',era='structured',record_kind='inference_candidate',phase='post_test_development',protocol='strict_lesion_disjoint',evaluation_split='validation',split_manifest=SPLIT,split_sha256=DIGEST,method='Add S40 to original fixed ensemble; equal four probabilities',ensemble_members=json.dumps(ids),ensemble_weights='[0.25,0.25,0.25,0.25]',epochs=0,image_size=224,seed=42,metrics_path=relative(out/'validation_metrics.json'),plots_dir=relative(out/'figures'),status='completed',decision='material_gate_passed' if passed else 'material_gate_failed',notes='One CPU-only combination. Already selected checkpoints; development validation, no test.',**{k:metrics[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
