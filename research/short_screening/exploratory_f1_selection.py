"""One fixed all-macro-F1-checkpoint fusion policy on saved exploratory scores."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from research.common import ROOT,CLASSES,sha256,relative,write_json,write_csv
from research.short_screening.exploratory_stack_v1 import MEMBERS,SPLIT
from research.strict_train import metric_report
from research.aggressive.core import predictions,retry_registry_upsert
from research.plots import metric_figures,comparison_figures

OUT=ROOT/'results/short_screening/s46_all_f1_checkpoint_fusion'


def main():
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    OUT.mkdir(parents=True,exist_ok=False)
    write_json(OUT/'PREDECLARED_PLAN.json',dict(members=MEMBERS,weights=[.25]*4,
        selection='All four models use their existing independently macro-F1-selected checkpoint; no per-model choice or epoch search',
        protocol='exploratory_image_level',scope='post-test development validation',
        material_gate='>=+.005 accuracy over S42; macro-F1 and MEL recall nondecrease',
        limitations='Both sets of base checkpoints were selected using this same validation cohort; not an independent test result.',test_loaded=False,gpu_used=False))
    ref=pd.read_csv(ROOT/'results/short_screening/s42_exploratory_stack_v1/equal_four_control/validation_predictions.csv')
    assert ref.image_id.is_unique and len(ref)==1503
    identities=pd.read_csv(ROOT/SPLIT,usecols=['image_id','split'])
    assert set(ref.image_id)==set(identities.loc[identities.split=='val','image_id'])
    locked=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','split'])
    assert not set(ref.image_id)&set(locked.loc[locked.split=='test','image_id'])
    arrays=[];sources=[]
    for rid in MEMBERS:
        path=ROOT/'results/structured_experiments'/rid/'validation_predictions_macro_f1.csv'
        checkpoint=ROOT/'checkpoints/structured'/rid/'best_macro_f1.pt'
        assert checkpoint.is_file()
        frame=pd.read_csv(path);assert frame.image_id.is_unique and set(frame.image_id)==set(ref.image_id)
        frame=frame.set_index('image_id').loc[ref.image_id];assert frame.true_class.tolist()==ref.true_class.tolist()
        p=frame[[f'p_{cl}' for cl in CLASSES]].to_numpy()
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        arrays.append(p);sources.append(dict(run=rid,prediction_file=relative(path),prediction_sha256=sha256(path),checkpoint=relative(checkpoint),checkpoint_sha256=sha256(checkpoint)))
    p=np.mean(arrays,axis=0);y=np.array([CLASSES.index(c) for c in ref.true_class])
    m=metric_report(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()))
    baseline=json.loads((ROOT/'results/short_screening/s42_exploratory_stack_v1/equal_four_control/validation_metrics.json').read_text())
    pred=predictions(ref.image_id,y,p);write_csv(OUT/'validation_predictions.csv',pred)
    write_csv(OUT/'validation_probabilities.csv',pd.DataFrame(pred)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
    write_json(OUT/'validation_metrics.json',m);metric_figures(m,OUT/'figures','S46 fixed F1 checkpoint rule | exploratory validation')
    saved=pd.read_csv(OUT/'validation_predictions.csv')
    assert abs(accuracy_score(saved.true_class,saved.predicted_class)-m['accuracy'])<1e-12
    assert abs(f1_score(saved.true_class,saved.predicted_class,labels=CLASSES,average='macro')-m['macro_f1'])<1e-12
    old=ref.true_class.eq(ref.predicted_class).to_numpy();new=p.argmax(1)==y
    passed=m['accuracy']-baseline['accuracy']>=.005-1e-12 and m['macro_f1']>=baseline['macro_f1'] and m['per_class']['mel']['recall']>=baseline['per_class']['mel']['recall']
    rows=[dict(display_name='S42 accuracy winners equal fusion',accuracy=baseline['accuracy'],macro_f1=baseline['macro_f1']),dict(display_name='S46 macro-F1 winners equal fusion',accuracy=m['accuracy'],macro_f1=m['macro_f1'])]
    comparison_figures(rows,OUT/'comparison_figures','Fixed checkpoint-selection policy comparison')
    summary=dict(status='completed',comparisons=rows,accuracy=m['accuracy'],macro_f1=m['macro_f1'],mel_recall=m['per_class']['mel']['recall'],gained=int((~old&new).sum()),lost=int((old&~new).sum()),net=int(new.sum()-old.sum()),material_gate_passed=bool(passed),sources=sources,test_loaded=False,gpu_used=False,decision='Fixed rule complete; no other checkpoint combinations or training')
    write_json(OUT/'summary.json',summary)
    retry_registry_upsert(dict(experiment_id='s46_all_f1_checkpoint_equal_fusion_exploratory_seed42',era='structured',record_kind='cpu_fusion_screen',phase='post_test_exploratory_development',protocol='exploratory_image_level',evaluation_split='validation',split_manifest=SPLIT,split_sha256=sha256(ROOT/SPLIT),method='Fixed all macro-F1 winner checkpoint rule; equal fusion',ensemble_members=json.dumps(MEMBERS),ensemble_weights='[0.25,0.25,0.25,0.25]',epochs=0,seed=42,metrics_path=relative(OUT/'validation_metrics.json'),plots_dir=relative(OUT/'figures'),status='completed',decision='material_gate_passed' if passed else 'material_gate_failed',notes='Same exploratory validation used for base checkpoint selection; no independent test claim.',**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2))


if __name__=='__main__':main()
