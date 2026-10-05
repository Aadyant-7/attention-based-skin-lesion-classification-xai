"""Three predeclared CPU follow-ups to S42. No continuous weight search."""
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
from research.common import ROOT, CLASSES, sha256, relative, write_json, write_csv
from research.short_screening.exploratory_stack_v1 import MEMBERS, SPLIT
from research.strict_train import metric_report
from research.aggressive.core import predictions, retry_registry_upsert
from research.plots import metric_figures, comparison_figures

OUT=ROOT/'results/short_screening/s43_bounded_exploratory_fusion'
PLAN=dict(members=MEMBERS,member_order='Tiny, Small, Dense201, V2S',
    methods={'strong_convnext_weights':[.35,.35,.15,.15],
             'tiny_emphasis_weights':[.4,.2,.2,.2],
             'equal_geometric_fusion':[.25,.25,.25,.25]},
    rationale={'strong_convnext_weights':'70% weight to the two strongest standalone backbones; retain complementary Dense/V2S',
               'tiny_emphasis_weights':'Favor Tiny, whose standalone macro-F1 exceeds Small; retain all other components',
               'equal_geometric_fusion':'Fixed log-probability pooling, not fitting weights or routing using labels'},
    probability_floor=1e-8,protocol='exploratory_image_level',evaluation='repeatedly used development validation',
    search_limit='Exactly three listed methods, no additional weights, checkpoints, clipping thresholds or cohorts',
    material_gate='>=+.005 absolute accuracy vs S42 with nondecreasing macro-F1 and melanoma recall',
    test_loaded=False,gpu_used=False)


def main():
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    OUT.mkdir(parents=True,exist_ok=False);write_json(OUT/'PREDECLARED_PLAN.json',PLAN)
    reference_path=ROOT/'results/short_screening/s42_exploratory_stack_v1/equal_four_control/validation_predictions.csv'
    reference=pd.read_csv(reference_path);assert reference.image_id.is_unique and len(reference)==1503
    y=np.array([CLASSES.index(cl) for cl in reference.true_class]);arrays=[];hashes={}
    manifest=pd.read_csv(ROOT/SPLIT,usecols=['image_id','lesion_id','split'])
    assert set(reference.image_id)==set(manifest.loc[manifest.split=='val','image_id'])
    locked=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','split'])
    assert not set(reference.image_id)&set(locked.loc[locked.split=='test','image_id'])
    for rid in MEMBERS:
        path=ROOT/'results/structured_experiments'/rid/'validation_predictions.csv'
        frame=pd.read_csv(path);assert frame.image_id.is_unique and set(frame.image_id)==set(reference.image_id)
        frame=frame.set_index('image_id').loc[reference.image_id]
        assert frame.true_class.tolist()==reference.true_class.tolist()
        p=frame[[f'p_{cl}' for cl in CLASSES]].to_numpy()
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        arrays.append(p);hashes[relative(path)]=sha256(path)
    baseline=reference[[f'p_{cl}' for cl in CLASSES]].to_numpy()
    np.testing.assert_allclose(np.mean(arrays,axis=0),baseline,atol=1e-12)
    bm=metric_report(y,baseline,float(-np.log(np.maximum(baseline[np.arange(len(y)),y],1e-12)).mean()))
    bc=baseline.argmax(1)==y;rows=[dict(display_name='S42 equal probability baseline',**{k:bm[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']})];decisions={}
    for index,(name,weights) in enumerate(PLAN['methods'].items(),start=43):
        weights=np.array(weights);assert np.isclose(weights.sum(),1)
        if name=='equal_geometric_fusion':p=softmax(sum(w*np.log(np.maximum(a,1e-8)) for w,a in zip(weights,arrays)),axis=1)
        else:p=sum(w*a for w,a in zip(weights,arrays))
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        m=metric_report(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()));folder=OUT/name
        write_json(folder/'validation_metrics.json',m);pred=predictions(reference.image_id,y,p)
        write_csv(folder/'validation_predictions.csv',pred)
        write_csv(folder/'validation_probabilities.csv',pd.DataFrame(pred)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
        metric_figures(m,folder/'figures',f'S{index} {name} | exploratory validation')
        saved=pd.read_csv(folder/'validation_predictions.csv')
        assert abs(accuracy_score(saved.true_class,saved.predicted_class)-m['accuracy'])<1e-12
        assert abs(f1_score(saved.true_class,saved.predicted_class,labels=CLASSES,average='macro')-m['macro_f1'])<1e-12
        rows.append(dict(display_name=name,**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
        correct=p.argmax(1)==y;gain=m['accuracy']-bm['accuracy']
        passed=gain>=.005-1e-12 and m['macro_f1']>=bm['macro_f1'] and m['per_class']['mel']['recall']>=bm['per_class']['mel']['recall']
        decisions[name]=dict(gained=int((~bc&correct).sum()),lost=int((bc&~correct).sum()),net=int(correct.sum()-bc.sum()),gain_pp=100*gain,mel_recall=m['per_class']['mel']['recall'],material_gate_passed=bool(passed))
        retry_registry_upsert(dict(experiment_id=f's{index}_{name}_exploratory_seed42',era='structured',record_kind='cpu_fusion_screen',phase='post_test_exploratory_development',protocol='exploratory_image_level',evaluation_split='validation',split_manifest=SPLIT,split_sha256=sha256(ROOT/SPLIT),method=name,ensemble_members=json.dumps(MEMBERS),ensemble_weights=json.dumps(weights.tolist()),epochs=0,seed=42,metrics_path=relative(folder/'validation_metrics.json'),plots_dir=relative(folder/'figures'),status='completed',decision='material_gate_passed' if passed else 'material_gate_failed',notes='One of exactly3predeclared CPU methods. Reuses original selected-model probabilities; repeated validation selection; no test result.',**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    comparison_figures(rows,OUT/'comparison_figures','S43–S45: three fixed exploratory fusion follow-ups')
    for path,digest in hashes.items():assert sha256(ROOT/path)==digest
    summary=dict(status='completed',comparisons=rows,decisions=decisions,source_prediction_sha256=hashes,
                 interpretation='Observed same-cohort developmental scores, not test or independent CV performance',
                 gpu_used=False,test_loaded=False,decision='Finite study complete; no automatic training or further weight trials')
    write_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
