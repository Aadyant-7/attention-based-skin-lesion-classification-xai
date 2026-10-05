"""Exactly two fixed robust-voting checks using preserved S46 model outputs."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from research.common import ROOT,CLASSES,sha256,relative,write_json,write_csv
from research.strict_train import metric_report
from research.aggressive.core import predictions,retry_registry_upsert
from research.plots import metric_figures,comparison_figures

OUT=ROOT/'results/short_screening/s47_s48_robust_vote_v1'
PARENT=ROOT/'results/short_screening/s46_all_f1_checkpoint_fusion'
PLAN=dict(methods=['hard_vote_probability_tiebreak','trimmed_middle_two'],
    hard_vote='Each model votes for its highest-probability class; highest vote count wins; equal soft mean breaks ties only',
    trimmed='For each class, sort four probabilities, discard highest/lowest, average middle two; normalize across classes',
    tiebreak_epsilon=1e-6,weights='fixed; no fitting',checkpoint_rule='Exact four preserved S46 macro-F1 winners',
    scope='Exploratory developmental validation, already used for model/epoch/method selection',
    gate='Accuracy above S46, macro-F1 nondecrease and melanoma recall nondecrease; report modest gains as modest',
    study_limit='Exactly two methods; no trim-fraction, epsilon, checkpoint or weight search',test_loaded=False,gpu_used=False)


def main():
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    OUT.mkdir(parents=True,exist_ok=False);write_json(OUT/'PREDECLARED_PLAN.json',PLAN)
    manifest=json.loads((PARENT/'candidate_manifest.json').read_text())
    reference=pd.read_csv(PARENT/'validation_predictions.csv');assert reference.image_id.is_unique and len(reference)==1503
    split='data/splits/exploratory/image_level_dev_v1.csv'
    identities=pd.read_csv(ROOT/split,usecols=['image_id','split'])
    assert set(reference.image_id)==set(identities.loc[identities.split=='val','image_id'])
    locked=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','split'])
    assert not set(reference.image_id)&set(locked.loc[locked.split=='test','image_id'])
    arrays=[]
    for source in manifest['sources']:
        assert sha256(ROOT/source['prediction_file'])==source['prediction_sha256']
        frame=pd.read_csv(ROOT/source['prediction_file']);assert frame.image_id.is_unique and set(frame.image_id)==set(reference.image_id)
        frame=frame.set_index('image_id').loc[reference.image_id]
        assert frame.true_class.tolist()==reference.true_class.tolist()
        p=frame[[f'p_{c}' for c in CLASSES]].to_numpy()
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5);arrays.append(p)
    stack=np.stack(arrays);baseline=stack.mean(0)
    np.testing.assert_allclose(baseline,reference[[f'p_{c}' for c in CLASSES]].to_numpy(),atol=1e-12)
    votes=np.eye(7)[stack.argmax(2)].mean(0)
    hard=(votes+1e-6*baseline)/(1+1e-6)
    trim=np.sort(stack,axis=0)[1:3].mean(0);assert (trim.sum(1)>0).all();trim/=trim.sum(1,keepdims=True)
    # Both rules are model-order invariant. Hard scores preserve unique plurality winners.
    reverse=stack[::-1]
    assert np.array_equal(hard.argmax(1),((np.eye(7)[reverse.argmax(2)].mean(0)+1e-6*reverse.mean(0))/(1+1e-6)).argmax(1))
    assert np.array_equal(trim.argmax(1),np.sort(reverse,axis=0)[1:3].mean(0).argmax(1))
    count=np.eye(7)[stack.argmax(2)].sum(0);maximum=count.max(1,keepdims=True)
    unique=(count==maximum).sum(1)==1;assert np.array_equal(hard.argmax(1)[unique],count.argmax(1)[unique])
    y=np.array([CLASSES.index(c) for c in reference.true_class]);bc=baseline.argmax(1)==y
    bm=json.loads((PARENT/'validation_metrics.json').read_text())
    rows=[dict(display_name='S46 equal soft vote',accuracy=bm['accuracy'],macro_f1=bm['macro_f1'])];decisions={}
    for index,(name,p) in enumerate([('hard_vote_probability_tiebreak',hard),('trimmed_middle_two',trim)],start=47):
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-7)
        m=metric_report(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()))
        m['score_definition']='Normalized voting scores, not calibrated class probabilities' if index==47 else 'Normalized trimmed class scores; not calibrated probabilities'
        folder=OUT/name;write_json(folder/'validation_metrics.json',m)
        pred=predictions(reference.image_id,y,p);write_csv(folder/'validation_predictions.csv',pred)
        scores=pd.DataFrame(pred)[['image_id']+[f'p_{c}' for c in CLASSES]].rename(columns={f'p_{c}':f'score_{c}' for c in CLASSES})
        write_csv(folder/'validation_decision_scores.csv',scores.to_dict('records'))
        metric_figures(m,folder/'figures',f'S{index} {name} | exploratory development')
        saved=pd.read_csv(folder/'validation_predictions.csv')
        assert abs(accuracy_score(saved.true_class,saved.predicted_class)-m['accuracy'])<1e-12
        assert abs(f1_score(saved.true_class,saved.predicted_class,labels=CLASSES,average='macro')-m['macro_f1'])<1e-12
        correct=p.argmax(1)==y
        passed=m['accuracy']>bm['accuracy']+1e-12 and m['macro_f1']>=bm['macro_f1'] and m['per_class']['mel']['recall']>=bm['per_class']['mel']['recall']
        decisions[name]=dict(gained=int((~bc&correct).sum()),lost=int((bc&~correct).sum()),net=int(correct.sum()-bc.sum()),mel_recall=m['per_class']['mel']['recall'],adoption_gate_passed=bool(passed))
        rows.append(dict(display_name=name,**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
        retry_registry_upsert(dict(experiment_id=f's{index}_{name}_exploratory_seed42',era='structured',record_kind='cpu_fusion_screen',phase='post_test_exploratory_development',protocol='exploratory_image_level',evaluation_split='validation',split_manifest=split,split_sha256=sha256(ROOT/split),method=name,ensemble_members=json.dumps([s['run'] for s in manifest['sources']]),epochs=0,seed=42,metrics_path=relative(folder/'validation_metrics.json'),plots_dir=relative(folder/'figures'),status='completed',decision='adoption_gate_passed' if passed else 'adoption_gate_failed',notes='Fixed robust vote; normalized decision scores are not calibrated probabilities; validation-selection optimism remains; no test.',**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    comparison_figures(rows,OUT/'comparison_figures','Two fixed robust-voting rules against S46')
    summary=dict(status='completed',comparisons=rows,decisions=decisions,source_manifest=relative(PARENT/'candidate_manifest.json'),test_loaded=False,gpu_used=False,decision='Two-method budget exhausted; no further voting-rule variants or automatic training')
    write_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
