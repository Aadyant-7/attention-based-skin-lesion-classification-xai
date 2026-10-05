"""One output-only conditional ResNet rescue rule; no labels used in routing."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from research.common import ROOT,CLASSES,sha256,relative,write_json,write_csv
from research.strict_train import metric_report
from research.aggressive.core import predictions,retry_registry_upsert
from research.plots import metric_figures,comparison_figures

OUT=ROOT/'results/short_screening/s52_disagreement_resnet_rescue'
PARENT=ROOT/'results/short_screening/s46_all_f1_checkpoint_fusion'
RESCUE='s19_resnet101_none_exploratory_seed42'
PLAN=dict(question='Does architecturally diverse ResNet101 help only on four-member disagreement cases?',
    rule='When fewer than3of4member argmax classes agree, use equal five probabilities including ResNet; otherwise retain S46 equal four',
    current_selection='Exact S46 macro-F1 winner checkpoints',rescue_selection='Preserved S19 raw-accuracy winner',
    no_label_routing=True,agreement_threshold=3,trigger_weights=[.2]*5,other_weights=[.25]*4,
    scope='Exploratory validation already used for selection; not an independent test',
    study_limit='One predefined agreement threshold, one rescue model, no weight/checkpoint/threshold alternatives',
    gate='Accuracy improvement with no macro-F1 or melanoma recall decline',test_loaded=False,gpu_used=False)


def main():
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    OUT.mkdir(parents=True,exist_ok=False);write_json(OUT/'PREDECLARED_PLAN.json',PLAN)
    manifest=json.loads((PARENT/'candidate_manifest.json').read_text())
    ref=pd.read_csv(PARENT/'validation_predictions.csv');assert len(ref)==1503 and ref.image_id.is_unique
    split='data/splits/exploratory/image_level_dev_v1.csv';ids=pd.read_csv(ROOT/split,usecols=['image_id','split'])
    assert set(ref.image_id)==set(ids.loc[ids.split=='val','image_id'])
    locked=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','split'])
    assert not set(ref.image_id)&set(locked.loc[locked.split=='test','image_id'])
    arrays=[];source_hashes={}
    for source in manifest['sources']:
        path=ROOT/source['prediction_file'];assert sha256(path)==source['prediction_sha256']
        frame=pd.read_csv(path);assert frame.image_id.is_unique and set(frame.image_id)==set(ref.image_id)
        frame=frame.set_index('image_id').loc[ref.image_id];assert frame.true_class.tolist()==ref.true_class.tolist()
        arrays.append(frame[[f'p_{cl}' for cl in CLASSES]].to_numpy());source_hashes[relative(path)]=sha256(path)
    path=ROOT/'results/structured_experiments'/RESCUE/'validation_predictions.csv'
    frame=pd.read_csv(path);assert frame.image_id.is_unique and set(frame.image_id)==set(ref.image_id)
    frame=frame.set_index('image_id').loc[ref.image_id];assert frame.true_class.tolist()==ref.true_class.tolist()
    rescue=frame[[f'p_{cl}' for cl in CLASSES]].to_numpy();source_hashes[relative(path)]=sha256(path)
    assert np.isfinite(rescue).all() and np.allclose(rescue.sum(1),1,atol=1e-5)
    checkpoint=ROOT/'checkpoints/structured'/RESCUE/'best.pt';assert checkpoint.is_file()
    stack=np.stack(arrays);base=stack.mean(0)
    np.testing.assert_allclose(base,ref[[f'p_{cl}' for cl in CLASSES]].to_numpy(),atol=1e-12)
    votes=np.eye(7)[stack.argmax(2)].sum(0);maximum=votes.max(1);trigger=maximum<3
    p=base.copy();p[trigger]=(stack.sum(0)[trigger]+rescue[trigger])/5
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    y=np.array([CLASSES.index(cl) for cl in ref.true_class]);m=metric_report(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()))
    write_json(OUT/'validation_metrics.json',m);pred=predictions(ref.image_id,y,p)
    write_csv(OUT/'validation_predictions.csv',pred);write_csv(OUT/'validation_probabilities.csv',pd.DataFrame(pred)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
    write_csv(OUT/'routing.csv',[dict(image_id=i,max_four_model_agreement=int(n),resnet_used=bool(t)) for i,n,t in zip(ref.image_id,maximum,trigger)])
    metric_figures(m,OUT/'figures','S52 fixed disagreement-only ResNet rescue | exploratory validation')
    saved=pd.read_csv(OUT/'validation_predictions.csv')
    assert abs(accuracy_score(saved.true_class,saved.predicted_class)-m['accuracy'])<1e-12
    assert abs(f1_score(saved.true_class,saved.predicted_class,labels=CLASSES,average='macro')-m['macro_f1'])<1e-12
    assert np.array_equal(confusion_matrix(saved.true_class,saved.predicted_class,labels=CLASSES),m['confusion_matrix'])
    old=base.argmax(1)==y;new=p.argmax(1)==y;bm=json.loads((PARENT/'validation_metrics.json').read_text())
    passed=m['accuracy']>bm['accuracy']+1e-12 and m['macro_f1']>=bm['macro_f1'] and m['per_class']['mel']['recall']>=bm['per_class']['mel']['recall']
    rows=[dict(display_name='S46 equal four reference',accuracy=bm['accuracy'],macro_f1=bm['macro_f1']),dict(display_name='S52 ResNet on disagreement only',accuracy=m['accuracy'],macro_f1=m['macro_f1'])]
    comparison_figures(rows,OUT/'comparison_figures','Fixed conditional fifth-model rescue')
    summary=dict(status='completed',comparisons=rows,routed_images=int(trigger.sum()),gained=int((~old&new).sum()),lost=int((old&~new).sum()),net=int(new.sum()-old.sum()),mel_recall=m['per_class']['mel']['recall'],adoption_gate_passed=bool(passed),source_prediction_sha256=source_hashes,rescue_checkpoint=relative(checkpoint),rescue_checkpoint_sha256=sha256(checkpoint),test_loaded=False,gpu_used=False,decision='Single rule evaluated; no alternative thresholds/rescue models or automatic training')
    for source,digest in source_hashes.items():assert sha256(ROOT/source)==digest
    write_json(OUT/'summary.json',summary)
    retry_registry_upsert(dict(experiment_id='s52_disagreement_resnet_rescue_exploratory_seed42',era='structured',record_kind='cpu_fusion_screen',phase='post_test_exploratory_development',protocol='exploratory_image_level',evaluation_split='validation',split_manifest=split,split_sha256=sha256(ROOT/split),method=PLAN['rule'],ensemble_members=json.dumps([s['run'] for s in manifest['sources']]+[RESCUE]),epochs=0,seed=42,metrics_path=relative(OUT/'validation_metrics.json'),plots_dir=relative(OUT/'figures'),status='completed',decision='adoption_gate_passed' if passed else 'adoption_gate_failed',notes=PLAN['scope']+'; routing uses model outputs only, not labels.',**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    print(json.dumps({k:v for k,v in summary.items() if k!='source_prediction_sha256'},indent=2))


if __name__=='__main__':main()
