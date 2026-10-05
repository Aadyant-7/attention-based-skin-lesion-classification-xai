"""Verify S59 artifacts and evaluate two fixed uses of its accuracy winner."""
import gc,json
import numpy as np
import pandas as pd
import torch,timm
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from research.common import ROOT,CLASSES,relative,sha256,write_json,write_csv
from research.strict_train import metric_report,tensor_nonfinite_names
from research.short_screening.supervised22k_transfer import OUT as RUN,CK,CFG,MODEL,fingerprints
from research.short_screening.panderm_fusion_v1 import aligned
from research.aggressive.core import predictions,retry_registry_upsert
from research.plots import metric_figures,comparison_figures

OUT=ROOT/'results/short_screening/s60_s61_s59_fusion'
REF=ROOT/'results/short_screening/s53_equal_five_b0_addition'
FOUR=ROOT/'results/short_screening/s46_all_f1_checkpoint_fusion'
RID='s59_convnext_tiny_supervised22k_exploratory_seed42'


def verify():
    torch.set_num_threads(4)
    cfg=json.loads(CFG.read_text());summary=json.loads((RUN/'summary.json').read_text())
    history=pd.read_csv(RUN/'history.csv')
    assert summary['status']=='completed' and history.epoch.tolist()==list(range(1,21))
    assert summary['best_epoch']==int(history.loc[history.val_accuracy.idxmax(),'epoch'])==20
    assert summary['best_macro_f1_epoch']==int(history.loc[history.val_macro_f1.idxmax(),'epoch'])==18
    model=timm.create_model(MODEL,pretrained=False,num_classes=7,drop_rate=.2,drop_path_rate=0.)
    checks=[]
    for filename,suffix,epoch in [('best.pt','',20),('best_macro_f1.pt','_macro_f1',18),('latest.pt',None,20)]:
        path=CK/filename;checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['config']==cfg and checkpoint['fingerprints']==fingerprints()
        assert not tensor_nonfinite_names(checkpoint['model'])
        model.load_state_dict(checkpoint['model'],strict=True)
        if suffix is None:
            assert len(checkpoint['history'])==20 and checkpoint['history'][-1]['epoch']==20
            assert not tensor_nonfinite_names(checkpoint['optimizer'])
            assert checkpoint['best']['epoch']==20 and checkpoint['f1best']['epoch']==18
        else:
            assert checkpoint['epoch']==epoch
            m=json.loads((RUN/f'validation_metrics{suffix}.json').read_text())
            assert checkpoint['metrics']==m
            p=pd.read_csv(RUN/f'validation_predictions{suffix}.csv')
            assert p.image_id.is_unique and len(p)==1503
            stored=pd.DataFrame(checkpoint['predictions'])
            assert stored.image_id.tolist()==p.image_id.tolist()
            assert stored.true_class.tolist()==p.true_class.tolist() and stored.predicted_class.tolist()==p.predicted_class.tolist()
            np.testing.assert_allclose(stored[[f'p_{c}' for c in CLASSES]],p[[f'p_{c}' for c in CLASSES]],atol=1e-12,rtol=0)
            assert abs(accuracy_score(p.true_class,p.predicted_class)-m['accuracy'])<1e-12
            assert abs(f1_score(p.true_class,p.predicted_class,labels=CLASSES,average='macro')-m['macro_f1'])<1e-12
            assert np.array_equal(confusion_matrix(p.true_class,p.predicted_class,labels=CLASSES),m['confusion_matrix'])
            assert p.predicted_class.tolist()==[CLASSES[j] for j in p[[f'p_{c}' for c in CLASSES]].to_numpy().argmax(1)]
            required=['confusion_matrix.csv','confusion_matrix_normalized.csv','per_class_metrics.csv','confusion_matrix.png','confusion_matrix.pdf','per_class_metrics.png','per_class_metrics.pdf']
            for name in required:assert (RUN/f'figures{suffix}'/name).is_file()
        checks.append(dict(path=relative(path),sha256=sha256(path),epoch=epoch,state_dict_strictly_loaded=True,finite=True))
        del checkpoint;gc.collect()
    for name in ['accuracy_curves','loss_curves','macro_f1_curve']:
        for extension in ['png','pdf']:assert (RUN/'figures'/f'{name}.{extension}').is_file()
    assert not (RUN/'failure.json').exists()
    write_json(RUN/'closeout_verification.json',dict(status='passed',checkpoints=checks,history_epochs=20,
        best_accuracy=summary['best_accuracy'],best_macro_f1=summary['best_macro_f1'],numerical_failure=False,
        validation_precision=sorted(set(history.validation_precision)),skipped_training_updates=int(history.skipped_updates.sum()),
        final_train_accuracy=float(history.iloc[-1].train_accuracy),test_loaded=False))
    return summary


def main():
    summary=verify()
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    OUT.mkdir(parents=True,exist_ok=False)
    plan=dict(checkpoint='S59 raw accuracy winner epoch20 only; no alternative F1 checkpoint or epoch search',
        experiments=['S60: equal-six S53 + S59','S61: equal-five replace S06 Tiny in S53 with S59'],
        rationale='S59 fixes24of96S53 errors; same-family replacement avoids adding correlated Tiny votes',
        adoption_gate='Accuracy improves over S53, macro-F1 and melanoma recall nondecreasing',
        material_gate='At least +.005 absolute accuracy over S53 with macro-F1 and melanoma recall nondecreasing',
        budget='Exactly two fixed combinations; no weights, thresholds, epochs, GPU inference or training',
        scope='Post-test exploratory development; repeated validation selection, not independent test',test_loaded=False)
    write_json(OUT/'PREDECLARED_PLAN.json',plan)
    ref=pd.read_csv(REF/'validation_predictions.csv');assert ref.image_id.is_unique and len(ref)==1503
    ids=ref.image_id.to_numpy();labels=ref.true_class.tolist();y=np.array([CLASSES.index(c) for c in labels])
    split=ROOT/'data/splits/exploratory/image_level_dev_v1.csv'
    cohort=pd.read_csv(split,usecols=['image_id','split']);assert set(ids)==set(cohort.loc[cohort.split=='val','image_id'])
    locked=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','split'])
    assert not set(ids)&set(locked.loc[locked.split=='test','image_id'])
    sources=json.loads((FOUR/'candidate_manifest.json').read_text())['sources']
    arrays=[]
    for source in sources:
        assert sha256(ROOT/source['prediction_file'])==source['prediction_sha256']
        arrays.append(aligned(ROOT/source['prediction_file'],ids,labels))
    b0='s03_efficientnet_b0_none_exploratory_seed42'
    path=ROOT/'results/structured_experiments'/b0/'validation_predictions.csv';checkpoint=ROOT/'checkpoints/structured'/b0/'best.pt'
    arrays.append(aligned(path,ids,labels));sources.append(dict(run=b0,prediction_file=relative(path),prediction_sha256=sha256(path),checkpoint=relative(checkpoint),checkpoint_sha256=sha256(checkpoint)))
    baseline=np.mean(arrays,axis=0)
    np.testing.assert_allclose(baseline,aligned(REF/'validation_predictions.csv',ids,labels),atol=1e-12)
    candidate=aligned(RUN/'validation_predictions.csv',ids,labels)
    cs=dict(run=RID,prediction_file=relative(RUN/'validation_predictions.csv'),prediction_sha256=sha256(RUN/'validation_predictions.csv'),checkpoint=relative(CK/'best.pt'),checkpoint_sha256=sha256(CK/'best.pt'),epoch=20)
    old=baseline.argmax(1)==y;new=candidate.argmax(1)==y
    bm=json.loads((REF/'validation_metrics.json').read_text())
    rows=[dict(display_name='S53 equal five reference',accuracy=bm['accuracy'],macro_f1=bm['macro_f1'])]
    studies=[]
    for number,name,p,members in [
        (60,'equal_six_s59_addition',(baseline*5+candidate)/6,sources+[cs]),
        (61,'replace_s06_with_s59',np.mean(arrays[1:]+[candidate],axis=0),sources[1:]+[cs])]:
        assert sources[0]['run']=='s06_convnext_tiny_none_exploratory_seed42'
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        folder=OUT/name;m=metric_report(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()))
        pred=predictions(ids,y,p);write_json(folder/'validation_metrics.json',m);write_csv(folder/'validation_predictions.csv',pred)
        write_csv(folder/'validation_probabilities.csv',pd.DataFrame(pred)[['image_id']+[f'p_{c}' for c in CLASSES]].to_dict('records'))
        write_json(folder/'candidate_manifest.json',dict(sources=members,weights=[1/len(members)]*len(members)))
        metric_figures(m,folder/'figures',f'S{number} fixed S59 fusion | exploratory validation')
        saved=pd.read_csv(folder/'validation_predictions.csv')
        assert abs(accuracy_score(saved.true_class,saved.predicted_class)-m['accuracy'])<1e-12
        assert abs(f1_score(saved.true_class,saved.predicted_class,labels=CLASSES,average='macro')-m['macro_f1'])<1e-12
        assert np.array_equal(confusion_matrix(saved.true_class,saved.predicted_class,labels=CLASSES),m['confusion_matrix'])
        correct=p.argmax(1)==y
        balance=m['macro_f1']>=bm['macro_f1'] and m['per_class']['mel']['recall']>=bm['per_class']['mel']['recall']
        passed=m['accuracy']>bm['accuracy']+1e-12 and balance
        material=m['accuracy']-bm['accuracy']>=.005-1e-12 and balance
        s=dict(experiment=f'S{number}',accuracy=m['accuracy'],macro_f1=m['macro_f1'],mel_recall=m['per_class']['mel']['recall'],
            gained=int((~old&correct).sum()),lost=int((old&~correct).sum()),net=int(correct.sum()-old.sum()),
            directional_gate_passed=bool(passed),material_gate_passed=bool(material))
        write_json(folder/'summary.json',s);studies.append(s)
        write_csv(folder/'class_comparison.csv',[dict(class_name=c,reference_recall=bm['per_class'][c]['recall'],candidate_recall=m['per_class'][c]['recall'],reference_f1=bm['per_class'][c]['f1'],candidate_f1=m['per_class'][c]['f1']) for c in CLASSES])
        rows.append(dict(display_name=f'S{number} {name}',accuracy=m['accuracy'],macro_f1=m['macro_f1']))
        retry_registry_upsert(dict(experiment_id=f's{number}_{name}_exploratory_seed42',era='structured',record_kind='cpu_fusion_screen',phase='post_test_exploratory_development',protocol='exploratory_image_level',evaluation_split='validation',split_manifest=relative(split),split_sha256=sha256(split),method=name+'; equal identity probability fusion; S59 accuracy winner epoch20',ensemble_members=json.dumps([s['run'] for s in members]),ensemble_weights=json.dumps([1/len(members)]*len(members)),epochs=0,seed=42,status='completed',decision='material_gate_passed' if material else 'material_gate_failed',metrics_path=relative(folder/'validation_metrics.json'),plots_dir=relative(folder/'figures'),notes=plan['scope'],**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    for source in sources+[cs]:assert sha256(ROOT/source['prediction_file'])==source['prediction_sha256']
    comparison_figures(rows,OUT/'comparison_figures','Two bounded S59 ensemble uses')
    result=dict(status='completed',s59_best_accuracy=summary['best_accuracy'],reference_accuracy=bm['accuracy'],reference_macro_f1=bm['macro_f1'],
        s59_fixes_reference_errors=int((~old&new).sum()),s59_errors_on_reference_correct=int((old&~new).sum()),studies=studies,test_loaded=False,gpu_used=False,
        decision='Two predeclared comparisons complete; no epoch extension or further searches')
    write_json(OUT/'summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
