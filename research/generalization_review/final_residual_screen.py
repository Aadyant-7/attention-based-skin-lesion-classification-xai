"""Two fixed learned CPU corrections, lesion-group crossfit on saved S83 validation.

No old-test predictions/clinical fields, GPU, images or checkpoints load. This is
post-development meta-OOF validation, not independent CNN cross-validation.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from research.common import ROOT, CLASSES, relative, sha256, write_csv, write_json
from research.plots import save, validate_metrics
from research.short_screening.lesion_bag_screen import report
from research.generalization_review.clinical_residual import crossfit_clinical
from research.generalization_review.top2_residual import crossfit_top2

OUT=ROOT/'results/diagnostics/s83_final_residual_screen_2026-10-12'
SOURCE=ROOT/'results/short_screening/s83_cbam_f1_addition_cpu/validation_predictions.csv'
METRICS=SOURCE.with_name('validation_metrics.json')
FREEZE=ROOT/'results/final_exploratory_freeze/v1/frozen_method.json'
SPLIT=ROOT/'data/splits/exploratory/image_level_dev_v1.csv'
META=ROOT/'data/raw/HAM10000/HAM10000_metadata.csv'
COLS=['p_'+c for c in CLASSES]
PLAN=dict(
    question='Can a conservative learned correction recover complementary S83 errors without discarding its strong image predictions?',
    candidates=['clinical_residual','top2_reranker'],control='intercept-only S83 log-probability residual',
    clinical_rule='softmax(log(P_S83)+b+W*fold-encoded age/sex/localization)',
    clinical_features='TRAIN-fold standardized age; TRAIN-vocabulary one-hot sex/site; missing/unseen neutral; no diagnosis type or identifiers',
    reranker_rule='Binary residual over fixed log(P_runner/P_leader), only when margin<=0.5 and member argmax disagreement; keep pair mass and all noneligible outputs',
    reranker_features='leader/runner one-hot classes plus six member runner-minus-leader probabilities',
    penalty='Mean unweighted cross-entropy +0.01/2*squared correction parameters; identical fixed L2 for every fit',
    optimizer='Zero initialization; L-BFGS-B, analytic gradients, maxiter500, ftol1e-12, gtol1e-7',
    folds=5,fold_method='StratifiedGroupKFold by lesion_id, shuffle, seed42',
    primary='All 1,503 original validation images, one held-out-meta prediction per image; no exclusion or altered denominator',
    gate=dict(minimum_accuracy_gain=.005,minimum_net_correct=8,macro_f1_non_decreasing=True,
              melanoma_recall_non_decreasing=True,akiec_recall_non_decreasing=True,minimum_positive_folds=4,
              clinical_must_also_improve_over_intercept_control=True),
    selection='Only passing candidates; accuracy then macro-F1, simpler image-only reranker on exact ties',
    no_parameter_grid=True,no_post_result_rescue=True,no_training_in_sample_probabilities=True,
    test_action='No candidate old-test scoring in this screen. A passing fixed deployment head requires a separate transparent retrospective assessment.',
    limitation='Base CNNs/checkpoints and methodology were selected on this repeatedly used validation cohort; grouped crossfit protects only the new head fitting',
    prior='Previous unrestricted stacking, feature fusion, scalar calibration and naive-Bayes metadata adjustments were unhelpful; these anchored residuals are distinct unproven hypotheses',
    known_prior_test_outcomes=True,training_epochs=0,gpu_used=False,new_model_inference=False)


def prepare():
    method=json.loads(FREEZE.read_text(encoding='utf-8'))
    caches=[ROOT/a['path'] for a in method['fp32_validation_cache_provenance']]
    cbam=ROOT/method['sources'][-1]['prediction_file']
    sources=[SOURCE,METRICS,FREEZE,SPLIT,META,cbam]+caches
    codes=[Path(__file__),ROOT/'research/generalization_review/clinical_residual.py',ROOT/'research/generalization_review/top2_residual.py',ROOT/'research/short_screening/lesion_bag_screen.py']
    frozen=dict(plan=PLAN,source_sha256={relative(p):sha256(p) for p in sources},
                code_sha256={relative(p):sha256(p) for p in codes})
    if (OUT/'plan.json').exists():
        assert json.loads((OUT/'plan.json').read_text(encoding='utf-8'))==frozen
    else:
        write_json(OUT/'plan.json',frozen)
    return method,frozen


def load_development(method):
    frame=pd.read_csv(SOURCE)
    assert len(frame)==1503 and frame.image_id.is_unique
    identity=pd.read_csv(SPLIT,usecols=['image_id','lesion_id','split'])
    val=identity.loc[identity.split=='val'].set_index('image_id').loc[frame.image_id]
    assert len(val)==1503 and val.lesion_id.tolist()==frame.lesion_id.tolist()
    p=frame[COLS].to_numpy(dtype=np.float64)
    parts=[]
    for item in method['fp32_validation_cache_provenance']:
        path=ROOT/item['path']
        assert sha256(path)==item['sha256']
        with np.load(path,allow_pickle=False) as cached:
            assert cached['ids'].tolist()==frame.image_id.tolist()
            parts.append(cached['probabilities'].astype(np.float32))
    cbam=pd.read_csv(ROOT/method['sources'][-1]['prediction_file']).set_index('image_id').loc[frame.image_id]
    parts.append(cbam[COLS].to_numpy(dtype=np.float32))
    stack=np.stack(parts)
    assert stack.shape==(6,1503,7) and np.isfinite(stack).all()
    np.testing.assert_allclose(stack.mean(0),p,atol=2e-7,rtol=2e-7)
    assert np.array_equal(stack.mean(0).argmax(1),p.argmax(1))
    # Parse clinical values for the 1,503 allowed validation originals only.
    # All other rows, including old-test clinical rows, are skipped beforehand.
    meta_ids=pd.read_csv(META,usecols=['image_id'])
    skip=set((meta_ids.index[~meta_ids.image_id.isin(set(frame.image_id))]+1).tolist())
    metadata=pd.read_csv(META,usecols=['image_id','age','sex','localization'],skiprows=lambda line:line in skip)
    assert metadata.image_id.is_unique and set(metadata.image_id)==set(frame.image_id)
    metadata=metadata.set_index('image_id').loc[frame.image_id].reset_index(drop=True)
    age=pd.to_numeric(metadata.age,errors='coerce')
    assert age.dropna().between(0,120).all()
    y=frame.true_class.map(dict(zip(CLASSES,range(7)))).to_numpy(dtype=int)
    assert frame.groupby('lesion_id').true_class.nunique().eq(1).all()
    return frame,p,stack,metadata,y


def package(name,frame,y,p,fold_id):
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-6)
    m=report(y,p)
    validate_metrics(m)
    m.update(observations=len(y),unit='image',evaluation='post-development grouped meta-OOF validation' if name!='s83_reference' else 'original reused exploratory validation',
             correct=int((p.argmax(1)==y).sum()),incorrect=int((p.argmax(1)!=y).sum()),
             weighted_f1=sum(v['support']*v['f1'] for v in m['per_class'].values())/len(y))
    folder=OUT/name
    write_json(folder/'validation_metrics.json',m)
    saved=frame[['image_id','lesion_id','true_class']].copy()
    saved['predicted_class']=[CLASSES[i] for i in p.argmax(1)]
    saved['correct']=p.argmax(1)==y
    saved['meta_holdout_fold']=fold_id
    saved[COLS]=p
    write_csv(folder/'validation_predictions.csv',saved.to_dict('records'))
    write_csv(folder/'validation_probabilities.csv',saved[['image_id','lesion_id']+COLS].to_dict('records'))
    write_csv(folder/'per_class_metrics.csv',[dict(class_name=c,**m['per_class'][c]) for c in CLASSES])
    raw=np.asarray(m['confusion_matrix'])
    normalized=raw/raw.sum(1,keepdims=True)
    for matrix_name,a in [('confusion_matrix',raw),('confusion_matrix_normalized',normalized)]:
        write_csv(folder/(matrix_name+'.csv'),[dict(true_class=c,**dict(zip(CLASSES,a[i].tolist()))) for i,c in enumerate(CLASSES)])
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
    ax.set(xticks=range(7),yticks=range(7),xticklabels=CLASSES,yticklabels=CLASSES,
           xlabel='Predicted class',ylabel='True class',title=name.replace('_',' ')+' | development only')
    for i in range(7):
        for j in range(7):
            ax.text(j,i,f'{normalized[i,j]:.2f}',ha='center',va='center',fontsize=8,
                    color='white' if normalized[i,j]>.55 else 'black')
    fig.colorbar(im,ax=ax,fraction=.045)
    fig.tight_layout()
    fig.savefig(folder/'confusion_matrix_normalized.png',dpi=250,bbox_inches='tight')
    plt.close(fig)
    reread=pd.read_csv(folder/'validation_predictions.csv')
    check=report(y,reread[COLS].to_numpy())
    assert check['confusion_matrix']==m['confusion_matrix'] and abs(check['macro_f1']-m['macro_f1'])<1e-12
    return m


def main():
    if (OUT/'summary.json').exists():
        print('Completed finite residual screen preserved; no repeat, grid or test scoring.');return
    tick=time.perf_counter()
    method,frozen=prepare()  # Freeze hypotheses/settings before first model fit.
    frame,p,stack,metadata,y=load_development(method)
    groups=frame.lesion_id.to_numpy()
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(p,y,groups))
    fold_id=np.full(len(y),-1)
    for k,(train,hold) in enumerate(folds):
        assert not set(groups[train])&set(groups[hold]) and set(y[train])==set(range(7))
        assert (fold_id[hold]==-1).all()
        fold_id[hold]=k
    assert (fold_id>=0).all()
    write_csv(OUT/'fold_assignments.csv',[dict(image_id=i,lesion_id=g,meta_holdout_fold=int(f)) for i,g,f in zip(frame.image_id,groups,fold_id)])
    with threadpool_limits(limits=2):
        clinical=crossfit_clinical(p,metadata,y,folds)
        top2,top2_records=crossfit_top2(p,stack,y,folds)
    write_json(OUT/'clinical_fold_parameters.json',clinical['fold_records'])
    write_json(OUT/'top2_fold_parameters.json',top2_records)
    arrays=dict(s83_reference=p,intercept_control=clinical['control_oof'],
                clinical_residual=clinical['clinical_oof'],top2_reranker=top2)
    metrics={name:package(name,frame,y,probs,fold_id) for name,probs in arrays.items()}
    assert metrics['s83_reference']['confusion_matrix']==json.loads(METRICS.read_text(encoding='utf-8'))['confusion_matrix']
    old=p.argmax(1)==y
    rows=[];fold_rows=[];changed_rows=[]
    for name,probs in arrays.items():
        guessed=probs.argmax(1);new=guessed==y;m=metrics[name]
        gained,lost=int((~old&new).sum()),int((old&~new).sum())
        gains=[]
        for k,(_,hold) in enumerate(folds):
            delta=float(new[hold].mean()-old[hold].mean());gains.append(delta)
            fold_rows.append(dict(method=name,fold=k,observations=len(hold),reference_accuracy=float(old[hold].mean()),candidate_accuracy=float(new[hold].mean()),accuracy_gain=delta))
        checks=dict(accuracy_gain=m['accuracy']-metrics['s83_reference']['accuracy']>=.005-1e-12,
                    net_correct=gained-lost>=8,macro_f1=m['macro_f1']>=metrics['s83_reference']['macro_f1']-1e-12,
                    melanoma_recall=m['per_class']['mel']['recall']>=metrics['s83_reference']['per_class']['mel']['recall']-1e-12,
                    akiec_recall=m['per_class']['akiec']['recall']>=metrics['s83_reference']['per_class']['akiec']['recall']-1e-12,
                    positive_folds=sum(v>0 for v in gains)>=4)
        if name=='clinical_residual':
            checks['clinical_information_gain']=m['accuracy']>metrics['intercept_control']['accuracy']+1e-12
        passed=name in PLAN['candidates'] and all(checks.values())
        rows.append(dict(method=name,accuracy=m['accuracy'],macro_precision=m['macro_precision'],macro_recall=m['macro_recall'],macro_f1=m['macro_f1'],
                         melanoma_recall=m['per_class']['mel']['recall'],akiec_recall=m['per_class']['akiec']['recall'],gained=gained,lost=lost,net=gained-lost,
                         positive_folds=sum(v>0 for v in gains),gate_passed=bool(passed),checks=checks))
        for i in np.flatnonzero(guessed!=p.argmax(1)):
            changed_rows.append(dict(method=name,image_id=frame.iloc[i].image_id,lesion_id=groups[i],meta_holdout_fold=int(fold_id[i]),true_class=CLASSES[y[i]],original_class=CLASSES[p[i].argmax()],candidate_class=CLASSES[guessed[i]],original_correct=bool(old[i]),candidate_correct=bool(new[i])))
    write_csv(OUT/'comparison.csv',[{k:v for k,v in row.items() if k!='checks'} for row in rows])
    write_csv(OUT/'fold_comparison.csv',fold_rows)
    write_csv(OUT/'changed_predictions.csv',changed_rows,fields=['method','image_id','lesion_id','meta_holdout_fold','true_class','original_class','candidate_class','original_correct','candidate_correct'])
    fig,axes=plt.subplots(1,2,figsize=(11,5.2))
    labels=['S83 reference','Intercept control','Clinical residual','Top-two reranker']
    for ax,key,title in zip(axes,['accuracy','macro_f1'],['Image-level accuracy','Macro-F1']):
        bars=ax.bar(range(4),[r[key] for r in rows],color=['#4178a6','#777777','#439476','#c07c3e'])
        ax.bar_label(bars,fmt='%.4f',padding=4,fontsize=9)
        ax.set(xticks=range(4),xticklabels=labels,ylim=(0,1.07),title=title,ylabel='Score')
        ax.tick_params(axis='x',labelsize=8,rotation=15)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('S83 learned residual study | 1,503 images | grouped meta-OOF development',fontsize=12)
    fig.text(.5,.015,'Base CNNs were selected on this cohort; these are not independent test or CNN cross-validation scores.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.95])
    save(fig,OUT/'figures','residual_comparison')
    for group in ('source_sha256','code_sha256'):
        assert all(sha256(ROOT/path)==h for path,h in frozen[group].items())
    passing=[r for r in rows if r['gate_passed']]
    chosen=sorted(passing,key=lambda r:(-r['accuracy'],-r['macro_f1'],r['method']!='top2_reranker'))[0]['method'] if passing else None
    summary=dict(status='completed',comparisons=rows,passing_candidates=[r['method'] for r in passing],chosen_candidate=chosen,
                 decision='Conditional development evidence only; no test scoring or automatic GPU' if passing else 'Both substantive hypotheses rejected; retain frozen S83 and stop this finite study',
                 validation_images=len(y),validation_lesions=len(set(groups)),folds=5,all_head_train_holdout_lesion_overlap=0,
                 meta_fitting_reuses_validation=True,base_checkpoint_selection_uses_same_validation=True,
                 clinical_rows_parsed=1503,diagnosis_type_or_identifiers_as_predictors=False,
                 candidate_old_test_scored=False,test_clinical_rows_parsed=False,training_epochs=0,gpu_used=False,new_model_inference=False,
                 original_sources_unchanged=True,elapsed_seconds=time.perf_counter()-tick)
    write_json(OUT/'summary.json',summary)
    print(pd.DataFrame([{k:v for k,v in row.items() if k!='checks'} for row in rows]).to_string(index=False))
    print(summary['decision'])


if __name__=='__main__':
    main()
