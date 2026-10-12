"""One retrospective label-free lesion mean of saved S83 test probabilities.

Changed input contract/evaluation unit; never substitutes for frozen single-image
S83. No image decoding, checkpoint loading, GPU, training or new test inference.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from research.common import ROOT, CLASSES, sha256, relative, write_json, write_csv
from research.plots import save
from research.short_screening.lesion_bag_screen import pool_views, report

OUT = ROOT/'results/diagnostics/s83_multiimage_lesion_review_2026-10-12'
SOURCE = ROOT/'results/final_exploratory_test_audit/v1/ensemble/test_predictions.csv'
METRICS = ROOT/'results/final_exploratory_test_audit/v1/ensemble/test_metrics.json'
COLS = ['p_'+c for c in CLASSES]
PLAN = dict(
    question='Can multiple already-held-out photographs of a lesion add useful information to fixed S83 predictions?',
    rule='Equal probability mean over every available image of each lesion within the original held-out cohort; singleton unchanged',
    input_contract='Known lesion grouping and all available distinct held-out images per lesion, not single-image input',
    grouping_uses='lesion_id only; never labels, correctness or confidence',
    primary='ALL1118 lesions, one decision per lesion; equal mean versus first lexicographic image-ID single-view control',
    secondary='ALL1503 images; broadcast pooled case prediction versus original saved S83 single-image prediction',
    reporting='Retrospective multi-image lesion-level assessment of saved S83 predictions; not a new untouched test',
    preserve='Original S83 single-image87.558217%/0.807031 result and all original files unchanged',
    gate='>=0.005 accuracy gain, nondecreasing macro-F1/melanoma/akiec recall for both matched endpoints; >=8 net correct gains for image endpoint',
    no_subsets=True, no_view_or_weight_search=True, no_other_pooling=True, no_cross_partition_companions=True,
    no_training=True, no_new_image_inference=True, previous_test_score_known=True,
    probability_arithmetic='CPU float64 on previously saved FP32 six-model probabilities; no new model logits',
    limitation='Lesion-level and image-level denominators/class mixes differ; a higher case score alone is not a model improvement')


def prepare():
    frozen = dict(plan=PLAN, source_sha256={relative(p):sha256(p) for p in [SOURCE, METRICS,
        ROOT/'results/final_exploratory_freeze/v1/frozen_method.json',
        ROOT/'results/final_exploratory_test_audit/v1/pre_inference_verification.json']},
        code_sha256={relative(p):sha256(p) for p in [Path(__file__),ROOT/'research/short_screening/lesion_bag_screen.py',ROOT/'research/plots.py']})
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text())==frozen
    else:
        write_json(path,frozen)
    return frozen


def figures(m, folder, title):
    x = np.arange(7)
    raw = np.array(m['confusion_matrix'])
    normalized = raw/raw.sum(1,keepdims=True)
    for name,a,fmt in [('confusion_matrix',raw,'d'),('confusion_matrix_normalized',normalized,'.2f')]:
        fig,ax=plt.subplots(figsize=(8,7))
        im=ax.imshow(a,cmap='Blues',vmin=0,vmax=1 if 'normalized' in name else None)
        ax.set(xticks=x,yticks=x,xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted class',ylabel='True class')
        for i in x:
            for j in x:
                ax.text(j,i,format(a[i,j],fmt),ha='center',va='center',color='white' if a[i,j]>.55*a.max() else 'black',fontsize=9)
        fig.colorbar(im,ax=ax,fraction=.045,pad=.04)
        ax.set_title(title,fontsize=11)
        fig.tight_layout()
        save(fig,folder,name)
        write_csv(folder/(name+'.csv'),[dict(true_class=c,**dict(zip(CLASSES,a[i].tolist()))) for i,c in enumerate(CLASSES)])
    fig,ax=plt.subplots(figsize=(9,5.8))
    for off,key in [(-1,'precision'),(0,'recall'),(1,'f1')]:
        ax.bar(x+off*.24,[m['per_class'][c][key] for c in CLASSES],.24,label=key)
    ax.set(xticks=x,xticklabels=CLASSES,ylim=(0,1.04),ylabel='Score')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False)
    fig.suptitle(title,fontsize=11,y=.97)
    fig.subplots_adjust(top=.79,bottom=.12,left=.09,right=.98)
    save(fig,folder,'per_class_metrics')
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.bar(CLASSES,[m['per_class'][c]['support'] for c in CLASSES])
    ax.set(ylabel='Support',title=title)
    fig.tight_layout()
    save(fig,folder,'class_support')


def endpoint(frame,y,p,folder,title,unit):
    m=report(y,p)
    m.update(correct=int((p.argmax(1)==y).sum()),incorrect=int((p.argmax(1)!=y).sum()),
             observations=len(y),unit=unit,role=PLAN['reporting'],weighted_f1=0.)
    m['weighted_f1']=sum(a['support']*a['f1'] for a in m['per_class'].values())/len(y)
    a=frame.copy()
    a['true_class']=[CLASSES[i] for i in y]
    a['predicted_class']=[CLASSES[i] for i in p.argmax(1)]
    a[COLS]=p
    write_json(folder/'metrics.json',m)
    write_csv(folder/'predictions.csv',a.to_dict('records'))
    write_csv(folder/'per_class_metrics.csv',[dict(class_name=c,**m['per_class'][c]) for c in CLASSES])
    figures(m,folder/'figures',title)
    return m


def changes(y,old,new):
    a,b=old.argmax(1)==y,new.argmax(1)==y
    gained=int((~a&b).sum()); lost=int((a&~b).sum())
    return dict(gained=gained,lost=lost,net=gained-lost)


def main():
    if (OUT/'summary.json').exists():
        print('Completed single-rule diagnostic preserved; no further search')
        return
    f=prepare()  # Freeze rule/units/control before deriving candidate probabilities.
    source=pd.read_csv(SOURCE)
    original=json.loads(METRICS.read_text())
    assert len(source)==1503 and source.image_id.is_unique
    p=source[COLS].to_numpy()
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    # Pool receives only group IDs and probabilities, no labels/correctness.
    broadcast,names,inverse,counts,bag=pool_views(source.lesion_id.to_numpy(),p)
    assert len(names)==1118
    # Canonical view is chosen by image ID alone, irrespective of outcome.
    ordered=source.reset_index(names='source_index').sort_values('image_id')
    first=ordered.drop_duplicates('lesion_id').set_index('lesion_id').loc[names]
    canonical_indices=first.source_index.to_numpy(dtype=int)
    canonical=p[canonical_indices]
    # Labels enter only consistency checks and scoring, after prediction aggregation.
    assert source.groupby('lesion_id').true_class.nunique().eq(1).all()
    y=source.true_class.map(dict(zip(CLASSES,range(7)))).to_numpy()
    lesion_y=y[canonical_indices]
    assert np.array_equal(lesion_y[inverse],y)
    control=report(y,p)
    assert np.isclose(control['accuracy'],original['accuracy'],atol=1e-12)
    assert np.isclose(control['macro_f1'],original['macro_f1'],atol=1e-12)
    lesion_frame=pd.DataFrame(dict(lesion_id=names,canonical_image_id=source.image_id.to_numpy()[canonical_indices],available_views=counts))
    image_frame=source[['image_id','lesion_id']].copy()
    image_frame['available_views']=counts[inverse]
    single=endpoint(lesion_frame,lesion_y,canonical,OUT/'canonical_single_lesion',
        'S83 retrospective diagnostic | first-ID view | all1118 lesions','lesion')
    multi=endpoint(lesion_frame,lesion_y,bag,OUT/'multiimage_lesion',
        'S83 retrospective multi-image diagnostic | equal mean | all1118 lesions','lesion')
    image=endpoint(image_frame,y,broadcast,OUT/'broadcast_image_endpoint',
        'S83 retrospective multi-image diagnostic | all1503 original images','image')
    pairs=[('lesion',[('First-ID single view',single),('All available views',multi)]),
           ('image',[('Original single-image S83',control),('Broadcast multi-image',image)])]
    rows=[]
    for unit,pair in pairs:
        fig,axes=plt.subplots(1,2,figsize=(10,4.5))
        for ax,key in zip(axes,['accuracy','macro_f1']):
            values=[m[key] for _,m in pair]
            ax.bar(range(2),values,color=['#4178a6','#439476'])
            ax.set(xticks=range(2),xticklabels=[name for name,_ in pair],ylim=(0,1.07),title=key.replace('_',' ').title())
            ax.tick_params(axis='x',labelsize=9)
            for i,value in enumerate(values):ax.text(i,value+.015,f'{value:.4f}',ha='center')
        fig.suptitle(f'Retrospective S83 diagnostic | matched {unit} endpoint',fontsize=12)
        fig.tight_layout()
        save(fig,OUT/'comparison_figures',unit+'_comparison')
        rows.extend(dict(unit=unit,method=name,observations=len(lesion_y) if unit=='lesion' else len(y),
                         **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}) for name,m in pair)
    write_csv(OUT/'comparison.csv',rows)
    checks=[]
    for base,candidate in [(single,multi),(control,image)]:
        checks.append(candidate['accuracy']-base['accuracy']>=.005-1e-12 and candidate['macro_f1']>=base['macro_f1']-1e-12
            and all(candidate['per_class'][c]['recall']>=base['per_class'][c]['recall']-1e-12 for c in ['mel','akiec']))
    image_changes=changes(y,p,broadcast)
    result=dict(status='completed',role=PLAN['reporting'],original_single_image_accuracy=control['accuracy'],
        original_single_image_macro_f1=control['macro_f1'],original_images=1503,all_lesions=1118,
        canonical_single_lesion=single,multiimage_lesion=multi,broadcast_image_endpoint=image,
        lesion_changes=changes(lesion_y,canonical,bag),image_changes=image_changes,
        gate_passed=bool(all(checks) and image_changes['net']>=8),
        view_count_distribution={str(n):int((counts==n).sum()) for n in np.unique(counts)},
        singleton_lesions=int((counts==1).sum()),multiimage_lesions=int((counts>1).sum()),
        multiimage_images=int(counts[counts>1].sum()),maximum_views=int(counts.max()),
        training=False,gpu_used=False,new_model_inference=False,test_images_decoded=False,
        saved_test_labels_used_for_scoring=True,previous_test_outcomes_known=True,untouched_new_test=False,
        frozen_S83_changed=False,original_source_sha256=f['source_sha256'],
        metric_warning='All1118 lesion and all1503 image scores use different units/class prevalence; do not substitute the higher value for original single-image test accuracy.')
    assert all(sha256(ROOT/path)==digest for path,digest in f['source_sha256'].items())
    write_json(OUT/'summary.json',result)
    print(json.dumps({k:result[k] for k in ['original_single_image_accuracy','all_lesions','lesion_changes','image_changes','gate_passed','view_count_distribution']}))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    main()
