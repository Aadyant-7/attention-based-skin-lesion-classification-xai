"""S70/S71: fixed same-lesion multi-image inference from saved validation scores.

This changes the input contract, not CNN parameters or ensemble weights. It is
never single-image test evidence. No images, checkpoints or test labels load.
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from research.common import ROOT, CLASSES, relative, sha256, write_csv, write_json
from research.plots import metric_figures, comparison_figures
from research.registry import upsert

OUT = ROOT / 'results/short_screening/s70_s71_same_lesion_multiimage'
P_COLS = [f'p_{c}' for c in CLASSES]
SOURCES = {
    's70': dict(protocol='exploratory_image_level',
                reference='results/short_screening/s53_equal_five_b0_addition',
                split='data/splits/exploratory/image_level_dev_v1.csv',
                split_sha256='75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb',
                display='S70 S53 same-lesion multi-image'),
    's71': dict(protocol='strict_lesion_disjoint',
                reference='results/final_strict/v1/ensemble',
                split='data/splits/split_assignments.csv',
                split_sha256='db1ce9f8b83f176001dd89fd332fb1668c7d2ba5f58d77157d293a09e2c2e696',
                display='S71 S31 same-lesion multi-image'),
}
PLAN = dict(
    question='Can distinct available dermoscopic images of the same lesion add information beyond single-image CNN/model fusion?',
    rule='Equal mean of existing ensemble probability vectors within each validation-only lesion_id; singleton unchanged',
    input_contract='Known lesion grouping and all available distinct images of that lesion in the validation partition; group IDs only, never labels',
    singleton_fallback='Original image probability vector',
    cross_partition_pooling=False,
    sources=SOURCES,
    comparisons='Paired image-weighted original versus broadcast pooled predictions on identical 1503 images within each protocol; additionally one canonical lexicographic-ID view versus bag prediction per lesion',
    gate=dict(accuracy_gain_absolute=.005,minimum_net_correct=8,
              macro_f1_non_decreasing=True,melanoma_recall_non_decreasing=True),
    advancement='Only a conditional multi-image development candidate if both protocols pass; never replace frozen single-image test or call this a single-image improvement',
    search='One predefined arithmetic mean; no confidence weighting, max/product alternatives, tuned subset or cross-partition companions',
    uncertainty='10000 paired lesion-cluster bootstrap replicates, seed42; descriptive after repeated validation selection, not independent significance evidence',
    literature=dict(url='https://doi.org/10.1111/exd.13777',
                    scope='Yap et al. studied clinical+dermoscopic modalities and metadata, not our repeated-dermoscopy arithmetic mean; conceptual motivation only, not protocol replication'),
    test_loaded=False,gpu_used=False,training_epochs=0,
)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'PREDECLARED_PLAN.json'
    if path.exists():
        assert json.loads(path.read_text()) == PLAN, 'Existing plan differs; refuse mutation'
    else:
        write_json(path, PLAN)


def pool_views(groups, probabilities):
    """Label-free and permutation-invariant; returns image and lesion scores."""
    groups = np.asarray(groups)
    assert groups.ndim == 1 and len(groups) == len(probabilities)
    p = np.asarray(probabilities, dtype=np.float64)
    assert p.shape == (len(groups), 7) and np.isfinite(p).all()
    assert (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-5)
    names, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    totals = np.zeros((len(names), 7), dtype=np.float64)
    np.add.at(totals, inverse, p)
    bag = totals/counts[:, None]
    return bag[inverse], names, inverse, counts, bag


def self_check():
    p = np.eye(7)[[0, 1, 2]].astype(float)
    pooled, names, inverse, counts, bag = pool_views(['b', 'a', 'b'], p)
    np.testing.assert_array_equal(counts, [1, 2])
    np.testing.assert_allclose(pooled, [.5*p[0]+.5*p[2], p[1], .5*p[0]+.5*p[2]])
    q = np.array([2, 0, 1])
    permuted = pool_views(np.array(['b', 'a', 'b'])[q], p[q])[0]
    np.testing.assert_allclose(permuted, pooled[q])


def report(y, p):
    predicted = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=range(7), zero_division=0)
    return dict(loss=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
                accuracy=float((predicted == y).mean()),
                macro_precision=float(precision.mean()),macro_recall=float(recall.mean()),
                macro_f1=float(f1.mean()),class_order=list(CLASSES),
                confusion_matrix=confusion_matrix(y,predicted,labels=range(7)).tolist(),
                per_class={c:dict(precision=float(precision[i]),recall=float(recall[i]),
                                  f1=float(f1[i]),support=int(support[i])) for i,c in enumerate(CLASSES)})


def save_endpoint(frame, y, p, path, title):
    m = report(y, p)
    write_json(path/'validation_metrics.json', m)
    saved = frame[['image_id','lesion_id','available_view_count']].copy()
    saved['true_class'] = [CLASSES[i] for i in y]
    saved['predicted_class'] = [CLASSES[i] for i in p.argmax(1)]
    saved[P_COLS] = p
    write_csv(path/'validation_predictions.csv', saved.to_dict('records'))
    write_csv(path/'validation_probabilities.csv', saved[['image_id','lesion_id']+P_COLS].to_dict('records'))
    metric_figures(m,path/'figures',title)
    reread = pd.read_csv(path/'validation_predictions.csv')
    assert reread.image_id.tolist() == frame.image_id.tolist()
    check = report(reread.true_class.map(dict(zip(CLASSES,range(7)))).to_numpy(),
                   reread[P_COLS].to_numpy())
    assert check['confusion_matrix'] == m['confusion_matrix']
    assert abs(check['macro_f1']-m['macro_f1']) < 1e-12
    return m


def cluster_interval(old, new, inverse, counts):
    delta = np.bincount(inverse,weights=new.astype(int)-old.astype(int),minlength=len(counts))
    rng = np.random.default_rng(42)
    replicates = []
    for _ in range(100):
        indices = rng.integers(0,len(counts),size=(100,len(counts)))
        replicates.extend((delta[indices].sum(1)/counts[indices].sum(1)).tolist())
    return dict(replicates=10000,seed=42,unit='lesion-cluster',
                accuracy_difference_95_percentile=np.quantile(replicates,[.025,.975]).tolist(),
                descriptive_only=True)


def run_one(key, c):
    target = OUT/key
    if (target/'summary.json').exists():
        return json.loads((target/'summary.json').read_text())
    prediction_path = ROOT/c['reference']/'validation_predictions.csv'
    metrics_path = ROOT/c['reference']/'validation_metrics.json'
    source_hashes = {relative(p):sha256(p) for p in [prediction_path,metrics_path,ROOT/c['split']]}
    assert sha256(ROOT/c['split']) == c['split_sha256']
    # Identity-only rows can include test; neither clinical fields nor test scores load.
    identity = pd.read_csv(ROOT/c['split'], usecols=['image_id','lesion_id','split'])
    val = identity.loc[identity.split=='val'].copy()
    assert len(val)==1503 and val.image_id.is_unique
    locked = pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','lesion_id','split'])
    test = locked.loc[locked.split=='test']
    assert not set(val.image_id)&set(test.image_id)
    assert not set(val.lesion_id)&set(test.lesion_id)
    ref = pd.read_csv(prediction_path, usecols=['image_id']+P_COLS)
    assert ref.image_id.is_unique and set(ref.image_id)==set(val.image_id)
    frame = val.set_index('image_id').loc[ref.image_id].reset_index()
    p = ref[P_COLS].to_numpy()
    # No label enters the pooling function. Compute before loading scoring labels.
    pooled, names, inverse, counts, bag = pool_views(frame.lesion_id.to_numpy(),p)
    frame['available_view_count'] = counts[inverse]
    labels = pd.read_csv(prediction_path,usecols=['image_id','true_class']).set_index('image_id').loc[ref.image_id]
    y = labels.true_class.map(dict(zip(CLASSES,range(7)))).to_numpy()
    assert np.isfinite(y).all()
    y = y.astype(int)
    audit = frame.assign(label=y).groupby('lesion_id').label.nunique()
    assert (audit==1).all(), 'Conflicting diagnosis inside known lesion group; fail without dropping cases'
    assert np.allclose(pooled[counts[inverse]==1],p[counts[inverse]==1],atol=1e-14)
    # A single-view lesion comparator chosen by ID, never correctness/confidence.
    canonical = frame.sort_values('image_id').drop_duplicates('lesion_id').index.to_numpy()
    canonical = np.array(sorted(canonical,key=lambda i:frame.iloc[i].lesion_id))
    assert frame.iloc[canonical].lesion_id.tolist()==names.tolist()
    base = report(y,p)
    saved_base = json.loads(metrics_path.read_text())
    assert base['confusion_matrix']==saved_base['confusion_matrix']
    assert abs(base['accuracy']-saved_base['accuracy'])<1e-12
    image = save_endpoint(frame,y,pooled,target/'image_weighted',c['display']+' | image-weighted validation')
    lesion_frame = frame.iloc[canonical].reset_index(drop=True)
    single = save_endpoint(lesion_frame,y[canonical],p[canonical],target/'lesion_canonical_single',c['display']+' | one fixed view per lesion')
    lesion = save_endpoint(lesion_frame,y[canonical],bag,target/'lesion_multiimage',c['display']+' | one score per lesion')
    old = p.argmax(1)==y
    new = pooled.argmax(1)==y
    gained,lost = int((~old&new).sum()),int((old&~new).sum())
    changed = p.argmax(1)!=pooled.argmax(1)
    assert not changed[counts[inverse]==1].any()
    changes = frame.assign(true_class=[CLASSES[i] for i in y],
                           old_class=[CLASSES[i] for i in p.argmax(1)],
                           new_class=[CLASSES[i] for i in pooled.argmax(1)],
                           old_correct=old,new_correct=new)
    write_csv(target/'prediction_changes.csv',changes.loc[changed].to_dict('records'),
              ['image_id','lesion_id','split','available_view_count','true_class','old_class','new_class','old_correct','new_correct'])
    rows = [dict(class_name=cl,reference_recall=base['per_class'][cl]['recall'],
                 candidate_recall=image['per_class'][cl]['recall'],
                 reference_f1=base['per_class'][cl]['f1'],candidate_f1=image['per_class'][cl]['f1'],
                 net_correct=int(new[y==i].sum()-old[y==i].sum())) for i,cl in enumerate(CLASSES)]
    write_csv(target/'class_comparison.csv',rows)
    comparison_figures([
        dict(display_name='Original single-image reference',accuracy=base['accuracy'],macro_f1=base['macro_f1']),
        dict(display_name='Same-lesion multi-image (changed input)',accuracy=image['accuracy'],macro_f1=image['macro_f1'])
    ],target/'comparison_figures',c['display']+' | same cohort, different available inputs')
    comparison_figures([
        dict(display_name='One fixed view per lesion',accuracy=single['accuracy'],macro_f1=single['macro_f1']),
        dict(display_name='All available validation views per lesion',accuracy=lesion['accuracy'],macro_f1=lesion['macro_f1'])
    ],target/'lesion_comparison_figures',c['display']+' | lesion-weighted endpoint')
    passed = (image['accuracy']>=base['accuracy']+.005-1e-12 and gained-lost>=8 and
              image['macro_f1']>=base['macro_f1']-1e-12 and
              image['per_class']['mel']['recall']>=base['per_class']['mel']['recall']-1e-12)
    summary = dict(status='completed',protocol=c['protocol'],reference=base,image_weighted_candidate=image,
                   canonical_single_lesion=single,multiimage_lesion=lesion,images=len(frame),lesions=len(names),
                   multiimage_lesions=int((counts>1).sum()),multiimage_images=int(counts[counts>1].sum()),
                   maximum_available_views=int(counts.max()),gained=gained,lost=lost,net_correct=gained-lost,
                   gate_passed=bool(passed),source_hashes=source_hashes,
                   bootstrap=cluster_interval(old,new,inverse,counts),input_contract=PLAN['input_contract'],
                   test_loaded=False,gpu_used=False,training_epochs=0)
    for source,digest in source_hashes.items():
        assert sha256(ROOT/source)==digest, 'An existing source changed during the study'
    write_json(target/'source_manifest.json',dict(**source_hashes,class_order=list(CLASSES),script_sha256=sha256(Path(__file__))))
    write_json(target/'verification.json',dict(status='passed',source_metrics_recomputed=True,
                    unique_image_ids=True,label_free_pooling=True,singletons_unchanged=True,
                    labels_constant_within_lesions=True,cross_partition_pooling=False,
                    original_test_image_overlap=0,original_test_lesion_overlap=0,
                    saved_predictions_metrics_recomputed=True,no_image_or_checkpoint_loading=True,
                    test_labels_read=False,gpu_used=False))
    upsert(dict(experiment_id=f'{key}_same_lesion_multiimage_validation_seed42',era='structured',
                record_kind='cpu_multiimage_inference_screen',phase='post_test_multiimage_development',
                protocol=c['protocol']+'_multiimage_lesion_bags',evaluation_split='validation',
                split_manifest=c['split'],split_sha256=c['split_sha256'],model='preserved_reference_ensemble',
                method=PLAN['rule'],epochs=0,seed=42,status='completed',
                decision='conditional_gate_passed' if passed else 'gate_failed',
                metrics_path=relative(target/'image_weighted/validation_metrics.json'),
                plots_dir=relative(target/'image_weighted/figures'),
                notes='Changed input contract: requires known same-lesion companion images; no labels used for prediction. Not single-image improvement, independent validation or test evidence. Original checkpoints/weights untouched.',
                **{k:image[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    write_json(target/'summary.json',summary)
    logging.info('%s accuracy %.4f%% F1 %.6f gained %d lost %d gate %s',key,image['accuracy']*100,image['macro_f1'],gained,lost,passed)
    return summary


def main():
    prepare()
    if (OUT/'summary.json').exists():
        print((OUT/'summary.json').read_text()); return
    lock=OUT/'run.lock'
    handle=lock.open('x')
    try:
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                            handlers=[logging.FileHandler(OUT/'run.log',encoding='utf-8'),logging.StreamHandler()])
        self_check()
        start=time.perf_counter()
        logging.info('START fixed S70/S71; CPU saved-probability arithmetic only; no test/GPU')
        results={key:run_one(key,c) for key,c in SOURCES.items()}
        both=all(r['gate_passed'] for r in results.values())
        summary=dict(status='completed',results=results,both_protocol_gates_passed=both,
                     decision='Conditional multi-image candidate only' if both else 'No retained-method change; stop multi-image route at this budget',
                     runtime_seconds=time.perf_counter()-start,gpu_used=False,test_loaded=False)
        write_json(OUT/'summary.json',summary)
        logging.info('COMPLETE decision=%s runtime=%.1fs',summary['decision'],summary['runtime_seconds'])
        print(json.dumps(summary,indent=2))
    finally:
        handle.close();lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else main()
