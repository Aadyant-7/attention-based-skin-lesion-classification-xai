"""One fixed validation-only follow-up to S83 arithmetic photo pooling.

No old-test files, images, checkpoints, GPU or training load. A failed gate ends
this candidate before any retrospective old-test assessment. The earlier S83
single-image and multi-image diagnostic remain unchanged.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.common import ROOT, CLASSES, relative, sha256, write_json, write_csv
from research.plots import save, validate_metrics
from research.short_screening.lesion_bag_screen import pool_views, report

OUT = ROOT/'results/diagnostics/s83_geometric_photo_screen_2026-10-12'
SOURCE = ROOT/'results/short_screening/s83_cbam_f1_addition_cpu/validation_predictions.csv'
METRICS = SOURCE.with_name('validation_metrics.json')
SPLIT = ROOT/'data/splits/exploratory/image_level_dev_v1.csv'
COLS = ['p_'+c for c in CLASSES]
PLAN = dict(
    question='Does agreement-sensitive geometric photo pooling materially outperform arithmetic photo pooling on existing S83 validation?',
    candidate='Normalize exp(mean(log(max(p, 1e-12)))) over all available validation photographs of each lesion; preserve singleton vectors exactly',
    epsilon=1e-12, arithmetic_control='Equal probability mean across the same available photographs',
    units='All validation lesions and all original validation images; broadcast lesion predictions for the image endpoint',
    grouping='lesion_id only; no labels, confidence, correctness, diagnosis or image appearance selects photographs',
    gate=dict(minimum_accuracy_gain=.005, minimum_net_image_gain=8,
              macro_f1_non_decreasing=True, melanoma_recall_non_decreasing=True,
              akiec_recall_non_decreasing=True, required_on_both_units=True),
    prior='Geometric fusion across models was previously unhelpful; across photographs is a distinct unproven ablation',
    origin='User authorized one further check after the earlier retrospective arithmetic diagnostic; all previous outcomes remain known',
    assessment='Exploratory validation development only, with existing lesion overlap; not independent confirmation',
    fail_action='Stop candidate; do not open old-test prediction files or select another pooling alternative',
    pass_action='Only permits a separately labelled retrospective old-test diagnostic; does not freeze or upgrade S83',
    no_grids=True, no_subsets=True, no_cross_partition_companions=True,
    no_training=True, no_new_model_inference=True, old_test_candidate_scored=False)


def geometric_pool(groups, probabilities):
    """Grouping and arithmetic use no ground-truth information."""
    p = np.asarray(probabilities, dtype=np.float64)
    _, names, inverse, counts, arithmetic = pool_views(groups, p)
    totals = np.zeros_like(arithmetic)
    np.add.at(totals, inverse, np.log(np.maximum(p, PLAN['epsilon'])))
    log_mean = totals/counts[:, None]
    geometric = np.exp(log_mean-log_mean.max(1, keepdims=True))
    geometric /= geometric.sum(1, keepdims=True)
    single_indices = np.flatnonzero(counts[inverse] == 1)
    geometric[inverse[single_indices]] = p[single_indices]
    return names, inverse, counts, arithmetic, geometric


def scored(y, p, unit):
    m = report(y, p)
    m.update(unit=unit, observations=len(y), correct=int((p.argmax(1)==y).sum()),
             incorrect=int((p.argmax(1)!=y).sum()),
             weighted_f1=sum(v['support']*v['f1'] for v in m['per_class'].values())/len(y),
             evaluation='Exploratory validation development only; repeatedly reused image-level split')
    validate_metrics(m)
    return m


def changed(y, original, candidate):
    old, new = original.argmax(1)==y, candidate.argmax(1)==y
    gained, lost = int((~old & new).sum()), int((old & ~new).sum())
    return dict(gained=gained, lost=lost, net=gained-lost)


def gate(base, candidate):
    return dict(accuracy=candidate['accuracy']-base['accuracy']>=.005-1e-12,
                macro_f1=candidate['macro_f1']>=base['macro_f1']-1e-12,
                melanoma_recall=candidate['per_class']['mel']['recall']>=base['per_class']['mel']['recall']-1e-12,
                akiec_recall=candidate['per_class']['akiec']['recall']>=base['per_class']['akiec']['recall']-1e-12)


def save_endpoint(frame, y, p, metrics, folder):
    frame = frame.copy()
    frame['true_class'] = [CLASSES[v] for v in y]
    frame['predicted_class'] = [CLASSES[v] for v in p.argmax(1)]
    frame['correct'] = p.argmax(1)==y
    frame[COLS] = p
    write_json(folder/'metrics.json', metrics)
    write_csv(folder/'predictions.csv', frame.to_dict('records'))
    write_csv(folder/'per_class_metrics.csv', [dict(class_name=c, **metrics['per_class'][c]) for c in CLASSES])
    raw = np.asarray(metrics['confusion_matrix'])
    normalized = raw/raw.sum(1, keepdims=True)
    for name, matrix in [('confusion_matrix', raw), ('confusion_matrix_normalized', normalized)]:
        write_csv(folder/(name+'.csv'), [dict(true_class=c, **dict(zip(CLASSES, matrix[i].tolist()))) for i,c in enumerate(CLASSES)])
    # Recompute from the actual saved predictions, including the unit/supports.
    saved = pd.read_csv(folder/'predictions.csv')
    check = report(saved.true_class.map(dict(zip(CLASSES, range(7)))).to_numpy(), saved[COLS].to_numpy())
    assert check['confusion_matrix'] == metrics['confusion_matrix']
    assert abs(check['macro_f1']-metrics['macro_f1']) < 1e-12


def main():
    if (OUT/'summary.json').exists():
        print('Completed fixed development screen preserved; no repeat or old-test scoring.')
        return
    frozen = dict(plan=PLAN, source_sha256={relative(p):sha256(p) for p in (SOURCE,METRICS,SPLIT)},
                  code_sha256={relative(p):sha256(p) for p in (Path(__file__), ROOT/'research/short_screening/lesion_bag_screen.py', ROOT/'research/plots.py')})
    if (OUT/'plan.json').exists():
        assert json.loads((OUT/'plan.json').read_text(encoding='utf-8')) == frozen
    else:
        write_json(OUT/'plan.json', frozen)
    frame = pd.read_csv(SOURCE)
    assert len(frame)==1503 and frame.image_id.is_unique
    identities = pd.read_csv(SPLIT, usecols=['image_id','lesion_id','split'])
    val = identities.loc[identities.split=='val'].set_index('image_id').loc[frame.image_id]
    assert len(val)==1503 and val.lesion_id.tolist()==frame.lesion_id.tolist()
    p = frame[COLS].to_numpy(dtype=np.float64)
    names,inverse,counts,arithmetic,geometric = geometric_pool(frame.lesion_id.to_numpy(), p)
    first = frame.reset_index(names='source_index').sort_values('image_id').drop_duplicates('lesion_id').set_index('lesion_id').loc[names]
    indices = first.source_index.to_numpy(dtype=int)
    # Ground truth enters only after the fixed candidate probabilities exist.
    assert frame.groupby('lesion_id').true_class.nunique().eq(1).all()
    y = frame.true_class.map(dict(zip(CLASSES, range(7)))).to_numpy()
    case_y = y[indices]
    assert np.array_equal(case_y[inverse], y)
    singleton = counts[inverse] == 1
    assert np.array_equal(geometric[inverse][singleton], p[singleton])
    original = scored(y, p, 'image')
    saved_original = json.loads(METRICS.read_text(encoding='utf-8'))
    assert original['confusion_matrix'] == saved_original['confusion_matrix']
    case_base,case_new = scored(case_y, arithmetic, 'lesion'), scored(case_y, geometric, 'lesion')
    image_base,image_new = scored(y, arithmetic[inverse], 'image'), scored(y, geometric[inverse], 'image')
    canonical = scored(case_y, p[indices], 'lesion')
    case_changes = changed(case_y, arithmetic, geometric)
    image_changes = changed(y, arithmetic[inverse], geometric[inverse])
    checks = dict(lesion=gate(case_base,case_new), image=gate(image_base,image_new),
                  minimum_net_image_gain=image_changes['net']>=8)
    passed = all(checks['lesion'].values()) and all(checks['image'].values()) and checks['minimum_net_image_gain']
    case_frame = pd.DataFrame(dict(lesion_id=names, available_views=counts, canonical_image_id=frame.image_id.to_numpy()[indices]))
    image_frame = frame[['image_id','lesion_id']].copy()
    image_frame['available_views'] = counts[inverse]
    for unit, data, target, base, candidate, base_p, candidate_p in [
        ('lesion',case_frame,case_y,case_base,case_new,arithmetic,geometric),
        ('image',image_frame,y,image_base,image_new,arithmetic[inverse],geometric[inverse])]:
        save_endpoint(data,target,base_p,base,OUT/unit/'arithmetic')
        save_endpoint(data,target,candidate_p,candidate,OUT/unit/'geometric')
    rows=[]
    fig, axes = plt.subplots(1,2,figsize=(11,5.3))
    for ax, unit, base, candidate in [(axes[0],'1,429 validation lesions',case_base,case_new),
                                     (axes[1],'1,503 validation images',image_base,image_new)]:
        x=np.arange(2)
        for offset,key,label in [(-.18,'accuracy','Accuracy'),(.18,'macro_f1','Macro-F1')]:
            values=[base[key],candidate[key]]
            bars=ax.bar(x+offset,values,width=.34,label=label)
            ax.bar_label(bars,fmt='%.4f',padding=4,fontsize=9)
        ax.set(xticks=x,xticklabels=['Arithmetic mean','Geometric mean'],ylim=(0,1.07),title=unit,ylabel='Score')
        ax.legend(loc='lower left')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('S83 fixed photo-pooling screen | exploratory validation only',fontsize=12)
    fig.text(.5,.015,'Same observations and photographs; no test scoring, training or weight search.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95])
    save(fig,OUT/'figures','validation_comparison')
    for unit,pair in [('lesion',[('First-ID single view',canonical),('Arithmetic mean',case_base),('Geometric mean',case_new)]),
                      ('image',[('Original single image',original),('Arithmetic mean',image_base),('Geometric mean',image_new)])]:
        for method,m in pair:
            rows.append(dict(unit=unit,method=method,observations=m['observations'],**{k:m[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1')}))
    write_csv(OUT/'comparison.csv', rows)
    for group in ('source_sha256','code_sha256'):
        assert all(sha256(ROOT/path)==digest for path,digest in frozen[group].items())
    result=dict(status='completed',gate_passed=bool(passed),checks=checks,
                decision='Eligible only for separate retrospective diagnostic' if passed else 'Rejected on development; stop before old-test scoring',
                original_image=original,canonical_lesion=canonical,
                arithmetic_lesion=case_base,geometric_lesion=case_new,
                arithmetic_image=image_base,geometric_image=image_new,
                lesion_changes=case_changes,image_changes=image_changes,
                validation_images=len(y),validation_lesions=len(names),multiimage_lesions=int((counts>1).sum()),
                multiimage_images=int(counts[counts>1].sum()),
                view_counts={str(n):int((counts==n).sum()) for n in np.unique(counts)},
                labels_used_for_scoring=True,labels_used_for_pooling=False,
                old_test_candidate_scored=False,old_test_files_loaded=False,
                training=False,gpu_used=False,new_model_inference=False,frozen_S83_changed=False,
                original_sources_unchanged=True)
    write_json(OUT/'summary.json',result)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps({k:result[k] for k in ('gate_passed','decision','lesion_changes','image_changes','checks')},indent=2))


if __name__=='__main__':
    main()
