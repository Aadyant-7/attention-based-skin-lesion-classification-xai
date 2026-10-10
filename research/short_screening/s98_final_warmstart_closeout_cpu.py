"""The one S97-predeclared replacement, then freeze unchanged S83. CPU only."""
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
from research.common import ROOT, CLASSES, relative, sha256, write_json, write_csv
from research.registry import upsert
from research.plots import comparison_figures
from research.final_cbam_development.protocol import verified_development
from research.final_cbam_development.runtime import require_launch_freeze
from research.final_cbam_development.artifacts import fp32_parts, COLS
from research.short_screening.lesion_bag_screen import report
from research.short_screening.s86_s87_saved_member_addition import aligned
from research.short_screening.s82_cbam_addition_cpu import package
from research.short_screening.s97_final_convnext_warmstart import config_signature, check_payload

OUT = ROOT/'results/short_screening/s98_final_warmstart_replacement_cpu'
REFERENCE = ROOT/'results/short_screening/s83_cbam_f1_addition_cpu'
FREEZE = ROOT/'results/final_exploratory_freeze/v1'
RID = 's98_s97_cbam_replacement_equal_six_exploratory_seed42'


def main():
    if (OUT/'summary.json').exists():
        print('Completed S98/final freeze preserved; no repeated scoring'); return
    started = time.perf_counter(); torch.set_num_threads(4)
    c, sig = config_signature(); _, val = verified_development(c); y = val.label.to_numpy()
    _, original_sig = require_launch_freeze()
    folder = ROOT/c['results']; ck = ROOT/c['checkpoints']
    summary = json.loads((folder/'training_summary.json').read_text())
    assert summary['status']=='completed' and summary['total_branch_epochs']==51
    p = torch.load(ck/'latest.pt', map_location='cpu', weights_only=False)
    check_payload(p, c, sig)
    assert p['epoch']==51 and p['best_macro_f1']['epoch']==33 and p['meaningful_stopping']['stop']
    del p; gc.collect()
    sources = json.loads((REFERENCE/'PREDECLARED_PLAN.json').read_text())['sources']
    assert len(sources)==6 and sources[-1]['run']==c['parent_run']
    candidate = dict(run=c['experiment_id'], selector='earliest standalone macro-F1 maximum', epoch=33,
                     inherited_winner=True, checkpoint=relative(ck/'best_macro_f1.pt'),
                     checkpoint_sha256=sha256(ck/'best_macro_f1.pt'),
                     prediction_file=relative(folder/'macro_f1_selected/validation_predictions.csv'),
                     prediction_sha256=sha256(folder/'macro_f1_selected/validation_predictions.csv'))
    plan = dict(experiment_id=RID, rule=c['fusion_rule'], maximum_fusions=1, weights=[1/6]*6,
                original_sources=sources, replacement_source=candidate, replaced_index=5,
                predeclared_training_plan='results/short_screening/final_convnext_warmstart_v1/PREDECLARED_PLAN.json',
                predeclared_training_plan_sha256=sha256(ROOT/'results/short_screening/final_convnext_warmstart_v1/PREDECLARED_PLAN.json'),
                reference_predictions_sha256=sha256(REFERENCE/'validation_predictions.csv'),
                probability_arithmetic='FP32 identity source probabilities; NumPy float32 arithmetic mean as S83',
                validation_size=len(val), split_sha256=c['split_sha256'], test_loaded=False, gpu_used=False,
                caveat='Inherited33 winner reproduces known S82 weights; not a newly trained improvement or independent evidence.')
    write_json(OUT/'PREDECLARED_PLAN.json', plan)
    parts = fp32_parts(original_sig, val)
    original_cbam = aligned(ROOT/sources[-1]['prediction_file'], val).astype(np.float32)
    baseline = np.mean(parts+[original_cbam], axis=0)
    np.testing.assert_allclose(baseline, aligned(REFERENCE/'validation_predictions.csv', val), atol=1e-7, rtol=0)
    reference_metrics = report(y, baseline)
    assert reference_metrics['accuracy']==.9401197604790419
    new_cbam = aligned(ROOT/candidate['prediction_file'], val).astype(np.float32)
    np.testing.assert_array_equal(new_cbam, aligned(ROOT/c['parent_predictions'], val).astype(np.float32))
    combined = np.mean(parts+[new_cbam], axis=0)
    metrics = package(OUT, val, combined, 'S98 fixed inherited CBAM replacement')
    previous82 = json.loads((ROOT/'results/short_screening/s82_cbam_addition_cpu/validation_metrics.json').read_text())
    assert metrics['confusion_matrix']==previous82['confusion_matrix']
    old = baseline.argmax(1); new = combined.argmax(1)
    gains = (old!=y)&(new==y); losses = (old==y)&(new!=y)
    result = dict(status='completed', **{k:metrics[k] for k in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']},
                  gained=int(gains.sum()), lost=int(losses.sum()), net_correct=int(gains.sum()-losses.sum()),
                  inherited_winner=True, reproduces_existing_s82=True, decision='retain_S83_no_improvement',
                  test_loaded=False, gpu_used=False, runtime_seconds=time.perf_counter()-started)
    assert metrics['accuracy'] < reference_metrics['accuracy']
    write_csv(OUT/'changed_predictions.csv', [dict(image_id=val.iloc[i].image_id, true_class=CLASSES[y[i]],
              reference_class=CLASSES[old[i]], candidate_class=CLASSES[new[i]], change='gained' if gains[i] else 'lost')
              for i in np.flatnonzero(gains|losses)], ['image_id','true_class','reference_class','candidate_class','change'])
    comparison_figures([dict(display_name='S83 retained equal-six', **{k:reference_metrics[k] for k in ['accuracy','macro_f1']}),
                        dict(display_name='S98 S97 inherited33 replacement', **{k:metrics[k] for k in ['accuracy','macro_f1']})],
                       OUT/'comparison_figures', 'Final fixed warm-start replacement | exploratory validation')
    for source in sources+[candidate]:
        assert sha256(ROOT/source['checkpoint'])==source['checkpoint_sha256']
        assert sha256(ROOT/source['prediction_file'])==source['prediction_sha256']
    write_json(OUT/'verification.json', dict(status='passed', reference_reproduced=True,
               sources_hash_verified=True, known_s82_confusion_reproduced=True, samples=1503,
               saved_metrics_recomputed=report(y, aligned(OUT/'validation_predictions.csv', val))['confusion_matrix']==metrics['confusion_matrix'],
               test_loaded=False, gpu_used=False))
    write_json(OUT/'summary.json', result)
    upsert(dict(experiment_id=RID, era='structured', record_kind='fixed_probability_fusion',
               phase='post_test_exploratory_final_training', protocol=c['protocol'], evaluation_split='validation',
               split_manifest=c['split_manifest'], split_sha256=c['split_sha256'], model='S83 with inherited S97 CBAM replacement',
               method=c['fusion_rule'], attention='CBAM in inherited winner33', image_size=224, seed=42, epochs=0,
               status='completed', decision=result['decision'], ensemble_members=json.dumps([s['run'] for s in sources[:5]]+[c['experiment_id']]),
               ensemble_weights=json.dumps([1/6]*6), config_path=relative(OUT/'PREDECLARED_PLAN.json'),
               metrics_path=relative(OUT/'validation_metrics.json'), plots_dir=relative(OUT/'figures'),
               confusion_matrix_path=relative(OUT/'figures/confusion_matrix.csv'),
               notes=plan['caveat']+' Repeated exploratory validation after earlier test outcomes; no new test inference.',
               **{k:metrics[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    # Freeze the existing winning method; do not borrow its score for new checkpoints.
    frozen_sources = []
    for source in sources:
        cp = torch.load(ROOT/source['checkpoint'], map_location='cpu', weights_only=False)
        cfg = cp['config']; epoch = cp.get('best_epoch', cp.get('epoch'))
        frozen_sources.append(dict(source, epoch=int(epoch), model=cfg['model'],
                                   pretrained_weights=cfg.get('weights', cfg.get('pretrained_weights')),
                                   weight=1/6, selection='macro_f1' if 'best_macro_f1' in source['checkpoint'] else 'accuracy'))
        del cp; gc.collect()
    freeze = dict(status='frozen_pending_test_audit_approval', source_experiment='s83_cbam_f1_addition_equal_six_exploratory_seed42',
                  protocol=c['protocol'], split_manifest=c['split_manifest'], split_sha256=c['split_sha256'],
                  validation_metrics=relative(REFERENCE/'validation_metrics.json'), validation_predictions=relative(REFERENCE/'validation_predictions.csv'),
                  validation_prediction_sha256=sha256(REFERENCE/'validation_predictions.csv'),
                  validation_accuracy=reference_metrics['accuracy'], validation_macro_f1=reference_metrics['macro_f1'],
                  sources=frozen_sources, class_order=list(CLASSES), image_size=224,
                  preprocessing='PIL RGB; square224 bilinear antialias; ImageNet normalization',
                  normalization_mean=c['normalization_mean'], normalization_std=c['normalization_std'],
                  inference_precision='FP32', views=['identity'], tf32=False, weights=[1/6]*6,
                  probability_fusion='Arithmetic mean of six FP32 softmax probability vectors; argmax in frozen class order',
                  fp32_validation_cache_provenance=original_sig['fp32_reference_caches'],
                  no_further_training_or_validation_search=True, test_loaded=False, test_evaluation_launched=False,
                  future_test_role='Repeated post-development audit of a previously evaluated cohort; not first independent test',
                  xai_role='Post-hoc explanations; no prediction correction or accuracy increase')
    write_json(FREEZE/'frozen_method.json', freeze)
    print(json.dumps(dict(**result, retained_accuracy=reference_metrics['accuracy'], retained_macro_f1=reference_metrics['macro_f1'], frozen_sources=frozen_sources), indent=2))


if __name__=='__main__':
    main()
