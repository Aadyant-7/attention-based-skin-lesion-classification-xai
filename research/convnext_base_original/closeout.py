"""Verify completed R203 artifacts and two fixed saved-probability fusions.

CPU only: no image decoding, model forward, training, test scoring or weight
search. Existing checkpoints, histories and source prediction files are read-only.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from research.common import ROOT, CLASSES, relative, sha256, write_csv, write_json
from research.plots import metric_figures, comparison_figures, validate_metrics
from research.final_cbam_development.runtime import require_launch_freeze
from research.final_cbam_development.artifacts import fp32_parts
from .core import config_signature, FREEZE, report
from .train import update_registries

COLS = ['p_' + c for c in CLASSES]
IDENTITY = ['image_id', 'lesion_id', 'true_class']
STUDY = ROOT / 'results/structured_experiments/convnext_base_original_v1'
REFERENCE = ROOT / 'results/short_screening/s83_cbam_f1_addition_cpu'
OUT = STUDY / 'ensemble_comparison'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def probabilities(path, val):
    frame = pd.read_csv(path)
    expected = val[['image_id', 'lesion_id', 'diagnosis']].rename(columns={'diagnosis': 'true_class'})
    if not frame[IDENTITY].equals(expected.reset_index(drop=True)) or frame.image_id.duplicated().any():
        raise ValueError('Saved prediction cohort, labels or order mismatch: ' + str(path))
    p = frame[COLS].to_numpy(dtype=np.float32)
    if p.shape != (1503, 7) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probability array: ' + str(path))
    if frame.predicted_class.tolist() != [CLASSES[k] for k in p.argmax(1)]:
        raise ValueError('Stored predictions differ from argmax')
    return frame, p


def metrics(y, p):
    loss = -np.log(np.maximum(p[np.arange(len(y)), y].astype(np.float64), 1e-12)).mean()
    return report(y, p, float(loss))


def verify_packages(folder, val):
    checked = []
    y = val.label.to_numpy(dtype=int)
    for name in ('', 'best_accuracy', 'best_macro_f1', 'raw_accuracy', 'raw_macro_f1',
                 'ema_accuracy', 'ema_macro_f1', 'final_raw', 'final_ema'):
        package = folder / name
        stored = read_json(package / 'validation_metrics.json')
        validate_metrics(stored)
        frame, p = probabilities(package / 'predictions.csv', val)
        recomputed = metrics(y, p)
        for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1'):
            if not np.isclose(stored[key], recomputed[key], atol=1e-12):
                raise ValueError('Saved metric mismatch: ' + name + '/' + key)
        if stored['confusion_matrix'] != recomputed['confusion_matrix'] or stored['per_class'] != recomputed['per_class']:
            raise ValueError('Class scores or confusion matrix mismatch')
        if stored['protocol'] != 'exploratory_image_level' or stored['evaluation_split'] != 'validation' or stored['test_loaded'] is not False:
            raise ValueError('Prediction package protocol metadata mismatch')
        probability_frame = pd.read_csv(package / 'probabilities.csv')
        if not probability_frame[IDENTITY].equals(frame[IDENTITY]) or not np.allclose(probability_frame[COLS], p, rtol=1e-6, atol=1e-12):
            raise ValueError('Separate probability CSV mismatch')
        cm = np.asarray(stored['confusion_matrix'])
        for filename, expected in (('confusion_matrix.csv', cm),
                                   ('confusion_matrix_normalized.csv', cm / cm.sum(1, keepdims=True))):
            saved = pd.read_csv(package / filename)
            if saved.iloc[:, 0].tolist() != list(CLASSES) or not np.allclose(saved[list(CLASSES)], expected, atol=1e-12):
                raise ValueError('Confusion CSV mismatch')
        scores = pd.read_csv(package / 'per_class_metrics.csv')
        if scores.iloc[:, 0].tolist() != list(CLASSES):
            raise ValueError('Class CSV order mismatch')
        for i, class_name in enumerate(CLASSES):
            for key in ('precision', 'recall', 'f1', 'support'):
                if not np.isclose(scores.iloc[i][key], stored['per_class'][class_name][key], atol=1e-12):
                    raise ValueError('Class CSV score mismatch')
        checked.append(dict(package=name or 'root', epoch=stored['epoch'], branch=stored['branch'],
                            accuracy=stored['accuracy'], macro_f1=stored['macro_f1'],
                            prediction_sha256=sha256(package / 'predictions.csv')))
    pngs, pdfs = list(folder.rglob('*.png')), list(folder.rglob('*.pdf'))
    if len(pngs) != 40 or len(pdfs) != 40:
        raise ValueError('Expected complete raw/EMA metric, curve and comparison figure package')
    for path in pngs:
        with Image.open(path) as img:
            img.verify()
    for path in pdfs:
        data = path.read_bytes()
        if not data.startswith(b'%PDF-') or b'%%EOF' not in data[-1024:]:
            raise ValueError('Invalid PDF export')
    result = dict(status='passed', prediction_packages=checked, sample_count=1503,
                  recomputed_metrics=True, class_scores_and_confusion_csvs_verified=True,
                  png_images_decoded=len(pngs), pdf_headers_and_eof_checked=len(pdfs),
                  test_images_opened=0, gpu_used=False)
    write_json(folder / 'artifact_verification.json', result)
    return result


def package(folder, val, p, rid, members, title, c):
    y = val.label.to_numpy(dtype=int)
    m = metrics(y, p)
    m.update(experiment_id=rid, protocol=c['protocol'], evaluation_split='validation',
             class_order=list(CLASSES), inference_precision='fp32', views=['identity'],
             ensemble_members=[x['run'] for x in members], ensemble_weights=[1 / len(members)] * len(members),
             correct=int((p.argmax(1) == y).sum()), incorrect=int((p.argmax(1) != y).sum()),
             test_loaded=False, gpu_used=False, loss_definition='probability NLL, floor 1e-12 for reporting only')
    frame = val[['image_id', 'lesion_id', 'diagnosis']].rename(columns={'diagnosis': 'true_class'}).copy()
    frame['predicted_class'] = [CLASSES[k] for k in p.argmax(1)]
    frame[COLS] = p
    write_json(folder / 'validation_metrics.json', m)
    write_csv(folder / 'validation_predictions.csv', frame.to_dict('records'))
    write_csv(folder / 'validation_probabilities.csv', frame[IDENTITY + COLS].to_dict('records'))
    write_csv(folder / 'per_class_metrics.csv', [dict(class_name=k, **m['per_class'][k]) for k in CLASSES])
    metric_figures(m, folder / 'figures', title + ' | exploratory validation')
    for filename in ('confusion_matrix.csv', 'confusion_matrix_normalized.csv'):
        write_csv(folder / filename, pd.read_csv(folder / 'figures' / filename).to_dict('records'))
    return m


def main():
    tick = time.perf_counter()
    c, signature = config_signature()
    if read_json(FREEZE)['signature'] != signature:
        raise ValueError('Frozen R203 recipe changed')
    folder = ROOT / c['results']
    history = pd.read_csv(folder / 'history.csv')
    if history.epoch.tolist() != list(range(1, 31)) or read_json(folder / 'progress.json')['status'] != 'completed':
        raise ValueError('R203 must be completed before fusion')
    manifest = pd.read_csv(ROOT / c['split_manifest'])
    val = manifest.loc[manifest.split == 'val'].reset_index(drop=True)
    verify_packages(folder, val)
    _, old_signature = require_launch_freeze()
    legacy_plan = read_json(REFERENCE / 'PREDECLARED_PLAN.json')
    if legacy_plan['split_sha256'] != c['split_sha256']:
        raise ValueError('Cannot fuse across protocols')
    sources = legacy_plan['sources']
    for source in sources:
        if sha256(ROOT / source['prediction_file']) != source['prediction_sha256'] or sha256(ROOT / source['checkpoint']) != source['checkpoint_sha256']:
            raise ValueError('Historical ensemble source changed')
    parts = fp32_parts(old_signature, val)
    _, cbam = probabilities(ROOT / sources[-1]['prediction_file'], val)
    parts.append(cbam)
    _, base = probabilities(folder / 'best_accuracy/predictions.csv', val)
    _, reference = probabilities(REFERENCE / 'validation_predictions.csv', val)
    reproduced = np.mean(parts, axis=0)
    if not np.allclose(reference, reproduced, atol=2e-7, rtol=1e-6) or not np.array_equal(reference.argmax(1), reproduced.argmax(1)):
        raise ValueError('Original S83 equal-six reference was not reproduced')
    new_source = dict(run=c['experiment_id'], checkpoint=c['checkpoints'] + '/best.pt',
                      checkpoint_sha256=sha256(ROOT / c['checkpoints'] / 'best.pt'), epoch=20, branch='raw',
                      selector='predeclared earliest maximum standalone accuracy; no ensemble epoch selection',
                      prediction_file=c['results'] + '/best_accuracy/predictions.csv',
                      prediction_sha256=sha256(folder / 'best_accuracy/predictions.csv'))
    small = next(i for i, s in enumerate(sources) if s['run'].startswith('s18_'))
    candidates = [dict(experiment_id='r204_r203_equal_seven_original_exploratory_seed42',
                       output='r204_equal_seven', display_name='R204 equal-seven: S83 + Base',
                       indices=list(range(6)), members=sources + [new_source]),
                  dict(experiment_id='r205_r203_small_replacement_equal_six_original_exploratory_seed42',
                       output='r205_replace_small_equal_six', display_name='R205 equal-six: Base replaces Small',
                       indices=[i for i in range(6) if i != small],
                       members=[s for i, s in enumerate(sources) if i != small] + [new_source])]
    plan = dict(question='Does fresh original-split ConvNeXt-V2 Base add useful complementary predictions?',
                protocol=c['protocol'], split_manifest=c['split_manifest'], split_sha256=c['split_sha256'],
                sample_count=1503, class_order=list(CLASSES), candidate_selection=new_source,
                experiments=candidates, reference_sources=sources,
                fp32_cache_receipts=old_signature['fp32_reference_caches'],
                fusion='Equal per-model averaging of saved FP32 identity probabilities; no learned gating',
                search='Exactly two fixed combinations: addition and capacity replacement; no weights, other checkpoints or extra trials',
                rationale='Base is stronger than older Small standalone; replacement tests capacity upgrade without increasing six-model count; addition tests diversity.',
                material_gate=dict(minimum_net_correct=8, minimum_accuracy_gain=.005,
                                   macro_f1_nondecreasing=True, melanoma_recall_nondecreasing=True),
                gpu_used=False, test_loaded=False, training_epochs=0,
                caveat='Repeated exploratory validation with train/validation lesion overlap and known historical test outcomes; not independent test evidence',
                analysis_code_sha256=sha256(Path(__file__)))
    if (OUT / 'FIXED_PLAN.json').exists() and read_json(OUT / 'FIXED_PLAN.json') != plan:
        raise ValueError('Fixed study plan changed; preserve existing results')
    write_json(OUT / 'FIXED_PLAN.json', plan)  # Persist choices before computing any fusion score.
    y = val.label.to_numpy(dtype=int)
    baseline = metrics(y, reference)
    old_metrics = read_json(REFERENCE / 'validation_metrics.json')
    if baseline['accuracy'] != old_metrics['accuracy'] or baseline['macro_f1'] != old_metrics['macro_f1']:
        raise ValueError('Historical S83 metrics differ from saved probabilities')
    old_correct, single_correct = reference.argmax(1) == y, base.argmax(1) == y
    complementarity = dict(base_fixes_s83_errors=int((~old_correct & single_correct).sum()),
                           s83_fixes_base_errors=int((old_correct & ~single_correct).sum()),
                           shared_errors=int((~old_correct & ~single_correct).sum()),
                           both_correct=int((old_correct & single_correct).sum()),
                           oracle_note='Correct-answer union is a diagnostic upper bound, never an actual ensemble result')
    classrows = []
    for i, name in enumerate(CLASSES):
        mask = y == i
        classrows.append(dict(class_name=name, support=int(mask.sum()),
                              base_fixes_s83_errors=int((mask & ~old_correct & single_correct).sum()),
                              s83_fixes_base_errors=int((mask & old_correct & ~single_correct).sum()),
                              shared_errors=int((mask & ~old_correct & ~single_correct).sum()),
                              r203_precision=read_json(folder / 'validation_metrics.json')['per_class'][name]['precision'],
                              r203_recall=read_json(folder / 'validation_metrics.json')['per_class'][name]['recall'],
                              r203_f1=read_json(folder / 'validation_metrics.json')['per_class'][name]['f1'],
                              s83_precision=baseline['per_class'][name]['precision'], s83_recall=baseline['per_class'][name]['recall'],
                              s83_f1=baseline['per_class'][name]['f1']))
    write_csv(OUT / 'complementarity_by_class.csv', classrows)
    write_json(OUT / 'complementarity.json', complementarity)
    rows = [dict(experiment_id=c['experiment_id'], display_name='R203 Base standalone',
                 accuracy=metrics(y, base)['accuracy'], macro_f1=metrics(y, base)['macro_f1']),
            dict(experiment_id='s83_cbam_f1_addition_equal_six_exploratory_seed42', display_name='S83 existing equal-six',
                 accuracy=baseline['accuracy'], macro_f1=baseline['macro_f1'])]
    results = []
    for candidate in candidates:
        p = np.mean([parts[i] for i in candidate['indices']] + [base], axis=0)
        destination = OUT / candidate['output']
        m = package(destination, val, p, candidate['experiment_id'], candidate['members'], candidate['display_name'], c)
        now_correct = p.argmax(1) == y
        gained, lost = int((~old_correct & now_correct).sum()), int((old_correct & ~now_correct).sum())
        passed = (gained - lost >= 8 and m['accuracy'] - baseline['accuracy'] >= .005 - 1e-12 and
                  m['macro_f1'] >= baseline['macro_f1'] and m['per_class']['mel']['recall'] >= baseline['per_class']['mel']['recall'])
        summary = dict(status='completed', **{k: m[k] for k in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'correct', 'incorrect')},
                       experiment_id=candidate['experiment_id'], display_name=candidate['display_name'],
                       gained=gained, lost=lost, net_correct=gained-lost,
                       accuracy_gain_pp=100 * (m['accuracy'] - baseline['accuracy']), material_gate_passed=bool(passed),
                       decision='material_gain' if passed else ('small_numerical_gain' if m['accuracy'] > baseline['accuracy'] else 'retain_reference'),
                       members=candidate['members'], ensemble_weights=m['ensemble_weights'],
                       melanoma_recall=m['per_class']['mel']['recall'], akiec_recall=m['per_class']['akiec']['recall'],
                       metrics_path=relative(destination / 'validation_metrics.json'), test_loaded=False, gpu_used=False)
        changed = []
        for i in np.flatnonzero(old_correct != now_correct):
            changed.append(dict(image_id=val.iloc[i].image_id, true_class=CLASSES[y[i]],
                                reference_class=CLASSES[reference.argmax(1)[i]], ensemble_class=CLASSES[p.argmax(1)[i]],
                                change='gained' if now_correct[i] else 'lost'))
        write_csv(destination / 'changed_predictions.csv', changed,
                  ['image_id', 'true_class', 'reference_class', 'ensemble_class', 'change'])
        write_json(destination / 'summary.json', summary)
        record = dict(experiment_id=candidate['experiment_id'], era='structured', record_kind='fixed_probability_fusion',
                      phase='original_exploratory_convnext_base', protocol=c['protocol'], evaluation_split='validation',
                      split_manifest=c['split_manifest'], split_sha256=c['split_sha256'], method=plan['fusion'],
                      model='+'.join(s['run'] for s in candidate['members']), attention='CBAM retained in S79 component',
                      ensemble_members=json.dumps([s['run'] for s in candidate['members']]), ensemble_weights=json.dumps(m['ensemble_weights']),
                      image_size=224, seed=42, epochs=0, status='completed', decision=summary['decision'],
                      config_path=relative(OUT / 'FIXED_PLAN.json'), metrics_path=summary['metrics_path'],
                      confusion_matrix_path=relative(destination / 'confusion_matrix.csv'), plots_dir=relative(destination / 'figures'),
                      runtime_seconds=time.perf_counter()-tick, notes=plan['caveat'],
                      **{k: m[k] for k in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')})
        update_registries(record)
        write_json(destination / 'record.json', record)
        rows.append(dict(experiment_id=candidate['experiment_id'], display_name=candidate['display_name'],
                         accuracy=m['accuracy'], macro_f1=m['macro_f1']))
        results.append(summary)
    comparison_figures(rows, OUT / 'comparison_figures', 'Two fixed Base fusions | original exploratory validation')
    write_csv(OUT / 'comparison.csv', rows)
    overall = dict(status='completed', reference_accuracy=baseline['accuracy'], reference_macro_f1=baseline['macro_f1'],
                   r203_accuracy=metrics(y, base)['accuracy'], r203_macro_f1=metrics(y, base)['macro_f1'],
                   complementarity=complementarity, experiments=results,
                   numerical_leader=max(rows, key=lambda r: (r['accuracy'], r['macro_f1'])),
                   runtime_seconds=time.perf_counter()-tick, gpu_used=False, test_loaded=False,
                   training_closed=True, extend_r203=False,
                   extension_reason='No higher accuracy after epoch20; near-perfect training and rising validation loss; cosine ended at1e-7',
                   caveat=plan['caveat'])
    write_json(OUT / 'summary.json', overall)
    write_json(OUT / 'verification.json', dict(status='passed', same_original_validation_cohort=True,
               original_reference_reproduced=True, sample_count=1503, class_order=list(CLASSES),
               source_checkpoints_and_predictions_hash_checked=True, fp32_cache_hashes_checked=True,
               finite_normalized_probabilities=True, exactly_two_fixed_fusions=True,
               fusion_outputs_sha256={relative(p): sha256(p) for p in OUT.rglob('validation_*.csv')},
               analysis_code_sha256=sha256(Path(__file__)), gpu_used=False, test_images_opened=0))
    print(json.dumps(overall, indent=2))


if __name__ == '__main__':
    main()
