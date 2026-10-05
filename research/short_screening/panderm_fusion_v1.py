"""Bounded PanDerm addition, with one predefined replacement fallback."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from research.common import ROOT, CLASSES, sha256, relative, write_json, write_csv
from research.strict_train import metric_report
from research.aggressive.core import predictions, retry_registry_upsert
from research.plots import metric_figures, comparison_figures

OUT = ROOT / 'results/short_screening/s54_s55_panderm_fusion'
FOUR = ROOT / 'results/short_screening/s46_all_f1_checkpoint_fusion'
FIVE = ROOT / 'results/short_screening/s53_equal_five_b0_addition'
SPLIT = ROOT / 'data/splits/exploratory/image_level_dev_v1.csv'
CACHE = ROOT / '.cache/panderm_base_image_level_v1/panderm_rbf_svc_val_probabilities.npz'
HEAD = ROOT / 'checkpoints/exploratory/panderm_base_image_level_v1/panderm_rbf_svc.joblib'
WEIGHTS = ROOT / '.cache/panderm_bb_data6_checkpoint-499.pth'
PLAN = dict(
    question='Can a domain-pretrained frozen transformer plus its existing SVM head improve the current ensemble?',
    first='S54: equal six members, S53 five plus PanDerm identity SVM',
    fallback='S55 only if S54 fails directional gate: equal five, replace B0 with PanDerm in S53',
    views=['identity'], pan_selection='Existing exploratory RBF SVM C=10, no refit or alternate heads',
    directional_gate='Accuracy improves over S53 with macro-F1 and melanoma recall nondecreasing',
    material_gate='At least +0.005 absolute accuracy over S53, macro-F1 and melanoma recall nondecreasing',
    scope='Post-test exploratory development; same validation used for previous checkpoint and method selection',
    limit='At most two fixed combinations; no weight, threshold, checkpoint or TTA search',
    test_loaded=False, gpu_used=False, new_training=False)


def aligned(path, ids, labels):
    f = pd.read_csv(path)
    assert f.image_id.is_unique and set(f.image_id) == set(ids)
    f = f.set_index('image_id').loc[ids]
    assert f.true_class.tolist() == labels
    p = f[[f'p_{c}' for c in CLASSES]].to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-5)
    return p


def main():
    if (OUT / 'summary.json').exists():
        print((OUT / 'summary.json').read_text()); return
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / 'PREDECLARED_PLAN.json', PLAN)
    ref = pd.read_csv(FIVE / 'validation_predictions.csv')
    assert len(ref) == 1503 and ref.image_id.is_unique
    ids = ref.image_id.to_numpy(); labels = ref.true_class.tolist()
    y = np.array([CLASSES.index(c) for c in labels])
    assignments = pd.read_csv(SPLIT, usecols=['image_id', 'split'])
    assert set(ids) == set(assignments.loc[assignments.split == 'val', 'image_id'])
    locked = pd.read_csv(ROOT / 'data/splits/split_assignments.csv', usecols=['image_id', 'split'])
    assert not set(ids) & set(locked.loc[locked.split == 'test', 'image_id'])
    manifest = json.loads((FOUR / 'candidate_manifest.json').read_text())
    sources = []
    arrays = []
    for s in manifest['sources']:
        path = ROOT / s['prediction_file']
        assert sha256(path) == s['prediction_sha256']
        assert sha256(ROOT / s['checkpoint']) == s['checkpoint_sha256']
        arrays.append(aligned(path, ids, labels)); sources.append(s)
    b0id = 's03_efficientnet_b0_none_exploratory_seed42'
    b0path = ROOT / 'results/structured_experiments' / b0id / 'validation_predictions.csv'
    b0cp = ROOT / 'checkpoints/structured' / b0id / 'best.pt'
    b0 = aligned(b0path, ids, labels)
    sources.append(dict(run=b0id, prediction_file=relative(b0path), prediction_sha256=sha256(b0path),
                        checkpoint=relative(b0cp), checkpoint_sha256=sha256(b0cp)))
    baseline = (sum(arrays) + b0) / 5
    np.testing.assert_allclose(baseline, aligned(FIVE / 'validation_predictions.csv', ids, labels), atol=1e-12)
    with np.load(CACHE, allow_pickle=False) as z:
        cached_ids = z['ids'].astype(str)
        assert len(set(cached_ids)) == 1503 and set(cached_ids) == set(ids)
        order = pd.Index(cached_ids).get_indexer(ids)
        assert np.array_equal(z['y'][order], y)
        pan = z['probabilities'][order]
    assert pan.shape == (1503, 7) and np.isfinite(pan).all() and (pan >= 0).all()
    assert np.allclose(pan.sum(1), 1, atol=1e-5)
    pan_source = dict(run='legacy_panderm_identity_rbf_svc', prediction_file=relative(CACHE), prediction_sha256=sha256(CACHE),
                      checkpoint=relative(HEAD), checkpoint_sha256=sha256(HEAD), backbone=relative(WEIGHTS), backbone_sha256=sha256(WEIGHTS),
                      class_order=list(CLASSES), source_protocol='results/exploratory/panderm_base_image_level_v1/protocol.json')
    bm = json.loads((FIVE / 'validation_metrics.json').read_text())
    old = baseline.argmax(1) == y
    pan_correct = pan.argmax(1) == y
    complement = dict(panderm_accuracy=float(pan_correct.mean()), fixes_reference_errors=int((~old & pan_correct).sum()),
                      errors_on_reference_correct=int((old & ~pan_correct).sum()))
    rows = [dict(display_name='S53 equal five reference', accuracy=bm['accuracy'], macro_f1=bm['macro_f1'])]
    studies = []

    def evaluate(number, name, p, members):
        target = OUT / name
        assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-5)
        m = metric_report(y, p, float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()))
        pred = predictions(ids, y, p)
        write_json(target / 'validation_metrics.json', m)
        write_csv(target / 'validation_predictions.csv', pred)
        write_csv(target / 'validation_probabilities.csv', pd.DataFrame(pred)[['image_id'] + [f'p_{c}' for c in CLASSES]].to_dict('records'))
        metric_figures(m, target / 'figures', f'S{number} fixed PanDerm fusion | exploratory validation')
        saved = pd.read_csv(target / 'validation_predictions.csv')
        assert abs(accuracy_score(saved.true_class, saved.predicted_class) - m['accuracy']) < 1e-12
        assert abs(f1_score(saved.true_class, saved.predicted_class, labels=CLASSES, average='macro') - m['macro_f1']) < 1e-12
        assert np.array_equal(confusion_matrix(saved.true_class, saved.predicted_class, labels=CLASSES), m['confusion_matrix'])
        new = p.argmax(1) == y
        balance = m['macro_f1'] >= bm['macro_f1'] and m['per_class']['mel']['recall'] >= bm['per_class']['mel']['recall']
        passed = m['accuracy'] > bm['accuracy'] + 1e-12 and balance
        material = m['accuracy'] - bm['accuracy'] >= .005 - 1e-12 and balance
        summary = dict(experiment=f'S{number}', accuracy=m['accuracy'], macro_precision=m['macro_precision'], macro_recall=m['macro_recall'],
                       macro_f1=m['macro_f1'], mel_recall=m['per_class']['mel']['recall'], gained=int((~old & new).sum()),
                       lost=int((old & ~new).sum()), net=int(new.sum() - old.sum()), directional_gate_passed=bool(passed), material_gate_passed=bool(material))
        write_json(target / 'summary.json', summary)
        write_json(target / 'candidate_manifest.json', dict(sources=members, weights=[1/len(members)]*len(members), inference='Preserved identity probabilities; no new GPU inference'))
        write_csv(target / 'class_comparison.csv', [dict(class_name=c, reference_recall=bm['per_class'][c]['recall'], candidate_recall=m['per_class'][c]['recall'],
                   reference_f1=bm['per_class'][c]['f1'], candidate_f1=m['per_class'][c]['f1']) for c in CLASSES])
        rows.append(dict(display_name=f'S{number} {name}', accuracy=m['accuracy'], macro_f1=m['macro_f1']))
        studies.append(summary)
        retry_registry_upsert(dict(experiment_id=f's{number}_{name}_exploratory_seed42', era='structured', record_kind='cpu_fusion_screen',
            phase='post_test_exploratory_development', protocol='exploratory_image_level', evaluation_split='validation',
            split_manifest=relative(SPLIT), split_sha256=sha256(SPLIT), method=name+'; fixed equal identity probability fusion',
            ensemble_members=json.dumps([s['run'] for s in members]), ensemble_weights=json.dumps([1/len(members)]*len(members)),
            epochs=0, seed=42, metrics_path=relative(target/'validation_metrics.json'), plots_dir=relative(target/'figures'),
            status='completed', decision='material_gate_passed' if material else 'material_gate_failed', notes=PLAN['scope'],
            **{k:m[k] for k in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']}))
        return passed

    passed = evaluate(54, 'equal_six_panderm_addition', (baseline*5+pan)/6, sources+[pan_source])
    if not passed:
        evaluate(55, 'replace_b0_with_panderm', (sum(arrays)+pan)/5, sources[:4]+[pan_source])
    for s in sources+[pan_source]:
        assert sha256(ROOT/s['prediction_file']) == s['prediction_sha256']
        assert sha256(ROOT/s['checkpoint']) == s['checkpoint_sha256']
    assert sha256(WEIGHTS) == pan_source['backbone_sha256']
    comparison_figures(rows, OUT/'comparison_figures', 'Bounded PanDerm addition / replacement study')
    summary = dict(status='completed', reference_accuracy=bm['accuracy'], reference_macro_f1=bm['macro_f1'],
                   complementarity=complement, studies=studies, test_loaded=False, gpu_used=False,
                   decision='Bounded predefined study complete; no additional combinations or automatic training')
    write_json(OUT/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
