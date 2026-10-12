"""One predeclared training-loss-derived correction, saved validation only."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from research.common import ROOT, CLASSES, sha256, write_csv, write_json

OUT = ROOT / 'results/diagnostics/generalization_review_2026-10-12/r202_weight_bias_check'
CONFIG = ROOT / 'research/image_level_replication/v2/convnext_base_fold00.json'
PREDICTIONS = ROOT / 'results/image_level_replication/v2/r202_convnextv2_base_fold00_seed42/best_accuracy/predictions.csv'
COLS = ['p_' + c for c in CLASSES]
PLAN = dict(method='One inverse training-class-weight probability correction, exponent exactly1',
    question='Does undoing the known weighted-CE class tilt improve the saved R202 natural-distribution validation predictions?',
    formula='q_corrected[c] = (q[c]/training_loss_weight[c]) / sum_k(q[k]/training_loss_weight[k])',
    rationale='At the population optimum of weighted cross entropy, q[c] is proportional to weight[c]*P(c|x). A trained finite network need not satisfy that idealization.',
    sources='Existing EMA epoch20 accuracy-selected R202 checkpoint predictions; original902 inner-validation images',
    search='One fixed rule; no exponent/temperature/threshold/weight grid or alternative checkpoint evaluation',
    gate=dict(minimum_accuracy_gain=.005, macro_f1_non_decreasing=True,
              melanoma_recall_non_decreasing=True, akiec_recall_non_decreasing=True),
    scope='Postdevelopment validation-only diagnostic. It cannot measure generalization or prove a test gain.',
    test_images_loaded=False, test_labels_loaded=False, gpu_used=False, checkpoint_loaded=False)


def metrics(frame, p):
    y = frame.true_class.map(dict(zip(CLASSES, range(7)))).to_numpy()
    pred = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=range(7), zero_division=0)
    return dict(accuracy=float((y == pred).mean()), correct=int((y == pred).sum()),
        macro_precision=float(precision.mean()), macro_recall=float(recall.mean()), macro_f1=float(f1.mean()),
        confusion_matrix=confusion_matrix(y, pred, labels=range(7)).tolist(),
        class_order=list(CLASSES),
        per_class={c: dict(precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]), support=int(support[i]))
                   for i, c in enumerate(CLASSES)})


def run():
    if (OUT / 'summary.json').exists():
        print('Completed single-rule check preserved; no repeat search')
        return
    if (OUT / 'plan.json').exists():
        assert json.loads((OUT / 'plan.json').read_text()) == PLAN
    else:
        write_json(OUT / 'plan.json', PLAN)
    c = json.loads(CONFIG.read_text())
    assert c['loss'].startswith('weighted_cross_entropy') and c['label_smoothing'] == 0
    frame = pd.read_csv(PREDICTIONS)
    assert len(frame) == 902 and frame.image_id.is_unique
    p = frame[COLS].to_numpy()
    assert [CLASSES[i] for i in p.argmax(1)] == frame.predicted_class.tolist()
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-5)
    weights = np.array(c['loss_class_weights'])
    assert weights.shape == (7,) and (weights > 0).all()
    corrected = p / weights
    corrected /= corrected.sum(1, keepdims=True)
    old, new = metrics(frame, p), metrics(frame, corrected)
    gain = new['accuracy'] - old['accuracy']
    checks = dict(material_accuracy_gain=gain >= .005 - 1e-12,
        macro_f1_non_decreasing=new['macro_f1'] >= old['macro_f1'] - 1e-12,
        melanoma_recall_non_decreasing=new['per_class']['mel']['recall'] >= old['per_class']['mel']['recall'] - 1e-12,
        akiec_recall_non_decreasing=new['per_class']['akiec']['recall'] >= old['per_class']['akiec']['recall'] - 1e-12)
    y = frame.true_class.map(dict(zip(CLASSES, range(7)))).to_numpy()
    gained = int(((p.argmax(1) != y) & (corrected.argmax(1) == y)).sum())
    lost = int(((p.argmax(1) == y) & (corrected.argmax(1) != y)).sum())
    out_frame = frame.copy()
    out_frame['predicted_class'] = [CLASSES[i] for i in corrected.argmax(1)]
    out_frame[COLS] = corrected
    write_csv(OUT / 'candidate_validation_predictions.csv', out_frame.to_dict('records'))
    write_json(OUT / 'baseline_metrics.json', old)
    write_json(OUT / 'candidate_metrics.json', new)
    write_csv(OUT / 'class_comparison.csv', [dict(class_name=cl,
        baseline_precision=old['per_class'][cl]['precision'], candidate_precision=new['per_class'][cl]['precision'],
        baseline_recall=old['per_class'][cl]['recall'], candidate_recall=new['per_class'][cl]['recall'],
        baseline_f1=old['per_class'][cl]['f1'], candidate_f1=new['per_class'][cl]['f1'], support=new['per_class'][cl]['support']) for cl in CLASSES])
    result = dict(status='completed', checks=checks, adopt=all(checks.values()),
        baseline_accuracy=old['accuracy'], candidate_accuracy=new['accuracy'], accuracy_gain_pp=100 * gain,
        baseline_macro_f1=old['macro_f1'], candidate_macro_f1=new['macro_f1'],
        correct_gained=gained, correct_lost=lost, net_correct=gained - lost,
        source_sha256={'research/image_level_replication/v2/convnext_base_fold00.json': sha256(CONFIG),
                       'results/image_level_replication/v2/r202_convnextv2_base_fold00_seed42/best_accuracy/predictions.csv': sha256(PREDICTIONS)},
        test_loaded=False, training=False, gpu_used=False, original_model_or_results_changed=False,
        conclusion='Candidate eligible for a separately frozen evaluation' if all(checks.values()) else 'Rejected; original EMA20 predictions/decision rule retained')
    write_json(OUT / 'summary.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
