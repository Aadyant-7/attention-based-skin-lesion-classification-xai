"""Summarize fixed data screen results; never select new parameters or load images."""
import json
import numpy as np
import pandas as pd
from research.common import ROOT,CLASSES,sha256,write_json

OUT=ROOT/'results/data_assisted/s66_s67_frozen_data_screen'


def main():
    s=json.loads((OUT/'summary.json').read_text());v=json.loads((OUT/'verification.json').read_text())
    assert s['status']=='completed' and v['status']=='passed' and not s['test_loaded'] and not s['gpu_used']
    rows=[]
    for stage in ['s66','s67']:
        for name,m in s[stage]['metrics'].items():
            rows.append(f"|{stage.upper()} {name}|{100*m['accuracy']:.4f}%|{m['macro_precision']:.6f}|{m['macro_recall']:.6f}|{m['macro_f1']:.6f}|")
    domain=[]
    for name,m in s['s66']['external_preflight_metrics'].items():
        domain.append(f"|{name}|{100*m['accuracy']:.4f}%|{m['macro_f1']:.6f}|")
    folds=pd.read_csv(OUT/'s67/fold_comparison.csv')
    fr='\n'.join(f"|{r.fold}|{100*r.ham_only_accuracy:.4f}%|{100*r.ham_plus_external_accuracy:.4f}%|{r.net_correct_change:+d}|" for r in folds.itertuples(index=False))
    classes=[]
    for i,cl in enumerate(CLASSES):
        fields=[]
        for name in ['ham_only','ham_plus_external']:
            cm=np.array(s['s66']['metrics'][name]['confusion_matrix']);tp=cm[i,i]
            fields.append(f"{tp/max(cm[i].sum(),1):.4f} / {2*tp/max(cm[:,i].sum()+cm[i].sum(),1):.4f}")
        classes.append(f"|{cl}|{fields[0]}|{fields[1]}|")
    failures=[]
    a=s['s66'];g=s['s67']
    if a['accuracy_delta']<.01-1e-12:failures.append('S66 accuracy gain below +1.0 percentage point')
    if a['macro_f1_delta']<.01-1e-12:failures.append('S66 macro-F1 gain below +0.01')
    if a['melanoma_recall_candidate']<a['melanoma_recall_control']-1e-12:failures.append('S66 melanoma recall declined')
    if g['accuracy_delta']<.01-1e-12:failures.append('S67 pooled accuracy gain below +1.0 percentage point')
    if g['macro_f1_delta']<-1e-12:failures.append('S67 pooled macro-F1 declined')
    if g['melanoma_recall_candidate']<g['melanoma_recall_control']-1e-12:failures.append('S67 pooled melanoma recall declined')
    if g['positive_folds']<2:failures.append('S67 fewer than two improving folds')
    assert bool(s['data_gate_passed'])==(not failures)
    decision='PASS: evidence supports preparing a bounded GPU proposal, not launching automatically.' if not failures else 'FAIL: do not spend GPU time training this fixed combined-data recipe.'
    next_action=('Prepare a matched bounded source-assisted fine-tuning proposal with an explicit compute cap and difficult-class gates. This frozen-feature result does not predict a93% ensemble gain.' if not failures else
        'Keep the curated external pool and all evidence, but reject GPU fine-tuning of this simple pooled-data recipe. Inspect source/class-shift evidence before a revised hypothesis; do not rescue this screen through subset, source-weight or hyperparameter searches. Metadata remains a separate deferred option.')
    report=f'''# S66/S67 external-data CPU screen — closeout

## Decision

**{decision}**

The current fine-tuned ensemble stays unchanged at93.6128% exploratory validation accuracy. These are frozen-feature diagnostic classifiers, not new ensemble scores or locked-test results.

## Fixed setup

- Fresh ImageNet1k ConvNeXt-Tiny features, CPU FP32, square224 bilinear identity preprocessing.
- Same weighted logistic regression, C=1, seed42, train-only scaling and normalized sqrt inverse-frequency sample weights.
- Control:7,009 HAM training images. Candidate: same HAM training plus2,823 fixed curated BCN training lesions.
- Exploratory validation: same1,503 HAM development images; no fit on these images.
- Confirmation: three fixed train-only lesion-group folds; the same external training partition is added to each candidate fit. No HAM lesion crosses fit/heldout boundaries within these folds.
- External500-lesion preflight validation: descriptive source-shift check only, never fitted.
- No parameter/source-subset/weight search, backpropagation, metadata fusion or GPU training. Historical S64 control metrics reproduced exactly.

## Results

|Diagnostic method|Accuracy|Macro precision|Macro recall|Macro-F1|
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

S66: **{100*a['accuracy_delta']:+.4f} percentage points**, macro-F1 {a['macro_f1_delta']:+.6f}; {a['gained']} correct predictions gained / {a['lost']} lost / net {a['net_correct_change']:+d}. Melanoma recall {100*a['melanoma_recall_control']:.4f}% → {100*a['melanoma_recall_candidate']:.4f}%.

S67 pooled: **{100*g['accuracy_delta']:+.4f} percentage points**, macro-F1 {g['macro_f1_delta']:+.6f}; {g['gained']} gained / {g['lost']} lost / net {g['net_correct_change']:+d}. Melanoma recall {100*g['melanoma_recall_control']:.4f}% → {100*g['melanoma_recall_candidate']:.4f}%; {g['positive_folds']}/3 folds improve.

|Group fold|HAM-only accuracy|HAM+external accuracy|Net correct change|
|---|---:|---:|---:|
{fr}

### Class behavior on exploratory validation

|Class|HAM-only recall / F1|HAM+external recall / F1|
|---|---:|---:|
{chr(10).join(classes)}

Full class-wise precision/recall/F1/support and raw/normalized confusion matrices are saved for both protocols and the external preflight check. Paired gain/loss image identities are saved; no visual/test-label review is required to obtain these counts.

### External preflight domain check

|Method|Accuracy on external500|Macro-F1|
|---|---:|---:|
{chr(10).join(domain)}

These source-specific scores have different case composition and cannot be compared as if they were matched HAM validation results. Learning BCN performance does not by itself prove improved HAM generalization. AK is subset supervision for HAM akiec; unknown test aliases and patient overlap remain unverified. This is post-test exploratory development, not a replacement test evaluation.

## Gate and next action

{'All predefined gates passed.' if not failures else chr(10).join('- '+x for x in failures)}

{next_action}

## Saved evidence

- `results/data_assisted/s66_s67_frozen_data_screen/`: plans, source/cache hashes, summary and verification.
- `s66/ham_only/`, `s66/ham_plus_external/`: exploratory metrics, predictions/probabilities, confusion/class PNG/PDF figures; external-preflight subfolders contain the source-shift reports.
- `s67/ham_only/`, `s67/ham_plus_external/`: train-only meta-OOF metrics, predictions/probabilities and figures; folds/assignments/class comparisons saved in `s67/`.
- `s66/comparison_figures/`, `s67/comparison_figures/`: paper-ready comparison CSV/PNG/PDF.
- `.cache/s66_s67_frozen_data_screen/`: reusable external features and all fitted CPU classifiers (local only).
- `results/master_experiment_registry.csv`: four completed diagnostic method entries, explicitly separate from CNN/ensemble/test performance.

Runtime: {s['runtime_seconds']/60:.2f} minutes. Original locked-test images/labels never loaded; no new original-test evaluation.
'''
    (ROOT/'research/data_assisted/S66_S67_CLOSEOUT.md').write_text(report,encoding='utf-8')
    heads={p.name:sha256(p) for p in (ROOT/'.cache/s66_s67_frozen_data_screen').glob('*.joblib')}
    assert len(heads)==8
    write_json(OUT/'closeout.json',dict(status='completed',gate_passed=s['data_gate_passed'],failed_gates=failures,
        serialized_classifier_hashes=heads,test_loaded=False,gpu_used=False,ensemble_changed=False))
    print(json.dumps(dict(gate_passed=s['data_gate_passed'],s66_accuracy_delta=a['accuracy_delta'],s67_accuracy_delta=g['accuracy_delta'],failed_gates=failures),indent=2))


if __name__=='__main__':main()
