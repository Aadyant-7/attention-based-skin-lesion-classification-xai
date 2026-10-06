"""Close out the fixed metadata probe without tuning it."""
import json
import pandas as pd
from research.common import ROOT


def main():
    out=ROOT/'results/data_assisted/s68_s69_metadata_screen'
    s=json.loads((out/'summary.json').read_text());v=json.loads((out/'verification.json').read_text())
    assert s['status']=='completed' and v['status']=='passed' and not s['test_loaded'] and not s['gpu_used']
    b=s['reference'];methods=[('S53 image only',b)]+[(n,m) for n,m in s['metrics'].items()]
    table='\n'.join(f"|{n}|{100*m['accuracy']:.4f}%|{m['macro_precision']:.6f}|{m['macro_recall']:.6f}|{m['macro_f1']:.6f}|{100*m['per_class']['mel']['recall']:.4f}%|" for n,m in methods)
    changes='\n'.join(f"- {r['method']}: {r['gained']} correct predictions gained / {r['lost']} lost / net {r['net_correct_change']:+d}; material gate {'passed' if r['material_gate_passed'] else 'failed'}." for r in s['comparisons'])
    cf=pd.read_csv(out/'class_comparison.csv')
    classes='\n'.join(f'|{r.method}|{r.class_name}|{r.reference_recall:.4f}|{r.candidate_recall:.4f}|{r.reference_f1:.4f}|{r.candidate_f1:.4f}|{r.gained-r.lost:+d}|' for r in cf.itertuples(index=False))
    report=f'''# S68/S69 metadata with the retained ensemble — closeout

## Decision

**Keep S53 image-only.** Neither fixed metadata adjustment passed the meaningful-improvement gate. No strength, bin, category or threshold search follows. No GPU training or original locked-test evaluation was performed.

## Matched results

|Method|Exploratory accuracy|Macro precision|Macro recall|Macro-F1|Melanoma recall|
|---|---:|---:|---:|---:|---:|
{table}

{changes}

Material gate fixed before scoring: at least8 net additional correct images, at least+0.005 absolute accuracy, no macro-F1 or melanoma-recall decline. The existing five CNNs, checkpoint selections, equal weights and224px identity preprocessing were unchanged.

## What was fitted

This is an interpretable **train-only metadata likelihood adjustment**, not a jointly trained multimodal neural network or a validation-fitted stacker. Existing out-of-fold training probabilities for all S53 members were unavailable; using in-sample outputs to fit a fusion head would not give a reliable estimate of deployment behavior.

Only age, sex and lesion localization were parsed for the8,512 permitted development images. Original locked-test clinical rows were skipped before parsing. Labels came from the development training manifest. Image/lesion IDs were used for alignment/group exclusions only; diagnosis-confirmation type and diagnosis-derived metadata were excluded as predictors.

- **{s['fit_lesions']:,} distinct training lesions** fitted the conditional categorical likelihood tables.
- **{s['excluded_shared_lesion_train_images']} training images** were excluded because their lesion appeared in validation; metadata fitting has zero validation-lesion overlap.
- Contradictory categorical metadata images excluded: {s['contradictory_fit_images']}.
- Age bins fixed:0–19,20–39,40–59,60–79,80–120. Sex/localization vocabularies came only from metadata-fit training lesions.
- One record per training lesion, Laplace alpha1 smoothing, fixed adjustment strength0.25.
- Missing/unseen values add no class evidence. Missingness is not a feature, and no validation-driven imputation/binning is fitted.

Formula: `P_new = softmax(log(P_S53) + 0.25 * sum_fields log(P(field | class)))`. Class priors are not multiplied again. The categorical conditional-likelihood construction follows [categorical naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes); the independence assumptions and fixed tempered fusion are approximations, not an exact proven posterior for these CNNs. Age/sex/location correlate with signals already encoded by images, so added likelihoods can double-count information or override a correct image decision.

## Class behavior

|Candidate|Class|Reference recall|Candidate recall|Reference F1|Candidate F1|Net correct|
|---|---|---:|---:|---:|---:|---:|
{classes}

The observed harm, especially melanoma/akiec tradeoffs, rejects these two recipes. It does not prove that every carefully trained multimodal model would fail. Earlier metadata work on a different cohort also failed to beat its image-only reference; no costly multimodal training is justified by this new probe.

## Limits and next decision

This is post-test exploratory development on the repeatedly used1,503-image validation cohort. Training-only metadata fitting avoids fitting the clinical adjustment on these labels, but the base CNN checkpoints and previous ensemble choices were selected on this cohort. These scores are not independent generalization or new test evidence. The original test report remains86.7598% and is not rerun.

External-data pooling, confidence calibration and metadata adjustments have not produced a material retained-method gain. Do not start another long run from these negative results or repeat small voting/metadata weight searches. Any next performance proposal needs a substantive image-representation/training hypothesis, a matched control, bounded compute and evidence that distinguishes it from techniques already rejected. The strict320px inference-only resolution screen also failed, so merely resizing an unchanged checkpoint is not a justified shortcut.

## Artifacts

`results/data_assisted/s68_s69_metadata_screen/` saves the fixed plan, source hashes, fitted categorical model/counts, metadata-fit/exclusion manifests, reference and both candidate metrics, predictions/probabilities, gain/loss details, class comparison and raw/normalized confusion matrices with PNG/PDF figures. `comparison_figures/` saves the model comparison. Registry entries: `s68_age_metadata_likelihood_exploratory_seed42` and `s69_age_sex_location_metadata_likelihood_exploratory_seed42`.

Numerical/class-order/identity/lesion-exclusion checks passed, and all reported metrics/confusion matrices were independently recomputed from saved predictions. Existing probabilities/checkpoints and the retained method were preserved. Runtime: {s['runtime_seconds']:.1f}s CPU processing, excluding Python import startup. The metadata model is saved as JSON; there is no new CNN checkpoint or epoch training curve.
'''
    (ROOT/'research/data_assisted/S68_S69_METADATA_CLOSEOUT.md').write_text(report,encoding='utf-8')
    print('Metadata closeout saved; retained ensemble unchanged.')


if __name__=='__main__':main()
