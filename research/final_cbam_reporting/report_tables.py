"""Paper/presentation tables from existing completed results; no inference/fitting."""
import json
import numpy as np
import pandas as pd
from research.common import ROOT, CLASSES, write_json, write_csv, atomic_text, sha256, relative
from research.plots import comparison_figures, validate_metrics
from .finalize import OUT, S79, OLD, RID, frozen


def main():
    frozen()
    registry=pd.read_csv(ROOT/'results/master_experiment_registry.csv',dtype=str).fillna('')
    models=[('s02_mobilenet_v3_large_none_exploratory_seed42','MobileNetV3-Large'),
        ('s03_efficientnet_b0_none_exploratory_seed42','EfficientNet-B0'),
        ('s05_efficientnet_b0_cbam_exploratory_seed42','EfficientNet-B0 + CBAM'),
        ('s06_convnext_tiny_none_exploratory_seed42','ConvNeXt-Tiny'),
        ('s10_efficientnet_v2_s_none_exploratory_seed42','EfficientNetV2-S'),
        ('s15_densenet201_none_exploratory_seed42','DenseNet201'),
        ('s18_convnext_small_none_exploratory_seed42','ConvNeXt-Small'),
        ('s19_resnet101_none_exploratory_seed42','ResNet101'),
        ('s79_convnext_tiny_cbam_exploratory_seed42','ConvNeXt-Tiny + CBAM (longer package)')]
    rows=[]
    for rid,name in models:
        r=registry.loc[registry.experiment_id==rid].iloc[0]
        assert r.status=='completed' and r.protocol=='exploratory_image_level' and r.evaluation_split=='validation'
        m=json.loads((ROOT/r.metrics_path).read_text());validate_metrics(m)
        for k in ['accuracy','macro_precision','macro_recall','macro_f1']:assert abs(float(r[k])-m[k])<1e-12
        rows.append(dict(display_name=name,experiment_id=rid,epochs=int(float(r.epochs)),accuracy_checkpoint_epoch=int(float(r.best_epoch)),
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']},
            source_metrics=r.metrics_path,source_sha256=sha256(ROOT/r.metrics_path),
            recipe_note='Changed attention/augmentation/duration/selection package; not pure CBAM ablation' if rid.startswith('s79') else 'Historical exploratory screening recipe; actual completed epoch budget shown'))
    table=OUT/'paper_tables';write_csv(table/'standalone_exploratory.csv',rows)
    comparison_figures(rows,table/'standalone_figures','Standalone exploratory validation | unequal actual epoch budgets disclosed')
    audit=json.loads((OUT/'ensemble/test_metrics.json').read_text());xai=json.loads((OUT/'xai/manifest.json').read_text())
    retained=json.loads((ROOT/'results/short_screening/s53_equal_five_b0_addition/validation_metrics.json').read_text())
    cbam=json.loads((S79/'fixed_ensemble/validation_metrics.json').read_text());old=json.loads((OLD/'ensemble/test_metrics.json').read_text())
    overview=[]
    for name,m,protocol,path in [
        ('S53 retained equal-five (without added CBAM)',retained,'Repeated exploratory image-level validation','results/short_screening/s53_equal_five_b0_addition/validation_metrics.json'),
        ('S80 CBAM-inclusive equal-five',cbam,'Repeated exploratory image-level validation',relative(S79/'fixed_ensemble/validation_metrics.json')),
        ('S80 fixed post-development test audit',audit,'Previously evaluated held-out cohort; post-development descriptive audit',relative(OUT/'ensemble/test_metrics.json')),
        ('S31 original strict locked-test evaluation',old,'Original strict methodology; first evaluation preserved',relative(OLD/'ensemble/test_metrics.json'))]:
        validate_metrics(m);overview.append(dict(display_name=name,protocol=protocol,**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']},source_metrics=path,source_sha256=sha256(ROOT/path)))
    write_csv(table/'protocol_results.csv',overview)
    class_rows=[]
    for name,m in [('S80 exploratory validation',cbam),('S80 post-development test audit',audit)]:
        for c in CLASSES:class_rows.append(dict(result=name,class_name=c,**m['per_class'][c]))
    write_csv(table/'s80_class_scores.csv',class_rows)
    text='# Final figures/tables and writing handoff — 9 October 2026\n\n'
    text+='## Model versus accuracy: exploratory validation\n\n| Model | Epochs completed | Accuracy checkpoint | Accuracy | Macro-F1 at that checkpoint |\n|---|---:|---:|---:|---:|\n'
    for r in rows:text+=f"| {r['display_name']} | {r['epochs']} | {r['accuracy_checkpoint_epoch']} | {100*r['accuracy']:.4f}% | {r['macro_f1']:.6f} |\n"
    text+='\nAll use the same7009/1503 image-level development partition. Actual budgets and training-package differences are disclosed; S79 is not a pure CBAM ablation. Macro-F1 here belongs to the accuracy checkpoint, not necessarily each model\'s separate F1 maximum. No valid comparable completed EfficientNet-B3 score is invented. All backbones already use transfer learning; do not portray these as models trained from scratch.\n\n'
    text+='## Main protocol/result table\n\n| Method and evaluation | Accuracy | Macro precision | Macro recall | Macro-F1 |\n|---|---:|---:|---:|---:|\n'
    for r in overview:text+=f"| {r['display_name']} | {100*r['accuracy']:.4f}% | {r['macro_precision']:.6f} | {r['macro_recall']:.6f} | {r['macro_f1']:.6f} |\n"
    text+='\nProtocol labels in `protocol_results.csv` must remain visible in the paper. The new audit does not supersede the original first evaluation or restore test independence. No score-triggered tuning follows it.\n\n'
    text+='## Recommended results wording\n\n'
    text+=f'"The retained heterogeneous ensemble achieved93.61% exploratory validation accuracy (macro-F1 0.8869). The separately evaluated CBAM-inclusive ensemble achieved93.35% exploratory validation accuracy (macro-F1 0.8770) and {100*audit["accuracy"]:.2f}% in a post-development audit of the previously evaluated held-out cohort (macro-F1 {audit["macro_f1"]:.4f}). Grad-CAM visualizations explain representative predictions without altering the classifier."\n\n'
    text+='Make validation prominent through a clearly labelled comparison figure and table, not an ambiguous general-performance headline. Highest observed tiny-gain/multi-image variants are historical secondary evidence, not retained single-image or test scores. CBAM benefit is mixed, not universally positive.\n\n'
    text+=f'## Completed explainability\n\n{xai["status"]}; {len(xai["cases"])} deterministic validation cases, covering melanoma/akiec correct and incorrect categories and remaining classes where available. Every case includes all-five branch Grad-CAM, raw/normalized arrays, overlays, input224, actual CBAM attention weights and label/confidence metadata. Cases: '+', '.join(c['image_id'] for c in xai['cases'])+'.\n\n'
    text+='The equal normalized-map composite is a display summary; neither attention nor Grad-CAM proves lesion segmentation, causal clinical reasoning or model correctness. Explainability does not increase accuracy.\n\n'
    text+='## File/figure index (repository-relative)\n\n'
    text+='- `results/final_cbam_reporting/v1/paper_tables/`: model comparison CSV plus PNG/PDF; protocol result CSV; S80 validation/test class-score CSV.\n'
    text+='- `results/final_cbam_reporting/v1/ensemble/`: audit metrics, predictions/probabilities, class scores and raw/normalized confusion PNG/PDF.\n'
    text+='- `results/final_cbam_reporting/v1/xai/<image_id>/gradcam_panel.{png,pdf}`: all-five explanations and composite; per-model folders hold overlays/maps and CBAM figures.\n'
    text+='- `results/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/figures/`: accuracy/loss/F1 curves, selected confusion and class figures. Its `fixed_ensemble/figures/` is S80 validation.\n'
    text+='- `results/short_screening/s53_equal_five_b0_addition/`: retained non-CBAM validation result and figures.\n'
    text+='- `results/final_locked_test/v1/`: unchanged original strict test evidence.\n'
    text+='- `results/master_experiment_registry.csv`: complete method/results index; S81 is explicitly post-development.\n'
    text+='- Local-only report references: `docs/private_reference/report_guidance_2026-10-09/REFERENCE_NOTES.md`; originals and private notes excluded from Git.\n\n'
    text+='## Next writing phase\n\nUse the university A4/report structure after reconciling later guide instructions: preliminary pages; introduction/objectives; literature review; dataset/protocols; requirements/design/methodology; experiments/results; XAI; discussion/limitations; conclusions/future work; references; appendix/individual contributions. Add timeline, roles, constraints, risk and budget where the required Project Plan format calls for them. Journal/conference paper formatting is a separate template. Team identities, actual contributions, signatures, venue, deadlines and any additional guide requirements must come from the user; do not fabricate them.\n\n'
    text+='Remaining work is manuscript/report composition, final layout/citation review and guide feedback. No new training, ensemble search or test evaluation is queued.\n'
    atomic_text(ROOT/'research/final_cbam_reporting/REPORTING_HANDOFF.md',text)
    write_json(table/'sources.json',dict(status='completed',standalone_count=len(rows),protocol_rows=overview,test_tuning=False,xai_cases=len(xai['cases'])))
    print('Verified paper tables/figures and reporting handoff created; no inference/training')


if __name__=='__main__':main()
