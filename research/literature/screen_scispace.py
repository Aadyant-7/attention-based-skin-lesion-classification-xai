"""Read-only SciSpace screening; run with bundled Python/openpyxl, never edits XLSX."""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'research/literature/incoming/SciSpace Literature Review.xlsx'
OUT = ROOT/'research/literature/scispace_analysis'
# Workbook row numbers, not publication identifiers. Decisions concern relevance,
# not validity of papers whose complete methods have not yet been inspected.
DECISIONS = {
2: ('candidate','DCENSnet','Three customized CNNs; defer expensive ensemble until single-model controls.'),
3: ('priority','ML-IGIA','Attention/ensemble reference; table captions conflict on six vs seven classes; label-dependent weights need development-only fitting.'),
4: ('candidate','SynthraXCoreNet','Six-CNN soft voting and calibration; compute cost motivates two-model baseline.'),
5: ('candidate','SBXception','Efficiency context; dataset column missing; custom Xception outside initial shortlist.'),
6: ('priority','Patch attention and diagnosis weighting','High-resolution attention and training loss balancing; mean sensitivity is not accuracy.'),
7: ('candidate','Attentional fusion ensemble','CNN diversity/attention; dataset absent in extracted field; verify before numeric comparison.'),
8: ('deferred','TransCon-Skin','Segmentation/contrastive ViT; missing dataset and unresolved evaluation; separate heavy research question.'),
9: ('candidate','Depth/channel feature fusion','HAM10000 mentioned in method; weighted P/R/F1 cannot replace macro-F1 or accuracy.'),
10: ('task_context','SDAM UNet','Segmentation/binary melanoma context; Dice/IoU/pixel accuracy not seven-class image accuracy.'),
11: ('candidate','Network-level fused architecture','Contrast enhancement/fusion; cost and Bayesian search unsuitable before controls.'),
12: ('candidate','MedFusionNet','ConvNeXt/ViT attention fusion; motivates modern CNN control, not immediate hybrid implementation.'),
13: ('candidate','Deep features and ML ensembles','Frozen feature classifiers may be cheap later; cohort/split missing.'),
14: ('candidate','Skin-CAD','Feature selection/XAI; binary and HAM scores must be separated.'),
15: ('deferred','DSCATNet','Dual-scale transformer; kappa is not accuracy; defer custom transformer.'),
16: ('candidate','EFAM-Net','ConvNeXt multiscale attention; isolate backbone before fusion changes.'),
17: ('candidate','EFFNet','Feature fusion/random forest; balancing order unknown; not EfficientNet despite name.'),
18: ('task_context','Modified EfficientNetV2L','ISIC2357 cohort differs from seven-class HAM; size/parameter claims require verification.'),
19: ('deferred','CARE','Rare-class attention with lesion boxes; annotations/segmentation are an additional requirement.'),
20: ('task_context','DeMAL-CNN','Metric attention on ISIC2016/2017/PH2; different class/task cohorts.'),
21: ('deferred','VAdaKD','Teacher/student distillation; need demonstrated strong teacher before extra training.'),
22: ('task_context','STViTDA-Net','ISIC2019 GAN/MAE/deformable attention; different task and substantial extra compute.'),
23: ('task_context','Fine-tuned CNN ensemble','ISIC2016 official binary cohort900/379; ensemble technique context only.'),
24: ('task_context','EDA-ResNet50','Extracted benign/malignant task; dual attention/XAI idea, not seven-class score.'),
25: ('task_context','PMJAF-Net','Segmentation only; Dice/Jaccard/pixel ACC excluded from classification comparison.'),
26: ('candidate','AWB loss','Multicenter imbalance strategy; duplicate DOI with row27, count once.'),
27: ('duplicate','AWB loss','Same DOI as row26 (case insensitive); publication year variation not independent evidence.'),
28: ('task_context','Explainable stacked ensemble','Binary melanoma; stacking must use out-of-fold development predictions.'),
29: ('task_context','GP-CNN-DTEL','ISIC2016 AP/ISIC2017 AUC; different metric/tasks; local/global context idea.'),
30: ('task_context','DermaKNet','ISBI2017/EDRA dermatological attribute supervision unavailable in our initial inputs.'),
31: ('deferred','MSMA','Multimodal image/text inputs; not our image-only question.'),
32: ('candidate','ConvNeXt-ST-AFF','ECA/fusion and artifact preprocessing; dataset missing; first establish CNN baseline.'),
33: ('deferred','Federated CNNs','Privacy/client simulation is separate contribution; SMOTEENN order/cohort unknown.'),
34: ('candidate','DeepSkin','Undersampling vs oversampling trade-off; reducing dataset changes effective budget/cohort.'),
35: ('review_context','Skin classification survey','Secondary survey for taxonomy; not an independent experimental result.'),
36: ('candidate','Optimized CNN activation','Validation score not final test; holdout called cross-validation; protocol requires checking.'),
37: ('deferred','Dual-task B0 segmentation','Approx85% classification versus93.38% segmentation accuracy; masks/dual-task separate question.'),
38: ('priority','IncepX-Ensemble','86% original versus98% balanced extracted scores; audit balancing before interpreting increase.'),
39: ('candidate','Wavelet/residual/ELM','Preprocessing/feature classifier candidate; extra factors and split unknown.'),
40: ('candidate','S-MobileNet','Segmentation/Mish/compression; task and cohort require verification.'),
41: ('priority','Soft attention and balancing','Extracted69.75% original versus95.94% SMOTE test; balancing sequence/support unresolved.'),
42: ('priority','Multi-scale fusion and loss','Class weighting/smoothing/resampling; hair-removal and segmented datasets separate ablations.'),
43: ('candidate','CNN features with RF/LR','Cheap feature-classifier idea; split/balancing and leakage unverified.'),
44: ('priority','Five CNN comparison','Extracted single93.20% versusensemble92.83%; retain negative ensemble outcomes.'),
45: ('candidate','Optimized CNN XAI','82% extracted; loss accuracy0.47% is ambiguous and not adopted as a metric.'),
46: ('candidate','Regularly spaced shifting','TTA/ensemble candidate; extracted10050 image count conflicts with foundational10015.'),
47: ('priority','13 transfer CNN comparison','Verified abstract82.9%; dataset spelling corrected only in notes, not source.'),
48: ('deferred','GAN fuzzy ensemble','Synthesis requires extra training and artifact control; initial losses simpler.'),
49: ('candidate','DenseNet/Inception ensemble','Transfer learning context; dataset/task not established by extracted field.'),
50: ('priority','Progressive class-wise attention','Class/multiscale attention candidate; verify complete protocol before comparison.'),
51: ('candidate','AlexNet transfer learning','ISIC2018 seven-class claimed in contribution; split/balancing not established.'),
}
VERIFIED = {
3: ('https://journals.sagepub.com/doi/10.1177/20552076241312936','Methods; preprocessing; IGIA Steps1-4; Tables16/17; Threats to validity'),
6: ('https://arxiv.org/abs/1905.02793','Author abstract: methods and mean-sensitivity improvements only'),
42: ('https://www.sciencedirect.com/science/article/pii/S0010482524006796','Publisher abstract/highlights: methods, ACC/AUC and hair-removal distinction only'),
47: ('https://pubmed.ncbi.nlm.nih.gov/35808463/','Primary author abstract: thirteen models and82.9%; full split unverified'),
50: ('https://arxiv.org/abs/2306.07300','Author abstract: progressive class attention and reported scores only'),
2: ('https://www.sciencedirect.com/science/article/pii/S1746809423011904','Indexed publisher abstract: ensemble and99.53% only; fulltext retrieval failed'),
}

def main():
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    w=openpyxl.load_workbook(SOURCE,read_only=True,data_only=False)
    assert w.sheetnames==['table1']
    cells=list(w.active.rows)
    assert len(cells)==51 and len(cells[0])==8
    assert not any(c.data_type=='f' for row in cells for c in row), 'Formula source needs cached-value audit'
    headers=[c.value for c in cells[0]]
    rows=[]; raw=[]
    for n,cellrow in enumerate(cells[1:],2):
        values=[c.value or '' for c in cellrow]
        status,label,reason=DECISIONS[n]
        m=re.search(r'(10\.\d{4,9}/.+?)'+str(n-1)+r'\.\s',values[0])
        doi=m.group(1).lower() if m else ''
        clean=lambda v: str(v).replace('Save to Notebook','').strip()
        text=' '.join(map(clean,values[1:])).lower()
        tags=[]
        for tag,pattern in {'attention':r'attention|squeeze.excitation','ensemble_fusion':r'ensemble|fusion',
                'transfer_learning':r'transfer learning|pre.train|fine.tun', 'imbalance':r'imbalan|balanc|oversampl|undersampl|smote|loss weight',
                'augmentation_preprocessing':r'augment|preprocess|contrast|hair|wavelet|filter',
                'xai':r'grad.?cam|shap|lime|saliency|integrated gradients',
                'transformer':r'transformer|\bvit\b|swin','distillation':r'distill',
                'synthetic_data':r'\bgan\b|generative adversarial'}.items():
            if re.search(pattern,text): tags.append(tag)
        url,scope=VERIFIED.get(n,('','Spreadsheet extraction only; primary protocol not verified'))
        rows.append(dict(workbook_sha256=digest,sheet='table1',source_range=f'A{n}:H{n}',paper_label=label,
            doi_candidate=doi,decision=status,reason=reason,method_tags=';'.join(tags),
            extracted_dataset=clean(values[1]),extracted_methods=clean(values[3]),extracted_results=clean(values[4]),
            evaluation_comparability='not established for our locked lesion-disjoint cohort',
            primary_url=url,verification_scope=scope,verification_date='2026-10-03'))
        raw.append(dict(row=n,**dict(zip(headers,values))))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'screening.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Full original extracted text stays local; review outputs contain selected evidence.
    cache=ROOT/'.cache/research_literature';cache.mkdir(parents=True,exist_ok=True)
    (cache/'scispace_extracted.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding='utf-8')
    unique=[r for r in rows if r['decision']!='duplicate']
    summary=dict(source_path=SOURCE.relative_to(ROOT).as_posix(),sha256=digest,sheet='table1',range='A1:H51',
        entries=len(rows),distinct_doi_candidates=len({r['doi_candidate'] for r in rows}),
        duplicate_rows=[27],formula_cells=0,decisions=dict(Counter(r['decision'] for r in rows)),
        tag_counts_extracted_text_unique_entries=dict(Counter(t for r in unique for t in r['method_tags'].split(';') if t)),
        counting_caveat='Keywords across all extracted columns including generic gaps/challenges; not counts of verified implementations or evidence of efficacy.',
        raw_extraction='.cache/research_literature/scispace_extracted.json',workbook_unchanged=hashlib.sha256(SOURCE.read_bytes()).hexdigest()==digest)
    (OUT/'provenance.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
