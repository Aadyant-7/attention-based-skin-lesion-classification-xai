"""Descriptive source/class composition only: no new model or search."""
import json
import numpy as np
import pandas as pd
from research.common import ROOT,CLASSES,write_csv,write_json
from research.short_screening.convnextv2_transfer import config,development


def main():
    out=ROOT/'results/data_assisted/s66_s67_frozen_data_screen'
    s=json.loads((out/'summary.json').read_text())
    train,_,_=development(config())
    ext=pd.read_csv(ROOT/'results/data_assisted/clearance_v1/pixel_screened_candidate_manifest.csv')
    ext=ext.loc[ext.external_partition=='external_train'];rows=[]
    for i,cl in enumerate(CLASSES):
        ham=train.loc[train.label==i];external=ext.loc[ext.class_name==cl]
        rows.append(dict(class_name=cl,ham_train_images=len(ham),ham_train_lesions=ham.lesion_id.nunique(),
            external_train_images_and_lesions=len(external),ham_train_fraction=len(ham)/len(train),
            external_train_fraction=len(external)/len(ext),combined_train_fraction=(len(ham)+len(external))/(len(train)+len(ext))))
    write_csv(out/'source_class_composition.csv',rows)
    write_json(out/'source_shift_audit.json',dict(kind='descriptive_not_causal',new_model_fitted=False,test_loaded=False,
        ham_validation_accuracy_control=s['s66']['metrics']['ham_only']['accuracy'],
        ham_validation_accuracy_candidate=s['s66']['metrics']['ham_plus_external']['accuracy'],
        external_preflight_accuracy_control=s['s66']['external_preflight_metrics']['ham_only']['accuracy'],
        external_preflight_accuracy_candidate=s['s66']['external_preflight_metrics']['ham_plus_external']['accuracy'],
        interpretation='Candidate learns external cases better but loses HAM accuracy and melanoma recall; consistent with domain/composition/label-coverage shift, not proof of one causal mechanism.',
        inference_limit='This rejects the fixed pooled frozen-feature recipe, not every possible source-adaptation/fine-tuning method. No alternative source weighting or subset search was performed.'))
    path=ROOT/'research/data_assisted/S66_S67_CLOSEOUT.md'
    report=path.read_text(encoding='utf-8')
    marker='## Descriptive source-shift audit'
    section='\n\n'+marker+'\n\n'
    section+='The added pool differs substantially in class composition and source. Raw HAM image counts also include repeated lesion views, whereas the external pool uses one view per lesion. The same class-weight formula is recomputed on each training set; class weights and effective regularization/data mass therefore change with the added cohort. This is a fixed data-recipe comparison, not proof that sample count alone caused the change.\n\n'
    section+='|Class|HAM train images / lesions|External train lesions|HAM image share|External share|Combined share|\n|---|---:|---:|---:|---:|---:|\n'
    section+='\n'.join(f"|{r['class_name']}|{r['ham_train_images']} / {r['ham_train_lesions']}|{r['external_train_images_and_lesions']}|{100*r['ham_train_fraction']:.2f}%|{100*r['external_train_fraction']:.2f}%|{100*r['combined_train_fraction']:.2f}%|" for r in rows)
    section+='\n\nLearning external cases while harming HAM validation and all three HAM group folds is consistent with source/domain and case-composition shift. It does not establish one exact cause, and frozen features cannot rule out all future adaptation methods. The practical decision is to avoid a speculative GPU run of this rejected pooled recipe. Audit CSV/JSON are saved beside the results.\n'
    if marker not in report:path.write_text(report+section,encoding='utf-8')
    print(json.dumps(dict(status='completed',new_model_fitted=False,test_loaded=False)))


if __name__=='__main__':main()
