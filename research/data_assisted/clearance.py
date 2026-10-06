"""Conservative identity/label/source clearance before external image acquisition."""
import hashlib,json,re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from research.common import ROOT,CLASSES,sha256,relative,write_json,write_csv

OUT=ROOT/'results/data_assisted/clearance_v1'
REF=ROOT/'.cache/isic_duplicate_references'
SOURCE=ROOT/'results/short_screening/external_isic2019_feasibility/candidate_identity_manifest.csv'


def main():
    if (OUT/'summary.json').exists():
        print((OUT/'summary.json').read_text());return
    OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(seed=42,source_scope='BCN lesion-prefix rows of the official ISIC2019 training release only',
        label_policy='AK -> akiec as subset supervision; SCC/UNK excluded; other six matching diagnostic categories mapped directly',
        exclusions=['Every HAM image identity and canonical downsampled alias','Every known HAM lesion identity',
            'Missing lesion IDs','All downsampled copies','Non-BCN sources for this bounded stage',
            'Whole lesion group if any member occurs in pinned published duplicate-deletion name lists',
            'Whole lesion group if any member is one of four forum-confirmed historical duplicate-label cases',
            'Conflicting mapped classes within a lesion group'],
        selection='One image per retained lesion, minimum SHA256(seed42:image_id), independent of quality/model predictions',
        external_partition='Fixed stratified85/15 lesion-group external train/preflight-validation; NOT a new locked test',
        scope='Metadata/name clearance and acquisition candidate only; no training, inference, accuracy optimization or test labels',
        pixel_gate='Selected images require decode, hashes and external/HAM-development duplicate review before pilot readiness',
        caveats=['Patient identifiers unavailable; patient overlap cannot be directly audited.',
            'Published duplicate lists are incomplete and removal-order dependent, not proof that all retained images are unique.',
            'No original test pixels/labels will be inspected; unknown aliases of original test images remain unverified.',
            'AK is a subset of HAM akiec, not full equivalent coverage; external SCC includes diagnoses that must not be indiscriminately merged.'])
    write_json(OUT/'PREDECLARED_PLAN.json',plan)
    ref=json.loads((REF/'manifest.json').read_text())
    published=set();evidence=[]
    for f in ref['files']:
        assert sha256(REF/f['local_file'])==f['sha256']
        if f['repo_path'].startswith('file_lists/'):
            names=set(re.findall(r'ISIC_\d+', (REF/f['local_file']).read_text()))
            published.update(names);evidence.append(dict(**f,identity_count=len(names)))
    data=pd.read_csv(SOURCE)
    ham=pd.read_csv(ROOT/'data/raw/HAM10000/HAM10000_metadata.csv',usecols=['image_id','lesion_id'])
    ham_ids=set(ham.image_id);ham_lesions=set(ham.lesion_id.dropna())
    assert data.image_id.is_unique and set(data.class_name)<=set(CLASSES)
    data['canonical_image_id']=data.image_id.str.replace('_downsampled','',regex=False)
    data['source']=data.lesion_id.fillna('').str.split('_').str[0]
    reasons={}
    for row in data.itertuples(index=False):
        flags=[]
        if row.canonical_image_id in ham_ids:flags.append('HAM_identity_or_alias')
        if row.lesion_id in ham_lesions:flags.append('HAM_lesion')
        if pd.isna(row.lesion_id):flags.append('missing_lesion_id')
        if row.image_id.endswith('_downsampled'):flags.append('downsampled_copy')
        if row.source!='BCN':flags.append('outside_BCN_stage_scope')
        if flags:reasons[row.image_id]=flags
    bcn=data.loc[~data.image_id.isin(reasons)].copy()
    confirmed={'ISIC_0069013','ISIC_0071017','ISIC_0067980','ISIC_0067502'}
    flagged_images=bcn.loc[bcn.canonical_image_id.isin(published|confirmed)]
    flagged_groups=set(flagged_images.lesion_id)
    conflicting=bcn.groupby('lesion_id').class_name.nunique()
    flagged_groups.update(conflicting.index[conflicting>1])
    for row in bcn.loc[bcn.lesion_id.isin(flagged_groups)].itertuples(index=False):
        reasons[row.image_id]=['published_flagged_or_conflicting_lesion_group']
    eligible=bcn.loc[~bcn.lesion_id.isin(flagged_groups)].copy()
    assert not set(eligible.canonical_image_id)&ham_ids and not set(eligible.lesion_id)&ham_lesions
    assert not eligible.image_id.str.endswith('_downsampled').any()
    assert eligible.groupby('lesion_id').class_name.nunique().max()==1
    eligible['selection_hash']=[hashlib.sha256(('42:'+s).encode()).hexdigest() for s in eligible.image_id]
    selected=eligible.sort_values(['lesion_id','selection_hash']).drop_duplicates('lesion_id').reset_index(drop=True)
    assert selected.image_id.is_unique and selected.lesion_id.is_unique
    assert selected.image_id.str.fullmatch(r'ISIC_\d+').all()
    train,val=next(StratifiedShuffleSplit(n_splits=1,test_size=.15,random_state=42).split(selected,selected.class_name))
    selected['external_partition']='external_train'
    selected.loc[val,'external_partition']='external_preflight_val'
    assert not set(selected.loc[train,'lesion_id'])&set(selected.loc[val,'lesion_id'])
    columns=['image_id','class_name','lesion_id','source','selection_hash','external_partition']
    write_csv(OUT/'metadata_eligible_manifest.csv',eligible[['image_id','class_name','lesion_id','source']].to_dict('records'))
    write_csv(OUT/'selected_one_view_manifest.csv',selected[columns].to_dict('records'))
    write_csv(OUT/'excluded_candidates.csv',[dict(image_id=i,reasons=';'.join(r)) for i,r in sorted(reasons.items())])
    write_csv(OUT/'published_name_hits.csv',flagged_images[['image_id','class_name','lesion_id']].to_dict('records'))
    rows=[]
    for cl in CLASSES:
        s=selected.loc[selected.class_name==cl]
        rows.append(dict(class_name=cl,eligible_images=int((eligible.class_name==cl).sum()),selected_lesions=len(s),
            external_train=int((s.external_partition=='external_train').sum()),external_preflight_val=int((s.external_partition=='external_preflight_val').sum())))
    write_csv(OUT/'class_counts.csv',rows)
    write_json(OUT/'source_manifest.json',dict(candidate_manifest=relative(SOURCE),candidate_sha256=sha256(SOURCE),
        official_ground_truth_sha256=sha256(ROOT/'.cache/external_isic2019_metadata/GroundTruth.csv'),
        official_metadata_sha256=sha256(ROOT/'.cache/external_isic2019_metadata/Metadata.csv'),
        duplicate_reference_repository=ref['repository'],duplicate_reference_commit=ref['commit'],duplicate_lists=evidence,
        sources={'HAM_category_definitions':'https://www.nature.com/articles/sdata2018161',
            'ISIC_label_subdivision':'https://forum.isic-archive.com/t/question-re-new-classification-category/1074',
            'Historical_duplicate_label_cases':'https://forum.isic-archive.com/t/a-list-of-duplicate-images-in-the-training-set/1141'},
        selected_manifest_sha256=sha256(OUT/'selected_one_view_manifest.csv'),class_order=list(CLASSES)))
    summary=dict(status='metadata_and_published_name_clearance_complete_pixel_gate_pending',candidate_images=len(data),
        initial_BCN_images=len(bcn),published_name_direct_hits=int(bcn.canonical_image_id.isin(published).sum()),
        confirmed_case_direct_hits=int(bcn.canonical_image_id.isin(confirmed).sum()),flagged_lesion_groups=len(flagged_groups),
        whole_group_images_excluded=int(bcn.lesion_id.isin(flagged_groups).sum()),eligible_BCN_images=len(eligible),
        selected_images=len(selected),selected_lesion_groups=len(selected),external_train_images=len(train),external_preflight_val_images=len(val),
        class_counts=rows,known_HAM_identity_overlap=0,known_HAM_lesion_overlap=0,external_partition_lesion_overlap=0,
        training_launched=False,gpu_used=False,test_loaded=False,ready_for_GPU_pilot=False,
        selected_manifest_sha256=sha256(OUT/'selected_one_view_manifest.csv'),limitations=plan['caveats'])
    write_json(OUT/'verification.json',dict(status='passed',source_hashes_checked=True,known_HAM_identity_and_lesion_overlap=0,
        no_downsampled_entries=True,one_image_per_lesion=True,no_external_partition_overlap=True,all_seven_classes_present=True,
        published_flagged_groups_excluded=True,test_loaded=False))
    write_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
