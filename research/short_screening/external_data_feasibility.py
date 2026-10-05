"""Metadata-only feasibility audit; excludes all HAM identities before parsing labels."""
import io,json,urllib.request
import numpy as np
import pandas as pd
from research.common import ROOT,CLASSES,sha256,relative,write_json,write_csv

OUT=ROOT/'results/short_screening/external_isic2019_feasibility'
CACHE=ROOT/'.cache/external_isic2019_metadata'
BASE='https://isic-archive.s3.amazonaws.com/challenges/2019/'
URLS={n:BASE+f'ISIC_2019_Training_{n}.csv' for n in ['GroundTruth','Metadata']}


def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    ham=pd.read_csv(ROOT/'data/raw/HAM10000/HAM10000_metadata.csv',usecols=['image_id','lesion_id'])
    assert len(ham)==10015 and ham.image_id.is_unique
    ham_ids=set(ham.image_id);ham_lesions=set(ham.lesion_id.dropna())
    for name,url in URLS.items():
        path=CACHE/f'{name}.csv'
        if not path.exists():
            with urllib.request.urlopen(url,timeout=45) as response:content=response.read(5_000_001)
            assert 0<len(content)<5_000_000
            path.write_bytes(content)
    gt=CACHE/'GroundTruth.csv';mp=CACHE/'Metadata.csv'
    identities=pd.read_csv(gt,usecols=['image'])
    metadata=pd.read_csv(mp,usecols=['image','lesion_id'])
    assert identities.image.is_unique and metadata.image.is_unique
    assert set(identities.image)==set(metadata.image) and len(identities)==25331
    excluded=set(metadata.loc[metadata.image.isin(ham_ids)|metadata.lesion_id.isin(ham_lesions),'image'])
    excluded.update(ham_ids)
    skip=set((identities.index[identities.image.isin(excluded)]+1).tolist())
    # Diagnosis values for HAM rows, including the original test cohort, are skipped.
    external=pd.read_csv(gt,skiprows=lambda line:line in skip)
    assert not set(external.image)&ham_ids
    label_columns=['MEL','NV','BCC','AK','BKL','DF','VASC','SCC','UNK']
    available=[c for c in label_columns if c in external.columns]
    assert {'MEL','NV','BCC','AK','BKL','DF','VASC','SCC'}<=set(available)
    a=external[available].to_numpy();assert np.isin(a,[0,1]).all() and (a.sum(1)==1).all()
    external['diagnosis_code']=external[available].idxmax(axis=1)
    mapping={'AK':'akiec','BCC':'bcc','BKL':'bkl','DF':'df','MEL':'mel','NV':'nv','VASC':'vasc'}
    eligible=external.loc[external.diagnosis_code.isin(mapping)].copy()
    eligible['class_name']=eligible.diagnosis_code.map(mapping)
    eligible=eligible.merge(metadata,on='image',validate='one_to_one')
    assert not set(eligible.lesion_id.dropna())&ham_lesions
    # Existing exploratory training labels only, excluding non-train rows before parsing.
    manifest=ROOT/'data/splits/exploratory/image_level_dev_v1.csv'
    split_ids=pd.read_csv(manifest,usecols=['image_id','split'])
    not_train=set((split_ids.index[split_ids.split!='train']+1).tolist())
    train=pd.read_csv(manifest,skiprows=lambda line:line in not_train)
    counts=[]
    for cl in CLASSES:
        current=int((train.diagnosis==cl).sum());additional=int((eligible.class_name==cl).sum())
        known=int(((eligible.class_name==cl)&eligible.lesion_id.notna()).sum())
        groups=int(eligible.loc[eligible.class_name==cl,'lesion_id'].nunique())
        counts.append(dict(class_name=cl,current_ham_training_images=current,current_ham_training_lesions=int(train.loc[train.diagnosis==cl,'lesion_id'].nunique()),
            external_candidate_images=additional,known_lesion_id_candidates=known,known_external_lesion_groups=groups,
            candidate_to_current_ratio=additional/current))
    write_csv(OUT/'candidate_class_counts.csv',counts)
    write_csv(OUT/'candidate_identity_manifest.csv',eligible[['image','class_name','lesion_id']].rename(columns={'image':'image_id'}).to_dict('records'))
    known=eligible.dropna(subset=['lesion_id'])
    by_group=known.groupby('lesion_id').class_name.nunique()
    summary=dict(status='metadata_feasibility_only',source='https://challenge.isic-archive.com/data/',training_csv_urls=URLS,
        source_hashes={name:sha256(CACHE/f'{name}.csv') for name in URLS},source_training_rows=len(identities),
        ham_identity_rows_excluded=len(skip),external_rows_after_identity_exclusion=len(external),
        unsupported_diagnosis_rows_excluded=len(external)-len(eligible),candidate_images=len(eligible),
        missing_lesion_ids=int(eligible.lesion_id.isna().sum()),known_lesion_id_candidates=int(eligible.lesion_id.notna().sum()),
        known_external_lesion_groups=int(len(by_group)),groups_with_conflicting_candidate_classes=int((by_group>1).sum()),
        maximum_images_per_known_lesion=int(known.groupby('lesion_id').size().max()),candidate_class_counts=counts,
        images_downloaded=False,training_launched=False,test_loaded=False,ham_label_columns_loaded=False,
        identity_overlap_with_all_ham=0,known_lesion_id_overlap_with_all_ham=0,
        limitations=['Image aliases/near duplicates and patient overlap not yet audited; metadata exclusion alone is insufficient clearance.',
            'AK to project akiec terminology requires documented mapping review; SCC and unknown are excluded.',
            'Different acquisition sources and label distributions; domain-shift or improvement not yet measured.',
            'This is not an accuracy experiment and cannot forecast final test accuracy.'],
        decision='Candidate for a separately scoped de-duplicated external-training feasibility stage; no automatic image download or GPU launch')
    write_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
