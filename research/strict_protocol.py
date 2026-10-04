"""Frozen strict development data. Test labels/images are never consumed."""
import csv
import json
import numpy as np
import pandas as pd
import torch
from .common import ROOT, CLASSES, sha256, write_json

SPLIT='data/splits/split_assignments.csv'
DIGEST='db1ce9f8b83f176001dd89fd332fb1668c7d2ba5f58d77157d293a09e2c2e696'
OUT=ROOT/'results/final_strict/v1'

def partition_metadata(root=ROOT):
    path=root/SPLIT
    if sha256(path)!=DIGEST: raise ValueError('Frozen strict manifest changed')
    # First read only identities/partition, including test identities for overlap.
    identities=pd.read_csv(path,usecols=['image_id','lesion_id','split'])
    if identities.isna().any().any() or identities.image_id.duplicated().any():
        raise ValueError('Invalid identities')
    counts=identities.split.value_counts().to_dict()
    if counts!={'train':7009,'val':1503,'test':1503}: raise ValueError('Unexpected strict cohort')
    lesions={s:set(identities.loc[identities.split==s,'lesion_id']) for s in counts}
    overlaps={f'{a}_{b}':len(lesions[a]&lesions[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    if any(overlaps.values()): raise ValueError('Strict lesion overlap')
    # Exclude test rows before parsing the diagnosis/label columns.
    excluded=set((identities.index[identities.split=='test']+1).tolist())
    dev=pd.read_csv(path,skiprows=lambda line:line in excluded)
    if not set(dev.split).issubset({'train','val'}): raise ValueError('Test leaked into development')
    if dev.isna().any().any() or not (dev.label==dev.diagnosis.map(dict(zip(CLASSES,range(7))))).all():
        raise ValueError('Development class mapping changed')
    report=dict(protocol='strict_lesion_disjoint',split_manifest=SPLIT,split_sha256=DIGEST,
        image_counts=counts,lesion_counts={s:len(v) for s,v in lesions.items()},lesion_overlaps=overlaps,
        test_labels_read=False,test_images_loaded=False,test_loader_instantiated=False)
    return dev,report

def development_data(config,root=ROOT):
    if config['protocol']!='strict_lesion_disjoint' or config['split_manifest']!=SPLIT or config['split_sha256']!=DIGEST:
        raise ValueError('Only frozen strict protocol allowed')
    dev,_=partition_metadata(root)
    files={}
    for folder in ('HAM10000_images_part_1','HAM10000_images_part_2'):
        directory=root/'data/raw/HAM10000'/folder
        if not directory.is_dir(): raise FileNotFoundError(directory)
        for p in directory.glob('*.jpg'):
            if p.stem in files: raise ValueError('Duplicate image filename')
            files[p.stem]=p
    dev['path']=[str(files[i]) for i in dev.image_id]
    # Verify only development metadata; no raw test labels are read.
    rawpath=root/'data/raw/HAM10000/HAM10000_metadata.csv'
    rawids=pd.read_csv(rawpath,usecols=['image_id'])
    dev_ids=set(dev.image_id)
    excluded=set((rawids.index[~rawids.image_id.isin(dev_ids)]+1).tolist())
    raw=pd.read_csv(rawpath,skiprows=lambda line:line in excluded,usecols=['image_id','lesion_id','dx'])
    raw=raw.set_index('image_id').rename(columns={'dx':'diagnosis'})
    if not raw.sort_index().equals(dev.set_index('image_id')[['lesion_id','diagnosis']].sort_index()):
        raise ValueError('Development metadata mismatch')
    train=dev.loc[dev.split=='train'].reset_index(drop=True)
    val=dev.loc[dev.split=='val'].reset_index(drop=True)
    counts=np.array([(train.label==i).sum() for i in range(7)])
    if (counts==0).any(): raise ValueError('Missing training class')
    weights=np.sqrt(len(train)/(7*counts));weights/=weights.mean()
    return train,val,torch.tensor(weights,dtype=torch.float32)

def verification():
    _,report=partition_metadata()
    write_json(OUT/'split_verification.json',report)
    return report
