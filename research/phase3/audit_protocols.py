"""Frozen split audit for two-stage evaluation; metadata only, never images."""
import json
import pandas as pd
from research.common import ROOT, CLASSES, SPLITS, sha256, write_json, write_csv


def main():
    strict=pd.read_csv(ROOT/SPLITS['strict_lesion_disjoint'])
    exploratory=pd.read_csv(ROOT/SPLITS['exploratory_image_level'])
    expected={'strict_lesion_disjoint':'db1ce9f8b83f176001dd89fd332fb1668c7d2ba5f58d77157d293a09e2c2e696',
              'exploratory_image_level':'75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb'}
    assert strict.query("split=='test'").equals(exploratory.query("split=='test'"))
    assert strict.image_id.is_unique and exploratory.image_id.is_unique
    assert strict.drop(columns='split').equals(exploratory.drop(columns='split'))
    summary={};rows=[]
    for protocol,frame in (('strict_lesion_disjoint',strict),('exploratory_image_level',exploratory)):
        assert sha256(ROOT/SPLITS[protocol])==expected[protocol]
        groups={s:set(frame.loc[frame.split==s,'lesion_id']) for s in ('train','val','test')}
        overlaps={a+'_'+b:len(groups[a]&groups[b]) for a,b in (('train','val'),('train','test'),('val','test'))}
        assert overlaps['train_test']==overlaps['val_test']==0
        counts={s:int((frame.split==s).sum()) for s in ('train','val','test')}
        assert counts=={'train':7009,'val':1503,'test':1503}
        affected=int(frame.loc[frame.split=='val','lesion_id'].isin(groups['train']).sum())
        if protocol=='strict_lesion_disjoint': assert overlaps['train_val']==affected==0
        else: assert overlaps['train_val']==563 and affected==596
        summary[protocol]=dict(path=SPLITS[protocol],sha256=expected[protocol],counts=counts,
            percentages={s:n*100/len(frame) for s,n in counts.items()},lesion_overlaps=overlaps,
            validation_images_with_training_lesion=affected,
            class_counts={s:{c:int(((frame.split==s)&(frame.diagnosis==c)).sum()) for c in CLASSES} for s in counts})
        rows.extend(dict(protocol=protocol,partition=s,class_name=c,count=summary[protocol]['class_counts'][s][c]) for s in counts for c in CLASSES)
    report=dict(status='verified_existing_protocols_reused',protocols=summary,test_identity_unchanged=True,
       strict_validation_images_in_exploratory_training=len(set(strict.query("split=='val'").image_id)&set(exploratory.query("split=='train'").image_id)),
       strict_confirmation_requires_fresh_training=True,new_split_created=False,test_images_opened=False,gpu_used=False)
    write_json(ROOT/'results/datasets/two_stage_protocol_audit.json',report)
    write_csv(ROOT/'results/datasets/two_stage_class_counts.csv',rows)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
