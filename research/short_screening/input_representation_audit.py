"""Bounded CPU development audit. Proxy measurements are not lesion masks."""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy import ndimage

from research.common import ROOT, CLASSES
from research.strict_protocol import development_data

OUT = ROOT/'results/short_screening/input_representation_audit_v1'


def measurements(row):
    with Image.open(row.path) as source:
        image = source.convert('RGB').resize((200, 150), Image.Resampling.BILINEAR)
    gray = np.asarray(image.convert('L'), dtype=float)/255
    yy, xx = np.mgrid[:150, :200]
    inner = ((xx-99.5)/80)**2 + ((yy-74.5)/60)**2 < 1
    edge = ~inner
    # Otsu intensity threshold: descriptive dark-region proxy, not segmentation.
    hist = np.histogram(gray[inner], bins=256, range=(0, 1))[0].astype(float)
    weight = hist.cumsum(); moment = (hist*np.arange(256)).cumsum()
    denom = weight*(weight[-1]-weight)
    between = np.divide((moment[-1]*weight-moment*weight[-1])**2, denom,
                        out=np.zeros(256), where=denom>0)
    threshold = between.argmax()/255
    mask = (gray < threshold) & inner
    labels, count = ndimage.label(mask)
    areas = np.bincount(labels.ravel()); areas[0] = 0
    candidates = np.unique(labels[55:95, 70:130]); candidates = candidates[candidates>0]
    chosen = max(candidates, key=lambda i: areas[i]) if len(candidates) else 0
    area = float(areas[chosen]/inner.sum()) if chosen else 0.
    blurred = np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(1)), dtype=float)/255
    return dict(image_id=row.image_id, lesion_id=row.lesion_id, diagnosis=row.diagnosis,
                split=row.split, dark_region_fraction_proxy=area,
                proxy_usable=bool(.01 < area < .75),
                dark_border_fraction=float((gray[edge]<.12).mean()),
                high_frequency_energy=float(np.mean((gray[inner]-blurred[inner])**2)),
                brightness=float(np.median(gray[inner])),
                contrast=float(np.std(gray[inner])))


def main():
    config=json.loads((ROOT/'research/short_screening/targeted_finetune_v1.json').read_text())
    train,val,_=development_data(config)
    with ThreadPoolExecutor(max_workers=4) as pool:
        measured=pd.DataFrame(pool.map(measurements,pd.concat([train,val]).itertuples(index=False)))
    probabilities=[]
    for rid in ['s28_efficientnet_b0_final_strict_seed42','s29_convnext_tiny_final_strict_seed42','s30_efficientnet_v2_s_final_strict_seed42']:
        frame=pd.read_csv(ROOT/'results/structured_experiments'/rid/'validation_predictions.csv').set_index('image_id').loc[val.image_id]
        assert frame.true_class.tolist()==val.diagnosis.tolist()
        probabilities.append(frame[[f'p_{c}' for c in CLASSES]].to_numpy())
    p=np.mean(probabilities,axis=0)
    outcomes=pd.DataFrame(dict(image_id=val.image_id,predicted_class=np.asarray(CLASSES)[p.argmax(1)],confidence=p.max(1)))
    measured=measured.merge(outcomes,on='image_id',how='left')
    measured['correct']=measured.diagnosis.eq(measured.predicted_class)
    summaries=[]
    for cl in CLASSES:
        tr=measured[(measured.split=='train')&(measured.diagnosis==cl)]
        va=measured[(measured.split=='val')&(measured.diagnosis==cl)]
        for feature in ['dark_region_fraction_proxy','high_frequency_energy','dark_border_fraction','contrast']:
            t=tr[tr.proxy_usable] if feature=='dark_region_fraction_proxy' else tr
            v=va[va.proxy_usable] if feature=='dark_region_fraction_proxy' else va
            threshold=float(t[feature].quantile(.25))
            lower=v[feature]<=threshold; upper=~lower
            summaries.append(dict(diagnosis=cl,feature=feature,threshold_source='training within-class 25th percentile',
                                  threshold=threshold,low_n=int(lower.sum()),other_n=int(upper.sum()),
                                  low_accuracy=float(v.loc[lower,'correct'].mean()),
                                  other_accuracy=float(v.loc[upper,'correct'].mean()),
                                  low_minus_other_pp=float(100*(v.loc[lower,'correct'].mean()-v.loc[upper,'correct'].mean()))))
    OUT.mkdir(parents=True,exist_ok=True)
    measured.to_csv(OUT/'development_image_features.csv',index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'class_conditioned_associations.csv',index=False)
    report=dict(train_images=len(train),validation_images=len(val),test_images_loaded=False,gpu_used=False,
                proxy_valid_fraction=float(measured.proxy_usable.mean()),
                caveats=['Dark connected intensity region is not a verified lesion mask; excludes many light/red lesions and may capture hair/artifacts.',
                         'Quartiles fixed from training separately for each class; associations do not identify causes or training benefits.',
                         'Repeated images per lesion and small subgroups limit precision; no p-values or significance claims.',
                         'Audit covers already-used strict validation; no independent evaluation created.'])
    (OUT/'audit_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print(summary[summary.diagnosis.isin(['mel','bkl'])].to_string(index=False))


if __name__=='__main__':main()
