"""CPU pixel-integrity/duplicate quarantine; original test images never opened."""
import argparse,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from research.common import ROOT,CLASSES,sha256,relative,write_json,write_csv

OUT=ROOT/'results/data_assisted/clearance_v1'
CACHE=ROOT/'.cache/data_assisted_clearance_v1'
HAM_CACHE=CACHE/'ham_development_fingerprints.csv'


def fingerprint(row):
    path=ROOT/row.path;data=path.read_bytes()
    byte_hash=hashlib.sha256(data).hexdigest()
    if hasattr(row,'sha256'):assert byte_hash==row.sha256
    with Image.open(path) as image:
        rgb=image.convert('RGB');width,height=rgb.size
        pixel_hash=hashlib.sha256(f'{width}x{height}|RGB'.encode()+rgb.tobytes()).hexdigest()
        grey=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
        block=dctn(grey,norm=None)[:8,:8];bits=(block>np.median(block)).ravel()
        phash=sum(int(b)<<(63-i) for i,b in enumerate(bits))
    return dict(image_id=row.image_id,path=row.path,byte_sha256=byte_hash,pixel_sha256=pixel_hash,
        phash64=f'{phash:016x}',width=width,height=height,grey_std=float(grey.std()))


def ham_fingerprints():
    CACHE.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'data/splits/split_assignments.csv'
    # No diagnosis columns, labels, or original test pixels enter this reader.
    ids=pd.read_csv(manifest,usecols=['image_id','split'])
    dev=ids.loc[ids.split!='test'].copy();test_ids=set(ids.loc[ids.split=='test','image_id'])
    assert len(dev)==8512 and dev.image_id.is_unique
    if HAM_CACHE.exists():
        meta=json.loads((CACHE/'ham_development_cache.json').read_text())
        assert meta['split_sha256']==sha256(manifest)
        assert meta['fingerprints_sha256']==sha256(HAM_CACHE)
        saved=pd.read_csv(HAM_CACHE,dtype={'phash64':str})
        assert set(saved.image_id)==set(dev.image_id) and not set(saved.image_id)&test_ids
        return saved
    paths={p.stem:p for part in ['HAM10000_images_part_1','HAM10000_images_part_2']
           for p in (ROOT/'data/raw/HAM10000'/part).glob('*.jpg')}
    dev['path']=[relative(paths[i]) for i in dev.image_id]
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(fingerprint,dev.itertuples(index=False)))
    write_csv(HAM_CACHE,records)
    write_json(CACHE/'ham_development_cache.json',dict(images=len(records),split_sha256=sha256(manifest),
        fingerprints_sha256=sha256(HAM_CACHE),runtime_seconds=time.perf_counter()-start,test_loaded=False))
    print(f'Cached {len(records)} development-only fingerprints in {time.perf_counter()-start:.1f}s',flush=True)
    return pd.DataFrame(records)


def bitcount(a):
    if hasattr(np,'bitwise_count'):return np.bitwise_count(a)
    table=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    return table[a.view(np.uint8).reshape(-1,8)].sum(1)


def main(cache_only=False):
    if not cache_only and (OUT/'pixel_clearance_summary.json').exists():
        print((OUT/'pixel_clearance_summary.json').read_text());return
    ham=ham_fingerprints()
    if cache_only:return
    acquisition=json.loads((OUT/'acquisition_summary.json').read_text());assert acquisition['status']=='completed'
    selected=pd.read_csv(OUT/'selected_one_view_manifest.csv')
    acquired=pd.read_csv(OUT/'acquired_image_manifest.csv')
    assert set(acquired.image_id)==set(selected.image_id) and len(acquired)==acquisition['expected_images']
    plan=dict(phash_bits=64,resize_grey=32,near_duplicate_flag_hamming_max=4,
        exact='Byte SHA256 or full decoded RGB plus dimensions SHA256',
        policy='Conservatively quarantine every selected external image with an exact or near-pHash flag; do not relabel cases or call hash collisions confirmed duplicates.',
        scope='Every selected external image versus8512 HAM development images, and within selected external set.',
        original_test_images_opened=False,original_test_labels_loaded=False,
        limitation='This cannot audit unknown aliases of original test images or patient identity; published name/source checks are separate.')
    write_json(OUT/'PIXEL_GATE_PLAN.json',plan)
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(fingerprint,acquired.itertuples(index=False)))
    external=pd.DataFrame(records).sort_values('image_id').reset_index(drop=True)
    write_csv(OUT/'external_image_fingerprints.csv',external.to_dict('records'))
    hhash=np.array([int(s,16) for s in ham.phash64],dtype=np.uint64)
    ehash=np.array([int(s,16) for s in external.phash64],dtype=np.uint64)
    hb=dict(zip(ham.byte_sha256,ham.image_id));hp=dict(zip(ham.pixel_sha256,ham.image_id))
    flags=[];nearest=[];quarantine=set()
    for i,row in enumerate(external.itertuples(index=False)):
        distances=bitcount(np.bitwise_xor(hhash,ehash[i]));j=int(distances.argmin());distance=int(distances[j])
        nearest.append(dict(external_image_id=row.image_id,nearest_HAM_development_image_id=ham.iloc[j].image_id,phash_hamming=distance))
        if row.byte_sha256 in hb or row.pixel_sha256 in hp:
            exact_id=hb.get(row.byte_sha256,hp.get(row.pixel_sha256))
            flags.append(dict(image_id=row.image_id,other_image_id=exact_id,scope='HAM_development',kind='exact_content',phash_hamming=distance));quarantine.add(row.image_id)
        elif distance<=4:
            flags.append(dict(image_id=row.image_id,other_image_id=ham.iloc[j].image_id,scope='HAM_development',kind='near_hash_candidate_not_confirmed',phash_hamming=distance));quarantine.add(row.image_id)
        distances=bitcount(np.bitwise_xor(ehash[i+1:],ehash[i]))
        for local in np.flatnonzero(distances<=4):
            k=i+1+int(local);other=external.iloc[k]
            exact=row.byte_sha256==other.byte_sha256 or row.pixel_sha256==other.pixel_sha256
            flags.append(dict(image_id=row.image_id,other_image_id=other.image_id,scope='selected_external',
                kind='exact_content' if exact else 'near_hash_candidate_not_confirmed',phash_hamming=int(distances[local])))
            quarantine.update([row.image_id,other.image_id])
    # Exact matches are caught even if an implementation/numerical hash difference exists.
    for column in ['byte_sha256','pixel_sha256']:
        duplicate=external.loc[external[column].duplicated(keep=False)]
        quarantine.update(duplicate.image_id)
    write_csv(OUT/'nearest_HAM_development_hash.csv',nearest)
    write_csv(OUT/'duplicate_candidate_flags.csv',flags if flags else [dict(image_id='',other_image_id='',scope='',kind='none',phash_hamming='')])
    approved=selected.loc[~selected.image_id.isin(quarantine)].merge(acquired[['image_id','path','sha256','width','height','mode']],on='image_id',validate='one_to_one')
    assert approved.image_id.is_unique and approved.lesion_id.is_unique and set(approved.class_name)==set(CLASSES)
    assert not set(approved.image_id)&set(ham.image_id)
    write_csv(OUT/'pixel_screened_candidate_manifest.csv',approved.to_dict('records'))
    write_csv(OUT/'quarantined_images.csv',[dict(image_id=i,reason='conservative_duplicate_candidate_quarantine') for i in sorted(quarantine)] or [dict(image_id='',reason='none')])
    counts=[]
    for cl in CLASSES:
        a=approved.loc[approved.class_name==cl]
        counts.append(dict(class_name=cl,images_and_lesions=len(a),external_train=int((a.external_partition=='external_train').sum()),
            external_preflight_val=int((a.external_partition=='external_preflight_val').sum())))
    write_csv(OUT/'pixel_screened_class_counts.csv',counts)
    result=dict(status='bounded_pixel_and_published_name_screen_complete',selected_images=len(selected),decoded_images=len(external),
        HAM_development_images_compared=len(ham),flagged_pairs=len(flags),quarantined_images=len(quarantine),
        retained_images=len(approved),retained_lesion_groups=len(approved),class_counts=counts,
        original_test_pixels_compared=False,test_loaded=False,gpu_used=False,training_launched=False,
        patient_overlap_verified=False,unknown_original_test_aliases_verified=False,
        ready_for_bounded_development_screen=True,not_a_certificate_of_final_test_independence=True,
        candidate_manifest_sha256=sha256(OUT/'pixel_screened_candidate_manifest.csv'),
        development_fingerprints_sha256=sha256(HAM_CACHE),runtime_seconds=time.perf_counter()-start)
    write_json(OUT/'pixel_clearance_summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache-development-only',action='store_true');args=parser.parse_args()
    main(args.cache_development_only)
