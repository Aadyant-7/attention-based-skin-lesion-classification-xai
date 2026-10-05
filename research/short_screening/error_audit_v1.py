"""Descriptive audit of S53 errors; no prediction changes, training or test access."""
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont,ImageOps
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.common import ROOT,CLASSES,relative,sha256,write_json,write_csv

OUT=ROOT/'results/short_screening/s62_s53_error_audit'
REF=ROOT/'results/short_screening/s53_equal_five_b0_addition'
FOUR=ROOT/'results/short_screening/s46_all_f1_checkpoint_fusion'
NAMES=['ConvNeXt-Tiny','ConvNeXt-Small','DenseNet201','EfficientNetV2-S','EfficientNet-B0']


def ece(y,p):
    conf=p.max(1);ok=p.argmax(1)==y;value=0.;bins=[]
    for j in range(10):
        mask=(conf>=j/10)&(conf<(j+1)/10 if j<9 else conf<=1)
        if not mask.any():continue
        a=float(ok[mask].mean());c=float(conf[mask].mean());value+=mask.mean()*abs(a-c)
        bins.append(dict(lower=j/10,upper=(j+1)/10,images=int(mask.sum()),accuracy=a,confidence=c))
    return float(value),bins


def savefig(fig,name):
    folder=OUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(folder/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(folder/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def main():
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    OUT.mkdir(parents=True,exist_ok=False)
    write_json(OUT/'AUDIT_SCOPE.json',dict(reference=relative(REF),class_order=list(CLASSES),
        purpose='Descriptive audit of errors, model outputs, image quality and preprocessing; no new method evaluation',
        constraints='Validation images only; no label changes, quality-based exclusion, threshold tuning or GPU work',
        quality='Pixel brightness, contrast, square-resized Laplacian variance and dark border fraction are proxies, not clinical diagnoses',
        image_selection='All96errors contact sheets; preprocessing sheet uses first8errors sorted by confidence',
        uncertainty='Class/quality/metadata associations can be confounded; this validation cohort has already been used for model selection'))
    ref=pd.read_csv(REF/'validation_predictions.csv');assert len(ref)==1503 and ref.image_id.is_unique
    ids=ref.image_id.to_numpy();y=ref.true_class.map(dict(zip(CLASSES,range(7)))).to_numpy()
    manifest=ROOT/'data/splits/exploratory/image_level_dev_v1.csv'
    identity=pd.read_csv(manifest,usecols=['image_id','lesion_id','split'])
    excluded=set((identity.index[identity.split=='test']+1).tolist())
    dev=pd.read_csv(manifest,skiprows=lambda line:line in excluded)
    assert set(dev.split)=={'train','val'}
    val=dev.loc[dev.split=='val'].set_index('image_id').loc[ids]
    assert val.diagnosis.tolist()==ref.true_class.tolist()
    train=dev.loc[dev.split=='train']
    strict=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','lesion_id','split'])
    test=strict.loc[strict.split=='test']
    assert not set(ids)&set(test.image_id) and not set(val.lesion_id)&set(test.lesion_id)
    metadata_path=ROOT/'data/raw/HAM10000/HAM10000_metadata.csv'
    metadata_ids=pd.read_csv(metadata_path,usecols=['image_id'])
    skip=set((metadata_ids.index[~metadata_ids.image_id.isin(set(ids))]+1).tolist())
    meta=pd.read_csv(metadata_path,skiprows=lambda line:line in skip,usecols=['image_id','dx_type','age','sex','localization']).set_index('image_id').loc[ids]
    sources=json.loads((FOUR/'candidate_manifest.json').read_text())['sources']
    b0='s03_efficientnet_b0_none_exploratory_seed42';b0path=ROOT/'results/structured_experiments'/b0/'validation_predictions.csv'
    sources.append(dict(run=b0,prediction_file=relative(b0path),prediction_sha256=sha256(b0path)))
    arrays=[];model_rows=[];recipes=[]
    for name,s in zip(NAMES,sources):
        path=ROOT/s['prediction_file'];assert sha256(path)==s['prediction_sha256']
        f=pd.read_csv(path);assert f.image_id.is_unique and set(f.image_id)==set(ids)
        f=f.set_index('image_id').loc[ids];assert f.true_class.tolist()==ref.true_class.tolist()
        p=f[[f'p_{c}' for c in CLASSES]].to_numpy();assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        arrays.append(p);a=p.argmax(1)==y;ec,bins=ece(y,p)
        model_rows.append(dict(model=name,run=s['run'],accuracy=float(a.mean()),mean_confidence=float(p.max(1).mean()),
            wrong_mean_confidence=float(p.max(1)[~a].mean()),nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
            ece10=ec,correct_on_reference_errors=0))
        cfg=json.loads((ROOT/'results/structured_experiments'/s['run']/'config.json').read_text())
        recipes.append(dict(run=s['run'],**{k:cfg.get(k) for k in ['image_size','preprocessing','normalization_mean','normalization_std','augmentation','loss','imbalance','learning_rate']}))
        assert cfg['image_size']==224 and cfg['normalization_mean']==[.485,.456,.406] and cfg['normalization_std']==[.229,.224,.225]
    stack=np.stack(arrays);p=stack.mean(0)
    np.testing.assert_allclose(p,ref[[f'p_{c}' for c in CLASSES]],atol=1e-12,rtol=0)
    pred=p.argmax(1);err=pred!=y;assert err.sum()==96
    votes=stack.argmax(2);counts=np.eye(7)[votes].sum(0);agreement=counts.max(1);ncorrect=(votes==y).sum(0)
    confidence=p.max(1);sorted_p=np.sort(p,axis=1);margin=sorted_p[:,-1]-sorted_p[:,-2]
    allwrong=(ncorrect==0);unanimous=(agreement==5)
    ec,bins=ece(y,p);write_csv(OUT/'ensemble_reliability.csv',bins)
    for i,row in enumerate(model_rows):row['correct_on_reference_errors']=int(((votes[i]==y)&err).sum())
    write_csv(OUT/'model_confidence.csv',model_rows);write_json(OUT/'preprocessing_recipes.json',recipes)
    rows=[];files={}
    for part in ['HAM10000_images_part_1','HAM10000_images_part_2']:
        for path in (ROOT/'data/raw/HAM10000'/part).glob('*.jpg'):
            if path.stem in set(ids):assert path.stem not in files;files[path.stem]=path
    assert len(files)==1503
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    image_hashes={};thumbs={};model_inputs={}
    for i,image_id in enumerate(ids):
        path=files[image_id]
        with Image.open(path) as im:
            im.load();width,height=im.size;rgb=im.convert('RGB');resized=rgb.resize((224,224),Image.Resampling.BILINEAR)
            gray=np.asarray(resized.convert('L'),dtype=np.float32)
            lap=-4*gray[1:-1,1:-1]+gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]
            edge=np.concatenate([gray[:8].ravel(),gray[-8:].ravel(),gray[:, :8].ravel(),gray[:,-8:].ravel()])
            if err[i]:thumbs[image_id]=ImageOps.contain(rgb,(224,168));model_inputs[image_id]=resized
        image_hashes[image_id]=sha256(path)
        row=dict(image_id=image_id,lesion_id=str(val.iloc[i].lesion_id),true_class=CLASSES[y[i]],predicted_class=CLASSES[pred[i]],correct=bool(not err[i]),
            confidence=float(confidence[i]),margin=float(margin[i]),true_probability=float(p[i,y[i]]),entropy=float(-(p[i]*np.log(np.maximum(p[i],1e-12))).sum()),
            max_vote_agreement=int(agreement[i]),members_correct=int(ncorrect[i]),all_members_wrong=bool(allwrong[i]),unanimous=bool(unanimous[i]),
            train_lesion_shared=bool(val.iloc[i].lesion_id in set(train.lesion_id)),width=width,height=height,aspect_ratio=width/height,
            brightness=float(gray.mean()),contrast=float(gray.std()),laplacian_variance=float(lap.var()),dark_border_fraction=float((edge<10).mean()),
            dx_type=str(meta.iloc[i].dx_type),localization=str(meta.iloc[i].localization),age=None if pd.isna(meta.iloc[i].age) else float(meta.iloc[i].age),sex=str(meta.iloc[i].sex),image_path=relative(path))
        for j,name in enumerate(NAMES):row[name+'_prediction']=CLASSES[votes[j,i]];row[name+'_confidence']=float(stack[j,i].max())
        rows.append(row)
    frame=pd.DataFrame(rows);write_csv(OUT/'all_validation_audit.csv',rows)
    errors=frame.loc[~frame.correct].sort_values(['confidence','image_id'],ascending=[False,True]);write_csv(OUT/'errors_96.csv',errors.to_dict('records'))
    groups=errors.groupby(['true_class','predicted_class']).size().reset_index(name='images').sort_values('images',ascending=False)
    write_csv(OUT/'error_confusions.csv',groups.to_dict('records'))
    strata=[]
    for key in ['train_lesion_shared','dx_type','max_vote_agreement']:
        for value,g in frame.groupby(key):strata.append(dict(group=key,value=str(value),images=len(g),errors=int((~g.correct).sum()),accuracy=float(g.correct.mean())))
    write_csv(OUT/'descriptive_strata.csv',strata)
    quality=[]
    for c in CLASSES:
        mask=frame.true_class.eq(c);a=frame.loc[mask&~frame.correct];b=frame.loc[mask&frame.correct]
        for feature in ['brightness','contrast','laplacian_variance','dark_border_fraction']:
            quality.append(dict(class_name=c,feature=feature,error_images=len(a),correct_images=len(b),error_median=float(a[feature].median()),correct_median=float(b[feature].median())))
    write_csv(OUT/'within_class_quality.csv',quality)
    pair=[]
    for i in range(5):
        for j in range(i+1,5):
            a=votes[i]!=y;b=votes[j]!=y;pair.append(dict(model_a=NAMES[i],model_b=NAMES[j],shared_errors=int((a&b).sum()),error_union=int((a|b).sum()),error_jaccard=float((a&b).sum()/(a|b).sum())))
    write_csv(OUT/'pairwise_error_overlap.csv',pair)
    class_rows=[]
    for c in CLASSES:
        mask=frame.true_class.eq(c);e=errors.loc[errors.true_class.eq(c)]
        class_rows.append(dict(class_name=c,train_images=int(train.diagnosis.eq(c).sum()),val_images=int(mask.sum()),errors=len(e),
            all_members_wrong=int(e.all_members_wrong.sum()),some_member_correct=int((e.members_correct>0).sum()),recall=float(frame.loc[mask].correct.mean())))
    write_csv(OUT/'class_error_summary.csv',class_rows)
    fig,ax=plt.subplots(figsize=(9,4));g=groups.head(10).iloc[::-1];ax.barh(g.true_class+' -> '+g.predicted_class,g.images);ax.set(xlabel='Wrong validation images',title='S53: largest confusions (descriptive)');savefig(fig,'largest_confusions')
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].hist(confidence[~err],bins=np.linspace(0,1,11),alpha=.6,label='Correct');axes[0].hist(confidence[err],bins=np.linspace(0,1,11),alpha=.7,label='Wrong');axes[0].legend();axes[0].set(xlabel='Ensemble confidence',ylabel='Images',title='Confidence distributions')
    axes[1].plot([r['confidence'] for r in bins],[r['accuracy'] for r in bins],'o-');axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set(xlabel='Mean confidence',ylabel='Observed accuracy',title='Fixed ten-bin reliability');savefig(fig,'confidence_and_reliability')
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(['All five wrong','At least one right'],[int((allwrong&err).sum()),int((~allwrong&err).sum())]);ax.set(ylabel='Reference errors',title='S53 errors: component correctness (not an achievable routing rule)');savefig(fig,'error_decomposition')
    review=OUT/'contact_sheets';review.mkdir()
    for page,start in enumerate(range(0,len(errors),16),1):
        sheet=Image.new('RGB',(960,1000),'white');d=ImageDraw.Draw(sheet)
        for cell,(_,r) in enumerate(errors.iloc[start:start+16].iterrows()):
            x=(cell%4)*240;y0=(cell//4)*250;image_id=r.image_id;thumb=thumbs[image_id]
            sheet.paste(thumb,(x+(240-thumb.width)//2,y0+5))
            d.text((x+6,y0+177),f'{image_id}\n{r.true_class} -> {r.predicted_class}\np={r.confidence:.2f}; votes={r.max_vote_agreement}/5\nmembers correct={r.members_correct}',font=font,fill='black')
        sheet.save(review/f'errors_{page:02}.png')
    sheet=Image.new('RGB',(960,1200),'white');d=ImageDraw.Draw(sheet)
    for cell,(_,r) in enumerate(errors.iloc[:8].iterrows()):
        x=(cell%2)*480;y0=(cell//2)*300;image_id=r.image_id;thumb=thumbs[image_id]
        sheet.paste(thumb,(x+(224-thumb.width)//2,y0+25+(224-thumb.height)//2));sheet.paste(model_inputs[image_id],(x+240,y0+25))
        d.text((x+5,y0),f'{image_id}: {r.true_class} -> {r.predicted_class}',font=font,fill='black')
        d.text((x+5,y0+255),'Original aspect ratio             Actual square224 input',font=font,fill='black')
    sheet.save(review/'preprocessing_top8.png')
    resolutions=frame.groupby(['width','height']).size().reset_index(name='images').to_dict('records')
    write_json(OUT/'source_verification.json',dict(split=relative(manifest),split_sha256=sha256(manifest),sources=sources,reference_sha256=sha256(REF/'validation_predictions.csv'),validation_image_sha256=image_hashes,test_images_opened=0,test_labels_loaded=False))
    for s in sources:assert sha256(ROOT/s['prediction_file'])==s['prediction_sha256']
    summary=dict(status='completed',reference_accuracy=float((~err).mean()),correct=int((~err).sum()),errors=int(err.sum()),
        all_members_wrong_errors=int((allwrong&err).sum()),some_member_correct_errors=int((~allwrong&err).sum()),unanimous_wrong=int((unanimous&err).sum()),
        high_confidence_wrong_ge_09=int(((confidence>=.9)&err).sum()),high_confidence_wrong_ge_08=int(((confidence>=.8)&err).sum()),
        mean_confidence_wrong=float(confidence[err].mean()),mean_confidence_correct=float(confidence[~err].mean()),ensemble_ece10=ec,
        confidence_error_ranking_auc=float(roc_auc_score(err,-confidence)),resolutions=resolutions,all_images_decoded=True,
        class_error_summary=class_rows,strata=strata,model_confidence=model_rows,test_loaded=False,gpu_used=False,new_predictions_generated=False,
        scope='Descriptive post-test exploratory validation audit, not a model improvement experiment')
    write_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
