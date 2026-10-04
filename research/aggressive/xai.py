"""Validation-only Grad-CAM of fixed ensemble probability; actual CBAM maps separate."""
import gc,json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from research.common import CLASSES,write_json,write_csv
from research.plots import save
from .core import OUT,CKPT,EnhancedModel,data,Images

def selected_cases(frame,max_cases=14):
    groups=[('correct_melanoma',(frame.true_class=='mel')&(frame.predicted_class=='mel')),
        ('incorrect_melanoma',(frame.true_class=='mel')&(frame.predicted_class!='mel')),
        ('melanoma_as_nv',(frame.true_class=='mel')&(frame.predicted_class=='nv')),
        ('correct_akiec',(frame.true_class=='akiec')&(frame.predicted_class=='akiec'))]
    for c in ['df','vasc','nv','bcc','bkl']:groups.append(('representative_'+c,frame.true_class==c))
    if 'disagreement' in frame:groups.append(('ensemble_disagreement',frame.disagreement))
    selected={};unavailable=[]
    for category,mask in groups:
        candidates=frame.loc[mask].sort_values('image_id')
        if candidates.empty:unavailable.append(category);continue
        row=candidates.iloc[0];selected.setdefault(row.image_id,[]).append(category)
    return list(selected.items())[:max_cases],unavailable

def generate(c,frame):
    import matplotlib.pyplot as plt
    _,val,_=data(c);dataset=Images(val,c);index={v:i for i,v in enumerate(val.image_id)}
    models=[];layers=[]
    for spec in c['models']:
        ck=torch.load(CKPT/spec['id']/'best.pt',map_location='cpu',weights_only=False)
        model=EnhancedModel(spec,False);model.load_state_dict(ck['model']);del ck;model=model.cuda().eval();models.append(model)
        layers.append(model.features[0].denseblock4 if spec['model']=='densenet201' else model.features[-1])
    cases,missing=selected_cases(frame);rows=[]
    for image_id,categories in cases:
        x,y,_=dataset[index[image_id]];x=x.unsqueeze(0).cuda().requires_grad_(True)
        row=frame.loc[frame.image_id==image_id].iloc[0];target=CLASSES.index(row.predicted_class)
        activations={};gradients={};handles=[];cbam={}
        def capture(j):
            def hook(_module,_input,output):
                activations[j]=output
                if output.requires_grad:output.register_hook(lambda grad:gradients.__setitem__(j,grad))
            return hook
        for j,layer in enumerate(layers):handles.append(layer.register_forward_hook(capture(j)))
        # Reuse the actual sigmoid modules: these are learned attention weights, not CAM.
        handles.append(models[0].attention[0].sigmoid.register_forward_hook(lambda _m,_i,o:cbam.__setitem__('channel',o.detach())))
        handles.append(models[0].attention[1].sigmoid.register_forward_hook(lambda _m,_i,o:cbam.__setitem__('spatial',o.detach())))
        try:
            for model in models:model.zero_grad(set_to_none=True)
            p=[model(x.float()).softmax(1) for model in models]
            ensemble=sum(weight*prob for weight,prob in zip(c['primary_weights'],p));ensemble[0,target].backward()
            heats=[]
            for j in range(3):
                weights=gradients[j].mean((2,3),keepdim=True)
                heat=torch.relu((weights*activations[j]).sum(1,keepdim=True))
                heat=torch.nn.functional.interpolate(heat,(224,224),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy()
                maximum=float(heat.max());heat=heat/maximum if maximum>0 else heat
                if not np.isfinite(heat).all():raise FloatingPointError('Invalid CAM')
                heats.append(heat)
            original=Image.open(val.iloc[index[image_id]].path).convert('RGB')
            base=np.asarray(original.resize((224,224),Image.Resampling.BILINEAR))/255.
            folder=OUT/'xai'/image_id;folder.mkdir(parents=True,exist_ok=True);original.save(folder/'original.png')
            composite=sum(weight*heat for weight,heat in zip(c['primary_weights'],heats))
            fig,axes=plt.subplots(1,5,figsize=(15,3))
            axes[0].imshow(base);axes[0].set_title('Original')
            for j,(name,heat) in enumerate([(s['model'],h) for s,h in zip(c['models'],heats)]+[('Composite display',composite)]):
                np.save(folder/(name+'_heatmap.npy'),heat)
                color=plt.get_cmap('jet')(heat)[...,:3];overlay=np.clip(.65*base+.35*color,0,1)
                Image.fromarray((color*255).astype('uint8')).save(folder/(name+'_heatmap.png'))
                Image.fromarray((overlay*255).astype('uint8')).save(folder/(name+'_overlay.png'))
                axes[j+1].imshow(overlay);axes[j+1].set_title(name)
            for ax in axes:ax.axis('off')
            fig.suptitle(f'{image_id}: true={CLASSES[y]}, predicted={row.predicted_class}, confidence={float(ensemble[0,target].detach()):.3f}')
            save(fig,folder,'gradcam_panel')
            spatial=cbam['spatial'].float();spatial=torch.nn.functional.interpolate(spatial,(224,224),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            np.save(folder/'cbam_spatial_attention.npy',spatial)
            Image.fromarray((spatial*255).astype('uint8')).save(folder/'cbam_spatial_attention.png')
            channel=cbam['channel'][0,:,0,0].float().cpu().numpy()
            write_csv(folder/'cbam_channel_attention.csv',[dict(channel=i,attention=float(a)) for i,a in enumerate(channel)])
            fig,axes=plt.subplots(1,2,figsize=(8,3));axes[0].imshow(spatial,cmap='viridis',vmin=0,vmax=1);axes[0].set_title('Actual CBAM spatial weights');axes[1].plot(channel);axes[1].set_title('Actual CBAM channel weights');save(fig,folder,'cbam_attention')
            # Bounded secondary XAI: patch occlusion on first two selected cases only.
            if len(rows)<2:
                occlusion=np.zeros((8,8));confidence=float(ensemble[0,target].detach())
                with torch.inference_mode():
                    for r in range(8):
                        for col in range(8):
                            altered=x.detach().clone();altered[:,:,r*28:(r+1)*28,col*28:(col+1)*28]=0
                            score=sum(w*m(altered).softmax(1)[0,target] for w,m in zip(c['primary_weights'],models))
                            occlusion[r,col]=confidence-float(score)
                np.save(folder/'occlusion_sensitivity.npy',occlusion)
                fig,ax=plt.subplots(figsize=(4,4));im=ax.imshow(occlusion,cmap='coolwarm');fig.colorbar(im,ax=ax);ax.set_title('Confidence drop | 28px occlusion');save(fig,folder,'occlusion_sensitivity')
            metadata=dict(image_id=image_id,categories=categories,true_class=CLASSES[y],predicted_class=row.predicted_class,confidence=float(ensemble[0,target].detach()),
                target='fixed weighted ensemble predicted-class probability',per_branch_confidence=[float(a[0,target].detach()) for a in p],
                layers=['EfficientNetB3 features[-1]','DenseNet201 features[0].denseblock4','ResNet101 features[-1] (layer4)'],
                interpretation='Branch CAM gradients derive from actual weighted probability. Composite averages normalized branch CAMs for display, not a unique/exact causal ensemble attribution. CBAM sigmoid weights are internal attention, not Grad-CAM; none proves clinical reasoning.')
            write_json(folder/'metadata.json',metadata);rows.append(metadata)
        finally:
            for handle in handles:handle.remove()
        del x,p,ensemble,activations,gradients;gc.collect()
    write_json(OUT/'xai/manifest.json',dict(status='completed',cases=rows,unavailable_categories=missing,selection='deterministic first image ID per category; absent categories disclosed; no favorable cherry-picking',test_images_used=False,lime='not run; no installed implementation required'))
    return dict(cases=len(rows),missing=missing)
