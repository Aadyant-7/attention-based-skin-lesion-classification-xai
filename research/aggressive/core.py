"""Shared enhanced recipe: train-only focal weighting, augmentation, CBAM and safe precision."""
import json,os,random
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from . import __name__ as PACKAGE
from research.common import ROOT,CLASSES,sha256,write_json,write_csv
from research.strict_protocol import partition_metadata,development_data
from research.models import ResearchClassifier,CHANNELS

CONFIG=ROOT/'research/aggressive/config.json'
OUT=ROOT/'results/aggressive_enhanced/v1'
CKPT=ROOT/'checkpoints/aggressive_enhanced/v1'

def retry_registry_upsert(row):
    """Windows spreadsheet/scanner locks can temporarily block atomic replacement."""
    from research.registry import upsert
    for attempt in range(16):
        try:return upsert(row)
        except PermissionError:
            if attempt==15:raise
            time.sleep(2)

def config():return json.loads(CONFIG.read_text(encoding='utf-8'))

def hashes():
    names=['research/aggressive/'+n for n in ['config.json','core.py','train.py','results.py','xai.py','prepare.py','stopping.py']]
    names+=['research/run_aggressive_enhanced_pipeline.py','research/common.py','research/train.py','research/strict_protocol.py','research/models.py','research/plots.py','research/registry.py','src/cbam.py']
    return {n:sha256(ROOT/n) for n in names}

def compatible_sources(saved):
    current=hashes()
    if saved==current:return True
    amendment=OUT/'stopping_policy_amendment.json'
    if not amendment.exists():return False
    record=json.loads(amendment.read_text())
    allowed=[record.get('original_source_hashes'),record.get('legacy_live_disk_hashes')]+record.get('additional_authorized_sources',[])
    return record.get('status')=='authorized' and saved in allowed and current==record['amended_source_hashes']

def data(c):
    # Existing checked reader excludes test rows BEFORE parsing any labels.
    train,val,_=development_data(dict(protocol='strict_lesion_disjoint',split_manifest=c['split_manifest'],split_sha256=c['split_sha256']))
    counts=np.array([(train.label==i).sum() for i in range(7)])
    # Data-only inverse-frequency formulation; no validation-fitted/manual multipliers.
    weights=np.clip(len(train)/(7*counts),1,20);weights/=weights.mean()
    return train,val,torch.tensor(weights,dtype=torch.float32)

def focal_per_sample(logits,labels,weights,gamma=2.2):
    logp=torch.nn.functional.log_softmax(logits.float(),dim=1)
    selected=logp.gather(1,labels[:,None]).squeeze(1)
    # -expm1(log p) accurately represents 1-p near p=1.
    return -weights[labels]*(-torch.expm1(selected)).pow(gamma)*selected

def mixed_focal_sum(logits,a,b,lam,weights,gamma):
    # Convex combination of two hard-target focal objectives, NOT hard-label loss on soft targets.
    return (lam*focal_per_sample(logits,a,weights,gamma)+(1-lam)*focal_per_sample(logits,b,weights,gamma)).sum()

def transforms_for(c,training=False,label=None):
    ops=[transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)]
    if training:
        minority=CLASSES[label] in ['akiec','df','vasc']
        intermediate=CLASSES[label] in ['bcc','mel']
        degree=20 if minority else (17 if intermediate else 15)
        jitter=.25 if minority else .2
        ops += [transforms.RandomHorizontalFlip(.7 if minority else .5),transforms.RandomVerticalFlip(.5 if minority else .3),
            transforms.RandomRotation(degree,interpolation=transforms.InterpolationMode.BILINEAR,fill=(195,139,145)),
            transforms.RandomAffine(0,translate=(.12,.12) if minority else (.1,.1),scale=(.85,1.15) if minority else (.9,1.1),
                interpolation=transforms.InterpolationMode.BILINEAR,fill=(195,139,145)),
            transforms.ColorJitter(brightness=jitter,contrast=jitter,saturation=.2,hue=.05)]
    ops += [transforms.ToTensor(),transforms.Normalize(c['normalization_mean'],c['normalization_std'])]
    if training:ops += [transforms.RandomErasing(p=.3 if minority else .2,scale=(.01,.05),ratio=(.5,2),value=0)]
    return transforms.Compose(ops)

class Images(torch.utils.data.Dataset):
    def __init__(self,frame,c,training=False):
        allowed={'train'} if training else {'val'}
        if not set(frame.split).issubset(allowed):raise ValueError('Only enhanced development images allowed')
        self.frame=frame;self.training=training
        self.transforms=[transforms_for(c,training,i) for i in range(7)] if training else [transforms_for(c)]
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]
        with Image.open(row.path) as image:x=self.transforms[int(row.label) if self.training else 0](image.convert('RGB'))
        return x,int(row.label),row.image_id

class FP32BN1(nn.BatchNorm1d):
    def forward(self,x):
        with torch.autocast(x.device.type,enabled=False):return super().forward(x.float())
class FP32BN2(nn.BatchNorm2d):
    def forward(self,x):
        with torch.autocast(x.device.type,enabled=False):return super().forward(x.float())

def safe_bn(module):
    for name,child in list(module.named_children()):
        if isinstance(child,(nn.BatchNorm1d,nn.BatchNorm2d)):
            cls=FP32BN1 if isinstance(child,nn.BatchNorm1d) else FP32BN2
            new=cls(child.num_features,eps=child.eps,momentum=child.momentum,affine=child.affine,track_running_stats=child.track_running_stats)
            new.load_state_dict(child.state_dict());setattr(module,name,new)
        else:safe_bn(child)

class EnhancedModel(ResearchClassifier):
    def __init__(self,spec,pretrained=True):
        super().__init__(spec['model'],spec['weights'] if pretrained else None,spec['attention'])
        d=CHANNELS[spec['model']]
        # Paper's dropout progression is across HEAD DEPTH, not an epoch schedule.
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.65),nn.Linear(d,512),FP32BN1(512),nn.ReLU(),
            nn.Dropout(.55),nn.Linear(512,256),FP32BN1(256),nn.ReLU(),nn.Dropout(.45),nn.Linear(256,7))
        safe_bn(self.features)
    def stage(self,epoch):
        stage='head' if epoch<=2 else ('last_stage' if epoch<=5 else 'full')
        for p in self.features.parameters():p.requires_grad_(stage=='full')
        if stage=='last_stage':
            last=self.features[0].denseblock4 if self.backbone_name=='densenet201' else self.features[-1]
            for p in last.parameters():p.requires_grad_(True)
        # Frozen BN buffers must remain frozen too; active BN computes/stores in FP32.
        for module in self.features.modules():
            if isinstance(module,nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):module.eval()
        return stage

def weighted_metrics(y,p,loss):
    from research.train import metric_report
    from sklearn.metrics import roc_auc_score
    m=metric_report(np.asarray(y),p,loss)
    support=np.array([m['per_class'][c]['support'] for c in CLASSES])
    m['weighted_f1']=float(np.dot(support,[m['per_class'][c]['f1'] for c in CLASSES])/support.sum())
    if (support>0).all():
        # Binary one-v-rest AUC for each class; no thresholds fitted.
        auc={c:float(roc_auc_score(np.asarray(y)==j,np.asarray(p)[:,j])) for j,c in enumerate(CLASSES)}
        m['per_class_auc']=auc;m['macro_auc']=float(np.mean(list(auc.values())))
    m['correct']=int(np.trace(m['confusion_matrix']));m['incorrect']=int(support.sum())-m['correct']
    return m

def predictions(ids,y,p):
    return [dict(image_id=i,true_class=CLASSES[t],predicted_class=CLASSES[int(np.argmax(a))],**{f'p_{c}':float(a[j]) for j,c in enumerate(CLASSES)}) for i,t,a in zip(ids,y,p)]

def stopping_state(reference,stale,accuracy,epoch,c):
    if accuracy>reference+c['min_delta']:reference=accuracy;stale=0
    elif epoch>=c['minimum_epochs']:stale+=1
    return reference,stale

def epoch_batches(n,seed):
    order=torch.randperm(n,generator=torch.Generator().manual_seed(seed)).tolist()
    batches=[order[i:i+32] for i in range(0,n,32)]
    # Keep every sample. A terminal singleton joins the preceding batch for safe head BN.
    if len(batches)>1 and len(batches[-1])==1:
        terminal=batches.pop();batches[-1].extend(terminal)
    return batches

def micro_ranges(n,micro):
    chunks=math.ceil(n/micro);width,remainder=divmod(n,chunks);start=0;out=[]
    for i in range(chunks):
        end=start+width+int(i<remainder);out.append((start,end));start=end
    if min(b-a for a,b in out)<2:raise ValueError('Head BatchNorm requires at least two training examples')
    return out

def verify_saved(rid,c):
    from research.plots import validate_metrics
    folder=OUT/rid;record=json.loads((folder/'record.json').read_text());m=json.loads((folder/'validation_metrics.json').read_text());validate_metrics(m)
    if record['status']!='completed' or record['config_sha256']!=sha256(CONFIG):raise ValueError('Incomplete/incompatible run')
    if sha256(CKPT/rid/'best.pt')!=record['checkpoint_sha256']:raise ValueError('Changed enhanced checkpoint')
    frame=pd.read_csv(folder/'validation_predictions.csv');_,val,_=data(c)
    if frame.image_id.tolist()!=val.image_id.tolist() or frame.true_class.tolist()!=val.diagnosis.tolist():raise ValueError('Validation cohort mismatch')
    p=frame[[f'p_{x}' for x in CLASSES]].to_numpy()
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid validation probabilities')
    computed=weighted_metrics(val.label.to_numpy(),p,m['loss'])
    if abs(computed['accuracy']-m['accuracy'])>1e-8 or abs(computed['macro_f1']-m['macro_f1'])>1e-8:raise ValueError('Metrics mismatch')
    return record,m,frame
