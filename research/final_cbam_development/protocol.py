"""CPU-testable policy shared with the future S79 runner. No training entry point."""
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from research.common import ROOT, CLASSES, sha256
from research.models import ResearchClassifier
from research.short_screening.feature_fusion import development

CONFIG=ROOT/'research/configs/final_cbam_development/s79_convnext_tiny_cbam_v1.json'


class QuarterTurn:
    def __call__(self, image):
        return image if (k:=int(torch.randint(4,())))==0 else image.transpose(
            [Image.Transpose.ROTATE_90,Image.Transpose.ROTATE_180,Image.Transpose.ROTATE_270][k-1])


class Images(torch.utils.data.Dataset):
    def __init__(self, frame, config, training=False):
        if not len(frame) or set(frame.split)!=({'train'} if training else {'val'}):
            raise ValueError('Only the declared development partition may enter this loader')
        self.frame=frame.copy()
        self.transform=transforms.Compose([
            transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
            *([transforms.RandomHorizontalFlip(.5),transforms.RandomVerticalFlip(.5),QuarterTurn()] if training else []),
            transforms.ToTensor(),transforms.Normalize(config['normalization_mean'],config['normalization_std'])])

    def __len__(self):return len(self.frame)

    def __getitem__(self,index):
        row=self.frame.iloc[index]
        with Image.open(row.path) as image:x=self.transform(image.convert('RGB'))
        return x,int(row.label),str(row.image_id)


def weights_from_training(frame):
    if set(frame.split)!={'train'}:raise ValueError('Weights must use training only')
    counts=np.bincount(frame.label.to_numpy(),minlength=7)
    if (counts<=0).any():raise ValueError('Missing training class')
    weights=np.sqrt(len(frame)/(7*counts));weights/=weights.mean()
    return torch.tensor(weights,dtype=torch.float32),counts.tolist()


def make_model(config,pretrained=True):
    return ResearchClassifier(config['model'],weights=config['weights'] if pretrained else None,
                              attention=config['attention'],dropout=config['head_dropout'])


def fresh_parameters(model):
    return {id(p) for module in (model.attention,model.head[-1]) for p in module.parameters()}


def stage(model,epoch,config):
    fresh=fresh_parameters(model);warm=epoch<=config['warmup_epochs'];model.train()
    for p in model.parameters():p.requires_grad_(not warm or id(p) in fresh)
    if warm:
        model.features.eval();model.head[1].eval()
    return 'attention_head_warmup' if warm else 'full_finetune'


def make_optimizer(model,config):
    fresh=fresh_parameters(model)
    return torch.optim.AdamW([
        {'params':[p for p in model.parameters() if id(p) not in fresh],'lr':config['learning_rate']['backbone']},
        {'params':[p for p in model.parameters() if id(p) in fresh],'lr':config['learning_rate']['head_attention']}],
        betas=tuple(config['optimizer_betas']),eps=config['optimizer_eps'],weight_decay=config['weight_decay'])


def make_scheduler(optimizer,config):
    c=config['scheduler']
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,**{k:c[k] for k in ['mode','factor','patience','threshold','threshold_mode','min_lr']})


def stopping(history,config):
    policy=config['stopping'];references={};last={};reductions=[];previous=0
    for row in history:
        epoch=int(row['epoch'])
        if epoch!=previous+1:raise ValueError('Non-contiguous committed history')
        previous=epoch
        for metric,delta in [('accuracy',policy['accuracy_delta']),('macro_f1',policy['macro_f1_delta'])]:
            value=float(row['val_'+metric])
            if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid metric')
            if metric not in references or value-references[metric]>=delta-1e-12:
                references[metric]=value;last[metric]=epoch
        if row.get('scheduler_lr_reduced',False):reductions.append(epoch)
    epoch=previous;last_meaningful=max(last.values(),default=0);stale=epoch-last_meaningful
    settled=bool(reductions) and epoch-reductions[-1]>=policy['lr_settle_epochs']
    flat=False;recent=history[-policy['plateau_window']:]
    if len(recent)==policy['plateau_window']:
        flat=True
        for metric,delta in [('accuracy',policy['accuracy_delta']),('macro_f1',policy['macro_f1_delta'])]:
            values=np.array([r['val_'+metric] for r in recent],dtype=float)
            flat &= float(np.polyfit(np.arange(len(values)),values,1)[0])*(len(values)-1)<delta
            flat &= float(values[-3:].mean()-values[:3].mean())<delta
    reason='running'
    if epoch>=config['max_epochs']:reason='epoch_cap'
    elif epoch>=config['minimum_epochs'] and settled and stale>=policy['patience']:
        reason='late_meaningful_plateau' if epoch>=policy['plateau_after_epoch'] and flat else 'meaningful_patience_exhausted'
    return dict(references=references,last_improvement_epochs=last,last_meaningful_epoch=last_meaningful,stale=stale,
                lr_reduction_epochs=reductions,lr_settled=settled,flat_or_declining=bool(flat),stop=reason!='running',reason=reason)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])


def check_resume(payload,config,signature):
    if payload.get('diagnostic_only') or payload.get('config')!=config or payload.get('signature')!=signature:
        raise ValueError('Diagnostic or mismatched checkpoint cannot resume')
    required=['epoch','model','optimizer','scheduler','scaler','rng','history','meaningful_stopping','optimizer_updates','best_accuracy','best_macro_f1']
    if any(k not in payload for k in required):raise ValueError('Incomplete checkpoint')
    epoch=payload['epoch']
    if not 0<=epoch<=config['max_epochs'] or len(payload['history'])!=epoch:
        raise ValueError('Checkpoint is not a committed epoch boundary')
    if payload['meaningful_stopping']!=stopping(payload['history'],config):
        raise ValueError('Stopping counters differ from committed history')
    def finite(obj):
        if torch.is_tensor(obj):return not obj.is_floating_point() or bool(torch.isfinite(obj).all())
        if isinstance(obj,dict):return all(finite(v) for v in obj.values())
        if isinstance(obj,(tuple,list)):return all(finite(v) for v in obj)
        if isinstance(obj,float):return math.isfinite(obj)
        return True
    if not finite(payload['model']) or not finite(payload['optimizer']):
        raise ValueError('Nonfinite model or optimizer state')


def verified_development(config):
    if config['class_order']!=list(CLASSES) or sha256(ROOT/config['split_manifest'])!=config['split_sha256']:
        raise ValueError('Frozen class order/split changed')
    return development()
