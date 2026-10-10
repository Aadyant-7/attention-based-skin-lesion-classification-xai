"""Fold-0 B3 adaptation. No outer-assessment dataset or training at import."""
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from research.common import ROOT, CLASSES, sha256, relative
from . import prepare_protocol as protocol

CONFIG = ROOT/'research/image_level_replication/v2/b3_fold00.json'
WEIGHT_FILE = ROOT/'.cache/torch/hub/checkpoints/efficientnet_b3_rwightman-b3899882.pth'
FREEZE = ROOT/'results/image_level_replication/v2/preparation/b3_fold00_launch_freeze.json'
CODE = ['research/image_level_replication/v2/core.py',
        'research/image_level_replication/v2/train_b3.py',
        'research/image_level_replication/v2/prepare_protocol.py',
        'research/image_level_replication/prepare_protocol.py',
        'research/common.py', 'research/plots.py', 'research/registry.py', 'research/train.py']


def config_signature(path=CONFIG):
    path = Path(path).resolve()
    if path != CONFIG:
        raise ValueError('Only the prepared fold-0 config is supported')
    c = json.loads(path.read_text(encoding='utf-8'))
    identity = dict(study_id=protocol.STUDY, protocol=protocol.PROTOCOL, fold=0, seed=42,
                    model='efficientnet_b3', weights='EfficientNet_B3_Weights.IMAGENET1K_V1',
                    image_size=112, batch_size=16, max_epochs=50, minimum_epochs=25,
                    full_fine_tuning=True, outer_assessment_enabled=False,
                    legacy_initialization_allowed=False, automatic_next_run=False,
                    validation_precision='fp32', training_precision='amp_bf16')
    for key, value in identity.items():
        if c.get(key) != value:
            raise ValueError('Invalid prepared recipe: '+key)
    if c['split_manifest'] != relative(protocol.DATA/'fold_00.csv'):
        raise ValueError('Wrong fold manifest')
    if c['split_sha256'] != sha256(protocol.DATA/'fold_00.csv'):
        raise ValueError('Changed prepared fold manifest')
    if not WEIGHT_FILE.is_file():
        raise FileNotFoundError('Fresh external B3 weights unavailable: '+str(WEIGHT_FILE))
    digest = sha256(WEIGHT_FILE)
    if not digest.startswith('b3899882'):
        raise ValueError('External ImageNet weight checksum mismatch')
    for key, base in [('results', 'results/image_level_replication/v2/'),
                      ('checkpoints', 'checkpoints/image_level_replication/v2/')]:
        if c[key] != base+c['experiment_id']:
            raise ValueError('Wrong isolated output path: '+key)
        (ROOT/c[key]).resolve().relative_to(ROOT)
    return c, dict(config_sha256=sha256(path),
                   code_sha256={p: sha256(ROOT/p) for p in CODE},
                   protocol_sha256=sha256(protocol.POLICY),
                   fold_manifest_sha256=c['split_sha256'], pretrained_sha256=digest)


def development_data(c):
    policy = json.loads(protocol.POLICY.read_text(encoding='utf-8'))
    if sha256(protocol.SOURCE) != policy['metadata_sha256']:
        raise ValueError('Original metadata changed')
    train, val = protocol.load_development(c['fold'])
    files = {}
    for folder in ('HAM10000_images_part_1', 'HAM10000_images_part_2'):
        for path in (protocol.SOURCE.parent/folder).glob('*.jpg'):
            if path.stem in files:
                raise ValueError('Duplicate original JPG')
            files[path.stem] = str(path)
    frames = []
    for rows, split, size in [(train, 'train', 8111), (val, 'val', 902)]:
        frame = pd.DataFrame(rows)
        frame['label'] = frame.label.astype(int)
        if len(frame) != size or set(frame.split) != {split} or frame.image_id.duplicated().any():
            raise ValueError('Wrong inner partition')
        if frame.diagnosis.map(dict(zip(CLASSES, range(7)))).tolist() != frame.label.tolist():
            raise ValueError('Wrong inner class order')
        frame['path'] = [files[i] for i in frame.image_id]
        frames.append(frame)
    if set(frames[0].image_id) & set(frames[1].image_id):
        raise ValueError('Original-image overlap in development')
    return tuple(frames)


class Images(Dataset):
    def __init__(self, frame, c, training=False):
        expected = 'train' if training else 'val'
        if set(frame.split) != {expected}:
            raise ValueError('Only inner '+expected+' originals are allowed')
        self.frame = frame.reset_index(drop=True)
        a = c['augmentation_parameters']
        ops = [transforms.Resize((112,112), interpolation=transforms.InterpolationMode.BILINEAR)]
        if training:
            ops += [transforms.RandomResizedCrop(112, scale=a['crop_scale'], ratio=(1.,1.)),
                    transforms.RandomHorizontalFlip(a['horizontal_flip']),
                    transforms.RandomVerticalFlip(a['vertical_flip']),
                    transforms.RandomApply([transforms.RandomRotation((90,90))],
                                           p=a['rotation90_probability']),
                    transforms.RandomAffine(0, translate=a['translate'], scale=a['expansion'],
                                            interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ColorJitter(brightness=a['brightness'], contrast=a['contrast'],
                                           saturation=a['saturation']),
                    transforms.RandomApply([transforms.GaussianBlur(3, sigma=a['blur_sigma'])],
                                           p=a['blur_probability'])]
        ops += [transforms.ToTensor(), transforms.Normalize(c['normalization_mean'], c['normalization_std'])]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            x = self.transform(image.convert('RGB'))
        if not torch.isfinite(x).all():
            raise FloatingPointError('Nonfinite input: '+str(row.image_id))
        return x, int(row.label), str(row.image_id)


def balanced_indices(frame, seed):
    """Exact equal exposure; each original included at least once each epoch."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(frame.label.to_numpy() == i) for i in range(7)]
    if any(len(g) == 0 for g in groups):
        raise ValueError('Absent training class')
    target = max(map(len, groups))
    output = []
    for group in groups:
        blocks = [rng.permutation(group) for _ in range(math.ceil(target/len(group)))]
        output.extend(np.concatenate(blocks)[:target].tolist())
    return rng.permutation(output).tolist()


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def epoch_loader(frame, c, epoch, training):
    seed = c['seed']+epoch*1009+(0 if training else 100000)
    sampler = balanced_indices(frame, seed) if training else None
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Images(frame,c,training), batch_size=c['batch_size'] if training else c['validation_batch_size'],
                      sampler=sampler, shuffle=False, num_workers=c['num_workers'],
                      drop_last=False, pin_memory=True, worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=False)


def make_model(c, pretrained=True):
    model = models.efficientnet_b3(weights=None)
    if pretrained:
        if not sha256(WEIGHT_FILE).startswith('b3899882'):
            raise ValueError('Invalid fresh pretrained weights')
        model.load_state_dict(torch.load(WEIGHT_FILE, map_location='cpu', weights_only=True), strict=True)
    model.classifier = nn.Sequential(nn.Linear(1536,4096), nn.ReLU(),
                                     nn.Linear(4096,4096), nn.ReLU(),
                                     nn.Dropout(c['head_dropout']), nn.Linear(4096,7))
    return model


def make_optimizer(model,c):
    return torch.optim.Adam(model.parameters(), lr=c['learning_rate'],
        betas=tuple(c['optimizer_betas']), eps=c['optimizer_eps'], weight_decay=c['weight_decay'])


def make_scheduler(opt,c):
    kw = {k:v for k,v in c['scheduler'].items() if k != 'name'}
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **kw)


def stopping(history,c):
    s = c['stopping']
    acc = f1 = None
    last = 0
    meaningful = []
    reductions = []
    for row in history:
        epoch = int(row['epoch'])
        a, f = float(row['val_accuracy']), float(row['val_macro_f1'])
        changed = False
        if acc is None or a-acc >= s['accuracy_delta']-1e-12:
            acc, changed = a, True
        if f1 is None or f-f1 >= s['macro_f1_delta']-1e-12:
            f1, changed = f, True
        if changed:
            last = epoch
            meaningful.append(epoch)
        if row.get('lr_after',row['lr']) < row['lr']-1e-12:
            reductions.append(epoch)
    epoch = int(history[-1]['epoch']) if history else 0
    stale = epoch-last
    settled = bool(reductions) and epoch-reductions[-1] >= s['lr_settle_epochs']
    reason = 'running'
    if epoch >= c['max_epochs']:
        reason = 'maximum_epochs'
    elif epoch >= c['minimum_epochs'] and stale >= s['patience'] and settled:
        reason = 'meaningful_plateau_after_lr_reduction'
    elif epoch >= s['late_guard_epoch'] and stale >= s['late_patience'] and settled:
        recent = history[-s['trend_window']:]
        x = np.arange(len(recent))
        slopes = [np.polyfit(x,[r[k] for r in recent],1)[0] for k in ('val_accuracy','val_macro_f1')]
        if slopes[0] <= s['accuracy_delta']/s['trend_window'] and slopes[1] <= s['macro_f1_delta']/s['trend_window']:
            reason = 'late_flat_trend_after_lr_reduction'
    return dict(stop=reason!='running', reason=reason, stale=stale, last_meaningful_epoch=last,
                accuracy_reference=acc, macro_f1_reference=f1,
                meaningful_improvement_epochs=meaningful, lr_reduction_epochs=reductions,
                lr_settled=settled)


def finite(obj):
    if torch.is_tensor(obj):
        return bool(torch.isfinite(obj).all())
    if isinstance(obj, dict):
        return all(finite(v) for v in obj.values())
    if isinstance(obj, (list,tuple)):
        return all(finite(v) for v in obj)
    if isinstance(obj, (float,np.floating)):
        return bool(np.isfinite(obj))
    return True


def capture_rng(cuda=False):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if cuda else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def report(y, probabilities, loss):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    y, p = np.asarray(y,dtype=int), np.asarray(probabilities)
    if p.shape != (len(y),7) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Invalid validation probabilities')
    if not np.isfinite(loss) or (y < 0).any() or (y > 6).any():
        raise ValueError('Invalid validation labels/loss')
    pred = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y,pred,labels=range(7),zero_division=0)
    return dict(accuracy=float(np.mean(y==pred)), macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()), macro_f1=float(f1.mean()),
                weighted_f1=float(np.average(f1,weights=support)), loss=float(loss),
                class_order=list(CLASSES), confusion_matrix=confusion_matrix(y,pred,labels=range(7)).tolist(),
                per_class={name:dict(precision=float(precision[i]),recall=float(recall[i]),
                    f1=float(f1[i]),support=int(support[i])) for i,name in enumerate(CLASSES)})
