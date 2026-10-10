"""Frozen R202 primitives. Importing this module never initializes CUDA."""
import copy
import json
import math
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from timm.models.convnext import checkpoint_filter_fn

from research.common import ROOT, CLASSES, sha256, relative
from . import prepare_protocol as protocol
from .core import (development_data as inner_data, finite, capture_rng,
                   restore_rng, report, seed_worker)

CONFIG = ROOT / 'research/image_level_replication/v2/convnext_base_fold00.json'
WEIGHT_FILE = ROOT / '.cache/torch/hub/checkpoints/convnextv2_base_22k_224_ema.pt'
FREEZE = ROOT / 'results/image_level_replication/v2/preparation/r202_convnext_base_launch_freeze.json'
MODEL = 'convnextv2_base.fcmae_ft_in22k_in1k'
AUTHOR_SHA256 = 'dd7a88d111c8af2295cb744f5bba888eca26759984b1cac81804ef876807ce17'
CODE = ['research/image_level_replication/v2/convnext_core.py',
        'research/image_level_replication/v2/train_convnext_base.py',
        'research/image_level_replication/v2/preflight_convnext_base.py',
        'research/image_level_replication/v2/core.py',
        'research/image_level_replication/v2/train_b3.py',
        'research/image_level_replication/v2/prepare_protocol.py',
        'research/image_level_replication/prepare_protocol.py',
        'research/common.py', 'research/plots.py', 'research/registry.py']


def config_signature(path=CONFIG):
    if __import__('pathlib').Path(path).resolve() != CONFIG:
        raise ValueError('Only the prepared R202 config is supported')
    c = json.loads(CONFIG.read_text(encoding='utf-8'))
    fixed = dict(experiment_id='r202_convnextv2_base_fold00_seed42',
                 study_id=protocol.STUDY, protocol=protocol.PROTOCOL,
                 fold=0, seed=42, model_name=MODEL, image_size=224,
                 max_epochs=50, head_warmup_epochs=2, num_classes=7,
                 effective_batch_size=32, outer_assessment_enabled=False,
                 legacy_initialization_allowed=False, automatic_next_run=False,
                 training_precision='amp_bf16', validation_precision='fp32')
    for key, expected in fixed.items():
        if c.get(key) != expected:
            raise ValueError('Unexpected R202 setting: ' + key)
    if c['class_order'] != list(CLASSES) or c['stopping']['automatic_early_stopping']:
        raise ValueError('Class order or fixed epoch budget changed')
    if c['batch_size'] * c['accumulation_steps'] != 32 or c['batch_size'] not in (4, 8):
        raise ValueError('Invalid preflight-selected effective batch')
    if c['split_manifest'] != relative(protocol.DATA / 'fold_00.csv') or c['split_sha256'] != sha256(protocol.DATA / 'fold_00.csv'):
        raise ValueError('Wrong frozen V2 split')
    if c['weights_file'] != relative(WEIGHT_FILE) or c['official_weights_url'] != timm.get_pretrained_cfg(MODEL).url:
        raise ValueError('Wrong external pretrained source')
    if sha256(WEIGHT_FILE) != AUTHOR_SHA256:
        raise ValueError('Downloaded author checkpoint differs from the recorded SHA256')
    for key, base in (('results', 'results/image_level_replication/v2/'),
                      ('checkpoints', 'checkpoints/image_level_replication/v2/')):
        if c[key] != base + c['experiment_id']:
            raise ValueError('Wrong isolated output path')
        (ROOT / c[key]).resolve().relative_to(ROOT)
    return c, dict(config_sha256=sha256(CONFIG), code_sha256={p: sha256(ROOT / p) for p in CODE},
                   protocol_sha256=sha256(protocol.POLICY), fold_manifest_sha256=c['split_sha256'],
                   pretrained_sha256=AUTHOR_SHA256,
                   timm_convnext_sha256=sha256(__import__('pathlib').Path(timm.__file__).parent / 'models/convnext.py'))


def development_data(c):
    train, val = inner_data(c)
    counts = np.bincount(train.label.to_numpy(), minlength=7)
    if (counts <= 0).any():
        raise ValueError('Missing training class')
    weights = np.sqrt(len(train) / (7 * counts))
    weights /= weights.mean()
    if not np.allclose(weights, c['loss_class_weights'], atol=1e-12):
        raise ValueError('Class weights differ from training-only counts')
    return train, val, torch.tensor(weights, dtype=torch.float32)


class QuarterTurn:
    def __call__(self, image):
        k = random.randrange(4)
        operation = (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)[k]
        return image if operation is None else image.transpose(operation)


class Images(Dataset):
    def __init__(self, frame, c, training=False):
        expected = 'train' if training else 'val'
        if set(frame.split) != {expected}:
            raise ValueError('Only inner ' + expected + ' originals can be decoded')
        self.frame = frame.reset_index(drop=True)
        ops = [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)]
        if training:
            a = c['augmentation_parameters']
            ops += [transforms.RandomHorizontalFlip(a['horizontal_flip']),
                    transforms.RandomVerticalFlip(a['vertical_flip']), QuarterTurn()]
        ops += [transforms.ToTensor(), transforms.Normalize(c['normalization_mean'], c['normalization_std'])]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as source:
            image = self.transform(source.convert('RGB'))
        return image, int(row.label), row.image_id


def epoch_loader(frame, c, epoch, training):
    generator = torch.Generator().manual_seed(c['seed'] + epoch * 1009 + (0 if training else 100000))
    return DataLoader(Images(frame, c, training), batch_size=c['batch_size'] if training else c['validation_batch_size'],
                      shuffle=training, drop_last=False, num_workers=c['num_workers'], pin_memory=training,
                      generator=generator, worker_init_fn=seed_worker, persistent_workers=False)


def make_model(c, pretrained=True):
    model = timm.create_model(MODEL, pretrained=False, num_classes=1000,
                             drop_rate=c['head_drop_rate'], drop_path_rate=c['drop_path_rate'])
    if pretrained:
        if sha256(WEIGHT_FILE) != AUTHOR_SHA256:
            raise ValueError('Pretrained file changed')
        source = torch.load(WEIGHT_FILE, map_location='cpu', weights_only=True)
        if isinstance(source, dict) and 'model_ema' in source and 'model' not in source:
            source = source['model_ema']
        state = checkpoint_filter_fn(source, model)
        if not finite(state):
            raise FloatingPointError('Nonfinite external pretrained state')
        model.load_state_dict(state, strict=True)
    model.reset_classifier(7)
    if sum(p.numel() for p in model.parameters()) != c['parameter_count_with_seven_class_head']:
        raise ValueError('Model parameter count differs from the native seven-class model')
    return model


def make_optimizer(model, c):
    groups = {}
    for name, parameter in model.named_parameters():
        if name.startswith('head.'):
            stage, base = 'head', c['head_lr']
        elif name.startswith('stem.'):
            stage = 'stem'
            base = c['backbone_lr'] * c['stage_lr_multipliers'][stage]
        elif name.startswith('stages.'):
            stage = 'stage' + name.split('.')[1]
            base = c['backbone_lr'] * c['stage_lr_multipliers'][stage]
        else:
            raise ValueError('Unassigned native model parameter: ' + name)
        decay = 0.0 if parameter.ndim == 1 or name.endswith(('.bias', '.gamma', '.beta')) else c['weight_decay']
        key = stage + ('_no_decay' if decay == 0 else '_decay')
        groups.setdefault(key, dict(params=[], group_name=key, stage=stage, base_lr=base,
                                    lr=base, weight_decay=decay))['params'].append(parameter)
    parameters = [p for g in groups.values() for p in g['params']]
    if len(parameters) != len({id(p) for p in parameters}) or {id(p) for p in parameters} != {id(p) for p in model.parameters()}:
        raise ValueError('Optimizer parameter coverage is not exact')
    return torch.optim.AdamW(list(groups.values()), betas=tuple(c['optimizer_betas']),
                            eps=c['optimizer_eps'], foreach=False)


def set_stage(model, c, epoch):
    warmup = epoch <= c['head_warmup_epochs']
    for name, p in model.named_parameters():
        p.requires_grad_(not warmup or name.startswith('head.'))
    if warmup:
        model.eval()
        model.head.train()
    else:
        model.train()
    return 'head_warmup' if warmup else 'full_fine_tuning'


def expected_lrs(optimizer, c, epoch):
    if not 0 <= epoch <= c['max_epochs']:
        raise ValueError('LR epoch outside the frozen run')
    if epoch <= 2:
        return {g['group_name']: g['base_lr'] if g['stage'] == 'head' else 0.0 for g in optimizer.param_groups}
    if epoch <= 5:
        return {g['group_name']: g['base_lr'] * (1.0 if g['stage'] == 'head' else (epoch - 2) / 3.0)
                for g in optimizer.param_groups}
    floor = c['scheduler']['minimum_factor']
    factor = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (c['max_epochs'] - 5)))
    return {g['group_name']: g['base_lr'] * factor for g in optimizer.param_groups}


def apply_learning_rates(optimizer, c, epoch):
    values = expected_lrs(optimizer, c, epoch)
    for g in optimizer.param_groups:
        g['lr'] = values[g['group_name']]
    return values


def schedule_state(optimizer, c, epoch):
    values = expected_lrs(optimizer, c, epoch)
    current = {g['group_name']: g['lr'] for g in optimizer.param_groups}
    if current != values:
        raise ValueError('Optimizer LRs differ from the deterministic epoch schedule')
    return dict(name=c['scheduler']['name'], last_epoch=epoch, group_lrs=current)


@torch.no_grad()
def update_ema(ema, model, decay):
    parameters = set(dict(model.named_parameters()))
    source = model.state_dict()
    for name, target in ema.state_dict().items():
        if name in parameters and target.is_floating_point():
            target.mul_(decay).add_(source[name].detach(), alpha=1 - decay)
        else:
            target.copy_(source[name])


def meaningful(history, c):
    acc = f1 = None
    last = 0
    epochs = []
    for row in history:
        a = max(row['val_accuracy'], row.get('ema_val_accuracy') or row['val_accuracy'])
        f = max(row['val_macro_f1'], row.get('ema_val_macro_f1') or row['val_macro_f1'])
        changed = False
        if acc is None or a - acc >= c['stopping']['accuracy_delta'] - 1e-12:
            acc, changed = a, True
        if f1 is None or f - f1 >= c['stopping']['macro_f1_delta'] - 1e-12:
            f1, changed = f, True
        if changed:
            last = row['epoch']
            epochs.append(last)
    epoch = history[-1]['epoch'] if history else 0
    return dict(automatic_early_stopping=False, stale=epoch - last, last_meaningful_epoch=last,
                accuracy_reference=acc, macro_f1_reference=f1, meaningful_improvement_epochs=epochs,
                stop=epoch >= c['max_epochs'], reason='maximum_epochs' if epoch >= c['max_epochs'] else 'running')
