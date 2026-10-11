"""R203 identity, original development partition and verified model primitives.

Historical R202 functions are imported read-only. No CUDA initialization occurs
on import, and no project-trained weights are used for fresh initialization.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm

from research.common import ROOT, CLASSES, SPLITS, relative, sha256
from research.train import development_data as original_development_data
from research.image_level_replication.v2.convnext_core import (
    Images, QuarterTurn, epoch_loader, make_model, make_optimizer, set_stage,
    expected_lrs, apply_learning_rates, schedule_state, update_ema, meaningful,
    finite, capture_rng, restore_rng, report, MODEL, AUTHOR_SHA256, WEIGHT_FILE)

CONFIG = ROOT / 'research/convnext_base_original/config.json'
FREEZE = ROOT / 'results/structured_experiments/convnext_base_original_v1/preparation/r203_launch_freeze.json'
REFERENCE = ROOT / 'results/short_screening/s83_cbam_f1_addition_cpu/validation_predictions.csv'
MANIFEST = 'data/splits/exploratory/image_level_dev_v1.csv'
MANIFEST_SHA = '75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb'
CODE = [
    'research/convnext_base_original/core.py',
    'research/convnext_base_original/train.py',
    'research/convnext_base_original/preflight.py',
    'research/image_level_replication/v2/convnext_core.py',
    'research/image_level_replication/v2/core.py',
    'research/image_level_replication/v2/train_b3.py',
    'research/image_level_replication/v2/prepare_protocol.py',
    'research/image_level_replication/prepare_protocol.py',
    'research/train.py', 'research/experiment.py', 'research/models.py',
    'research/common.py', 'research/plots.py', 'research/registry.py']


def config_signature(path=CONFIG):
    if Path(path).resolve() != CONFIG:
        raise ValueError('Only the prepared R203 config is supported')
    c = json.loads(CONFIG.read_text(encoding='utf-8'))
    fixed = dict(experiment_id='r203_convnextv2_base_original_exploratory_seed42',
                 study_id='convnext_base_original_v1', protocol='exploratory_image_level',
                 seed=42, model_name=MODEL, image_size=224, max_epochs=30,
                 minimum_epochs=30, head_warmup_epochs=2, num_classes=7,
                 effective_batch_size=32, outer_assessment_enabled=False,
                 legacy_initialization_allowed=False, automatic_next_run=False,
                 test_evaluation_enabled=False, evaluation_split='validation',
                 train_count=7009, validation_count=1503,
                 training_precision='amp_bf16', validation_precision='fp32')
    for key, expected in fixed.items():
        if c.get(key) != expected:
            raise ValueError('Unexpected R203 setting: ' + key)
    if c['class_order'] != list(CLASSES) or c['stopping']['automatic_early_stopping']:
        raise ValueError('Class order or fixed epoch budget changed')
    if c['batch_size'] * c['accumulation_steps'] != 32 or c['batch_size'] not in (4, 8):
        raise ValueError('Invalid effective batch')
    if c['split_manifest'] != MANIFEST or c['split_sha256'] != MANIFEST_SHA or sha256(ROOT / MANIFEST) != MANIFEST_SHA:
        raise ValueError('Wrong or changed original exploratory split')
    if c['weights_file'] != relative(WEIGHT_FILE) or c['official_weights_url'] != timm.get_pretrained_cfg(MODEL).url:
        raise ValueError('Wrong external pretrained source')
    if sha256(WEIGHT_FILE) != AUTHOR_SHA256:
        raise ValueError('Author checkpoint differs from its recorded SHA256')
    for key in ('results', 'checkpoints'):
        expected = key + '/structured_experiments/convnext_base_original_v1/' + c['experiment_id']
        if c[key] != expected:
            raise ValueError('Wrong isolated R203 output path')
        (ROOT / c[key]).resolve().relative_to(ROOT)
    signature = dict(config_sha256=sha256(CONFIG),
                     code_sha256={p: sha256(ROOT / p) for p in CODE},
                     split_manifest_sha256=MANIFEST_SHA,
                     locked_reference_manifest_sha256=sha256(ROOT / SPLITS['strict_lesion_disjoint']),
                     locked_recipe_sha256=sha256(ROOT / 'research/phase2/recipe_v1.json'),
                     raw_metadata_sha256=sha256(ROOT / 'data/raw/HAM10000/HAM10000_metadata.csv'),
                     reference_predictions_sha256=sha256(REFERENCE),
                     pretrained_sha256=AUTHOR_SHA256,
                     timm_convnext_sha256=sha256(Path(timm.__file__).parent / 'models/convnext.py'))
    return c, signature


def original_split_rows(c):
    """Metadata only; no assessment image paths or decoded pixels."""
    return pd.read_csv(ROOT / c['split_manifest'], usecols=['image_id', 'lesion_id', 'split'])


def development_data(c):
    train, val, _ = original_development_data(c)
    counts = np.bincount(train.label.to_numpy(), minlength=7)
    if counts.tolist() != [229, 359, 769, 81, 778, 4693, 100]:
        raise ValueError('Original training support changed')
    weights = np.sqrt(len(train) / (7 * counts))
    weights /= weights.mean()
    if not np.allclose(weights, c['loss_class_weights'], rtol=1e-12, atol=1e-12):
        raise ValueError('Class weights differ from original training counts')
    reference = pd.read_csv(REFERENCE, usecols=['image_id', 'lesion_id', 'true_class'])
    if (reference.image_id.tolist() != val.image_id.tolist() or
            reference.lesion_id.tolist() != val.lesion_id.tolist() or
            reference.true_class.tolist() != val.diagnosis.tolist()):
        raise ValueError('Validation identity/order differs from the saved original ensemble')
    return train, val, torch.tensor(weights, dtype=torch.float32)
