"""One separately frozen R202 outer assessment; --prepare is CPU-only."""
import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from research.common import ROOT, CLASSES, relative, sha256, write_json, write_csv
from research.image_level_replication.v2.convnext_core import CONFIG, config_signature, make_model
from research.final_cbam_development.runtime import deterministic_cuda
from research.generalization_review.check_weight_bias import metrics

OUT = ROOT / 'results/image_level_replication/v2/r202_outer_assessment_v1'
CHECKPOINT = ROOT / 'checkpoints/image_level_replication/v2/r202_convnextv2_base_fold00_seed42/best.pt'
COLS = ['p_' + c for c in CLASSES]
CODE = ['research/generalization_review/evaluate_r202.py',
        'research/generalization_review/check_weight_bias.py',
        'research/final_cbam_development/runtime.py', 'research/plots.py']


def selected():
    c, sig = config_signature()
    z = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert z['config'] == c and z['signature'] == sig
    assert z['branch'] == 'ema' and z['epoch'] == 20 and z['checkpoint_kind'] == 'accuracy'
    assert z['class_order'] == list(CLASSES) and z['inference_precision'] == 'fp32'
    assert z['outer_assessment_loaded'] is False and z['selection_partition'] == 'inner_validation_fold00'
    assert all(torch.isfinite(v).all() for v in z['model'].values() if v.is_floating_point())
    return c, sig, z


def identities(c):
    a = pd.read_csv(ROOT / c['split_manifest'], usecols=['image_id', 'lesion_id', 'split'])
    assert a.image_id.is_unique
    train, val, outer = [a.loc[a.split == split].reset_index(drop=True) for split in ['train', 'val', 'test']]
    assert (len(train), len(val), len(outer)) == (8111, 902, 1002)
    assert not set(outer.image_id) & set(pd.concat([train, val]).image_id)
    return train, val, outer


def prepare():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    c, sig, z = selected()
    train, val, outer = identities(c)
    model = make_model(c, pretrained=False)
    model.load_state_dict(z['model'], strict=True)
    model.eval().requires_grad_(False)
    # Shape/state validation only: no inference or assessment pixels/labels.
    files = {p.stem: p for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']
             for p in (ROOT / 'data/raw/HAM10000' / part).glob('*.jpg')}
    assert all(image_id in files for image_id in outer.image_id)
    correction = json.loads((ROOT / 'results/diagnostics/generalization_review_2026-10-12/r202_weight_bias_check/summary.json').read_text())
    assert correction['adopt'] is False
    f = dict(status='prepared_not_evaluated', model='R202 ConvNeXt-V2 Base only; not S83 ensemble',
        checkpoint=relative(CHECKPOINT), checkpoint_sha256=sha256(CHECKPOINT), branch='ema', epoch=20,
        selector='Original inner-validation accuracy winner; unchanged', config=c, signature=sig,
        class_order=list(CLASSES), preprocessing=c['preprocessing'], precision='FP32', tf32=False,
        views=['identity'], fusion='none; one model', probability_correction='none; fixed class-weight correction rejected on validation',
        code_sha256={name: sha256(ROOT / name) for name in CODE}, seed=42, batch_size=8,
        outer_image_ids=outer.image_id.tolist(), outer_lesion_ids=outer.lesion_id.tolist(), cohort_size=1002,
        train_images=8111, inner_validation_images=902, training_outer_image_overlap=0, inner_validation_outer_image_overlap=0,
        training_outer_shared_lesions=len(set(train.lesion_id) & set(outer.lesion_id)),
        outer_images_with_training_lesion=int(outer.lesion_id.isin(set(train.lesion_id)).sum()),
        evaluation_claim='Post-development single-fold image-level outer assessment with natural lesion overlap; not S83 test or completed K10',
        prior_project_test_outcomes_known=True, labels_read_during_preparation=False, images_decoded_during_preparation=0,
        training=False, no_selection_on_outer_labels=True, no_automatic_next_experiment=True)
    path = OUT / 'frozen_assessment.json'
    if path.exists():
        assert json.loads(path.read_text()) == f
    else:
        write_json(path, f)
    assert not torch.cuda.is_initialized()
    print('R202 prepared: EMA20;1002 unused assessment images;389 share training lesions;0 image overlap; no test labels/pixels read.')


def frozen():
    f = json.loads((OUT / 'frozen_assessment.json').read_text())
    c, sig = config_signature()
    assert c == f['config'] and sig == f['signature']
    assert sha256(CHECKPOINT) == f['checkpoint_sha256']
    assert f['code_sha256'] == {name: sha256(ROOT / name) for name in CODE}
    _, _, outer = identities(c)
    assert outer.image_id.tolist() == f['outer_image_ids'] and outer.lesion_id.tolist() == f['outer_lesion_ids']
    return f, c, outer


class AssessmentImages(Dataset):
    """No label columns are accepted by this inference loader."""
    def __init__(self, outer, c):
        assert list(outer.columns) == ['image_id', 'lesion_id', 'split']
        assert set(outer.split) == {'test'}
        self.ids = outer.image_id.tolist()
        self.files = {p.stem: p for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']
                      for p in (ROOT / 'data/raw/HAM10000' / part).glob('*.jpg')}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(), transforms.Normalize(c['normalization_mean'], c['normalization_std'])])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        with Image.open(self.files[image_id]) as image:
            x = self.transform(image.convert('RGB'))
        return x, image_id


def evaluate():
    if (OUT / 'completion.json').exists():
        print('Completed assessment preserved; no repeat inference')
        return
    f, c, outer = frozen()
    with (OUT / 'inference_started.json').open('x', encoding='utf-8') as receipt:
        json.dump(dict(freeze_sha256=sha256(OUT / 'frozen_assessment.json'), pid=os.getpid(), repeats_allowed=False), receipt)
    deterministic_cuda(42)
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    logger = logging.getLogger('r202_outer')
    logger.setLevel(logging.INFO)
    for handler in [logging.FileHandler(OUT / 'assessment.log'), logging.StreamHandler()]:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    tick = time.perf_counter()
    _, _, z = selected()
    model = make_model(c, pretrained=False)
    model.load_state_dict(z['model'], strict=True)
    del z
    model.eval().requires_grad_(False)
    model = model.cuda().float()
    outputs, ids = [], []
    logger.info('START one R202 EMA20 outer pass; FP32 bicubic224 identity;1002 images; labels absent from loader')
    with torch.inference_mode():
        for batch, (x, image_ids) in enumerate(DataLoader(AssessmentImages(outer, c), batch_size=8, num_workers=0, shuffle=False)):
            logits = model(x.cuda().float())
            p = logits.softmax(1)
            assert torch.isfinite(logits).all() and torch.isfinite(p).all()
            outputs.append(p.cpu().numpy())
            ids.extend(image_ids)
            if batch == 0 or (batch + 1) % 40 == 0:
                logger.info('Inference batch%d/126', batch + 1)
    assert ids == f['outer_image_ids']
    p = np.concatenate(outputs)
    probabilities = outer[['image_id', 'lesion_id']].copy()
    probabilities[COLS] = p
    write_csv(OUT / 'probabilities_unscored.csv', probabilities.to_dict('records'))
    # Decode assessment labels only after the one complete probability pass is saved.
    roles = pd.read_csv(ROOT / c['split_manifest'], usecols=['image_id', 'split'])
    excluded = set((roles.index[roles.split != 'test'] + 1).tolist())
    labeled = pd.read_csv(ROOT / c['split_manifest'], skiprows=lambda row: row in excluded)
    assert labeled.image_id.tolist() == ids
    assert labeled.label.tolist() == labeled.diagnosis.map(dict(zip(CLASSES, range(7)))).tolist()
    frame = outer[['image_id', 'lesion_id']].copy()
    frame['true_class'] = labeled.diagnosis.tolist()
    frame['predicted_class'] = [CLASSES[i] for i in p.argmax(1)]
    frame[COLS] = p
    m = metrics(frame, p)
    y = labeled.label.to_numpy()
    m.update(loss=float(-np.log(np.maximum(p[np.arange(len(p)), y], 1e-12)).mean()),
             weighted_f1=sum(v['support'] * v['f1'] for v in m['per_class'].values()) / len(frame),
             evaluation_claim=f['evaluation_claim'], cohort_size=len(frame), branch='ema', selected_epoch=20,
             incorrect=len(frame) - m['correct'], melanoma_recall=m['per_class']['mel']['recall'],
             akiec_recall=m['per_class']['akiec']['recall'])
    from research.plots import metric_figures
    metric_figures(m, OUT / 'figures', 'R202 EMA20 | single-fold outer assessment; lesion overlap')
    write_json(OUT / 'assessment_metrics.json', m)
    write_csv(OUT / 'predictions.csv', frame.to_dict('records'))
    write_csv(OUT / 'per_class_metrics.csv', [dict(class_name=cl, **m['per_class'][cl]) for cl in CLASSES])
    train, _, _ = identities(c)
    seen = frame.lesion_id.isin(set(train.lesion_id)).to_numpy()
    write_json(OUT / 'lesion_exposure_subgroups.json', {name: metrics(frame.loc[mask], p[mask])
        for name, mask in [('training_lesion_represented', seen), ('training_lesion_absent', ~seen)]})
    assert sha256(CHECKPOINT) == f['checkpoint_sha256']
    from research.registry import upsert
    upsert(dict(experiment_id='r202_fold00_outer_assessment_seed42', era='structured', record_kind='outer_assessment',
        phase='image_level_replication_v2', protocol='postdevelopment_image_level_outer_assessment_overlap_allowed',
        evaluation_split='outer_assessment_fold00', split_manifest=c['split_manifest'], split_sha256=c['split_sha256'],
        method='Frozen R202 EMA20; FP32 identity; no probability correction', model=c['model_name'],
        pretrained_weights=c['weights'], attention=c['attention'], image_size=224, preprocessing=c['preprocessing'],
        seed=42, epochs=50, best_epoch=20, checkpoint=relative(CHECKPOINT), checkpoint_sha256=f['checkpoint_sha256'],
        checkpoint_available_local=True, config_path=relative(OUT / 'frozen_assessment.json'),
        metrics_path=relative(OUT / 'assessment_metrics.json'), plots_dir=relative(OUT / 'figures'),
        status='completed', decision='separate_internal_evaluation_not_S83',
        notes=f['evaluation_claim'], **{k:m[k] for k in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']}))
    write_json(OUT / 'completion.json', dict(status='completed', accuracy=m['accuracy'], macro_f1=m['macro_f1'],
        original_checkpoint_unchanged=True, no_training=True, one_inference_pass=True,
        labels_used_only_after_saved_inference=True, method_changed_after_scoring=False, seconds=time.perf_counter() - tick))
    logger.info('COMPLETE accuracy=%.6f macroF1=%.6f; no subsequent method change', m['accuracy'], m['macro_f1'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else evaluate()
