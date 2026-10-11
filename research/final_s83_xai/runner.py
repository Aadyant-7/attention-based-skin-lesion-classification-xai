"""Exact frozen S83 validation-only Grad-CAM. No training or test entry point."""
import argparse
import gc
import hashlib
import html
import json
import logging
import re
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from research.common import ROOT, CLASSES, relative, sha256, write_csv, write_json, atomic_text
from research.models import ResearchClassifier
from research.plots import save
from research.final_cbam_development.protocol import Images
from research.final_cbam_development.runtime import deterministic_cuda
from research.short_screening.feature_fusion import development

OUT = ROOT / 'results/final_s83_xai/v1'
FREEZE = ROOT / 'results/final_exploratory_freeze/v1/frozen_method.json'
FREEZE_SHA = '8b9d2125aedfe69f2aecc2d401ac0add580bd00a6f1b0b60a13d6e806f667e6b'
COLS = ['p_' + c for c in CLASSES]
CODE = ['research/final_s83_xai/runner.py', 'research/final_s83_xai/PLAN.md',
        'research/models.py', 'src/cbam.py', 'research/plots.py',
        'research/final_cbam_development/protocol.py',
        'research/final_cbam_development/runtime.py', 'research/short_screening/feature_fusion.py']
LABELS = ['S06 ConvNeXt-Tiny', 'S18 ConvNeXt-Small', 'S15 DenseNet201',
          'S10 EfficientNetV2-S', 'S03 EfficientNet-B0', 'S79 ConvNeXt-Tiny + CBAM']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tree_hashes(folder):
    return {relative(p): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def frozen_method():
    require(sha256(FREEZE) == FREEZE_SHA, 'S83 freeze changed')
    f = json.loads(FREEZE.read_text(encoding='utf-8'))
    require(f['source_experiment'] == 's83_cbam_f1_addition_equal_six_exploratory_seed42', 'Wrong ensemble')
    require(list(CLASSES) == f['class_order'] and len(f['sources']) == 6, 'Class/member mismatch')
    require(f['inference_precision'] == 'FP32' and f['tf32'] is False and f['views'] == ['identity'], 'Inference changed')
    require(f['image_size'] == 224 and f['weights'] == [1 / 6] * 6, 'Preprocessing/fusion changed')
    require(sha256(ROOT / f['split_manifest']) == f['split_sha256'], 'Split changed')
    require(sha256(ROOT / f['validation_predictions']) == f['validation_prediction_sha256'], 'S83 predictions changed')
    return f


def reference(f, val):
    frame = pd.read_csv(ROOT / f['validation_predictions'])
    require(frame.image_id.tolist() == val.image_id.tolist(), 'Validation order changed')
    require(frame.true_class.tolist() == [CLASSES[int(y)] for y in val.label], 'Validation labels disagree')
    arrays, provenance = [], []
    for member in f['sources']:
        cache = next((a for a in f['fp32_validation_cache_provenance'] if a['run'] == member['run']), None)
        if cache:
            path = ROOT / cache['path']
            require(sha256(path) == cache['sha256'], 'FP32 cache changed')
            with np.load(path, allow_pickle=False) as z:
                require(z['ids'].tolist() == val.image_id.tolist(), 'Cache cohort differs')
                require(np.array_equal(z['y'], val.label.to_numpy()), 'Cache labels differ')
                require(z['checkpoint_sha256'].item() == member['checkpoint_sha256'], 'Cache checkpoint differs')
                p = z['probabilities'].copy()
            provenance.append(dict(run=member['run'], path=cache['path'], sha256=cache['sha256']))
        else:
            path = ROOT / member['prediction_file']
            require(sha256(path) == member['prediction_sha256'], 'S79 reference changed')
            a = pd.read_csv(path)
            require(a.image_id.tolist() == val.image_id.tolist(), 'S79 cohort differs')
            p = a[COLS].to_numpy(dtype=np.float32)
            provenance.append(dict(run=member['run'], path=member['prediction_file'], sha256=member['prediction_sha256']))
        require(p.dtype == np.float32 and p.shape == (1503, 7) and np.isfinite(p).all(), 'Invalid FP32 reference')
        require((p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-5), 'Invalid reference probabilities')
        arrays.append(p)
    saved = frame[COLS].to_numpy(dtype=np.float32)
    difference = float(np.abs(np.mean(arrays, axis=0) - saved).max())
    require(difference == 0, 'Frozen references do not exactly reproduce S83')
    require([CLASSES[int(i)] for i in saved.argmax(1)] == frame.predicted_class.tolist(), 'S83 argmax differs')
    return frame, arrays, provenance, difference


def select_cases(frame):
    ranked = frame.copy()
    ranked['rank'] = ranked.image_id.map(lambda s: hashlib.sha256(('S83_XAI_v1|seed42|' + s).encode()).hexdigest())
    ranked = ranked.sort_values(['rank', 'image_id'])
    cases = []

    def take(mask, category):
        rows = ranked.loc[mask & ~ranked.image_id.isin([c['image_id'] for c in cases])]
        require(len(rows) > 0, 'Missing predeclared case category: ' + category)
        row = rows.iloc[0]
        cases.append(dict(image_id=row.image_id, category=category, true_class=row.true_class,
                          predicted_class=row.predicted_class, target_class=row.predicted_class,
                          correct=bool(row.true_class == row.predicted_class), selection_rank=row['rank']))

    for c in ['mel', 'akiec']:
        take((ranked.true_class == c) & (ranked.predicted_class == c), 'correct_' + c)
        take((ranked.true_class == c) & (ranked.predicted_class != c), 'incorrect_' + c)
    for c in ['bcc', 'bkl', 'df', 'nv', 'vasc']:
        take(ranked.true_class == c, 'representative_' + c)
    for i in range(3):
        take(ranked.true_class != ranked.predicted_class, 'additional_error_' + str(i + 1))
    require(len(cases) == 12 and len({c['image_id'] for c in cases}) == 12, 'Wrong case count')
    require({c['true_class'] for c in cases} == set(CLASSES), 'Missing true class')
    return cases


def payload(member, f):
    path = ROOT / member['checkpoint']
    require(sha256(path) == member['checkpoint_sha256'], 'Frozen checkpoint changed')
    z = torch.load(path, map_location='cpu', weights_only=False)
    c = z['config']
    require(list(z['class_order']) == f['class_order'], 'Checkpoint classes differ')
    require(z['best_epoch'] == member['epoch'] and z['selection_metric'] == member['selection'], 'Checkpoint selector differs')
    for key in ['split_manifest', 'split_sha256', 'image_size', 'normalization_mean', 'normalization_std']:
        require(c[key] == f[key], 'Checkpoint config differs: ' + key)
    require(c['model'] == member['model'] and c['weights'] == member['pretrained_weights'], 'Model/weights differ')
    attention = 'cbam' if member['run'].startswith('s79_') else 'none'
    require(c['attention'] == attention and c['head_dropout'] == .2 and c['seed'] == 42, 'Attention/dropout/seed differ')
    return z, c


def prepare():
    require(not (OUT / 'plan.json').exists(), 'Prepared plan exists; verify or generate without overwriting it')
    f = frozen_method()
    _, val = development()
    frame, _, provenance, difference = reference(f, val)
    cases = select_cases(frame)
    members = []
    torch.set_num_threads(2)
    for source, label in zip(f['sources'], LABELS):
        z, c = payload(source, f)
        require(all(torch.isfinite(v).all() for v in z['model'].values() if v.is_floating_point()), 'Nonfinite checkpoint')
        members.append(dict(source, display_name=label, attention=c['attention'], head_dropout=c['head_dropout']))
        del z
        gc.collect()
    for case in cases:
        row = val.loc[val.image_id == case['image_id']].iloc[0]
        case.update(input_sha256=sha256(row.path), lesion_id=row.lesion_id)
    preserved = {}
    for folder in ['results/final_cbam_reporting/v1/xai', 'results/final_locked_test/v1',
                   'results/final_exploratory_test_audit/v1']:
        preserved.update(tree_hashes(ROOT / folder))
    plan = dict(status='prepared', method=f['source_experiment'], freeze_path=relative(FREEZE), freeze_sha256=FREEZE_SHA,
                seed=42, cohort='exploratory_validation', selection='SHA256 S83_XAI_v1|seed42|image_id within predeclared categories',
                cases=cases, members=members, reference_provenance=provenance,
                complete_reference_max_difference=difference, validation_cohort_size=1503,
                gradient_target='(1/6) * branch softmax probability of saved S83 predicted class',
                code_sha256={p: sha256(ROOT / p) for p in CODE}, preserved_artifact_sha256=preserved,
                test_images_used=False, training=False, optimizer=False, predictions_changed=False)
    write_json(OUT / 'plan.json', plan)
    write_csv(OUT / 'case_index.csv', cases)
    print('Prepared exact S83: six checkpoints authenticated; all1503 FP32 vectors reproduced exactly;12 validation cases frozen', flush=True)


def read_plan(check_sources=True):
    plan = json.loads((OUT / 'plan.json').read_text(encoding='utf-8'))
    f = frozen_method()
    require(plan['code_sha256'] == {p: sha256(ROOT / p) for p in CODE}, 'XAI implementation/plan changed')
    require([m['checkpoint_sha256'] for m in plan['members']] == [m['checkpoint_sha256'] for m in f['sources']], 'Plan checkpoints differ')
    if check_sources:
        for m in plan['members']:
            require(sha256(ROOT / m['checkpoint']) == m['checkpoint_sha256'], 'Checkpoint mutated')
        for path, digest in plan['preserved_artifact_sha256'].items():
            require(sha256(ROOT / path) == digest, 'Historical XAI/test artifact changed: ' + path)
    return plan, f


def gradcam(model, x, target):
    layer = model.features[0].denseblock4 if model.backbone_name == 'densenet201' else model.features[-1]
    name = 'features.0.denseblock4' if model.backbone_name == 'densenet201' else 'features.' + str(len(model.features) - 1)
    state, handles = {}, []

    def capture(_module, _inputs, output):
        state['activation'] = output
        output.register_hook(lambda gradient: state.__setitem__('gradient', gradient))

    handles.append(layer.register_forward_hook(capture))
    if isinstance(model.attention, torch.nn.Sequential):
        for i, key in [(0, 'channel'), (1, 'spatial')]:
            handles.append(model.attention[i].sigmoid.register_forward_hook(
                lambda _m, _i, o, k=key: state.__setitem__(k, o.detach())))
    try:
        logits = model(x)
        require(torch.isfinite(logits).all(), 'Nonfinite logits')
        p = logits.softmax(1)
        (p[0, target] / 6).backward()
        a, g = state['activation'], state['gradient']
        require(a.ndim == 4 and a.shape == g.shape and torch.isfinite(a).all() and torch.isfinite(g).all(), 'Invalid activation/gradient')
        native = torch.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
        raw = torch.nn.functional.interpolate(native, (224, 224), mode='bilinear', align_corners=False)[0, 0].detach().cpu().numpy()
        maximum = float(raw.max())
        heat = raw / maximum if maximum > 0 else raw.copy()
        attention = {}
        if 'spatial' in state:
            attention['spatial_native'] = state['spatial'][0, 0].cpu().numpy()
            attention['spatial'] = torch.nn.functional.interpolate(state['spatial'], (224, 224), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            attention['channel'] = state['channel'][0, :, 0, 0].cpu().numpy()
        require(all(np.isfinite(v).all() for v in [raw, heat, *attention.values()]), 'Nonfinite map')
        require(all((v >= 0).all() and (v <= 1).all() for v in attention.values()), 'Invalid CBAM gate')
        require(all(p.grad is None for p in model.parameters()), 'Unexpected parameter gradient')
        meta = dict(target_layer=name, activation_shape=list(a.shape), gradient_shape=list(g.shape),
                    raw_cam_max=maximum, degenerate_map=maximum <= 0)
        return p.detach().cpu().numpy()[0], raw, heat, attention, meta
    finally:
        for handle in handles:
            handle.remove()


def overlay(base, heat):
    import matplotlib.pyplot as plt
    return np.clip(.65 * base + .35 * plt.get_cmap('magma')(heat)[..., :3], 0, 1)


def generate():
    # Completion check occurs before any CUDA work. Never regenerate a finished package.
    if (OUT / 'manifest.json').exists():
        print('Completed S83 XAI preserved; run --verify for a read-only check', flush=True)
        return
    require(not (OUT / 'generation_started.json').exists(), 'Partial run preserved; inspect before any recovery')
    plan, f = read_plan()
    _, val = development()
    frame, refs, _, _ = reference(f, val)
    require(plan['cases'] == select_cases_with_inputs(frame, val), 'Case selection/input changed')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    deterministic_cuda(42)
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.set_num_threads(2)
    log = logging.getLogger('s83_xai')
    log.setLevel(logging.INFO)
    for handler in [logging.FileHandler(OUT / 'xai.log'), logging.StreamHandler()]:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        log.addHandler(handler)
    tick = time.perf_counter()
    write_json(OUT / 'generation_started.json', dict(plan_sha256=sha256(OUT / 'plan.json'), training=False, test_images_used=False))
    indices = dict(zip(val.image_id, range(len(val))))
    selected = val.loc[val.image_id.isin([c['image_id'] for c in plan['cases']])].reset_index(drop=True)
    dataset = Images(selected, f, training=False)
    local_indices = dict(zip(selected.image_id, range(len(selected))))
    data = {c['image_id']: dict(probabilities=[], heats=[], branches=[]) for c in plan['cases']}
    peak = 0
    log.info('START S83 explanations:12 validation images x6 frozen branches; FP32; no test/training')
    for slot, member in enumerate(plan['members']):
        z, c = payload(member, f)
        model = ResearchClassifier(c['model'], weights=None, attention=c['attention'], dropout=c['head_dropout'])
        model.load_state_dict(z['model'], strict=True)
        del z
        model.eval().requires_grad_(False)
        model = model.cuda().float()
        torch.cuda.reset_peak_memory_stats()
        for case in plan['cases']:
            image_id = case['image_id']
            target = CLASSES.index(case['target_class'])
            x, _, loaded_id = dataset[local_indices[image_id]]
            require(loaded_id == image_id, 'Wrong validation image')
            x = x.unsqueeze(0).cuda().float().requires_grad_(True)
            p, raw, heat, attention, meta = gradcam(model, x, target)
            ref = refs[slot][indices[image_id]]
            require(np.allclose(p, ref, atol=1e-5, rtol=1e-5), 'Branch FP32 replay mismatch: ' + member['run'])
            base_path = val.iloc[indices[image_id]].path
            with Image.open(base_path) as image:
                base = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)) / 255.
            folder = OUT / image_id / member['run']
            folder.mkdir(parents=True, exist_ok=False)
            np.save(folder / 'raw_gradcam.npy', raw)
            np.save(folder / 'normalized_gradcam.npy', heat)
            Image.fromarray((plt.get_cmap('magma')(heat)[..., :3] * 255).astype('uint8')).save(folder / 'heatmap.png')
            Image.fromarray((overlay(base, heat) * 255).astype('uint8')).save(folder / 'overlay.png')
            if attention:
                np.save(folder / 'cbam_spatial_native.npy', attention['spatial_native'])
                np.save(folder / 'cbam_spatial_weights.npy', attention['spatial'])
                write_csv(folder / 'cbam_channel_weights.csv', [dict(channel=i, weight=float(v)) for i, v in enumerate(attention['channel'])])
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
                axes[0].imshow(attention['spatial'], cmap='viridis', vmin=0, vmax=1)
                axes[0].set_title('S79 CBAM spatial sigmoid gate (not CAM)')
                axes[0].axis('off')
                axes[1].plot(attention['channel'], linewidth=.8)
                axes[1].set(xlabel='Channel index', ylabel='Sigmoid weight', ylim=(0, 1), title='All768 actual channel weights')
                fig.tight_layout()
                save(fig, folder, 'cbam_attention')
            meta.update(run=member['run'], display_name=member['display_name'], checkpoint=member['checkpoint'],
                        checkpoint_sha256=member['checkpoint_sha256'], epoch=member['epoch'], weight=1 / 6,
                        target_class=case['target_class'], probabilities=p.tolist(),
                        reference_max_difference=float(abs(p - ref).max()))
            write_json(folder / 'metadata.json', meta)
            data[image_id]['probabilities'].append(p)
            data[image_id]['heats'].append(heat)
            data[image_id]['branches'].append(meta)
            del x
        peak = max(peak, torch.cuda.max_memory_allocated())
        del model
        gc.collect()
        torch.cuda.empty_cache()
        log.info('SAVED %s:12 maps; checkpoint unchanged', member['display_name'])
    replay = []
    with PdfPages(OUT / 'S83_GradCAM_Atlas.pdf') as atlas:
        atlas.infodict()['Title'] = 'S83 frozen equal-six ensemble | Validation Grad-CAM and CBAM'
        for case in plan['cases']:
            image_id = case['image_id']
            folder = OUT / image_id
            p = np.mean(data[image_id]['probabilities'], axis=0)
            saved = frame.loc[frame.image_id == image_id, COLS].to_numpy(dtype=np.float32)[0]
            require(np.allclose(p, saved, atol=1e-5, rtol=1e-5), 'S83 FP32 replay mismatch')
            require(CLASSES[int(p.argmax())] == case['predicted_class'], 'S83 prediction changed')
            heat = np.mean(data[image_id]['heats'], axis=0)
            np.save(folder / 'display_composite.npy', heat)
            with Image.open(val.iloc[indices[image_id]].path) as image:
                base = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)) / 255.
            Image.fromarray((base * 255).astype('uint8')).save(folder / 'input_224.png')
            Image.fromarray((overlay(base, heat) * 255).astype('uint8')).save(folder / 'display_composite_overlay.png')
            fig, axes = plt.subplots(3, 3, figsize=(10, 11))
            axes = axes.ravel()
            axes[0].imshow(base)
            axes[0].set_title('Input RGB224', fontsize=11)
            for ax, member in zip(axes[1:7], plan['members']):
                with Image.open(folder / member['run'] / 'overlay.png') as image:
                    ax.imshow(np.array(image))
                ax.set_title(member['display_name'], fontsize=10)
            axes[7].imshow(overlay(base, heat))
            axes[7].set_title('Mean normalized maps\nDisplay composite only', fontsize=10)
            spatial = np.load(folder / plan['members'][5]['run'] / 'cbam_spatial_weights.npy')
            axes[8].imshow(spatial, cmap='viridis', vmin=0, vmax=1)
            axes[8].set_title('S79 CBAM spatial gate\nInternal attention, not CAM', fontsize=10)
            for ax in axes:
                ax.axis('off')
            fig.suptitle(f"S83 | {image_id}\nTrue: {case['true_class']}   Prediction/Grad-CAM target: {case['predicted_class']}   Probability: {p.max():.3f}", fontsize=13)
            fig.text(.5, .018, 'Qualitative validation example. All six maps target the same ensemble prediction.\nHotter colors = larger positive CAM within each branch; scales are normalized separately.', ha='center', fontsize=9)
            fig.subplots_adjust(top=.89, bottom=.07, hspace=.23, wspace=.05)
            atlas.savefig(fig, bbox_inches='tight')
            save(fig, folder, 'gradcam_panel')
            meta = dict(case, branches=data[image_id]['branches'], probabilities=p.tolist(),
                        saved_probabilities=saved.tolist(), max_probability_difference=float(abs(p - saved).max()),
                        gradient_target=plan['gradient_target'],
                        composite_rule='Equal mean of branch-normalized positive maps; display summary, not exact causal attribution',
                        attention_note='Actual sigmoid gates, not segmentation or proof of clinical reasoning')
            write_json(folder / 'metadata.json', meta)
            replay.append(dict(image_id=image_id, true_class=case['true_class'], predicted_class=case['predicted_class'],
                               max_probability_difference=meta['max_probability_difference'], **dict(zip(COLS, p.tolist()))))
    write_csv(OUT / 'selected_probability_replay.csv', replay)
    gallery(plan)
    write_json(OUT / 'generation_completed.json', dict(status='generated_pending_verification', seconds=time.perf_counter() - tick,
        device=torch.cuda.get_device_name(0), torch_version=torch.__version__, peak_allocated_mib=peak / 1024 ** 2,
        precision='FP32', tf32=False, parameter_updates=0, test_images_used=False))
    verify()
    log.info('COMPLETED S83 XAI:72 Grad-CAM maps;12 CBAM pairs; unchanged predictions')


def select_cases_with_inputs(frame, val):
    cases = select_cases(frame)
    for case in cases:
        row = val.loc[val.image_id == case['image_id']].iloc[0]
        case.update(input_sha256=sha256(row.path), lesion_id=row.lesion_id)
    return cases


def gallery(plan):
    sections = []
    for case in plan['cases']:
        image_id = html.escape(case['image_id'])
        sections.append(f'<section><h2>{image_id} | true {case["true_class"]} | predicted {case["predicted_class"]}</h2>'
                        f'<p>{case["category"]} | <a href="{image_id}/gradcam_panel.pdf">Case PDF</a> | '
                        f'<a href="{image_id}/metadata.json">Provenance</a></p><img src="{image_id}/gradcam_panel.png" alt="Six branch Grad-CAM panel for {image_id}"></section>')
    text = '<!doctype html><html lang="en"><meta charset="utf-8"><title>S83 Grad-CAM gallery</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;background:#f6f7f9;color:#17212b}section{background:white;padding:24px;margin:28px 0;border-radius:10px}img{width:100%;height:auto}a{color:#135c9b}h1{font-size:32px}h2{font-size:20px}</style><h1>Frozen S83: Grad-CAM + CBAM</h1>'
    text += '<p>Equal-six probability ensemble; FP32 identity RGB224. Twelve preselected exploratory validation examples. No training or test evaluation.</p>'
    text += '<p><a href="S83_GradCAM_Atlas.pdf">Open the12-page atlas</a> | <a href="verification.json">Verification receipt</a> | <a href="README.md">Method and limitations</a></p>'
    text += '<p>All branch maps explain the saved ensemble prediction. Independently normalized maps and their display composite are qualitative, not calibrated localization or proof of medical reasoning. CBAM panels show actual internal sigmoid gates.</p>'
    atomic_text(OUT / 'gallery.html', text + ''.join(sections) + '</html>')


def verify():
    plan, f = read_plan()
    _, val = development()
    frame, refs, _, _ = reference(f, val)
    require(plan['cases'] == select_cases_with_inputs(frame, val), 'Selected inputs changed')
    stats, degenerate, branch_diffs, ensemble_diffs = [], [], [], []
    for case in plan['cases']:
        folder = OUT / case['image_id']
        meta = json.loads((folder / 'metadata.json').read_text())
        require(meta['target_class'] == case['predicted_class'], 'CAM target changed')
        require([b['run'] for b in meta['branches']] == [m['run'] for m in plan['members']], 'Missing/colliding branches')
        index = val.index[val.image_id == case['image_id']][0]
        arrays, heats = [], []
        for slot, (b, m) in enumerate(zip(meta['branches'], plan['members'])):
            branch = folder / m['run']
            require(b['weight'] == 1 / 6 and b['checkpoint_sha256'] == m['checkpoint_sha256'] and b['epoch'] == m['epoch'], 'Branch provenance changed')
            require(b['target_class'] == case['target_class'], 'Branch target changed')
            p = np.array(b['probabilities'], dtype=np.float32)
            require(np.isfinite(p).all() and (p >= 0).all() and np.isclose(p.sum(), 1, atol=1e-5), 'Invalid replay probabilities')
            require(np.allclose(p, refs[slot][index], atol=1e-5, rtol=1e-5), 'Branch reference mismatch')
            branch_diffs.append(float(abs(p - refs[slot][index]).max()))
            raw, heat = [np.load(branch / name) for name in ['raw_gradcam.npy', 'normalized_gradcam.npy']]
            require(all(a.shape == (224, 224) and np.isfinite(a).all() and (a >= 0).all() for a in [raw, heat]), 'Invalid CAM')
            require((heat <= 1).all(), 'Invalid normalized CAM')
            zero = float(raw.max()) <= 0
            require(zero == b['degenerate_map'] and np.allclose(heat, raw if zero else raw / raw.max()), 'CAM normalization/zero flag mismatch')
            if zero:
                degenerate.append(dict(image_id=case['image_id'], run=m['run']))
            if m['attention'] == 'cbam':
                native = np.load(branch / 'cbam_spatial_native.npy')
                spatial = np.load(branch / 'cbam_spatial_weights.npy')
                channel = pd.read_csv(branch / 'cbam_channel_weights.csv')
                require(native.shape == (7, 7) and spatial.shape == (224, 224) and len(channel) == 768, 'CBAM shapes differ')
                require(channel.channel.tolist() == list(range(768)), 'CBAM channel indices differ')
                require(all(np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all() for a in [native, spatial, channel.weight.to_numpy()]), 'Invalid CBAM gates')
            arrays.append(p)
            heats.append(heat)
        p = np.mean(arrays, axis=0)
        saved = frame.loc[frame.image_id == case['image_id'], COLS].to_numpy(dtype=np.float32)[0]
        require(np.allclose(p, saved, atol=1e-5, rtol=1e-5) and CLASSES[int(p.argmax())] == case['predicted_class'], 'Ensemble replay changed')
        ensemble_diffs.append(float(abs(p - saved).max()))
        require(np.allclose(np.load(folder / 'display_composite.npy'), np.mean(heats, axis=0)), 'Composite mismatch')
        stats.append(dict(image_id=case['image_id'], correct=case['correct'], true_class=case['true_class'], predicted_class=case['predicted_class']))
    images = list(OUT.rglob('*.png'))
    for path in images:
        with Image.open(path) as image:
            image.verify()
    # Use the bundled Poppler command; no new dependency in the training venv.
    pdfs = list(OUT.rglob('*.pdf'))
    require(len(pdfs) == 25, 'Missing case/attention/atlas PDFs')
    for path in pdfs:
        info = subprocess.check_output(['pdfinfo', str(path)], text=True)
        pages = re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE)
        require(pages is not None and int(pages[1]) == (12 if path.name == 'S83_GradCAM_Atlas.pdf' else 1), 'PDF page count differs')
    require(len(images) == 192, 'Unexpected PNG count')
    result = dict(status='passed', cases=12, branch_gradcams=72, cbam_cases=12, true_classes=list(CLASSES),
        full_1503_reference_max_difference=0.0, branch_replay_max_difference=max(branch_diffs),
        ensemble_replay_max_difference=max(ensemble_diffs), label_changes=0, degenerate_maps=degenerate,
        pngs_verified=len(images), pdfs_verified=len(pdfs), atlas_pages=12,
        frozen_checkpoints_unchanged=True, frozen_method_unchanged=True, historical_xai_and_test_artifacts_unchanged=True,
        input_hashes_verified=True, parameters_updated=False, test_images_used=False,
        numerical_tolerance=dict(atol=1e-5, rtol=1e-5), visual_review='Required separately before publication')
    write_json(OUT / 'verification.json', result)
    manifest = dict(status='completed', source_experiment=f['source_experiment'], plan_sha256=sha256(OUT / 'plan.json'),
                    cases=stats, verification_path=relative(OUT / 'verification.json'),
                    artifact_sha256={relative(p): sha256(p) for p in sorted(OUT.rglob('*'))
                                     if p.is_file() and p.name not in ['manifest.json', 'xai.log', 'verification.json']})
    # A verify rerun may refresh only this new receipt, never old evidence/model state.
    write_json(OUT / 'manifest.json', manifest)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ['prepare', 'generate', 'verify']:
        group.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.generate:
        generate()
    else:
        verify()


if __name__ == '__main__':
    main()
