"""One authorized frozen S83 six-model audit. No training or test-driven tuning."""
import argparse
import gc
import hashlib
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from research.common import ROOT, CLASSES, relative, sha256, write_json, write_csv, atomic_text
from research.strict_protocol import SPLIT, DIGEST
from research.strict_train import run_lock, runtime_versions
from research.evaluate_final_locked_test import LockedImages
from research.final_cbam_reporting.finalize import cpu_model, scored_package, tree_hashes
from research.final_cbam_development.runtime import deterministic_cuda
from research.short_screening.feature_fusion import development
from research.plots import comparison_figures
from research.registry import upsert

OUT = ROOT/'results/final_exploratory_test_audit/v1'
METHOD = ROOT/'results/final_exploratory_freeze/v1/frozen_method.json'
RID = 's99_s83_frozen_six_postdevelopment_test_audit_seed42'
COLS = [f'p_{c}' for c in CLASSES]
CODE = [relative(__file__), 'research/models.py', 'src/cbam.py', 'research/common.py',
        'research/evaluate_final_locked_test.py', 'research/final_cbam_reporting/finalize.py',
        'research/short_screening/lesion_bag_screen.py', 'research/plots.py',
        'research/final_cbam_development/runtime.py', 'research/strict_train.py',
        'research/strict_protocol.py', 'research/short_screening/feature_fusion.py',
        'research/registry.py']
OLD = [ROOT/'results/final_locked_test/v1', ROOT/'results/final_cbam_reporting/v1']


def prepare():
    if (OUT/'inference_started.json').exists():
        raise RuntimeError('Audit already started; do not rewrite its pre-inference freeze')
    torch.set_num_threads(4)
    f = json.loads(METHOD.read_text())
    assert f['source_experiment']=='s83_cbam_f1_addition_equal_six_exploratory_seed42'
    assert f['class_order']==list(CLASSES) and f['weights']==[1/6]*6
    assert f['inference_precision']=='FP32' and f['views']==['identity'] and not f['tf32']
    assert f['image_size']==224 and f['normalization_mean']==[.485,.456,.406] and f['normalization_std']==[.229,.224,.225]
    assert sha256(ROOT/f['split_manifest'])==f['split_sha256']
    assert sha256(ROOT/f['validation_predictions'])==f['validation_prediction_sha256']
    assert sha256(ROOT/SPLIT)==DIGEST
    identities = pd.read_csv(ROOT/SPLIT, usecols=['image_id','lesion_id','split'])
    test = identities.loc[identities.split=='test'].reset_index(drop=True)
    train, val = development(); dev = pd.concat([train,val])
    assert len(test)==1503 and test.image_id.is_unique and not test.isna().any().any()
    assert not set(test.image_id)&set(dev.image_id)
    assert not set(test.lesion_id)&set(dev.lesion_id)
    expected = ['convnext_tiny','convnext_small','densenet201','efficientnet_v2_s','efficientnet_b0','convnext_tiny']
    assert [s['model'] for s in f['sources']]==expected
    members = []
    for i, s in enumerate(f['sources']):
        path = ROOT/s['checkpoint']; assert sha256(path)==s['checkpoint_sha256']
        cp = torch.load(path, map_location='cpu', weights_only=False); c = cp['config']
        assert cp['best_epoch']==s['epoch'] and cp['selection_metric']==s['selection']
        assert list(cp['class_order'])==list(CLASSES)
        assert c['split_manifest']==f['split_manifest'] and c['split_sha256']==f['split_sha256']
        assert c['model']==s['model'] and c['weights']==s['pretrained_weights']
        assert c['attention']==('cbam' if i==5 else 'none')
        assert c['image_size']==224 and c['normalization_mean']==f['normalization_mean'] and c['normalization_std']==f['normalization_std']
        m = dict(s, sha256=s['checkpoint_sha256'], attention=c['attention'], head_dropout=c['head_dropout'])
        del cp; gc.collect()
        net = cpu_model(m)
        with torch.inference_mode(): z = net(torch.zeros(1,3,224,224))
        assert z.shape==(1,7) and torch.isfinite(z).all()
        del net,z; gc.collect(); members.append(m)
    prepared = dict(method=f, frozen_method_sha256=sha256(METHOD), members=members,
                    validation_metrics_sha256=sha256(ROOT/f['validation_metrics']),
                    validation_predictions_sha256=sha256(ROOT/f['validation_predictions']),
                    class_order=list(CLASSES), normalization_mean=f['normalization_mean'], normalization_std=f['normalization_std'],
                    cohort_size=1503, test_manifest=SPLIT, test_manifest_sha256=DIGEST,
                    test_image_ids=test.image_id.tolist(), test_lesion_count=int(test.lesion_id.nunique()),
                    test_identity_sha256=hashlib.sha256('\n'.join(test.image_id).encode()).hexdigest(),
                    training_validation_to_test_image_overlap=0, training_validation_to_test_lesion_overlap=0,
                    identity_only_preparation=True, test_labels_loaded=False, test_images_loaded=False,
                    precision='FP32', autocast=False, tf32=False, views=['identity'], batch_size=16, workers=0,
                    weights=[1/6]*6, code_sha256={p:sha256(ROOT/p) for p in CODE},
                    old_artifacts={relative(p):tree_hashes(p) for p in OLD},
                    original_first_test_report_sha256=sha256(ROOT/'research/FINAL_LOCKED_TEST_RESULTS.md'),
                    authorization='User sure go! after review of exact frozen S83 six-model method',
                    pristine_first_test=False, final_s83_audit=True, future_test_tuning=False,
                    probability_fusion='NumPy float32 mean of six FP32 softmax vectors in frozen member order')
    write_json(OUT/'pre_inference_verification.json', prepared)
    write_json(OUT/'cpu_preflight.json', dict(status='passed', frozen_checkpoints_strict_loaded=True,
               synthetic_cpu_forward_finite=True, class_epoch_selection_preprocessing_verified=True,
               cohort_size=1503, zero_development_to_test_lesion_overlap=True,
               cuda_initialized=torch.cuda.is_initialized(), labels_read=False, test_images_loaded=False))
    assert not torch.cuda.is_initialized()
    print('CPU preflight passed: six unchanged checkpoints;1503 identities;zero development/test image and lesion overlap; no test labels/images read', flush=True)


def frozen():
    f = json.loads((OUT/'pre_inference_verification.json').read_text())
    assert sha256(METHOD)==f['frozen_method_sha256']
    assert sha256(ROOT/SPLIT)==f['test_manifest_sha256']
    assert sha256(ROOT/f['method']['split_manifest'])==f['method']['split_sha256']
    assert sha256(ROOT/f['method']['validation_metrics'])==f['validation_metrics_sha256']
    assert sha256(ROOT/f['method']['validation_predictions'])==f['validation_predictions_sha256']
    assert f['code_sha256']=={p:sha256(ROOT/p) for p in CODE}
    assert sha256(ROOT/'research/FINAL_LOCKED_TEST_RESULTS.md')==f['original_first_test_report_sha256']
    for folder, hashes in f['old_artifacts'].items():
        assert tree_hashes(ROOT/folder)==hashes
    for m in f['members']:
        assert sha256(ROOT/m['checkpoint'])==m['sha256']
    return f


def score():
    f = frozen(); done = json.loads((OUT/'inference_completed.json').read_text())
    assert done['model_passes']==6 and len(done['probability_files'])==6
    # Test labels are parsed only after every frozen model has saved its full pass.
    ids = pd.read_csv(ROOT/SPLIT, usecols=['image_id','split'])
    excluded = set((ids.index[ids.split!='test']+1).tolist())
    cohort = pd.read_csv(ROOT/SPLIT, skiprows=lambda row:row in excluded).reset_index(drop=True)
    assert cohort.image_id.tolist()==f['test_image_ids']
    assert cohort.label.tolist()==cohort.diagnosis.map(dict(zip(CLASSES,range(7)))).tolist()
    arrays = []; comparison = []
    for path, m, digest in zip(done['probability_files'], f['members'], done['probability_sha256']):
        assert sha256(ROOT/path)==digest
        frame = pd.read_csv(ROOT/path); assert frame.image_id.tolist()==f['test_image_ids']
        p = frame[COLS].to_numpy(dtype=np.float32); arrays.append(p)
        title = m['model']+(' + CBAM' if m['attention']=='cbam' else '')
        metrics = scored_package(OUT/m['run'], cohort.label.to_numpy(), p, cohort, title)
        comparison.append(dict(display_name=title, **{k:metrics[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']}))
    final = np.mean(arrays, axis=0)
    metrics = scored_package(OUT/'ensemble', cohort.label.to_numpy(), final, cohort, 'Frozen S83 equal-six')
    comparison.append(dict(display_name='S83 equal-six + CBAM', **{k:metrics[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']}))
    write_csv(OUT/'constituent_test_comparison.csv', comparison)
    comparison_figures(comparison, OUT/'comparison_figures', 'Frozen S83 constituents and ensemble | descriptive test audit')
    val = json.loads((ROOT/f['method']['validation_metrics']).read_text())
    comparison_figures([dict(display_name='S83 exploratory validation', **{k:val[k] for k in ['accuracy','macro_f1']}),
                       dict(display_name='S83 post-development test audit', **{k:metrics[k] for k in ['accuracy','macro_f1']})],
                       OUT/'validation_test_comparison', 'Frozen S83 | descriptive cohort comparison')
    result = dict(status='completed', **{k:metrics[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','correct','incorrect']},
                  melanoma_recall=metrics['per_class']['mel']['recall'], akiec_recall=metrics['per_class']['akiec']['recall'],
                  validation_accuracy=val['accuracy'], validation_macro_f1=val['macro_f1'],
                  test_minus_validation_accuracy_pp=100*(metrics['accuracy']-val['accuracy']),
                  test_minus_validation_macro_f1=metrics['macro_f1']-val['macro_f1'],
                  inference_seconds=done['inference_seconds'], fresh_model_passes=6, images_per_model=1503,
                  cohort_previously_evaluated=True, pristine_first_test=False,
                  method_changed_after_test=False, checkpoint_hashes_unchanged=True,
                  old_test_artifacts_unchanged=True, performance_work_finished=True)
    frozen()  # Recheck sources and prior evidence after all output generation.
    write_json(OUT/'summary.json', result)
    write_json(OUT/'completion.json', result)
    upsert(dict(experiment_id=RID, era='structured', record_kind='final_postdevelopment_test_audit',
               phase='final_exploratory_reporting', protocol='exploratory_trained_postdevelopment_test_audit',
               evaluation_split='previously_evaluated_heldout_test', split_manifest=SPLIT, split_sha256=DIGEST,
               method='Frozen S83 equal-six FP32 identity inference; no training/test tuning',
               model='+'.join(m['model'] for m in f['members']), attention='CBAM in S79 only',
               ensemble_members=json.dumps([m['run'] for m in f['members']]), ensemble_weights=json.dumps([1/6]*6),
               image_size=224, seed=42, epochs=0, status='completed', decision='performance_work_finished_no_method_change',
               config_path=relative(OUT/'pre_inference_verification.json'), metrics_path=relative(OUT/'ensemble/test_metrics.json'),
               plots_dir=relative(OUT/'ensemble/figures'), confusion_matrix_path=relative(OUT/'ensemble/figures/confusion_matrix.csv'),
               runtime_seconds=done['inference_seconds'], source_sha256=sha256(OUT/'ensemble/test_metrics.json'),
               notes='Final recorded S83 audit of previously evaluated cohort; repeated exploratory selection after known prior test outcomes; no pristine first-test claim or later method tuning.',
               **{k:metrics[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    classes = '| Class | Precision | Recall | F1 | Support |\n|---|---:|---:|---:|---:|\n'
    for name, a in metrics['per_class'].items():
        classes += f"| {name} | {a['precision']:.6f} | {a['recall']:.6f} | {a['f1']:.6f} | {a['support']} |\n"
    table = '| Frozen component | Accuracy | Macro-F1 |\n|---|---:|---:|\n'
    for r in comparison:
        table += f"| {r['display_name']} | {100*r['accuracy']:.4f}% | {r['macro_f1']:.6f} |\n"
    text = f'''# Final S83 six-model test audit

10 October2026. User authorized this final fixed-method audit. The original cohort was already evaluated in earlier S31/S81 stages. This is a **repeated post-development test audit**, not another first untouched independent test. The original first-test report/artifacts and earlier audit remain unchanged. Test results are not used to change any model, checkpoint, preprocessing or ensemble rule afterward.

## Frozen method

S06 Tiny F1 epoch18 + S18 Small F1 epoch17 + S15 DenseNet201 F1 epoch14 + S10 EfficientNetV2-S F1 epoch15 + S03 B0 accuracy epoch20 + S79 Tiny-CBAM F1 epoch35. Equal1/6 probability averaging; FP32 identity224 RGB bilinear square/ImageNet normalization; no TF32, AMP, TTA, multi-resolution, calibration, metadata or inference augmentation. Six fresh passes with labels absent from the inference loader. Standalone scores use those same saved probabilities, not additional inference or checkpoint selection.

Exact hashes, epochs, class order, source/config freeze,1503-image identities and zero training/validation-to-test image/lesion overlap were verified before inference in `results/final_exploratory_test_audit/v1/pre_inference_verification.json`. The development split itself is image-level and can share lesions between training/validation.

## Final result

- Accuracy: **{100*metrics['accuracy']:.4f}%**; macro-F1: **{metrics['macro_f1']:.6f}**.
- Macro precision:{metrics['macro_precision']:.6f}; macro recall:{metrics['macro_recall']:.6f}; weighted-F1:{metrics['weighted_f1']:.6f}.
- Correct:{metrics['correct']}; incorrect:{metrics['incorrect']}; cohort1503.
- Melanoma recall:{100*result['melanoma_recall']:.2f}%; akiec recall:{100*result['akiec_recall']:.2f}%.

{classes}

Rows=true, columns=predicted; class order{list(CLASSES)}:

```text
{np.asarray(metrics['confusion_matrix'])}
```

## Constituent descriptions from the same passes

{table}

## Validation versus audit

S83 exploratory validation:{100*val['accuracy']:.4f}% /{val['macro_f1']:.6f}; final test audit:{100*metrics['accuracy']:.4f}% /{metrics['macro_f1']:.6f}. Audit minus validation:{result['test_minus_validation_accuracy_pp']:+.4f} accuracy percentage points /{result['test_minus_validation_macro_f1']:+.6f} macro-F1. This is a descriptive generalization gap across different cohorts. Highest repeatedly selected image-level validation performance is not an independent test estimate and must not be advertised as test accuracy.

Full metrics, complete predictions/probabilities, seven-class scores, raw/normalized confusion matrices and PNG/PDF figures: `results/final_exploratory_test_audit/v1/ensemble/`. Member packages, comparison tables/figures, durable inference receipts and source verification are in the parent folder. No checkpoints were overwritten. Performance experiments are finished; next work is exact-method Grad-CAM/XAI and final report/paper figures and writing.
'''
    atomic_text(ROOT/'research/FINAL_S83_TEST_AUDIT.md', text)
    print(json.dumps(result, indent=2), flush=True)


def evaluate():
    f = frozen()
    if (OUT/'inference_started.json').exists():
        raise RuntimeError('This S83 audit already started; another inference pass is prohibited. Use saved outputs only.')
    deterministic_cuda(42); assert torch.cuda.is_available()
    identities = pd.read_csv(ROOT/SPLIT, usecols=['image_id','lesion_id','split'])
    test = identities.loc[identities.split=='test'].reset_index(drop=True)
    assert test.image_id.tolist()==f['test_image_ids']
    with (OUT/'inference_started.json').open('x', encoding='utf-8') as receipt:
        json.dump(dict(pid=os.getpid(), started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                       freeze_sha256=sha256(OUT/'pre_inference_verification.json'),
                       final_s83_audit=True, pristine_first_test=False, automatic_retry=False), receipt)
        receipt.flush(); os.fsync(receipt.fileno())
    write_json(OUT/'environment.json', dict(runtime=runtime_versions(), device=torch.cuda.get_device_name(0)))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(OUT/'audit.log'),logging.StreamHandler()])
    logger = logging.getLogger('s83_final_audit'); tick = time.perf_counter(); outputs = []
    try:
        dataset = LockedImages(test, f)
        logger.info('START S83 final audit: six frozen FP32 identity passes; no training or label use in inference')
        for m in f['members']:
            target = OUT/m['run']/'inference_probabilities_unscored.csv'
            if target.exists(): raise FileExistsError('Member already inferred; preserve evidence')
            model = cpu_model(m).cuda().eval(); rows = []
            with torch.inference_mode():
                for i,(x, image_ids) in enumerate(DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0,pin_memory=True)):
                    x = x.cuda(non_blocking=True).float(); z = model(x); p = z.softmax(1)
                    if not torch.isfinite(z).all() or not torch.isfinite(p).all():
                        raise FloatingPointError('Nonfinite FP32 output; no fallback or clipping')
                    rows.extend(dict(image_id=image_id, **dict(zip(COLS,a.tolist()))) for image_id,a in zip(image_ids,p.cpu()))
                    if i==0:
                        write_json(OUT/'startup_confirmation.json', dict(pid=os.getpid(), gpu_inference_active=True,
                                   logging_active=True, output_generation_active=True, labels_used=False, member=m['run']))
                    if i==0 or (i+1)%30==0: logger.info('%s batch%d/94', m['run'], i+1)
            assert [r['image_id'] for r in rows]==f['test_image_ids']
            write_csv(target, rows); outputs.append(relative(target))
            logger.info('SAVED %s:1503 probability vectors', m['run'])
            del model,x,z,p; gc.collect(); torch.cuda.empty_cache()
        write_json(OUT/'inference_completed.json', dict(status='completed', model_passes=6, images_per_model=1503,
                   labels_used_during_inference=False, probability_files=outputs,
                   probability_sha256=[sha256(ROOT/p) for p in outputs], inference_seconds=time.perf_counter()-tick))
        score(); logger.info('COMPLETED final S83 audit; no method change, next experiment or repeated inference')
    except BaseException as exc:
        write_json(OUT/'failure.json', dict(error=repr(exc), no_automatic_retry=True)); logger.exception('Preserved evidence; no second inference'); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true'); mode.add_argument('--run',action='store_true'); mode.add_argument('--report',action='store_true')
    a = parser.parse_args()
    with run_lock(RID):
        if a.check: prepare()
        elif a.run: evaluate()
        else: score()


if __name__=='__main__':
    main()
