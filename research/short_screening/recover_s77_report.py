"""Recover S77's post-scoring path typo without inference, fitting or new tuning."""
import csv
import io
import json
import subprocess
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research.common import ROOT, sha256, write_json
from research.plots import validate_metrics
from research.short_screening.panderm_large_screen import OUT, CACHE, FILES, S53, COLS, MODEL_SOURCE, SPLIT
from research.short_screening.feature_fusion import kernel
from research.short_screening.lesion_bag_screen import report


def main():
    if (OUT/'summary.json').exists():
        print('Completed recovery preserved');return
    sig=json.loads((OUT/'signature.json').read_text())
    assert sha256(OUT/'runner_at_execution.py.txt')==sig['runner_sha256']
    assert sha256(MODEL_SOURCE)==sig['source_sha256'] and sha256(SPLIT)==sig['split_sha256']
    assert sha256(ROOT/'research/short_screening/feature_fusion.py')==sig['shared_runner_sha256']
    assert all(sha256(FILES[k])==v for k,v in sig['weights_sha256'].items())
    assert all(sha256(ROOT/s['path'])==s['sha256'] for s in json.loads((OUT/'feature_cache_manifest.json').read_text())['files'])
    metrics={};frames={}
    for kind in FILES:
        folder=OUT/kind;f=pd.read_csv(folder/'validation_predictions.csv');frames[kind]=f
        m=json.loads((folder/'validation_metrics.json').read_text());validate_metrics(m)
        with np.load(CACHE/f'{kind}_val.npz',allow_pickle=False) as arr:
            assert arr['ids'].tolist()==f.image_id.tolist()
            y=arr['y'];features=arr['features'].astype(np.float64)
        p=f[COLS].to_numpy();recomputed=report(y,p)
        assert recomputed['confusion_matrix']==m['confusion_matrix']
        for key in ['accuracy','macro_precision','macro_recall','macro_f1']:
            assert abs(m[key]-recomputed[key])<1e-12
        meta=json.loads((folder/'fit_metadata.json').read_text())
        assert sha256(ROOT/meta['checkpoint'])==meta['checkpoint_sha256']
        saved=joblib.load(ROOT/meta['checkpoint'])
        with threadpool_limits(limits=4):
            x=saved['scaler'].transform(features)*np.sqrt(saved['gamma'])
            np.testing.assert_allclose(saved['classifier'].predict_proba(kernel(x,saved['training_features'])),p,atol=1e-12,rtol=1e-12)
        metrics[kind]=m
    ref=pd.read_csv(S53/'validation_predictions.csv').set_index('image_id').loc[frames['large'].image_id]
    assert ref.true_class.tolist()==frames['large'].true_class.tolist()
    refp=ref[COLS].to_numpy();refm=report(y,refp)
    a,b=metrics['base'],metrics['large']
    eligible=b['accuracy']>=a['accuracy']+.01-1e-12 and b['macro_f1']>=a['macro_f1'] and b['per_class']['mel']['recall']>=a['per_class']['mel']['recall'] and b['accuracy']>=.90
    assert not eligible, 'This recovery is for the observed failed gate only; no new ensemble is computed'
    correct=refp.argmax(1)==y;large_correct=frames['large'][COLS].to_numpy().argmax(1)==y
    complement=dict(s53_errors_large_fixes=int((~correct&large_correct).sum()),s53_correct_large_misses=int((correct&~large_correct).sum()))
    old=list(csv.DictReader(io.StringIO(subprocess.check_output(['git','show','HEAD:results/master_experiment_registry.csv']).decode('utf-8-sig'))))
    now=list(csv.DictReader(open(ROOT/'results/master_experiment_registry.csv',encoding='utf-8-sig',newline='')))
    lookup={x['experiment_id']:x for x in now};assert len(lookup)==len(now)
    assert all(lookup[x['experiment_id']]==x for x in old)
    summary=dict(status='completed',metrics=metrics,reference_metrics=refm,s78_condition_passed=False,material_fusion_gate=False,
                 complementarity=complement,backbone_training=False,test_loaded=False,
                 decision='Large failed predefined standalone gate; S78 not run; retain S53; no search rescue',
                 recovery='Post-scoring WindowsPath indexing typo corrected; original artifacts retained; no inference/refit')
    write_json(OUT/'summary.json',summary)
    write_json(OUT/'verification.json',dict(status='passed',source_and_cache_hashes_unchanged=True,head_fitting_train_only=True,
        saved_heads_reproduce_predictions=True,metrics_recomputed=True,test_labels_read=False,test_images_loaded=False,backbone_updates=0,
        prior_registry_rows_unchanged=len(old),new_rows=len(now)-len(old)))
    write_json(OUT/'recovery_amendment.json',dict(original_runner_sha256=sig['runner_sha256'],
        corrected_runner_sha256=sha256(ROOT/'research/short_screening/panderm_large_screen.py'),
        recovery_runner_sha256=sha256(__file__),reason='ROOT[path] corrected to ROOT/path in final hash check; completed-run guard moved before prepare',
        original_signature_preserved=True,inference_repeated=False,classifier_refitted=False,metrics_changed=False))
    write_json(OUT/'progress.json',dict(status='completed',active_process=False,material_fusion_gate=False))
    print(json.dumps(complement));print('Recovery verified; no inference, fitting or ensemble performed')


if __name__=='__main__':main()
