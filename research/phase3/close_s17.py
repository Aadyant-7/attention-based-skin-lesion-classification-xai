from pathlib import Path
import json,numpy as np,pandas as pd
from research.common import ROOT,CLASSES,sha256,write_json,write_csv
from research.fuse import preflight
from research.phase3.close_s02 import verify_predictions,verify_figures,load
from research.registry import read_registry,upsert
from research.plots import comparison_figures
rid='s17_s06_s15_equal_probability_exploratory_seed42';p=ROOT/'results/structured_experiments'/rid
c=load(p/'config.json');m=load(p/'validation_metrics.json');r=load(p/'record.json')
f,v,parents,prob=preflight(c)
cm=verify_predictions(p/'validation_predictions.csv',m,v);verify_figures(p/'figures',cm,m)
found=pd.read_csv(p/'validation_predictions.csv').set_index('image_id').loc[v.index]
assert np.allclose(found[[f'p_{a}' for a in CLASSES]],sum(w*a for w,a in zip(c['weights'],prob)),rtol=0,atol=1e-12)
from research.train import metric_report
labels=v.label.to_numpy();counts=f.query("split=='train'").diagnosis.value_counts();w=np.sqrt(7009/np.array([counts[a] for a in CLASSES]));w/=w.mean()
loss=float(np.average(-np.log(np.clip(sum(wi*a for wi,a in zip(c['weights'],prob))[np.arange(len(v)),labels],1e-12,1)),weights=w[labels]))
assert metric_report(labels,found[[f'p_{a}' for a in CLASSES]].to_numpy(),loss)==m
ok=found.predicted_class.to_numpy()==v.diagnosis.to_numpy();gains={};rows=[];classrows=[]
refs={'S06':'s06_convnext_tiny_none_exploratory_seed42','S10':'s10_efficientnet_v2_s_none_exploratory_seed42','S12':'s12_s03_s06_s10_equal_probability_exploratory_seed42','S13 FP32 identity':'s13_s12_four_flip_tta_exploratory_seed42','S15':'s15_densenet201_none_exploratory_seed42','S17':rid}
for name,parent in refs.items():
 q=ROOT/'results/structured_experiments'/parent
 suffix='_identity_fp32' if name=='S13 FP32 identity' else ''
 mm=load(q/f'validation_metrics{suffix}.json');pr=pd.read_csv(q/f'validation_predictions{suffix}.csv').set_index('image_id').loc[v.index]
 oo=pr.predicted_class.to_numpy()==v.diagnosis.to_numpy()
 gains[name]=dict(fixed=int((ok&~oo).sum()),broken=int((~ok&oo).sum()),net=int(ok.sum()-oo.sum()))
 rows.append(dict(display_name=name,accuracy=mm['accuracy'],macro_f1=mm['macro_f1']))
 classrows.extend(dict(method=name,class_name=a,**mm['per_class'][a]) for a in CLASSES)
r.update(decision='exclude_from_final_ensemble_marginal_gain',notes=r['notes']+' CPU closeout verified exact probabilities, recomputed metrics/loss, matrices and class figures. Only +2 correct vs FP32 reference, below predeclared material-gain gate; no DenseNet weighting experiments.')
before=[x for x in read_registry() if x['experiment_id']!=rid];upsert(r);write_json(p/'record.json',r);assert before==[x for x in read_registry() if x['experiment_id']!=rid]
report=dict(status='verified_completed',metrics=m,paired_gains=gains,checkpoint_identities=c['checkpoint_sha256'],prediction_hashes=c['prediction_sha256'],test_images_loaded=False,gpu_used=False,decision=r['decision'])
write_json(p/'closeout_verification.json',report)
out=ROOT/'results/model_comparison/structured/s15_backbone_review';comparison_figures(rows,out,'Same exploratory validation | S15 and one fixed fusion');write_csv(out/'per_class_comparison.csv',classrows)
print(json.dumps(dict(accuracy=m['accuracy'],macro_f1=m['macro_f1'],paired_gains=gains),indent=2))
