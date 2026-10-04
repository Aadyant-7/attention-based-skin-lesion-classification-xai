"""Bounded final architecture evidence, saved probabilities only; never loads test images."""
import itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import ROOT,CLASSES,sha256,write_json,write_csv,atomic_text
from .registry import FIELDS,read_registry,upsert
from .plots import metric_figures,comparison_figures
from .train import metric_report,validate_config
from .phase3.close_s02 import load,verify_predictions,verify_figures,verify_launch_code

OUT=ROOT/'results/model_comparison/final_architecture_selection'
BATCH=ROOT/'results/architecture_selection/final_v1'
CONFIG=ROOT/'research/configs/final_architecture_selection_v1.json'
OLD=['s02_mobilenet_v3_large_none_exploratory_seed42','s03_efficientnet_b0_none_exploratory_seed42','s05_efficientnet_b0_cbam_exploratory_seed42','s10_efficientnet_v2_s_none_exploratory_seed42','s15_densenet201_none_exploratory_seed42','s06_convnext_tiny_none_exploratory_seed42','s18_convnext_small_none_exploratory_seed42']
TINY,SMALL,V2,B0=OLD[5],OLD[6],OLD[3],OLD[1]
S12='s12_s03_s06_s10_equal_probability_exploratory_seed42'
DENSE_FUSION='s17_s06_s15_equal_probability_exploratory_seed42'
REF='s13_s12_identity_fp32_control_exploratory_seed42'
FAMILY={'efficientnet_b0':'compound-scaled CNN','efficientnet_b3':'compound-scaled CNN','efficientnet_v2_s':'fused/MBConv CNN','mobilenet_v3_large':'mobile CNN','densenet201':'dense CNN','convnext_tiny':'modern CNN','convnext_small':'modern CNN','resnet101':'residual CNN'}

def val_frame():
 return pd.read_csv(ROOT/'data/splits/exploratory/image_level_dev_v1.csv').query("split=='val'").set_index('image_id')

def evidence(rid):
 if rid==REF:
  p=ROOT/'results/structured_experiments/s13_s12_four_flip_tta_exploratory_seed42'
  r=next(r for r in read_registry() if r['experiment_id']==REF)
  m=load(p/'validation_metrics_identity_fp32.json');csv=p/'validation_predictions_identity_fp32.csv'
 else:
  p=ROOT/'results/structured_experiments'/rid;r=load(p/'record.json');m=load(p/'validation_metrics.json');csv=p/'validation_predictions.csv'
 assert r['status']=='completed' and r['protocol']=='exploratory_image_level' and r['evaluation_split']=='validation'
 assert r['split_sha256']==sha256(ROOT/'data/splits/exploratory/image_level_dev_v1.csv')
 assert sha256(ROOT/r['metrics_path'])==r['source_sha256']
 v=val_frame();verify_predictions(csv,m,v)
 found=pd.read_csv(csv).set_index('image_id').loc[v.index]
 prob=found[[f'p_{c}' for c in CLASSES]].to_numpy()
 calculated=metric_report(v.label.to_numpy(),prob,m['loss'])
 for key in ('accuracy','macro_precision','macro_recall','macro_f1'):
  assert np.isclose(m[key],calculated[key],rtol=0,atol=1e-12)
 return dict(id=rid,record=r,metrics=m,prob=prob,csv=csv,ok=found.predicted_class.to_numpy()==v.diagnosis.to_numpy())

def verify_training(rid):
 p=ROOT/'results/structured_experiments'/rid;c=load(p/'config.json');r=load(p/'record.json');validate_config(c)
 assert r['status']=='completed';h=pd.read_csv(p/'history.csv')
 assert h.epoch.tolist()==list(range(1,len(h)+1)) and len(h)<=20 and len(h)==r['epochs']
 assert np.isfinite(h.select_dtypes(include='number').to_numpy()).all()
 ck=ROOT/'checkpoints/structured'/rid;k=torch.load(ck/'latest.pt',map_location='cpu',weights_only=False)
 assert k['config']==c and all(torch.isfinite(t).all() for t in k['model'].values())
 verify_launch_code(k,load(p/'environment.json'))
 pd.testing.assert_frame_equal(pd.DataFrame(k['history']),h,check_exact=False,rtol=1e-12,atol=1e-12)
 v=val_frame()
 for metric,suffix,file,statekey in [('accuracy','','best.pt','best'),('macro_f1','_macro_f1','best_macro_f1.pt','secondary_best')]:
  m=load(p/f'validation_metrics{suffix}.json');cm=verify_predictions(p/f'validation_predictions{suffix}.csv',m,v)
  w=torch.load(ck/file,map_location='cpu',weights_only=False);s=k[statekey]
  assert w['best_epoch']==s['epoch']==int(h.loc[h['val_'+metric].idxmax(),'epoch'])
  assert w['config']==c and w['metrics']==s['metrics']==m
  assert all(torch.isfinite(t).all() and torch.equal(t,s['model'][n]) for n,t in w['model'].items())
  pd.testing.assert_frame_equal(pd.DataFrame(s['predictions']),pd.read_csv(p/f'validation_predictions{suffix}.csv'),check_exact=False,rtol=1e-12,atol=1e-12)
  if not suffix:assert sha256(ck/file)==r['checkpoint_sha256']
  folder=p/('figures_macro_f1' if suffix else 'figures')
  if suffix and not folder.exists():metric_figures(m,folder,rid+' | macro-F1 winner')
  verify_figures(folder,cm,m,curves=not suffix)
  del w
 verification=dict(status='verified_completed',epochs=len(h),source_hashes=k['code_hashes'],checkpoint_hashes={f.name:sha256(f) for f in ck.glob('*.pt')},final_metrics={a:float(h.iloc[-1]['val_'+a]) for a in ['accuracy','macro_precision','macro_recall','macro_f1','loss']},test_images_loaded=False,gpu_used=False)
 del k
 write_json(p/'closeout_verification.json',verification)
 r.update(decision='verified_final_selection_candidate',source_availability='original');upsert(r);write_json(p/'record.json',r)
 return verification

def overlap(a,b):
 v=val_frame();both=~a['ok']&~b['ok'];classes={}
 for c in CLASSES:
  ix=v.diagnosis.to_numpy()==c
  classes[c]=dict(support=int(ix.sum()),both_wrong=int((both&ix).sum()),a_fixes_b=int((a['ok']&~b['ok']&ix).sum()),b_fixes_a=int((~a['ok']&b['ok']&ix).sum()))
 pa=a['prob'].argmax(1);pb=b['prob'].argmax(1)
 return dict(a=a['id'],b=b['id'],a_errors=int((~a['ok']).sum()),b_errors=int((~b['ok']).sum()),both_errors=int(both.sum()),a_fixes_b=int((a['ok']&~b['ok']).sum()),b_fixes_a=int((~a['ok']&b['ok']).sum()),agreement=float((pa==pb).mean()),disagreement=float((pa!=pb).mean()),per_class=classes,scope='diagnostic overlap; no oracle routing or inference use of true labels')

def qualified_sets(pool,new_ids):
 """Predeclared rules select <=2 pairs, <=3 triples, <=1 four; no metric peek."""
 new=[pool[r] for r in new_ids if r in pool and pool[r]['metrics']['accuracy']>=.88 and pool[r]['metrics']['macro_f1']>=.79 and int((pool[r]['ok']&~pool[TINY]['ok']).sum())>=20]
 new.sort(key=lambda e:(-e['metrics']['accuracy'],-e['metrics']['macro_f1'],e['id']))
 sets=[]
 def add(members):
  members=list(dict.fromkeys(members))
  if len(members) in (2,3,4) and frozenset(members) not in [frozenset(s) for s in sets]:sets.append(members)
 add([SMALL,V2,B0]) # one fixed replacement of Tiny in existing S12
 if new:
  strongest=new[0]['id'];add([TINY,strongest]);add([SMALL,strongest])
  thirds=[x for x in [V2]+[e['id'] for e in new[1:]] if x not in (SMALL,strongest)]
  thirds.sort(key=lambda x:(-int((pool[x]['ok']&~(pool[SMALL]['ok']|pool[strongest]['ok'])).sum()),-pool[x]['metrics']['macro_f1'],x))
  if thirds:add([SMALL,strongest,thirds[0]])
  if len(new)>1:
   second=new[1]['id'];add([TINY,strongest,second])
   union3=pool[SMALL]['ok']|pool[strongest]['ok']|pool[second]['ok']
   if int((pool[V2]['ok']&~union3).sum())>=10:add([SMALL,strongest,second,V2])
 assert sum(len(s)==2 for s in sets)<=2 and sum(len(s)==3 for s in sets)<=3 and sum(len(s)==4 for s in sets)<=1
 return sets

def templates(members,pool):
 """Exactly two non-equal weights; frozen strength ordering, no optimizer."""
 ranked=sorted(members,key=lambda r:(-pool[r]['metrics']['accuracy'],-pool[r]['metrics']['macro_f1'],r))
 n=len(members);raw={2:[[.6,.4],[.4,.6]],3:[[.4,.4,.2],[.5,.25,.25]],4:[[.4,.2,.2,.2],[.3,.3,.2,.2]]}[n]
 return [[dict(zip(ranked,w))[r] for r in members] for w in raw]

def fusion(rid,members,weights,pool):
 p=ROOT/'results/structured_experiments'/rid
 c=dict(experiment_id=rid,recipe_version='final_fixed_fusion_v1',protocol='exploratory_image_level',class_order=list(CLASSES),parent_run_ids=members,weights=weights,prediction_sha256={r:sha256(pool[r]['csv']) for r in members},checkpoint_sha256={r:pool[r]['record']['checkpoint_sha256'] for r in members},split_manifest='data/splits/exploratory/image_level_dev_v1.csv',split_sha256=sha256(ROOT/'data/splits/exploratory/image_level_dev_v1.csv'))
 assert np.isclose(sum(weights),1) and len(weights)==len(members)
 if p.exists():assert load(p/'config.json')==c,'Fusion identity changed'
 p.mkdir(parents=True,exist_ok=True);write_json(p/'config.json',c)
 v=val_frame();prob=sum(w*pool[r]['prob'] for w,r in zip(weights,members))
 train=pd.read_csv(ROOT/c['split_manifest']).query("split=='train'");counts=train.diagnosis.value_counts();cw=np.sqrt(len(train)/np.array([counts[c] for c in CLASSES]));cw/=cw.mean()
 labels=v.label.to_numpy();loss=float(np.average(-np.log(np.clip(prob[np.arange(len(v)),labels],1e-12,1)),weights=cw[labels]));m=metric_report(labels,prob,loss)
 pred=[dict(image_id=i,true_class=CLASSES[y],predicted_class=CLASSES[int(a.argmax())],**{f'p_{c}':float(a[j]) for j,c in enumerate(CLASSES)}) for i,y,a in zip(v.index,labels,prob)]
 write_json(p/'validation_metrics.json',m);write_csv(p/'validation_predictions.csv',pred);metric_figures(m,p/'figures',rid+' | fixed soft voting')
 row={f:'' for f in FIELDS};row.update(experiment_id=rid,era='structured',record_kind='fixed_probability_fusion',phase='final_architecture_selection',protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],method='Bounded final fixed fusion; no grid/class weights',model='+'.join(pool[r]['record']['model'] for r in members),attention='none',image_size=224,seed=42,ensemble_members=json.dumps(members),ensemble_weights=json.dumps(weights),checkpoint=json.dumps([pool[r]['record']['checkpoint'] for r in members]),checkpoint_available_local=True,config_path=(p/'config.json').relative_to(ROOT).as_posix(),metrics_path=(p/'validation_metrics.json').relative_to(ROOT).as_posix(),source_sha256=sha256(p/'validation_metrics.json'),source_availability='original',plots_dir=(p/'figures').relative_to(ROOT).as_posix(),confusion_matrix_path=(p/'figures/confusion_matrix.csv').relative_to(ROOT).as_posix(),status='completed',decision='bounded_final_candidate',notes='Saved parent accuracy winners; no training/checkpoint or test loader; validation-selected members/weights; arithmetic on stored probabilities, not new FP32 inference.',val_loss=loss,**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']})
 upsert(row);write_json(p/'record.json',row)
 cm=verify_predictions(p/'validation_predictions.csv',m,v);verify_figures(p/'figures',cm,m)
 write_json(p/'closeout_verification.json',dict(status='verified_completed',exact_probabilities_sha256=sha256(p/'validation_predictions.csv'),parent_prediction_hashes=c['prediction_sha256'],parent_checkpoint_hashes=c['checkpoint_sha256'],test_images_loaded=False,gpu_used=False))
 return evidence(rid)

def materially_better(m,ref):
 accuracy_gain=m['accuracy']>=ref['accuracy']+.01 and m['macro_f1']>=ref['macro_f1']
 balance_gain=m['macro_f1']>=ref['macro_f1']+.01 and m['accuracy']>=ref['accuracy']
 protected=all(m['per_class'][c]['recall']>=ref['per_class'][c]['recall']-limit for c,limit in [('akiec',.05),('mel',.03),('df',.10),('vasc',.10)] if c in ref['per_class'])
 return (accuracy_gain or balance_gain) and protected

def md_table(rows,fields):
 return '| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n'+''.join('| '+' | '.join(str(r.get(f,'')) for f in fields)+' |\n' for r in rows)

def postprocess(manifest,status):
 OUT.mkdir(parents=True,exist_ok=True)
 new_ids=[c['id'] for c in manifest['candidates'] if status['candidates'][c['id']]['status']=='completed']
 pool={r:evidence(r) for r in OLD+new_ids}
 for r in [S12,REF,DENSE_FUSION]:pool[r]=evidence(r)
 rows=[]
 for rid in OLD+new_ids:
  e=pool[rid];r=e['record'];p=ROOT/'results/structured_experiments'/rid;c=load(p/'config.json');summary=load(p/'training_summary.json')
  rows.append(dict(Model=c['model'],Architecture_family=FAMILY[c['model']],Pretrained_weights=c['weights'],Transfer_learning='ImageNet; full fine-tuning',Attention=c['attention'],Input_size=c['image_size'],Training_images=7009,Validation_images=1503,Test_status='1503 locked images; NOT evaluated',Protocol='exploratory image-level; shared lesions',Best_epoch=r['best_epoch'],Accuracy=e['metrics']['accuracy'],Macro_precision=e['metrics']['macro_precision'],Macro_recall=e['metrics']['macro_recall'],Macro_F1=e['metrics']['macro_f1'],Parameters=summary['parameters'],Runtime_seconds=r['runtime_seconds'],Training_epochs=r['epochs'],Outcome=('accuracy gain with macro-F1 trade-off' if rid==SMALL else 'weaker standalone than Tiny; complementary value separate' if e['metrics']['accuracy']<pool[TINY]['metrics']['accuracy'] else 'competitive standalone; ensemble gain not assumed'),experiment_id=rid))
 for c in manifest['candidates']:
  if c['id'] not in new_ids:
   rows.append(dict(Model=load(ROOT/c['config'])['model'],experiment_id=c['id'],Outcome='FAILED/UNVERIFIED: '+status['candidates'][c['id']].get('reason','unknown')))
 write_csv(OUT/'model_vs_accuracy.csv',rows)
 atomic_text(ROOT/'research/MODEL_VS_ACCURACY.md','# Final standalone architecture comparison\n\nSame exploratory validation; seven classes; no test results. Accuracy-selected checkpoints; native pretrained packages differ. Failed candidates disclosed.\n\n'+md_table(rows,list(rows[0].keys())))
 comparison_figures([dict(display_name=r['Model']+' '+r['Attention'],accuracy=r['Accuracy'],macro_f1=r['Macro_F1']) for r in rows if 'Accuracy' in r],OUT,'Final architecture screening | same exploratory validation')
 serious=[r for r in [TINY,SMALL,V2,B0]+new_ids if r in pool]
 pairwise=[overlap(pool[a],pool[b]) for a,b in itertools.combinations(serious,2)]
 write_json(OUT/'complementarity.json',pairwise);write_csv(OUT/'complementarity.csv',[{k:v for k,v in o.items() if k!='per_class'} for o in pairwise]);write_csv(OUT/'class_complementarity.csv',[dict(a=o['a'],b=o['b'],class_name=c,**v) for o in pairwise for c,v in o['per_class'].items()])
 declaration=OUT/'predeclared_equal_sets.json';sets=qualified_sets(pool,new_ids)
 declared=dict(member_sets=sets,parent_hashes={r:sha256(pool[r]['csv']) for s in sets for r in s},selection_rules=manifest['ensemble_policy'])
 if declaration.exists():assert load(declaration)==declared
 else:write_json(declaration,declared)
 ens=[]
 for j,s in enumerate(sets):
  rid=f's{20+j:02d}_final_equal_{j+1:02d}_exploratory_seed42';e=fusion(rid,s,[1/len(s)]*len(s),pool);ens.append(e);pool[rid]=e
 if ens:
  ref=pool[REF]['metrics']
  viable=[e for e in ens if e['metrics']['accuracy']>=ref['accuracy'] and e['metrics']['macro_f1']>=ref['macro_f1']-.002] or ens
  strongest=max(viable,key=lambda e:(e['metrics']['macro_f1'],e['metrics']['accuracy'],-len(json.loads(e['record']['ensemble_members']))))
  members=json.loads(strongest['record']['ensemble_members']);weights=templates(members,pool)
  d=dict(member_set=members,equal_run=strongest['id'],fixed_weights=weights,reason='One strongest equal set; parent strength ranking frozen; exactly2 additional templates')
  wp=OUT/'predeclared_weighted_set.json'
  if wp.exists():assert load(wp)==d
  else:write_json(wp,d)
  for j,w in enumerate(weights):
   rid=f's{26+j:02d}_final_weighted_{j+1:02d}_exploratory_seed42';e=fusion(rid,members,w,pool);ens.append(e);pool[rid]=e
 candidates=[pool[r] for r in [TINY,SMALL,S12,REF,DENSE_FUSION]+new_ids]+ens
 ref=pool[REF]['metrics'];better=[e for e in candidates if materially_better(e['metrics'],ref)]
 selected=max(better,key=lambda e:(e['metrics']['macro_f1'],e['metrics']['accuracy'])) if better else pool[REF]
 comparisons=[dict(display_name=e['id'].split('_')[0].upper()+(' FP32 identity' if e['id']==REF else ''),experiment_id=e['id'],accuracy=e['metrics']['accuracy'],macro_precision=e['metrics']['macro_precision'],macro_recall=e['metrics']['macro_recall'],macro_f1=e['metrics']['macro_f1'],selected=e['id']==selected['id']) for e in candidates]
 write_csv(OUT/'final_method_comparison.csv',comparisons);comparison_figures(comparisons,OUT/'ensemble_comparison','Final bounded comparison | no additional architectures after freeze')
 r=selected['record'];members=json.loads(r['ensemble_members']) if r['ensemble_members'] else [selected['id']];weights=json.loads(r['ensemble_weights']) if r['ensemble_members'] else [1.]
 models=[dict(run_id=m,model=pool[m]['record']['model'],weights=pool[m]['record']['pretrained_weights'],checkpoint=pool[m]['record']['checkpoint'],checkpoint_sha256=pool[m]['record']['checkpoint_sha256']) for m in members]
 frozen=dict(status='architecture_selection_closed',selected_run=selected['id'],models=models,number_of_models=len(models),ensemble_method='fixed probability soft voting' if len(models)>1 else 'single model',ensemble_weights=weights,metrics=selected['metrics'],metrics_path=r['metrics_path'],confusion_matrix_reference=r['confusion_matrix_path'],protocol='exploratory_image_level',split_sha256=r['split_sha256'],material_gain=bool(better),material_gate=manifest['material_gate'],batch_candidate_status=status['candidates'],test_evaluated=False,architecture_search_closed=True,no_more_backbones=True)
 freeze=OUT/'architecture_freeze.json'
 if freeze.exists():assert load(freeze)==frozen,'Frozen selection cannot change'
 else:write_json(freeze,frozen)
 doc='# FINAL ARCHITECTURE SELECTION\n\nARCHITECTURE SELECTION IS CLOSED. No further backbone proposals.\n\n'+json.dumps(frozen,indent=2)+'\n\n## Why these models\n'
 for model in models:
  e=pool[model['run_id']]
  strengths=sorted(CLASSES,key=lambda c:-e['metrics']['per_class'][c]['f1'])[:3]
  class_text=', '.join(c+' F1='+format(e['metrics']['per_class'][c]['f1'],'.4f') for c in strengths)
  fixes={other:int((e['ok']&~pool[other]['ok']).sum()) for other in members if other!=model['run_id']}
  doc+=f"- {model['model']}: accuracy {e['metrics']['accuracy']:.6f}, macro-F1 {e['metrics']['macro_f1']:.6f}; {FAMILY[model['model']]}. Strongest class scores: {class_text}. Corrects these other-member errors: {fixes}. Fixed ensemble selection is supported by saved joint performance, not standalone rank alone.\n"

 doc+='\n## Standalone evidence and rejection context\n'+md_table(rows,['Model','Attention','Accuracy','Macro_F1','Outcome'])+'\n'+('A material balanced gain passed the frozen gate.' if better else 'No candidate passed the material balanced-gain gate; retain the proven S12 FP32 identity reference rather than chase a few images.')+'\n\n## Rejected models and alternatives\n'+md_table(comparisons,['experiment_id','accuracy','macro_f1','selected'])+'\nUnselected higher/lower accuracy methods are rejected as the final choice because they fail the material/balance/complexity gate; retain all research evidence. MobileNet/B0 controls and CBAM ablation remain valid; standalone DenseNet was weaker. Candidate failures are disclosed above.\n\n## Class-wise performance\n'+md_table([dict(class_name=c,**v) for c,v in selected['metrics']['per_class'].items()],['class_name','precision','recall','f1','support'])+'\n## Limitations\n7009train/1503val/1503lockedtest. Train/val share563lesions affecting596validation images. Many prior development decisions increase selection bias. Bounded weighting is still validation-based method selection, not independent evidence. No statistical superiority or robustness claim from one seed. FP32 identity reference reused; new fusions use stored AMP parent probabilities. Test untouched. Strict confirmation must start fresh from external ImageNet weights; exploratory checkpoints cannot confirm lesion independence. No new architecture follows this report.\n'
 atomic_text(ROOT/'research/FINAL_ARCHITECTURE_SELECTION.md',doc)
 prepare_recipe(frozen,pool,rows)
 write_json(OUT/'pipeline_verification.json',dict(status='completed',equal_candidates=len(sets),additional_weighted_candidates=min(2,len(ens)),model_count=len(models),test_images_loaded=False,gpu_postprocessing=False,architecture_frozen=True))
 return frozen

def prepare_recipe(frozen,pool,rows):
 techniques=[('ImageNet transfer learning','ADOPT','Strong local and literature evidence; fresh external initialization for strict confirmation.'),('Full fine-tuning / progressive unfreezing','ADOPT / OPTIONAL','Full fine-tune retained; progressive unfreezing optional, not silently combined.'),('Discriminative LR','ADOPT','Preserve3e-5 backbone/1e-4 head philosophy.'),('Class-weighted cross-entropy','ADOPT','Train-only sqrt inverse frequency; no validation fitted class weights.'),('Focal/class-weighted focal','REJECT for default','Literature support but no matched local benefit; historical packages confound effects.'),('Minority/class-specific augmentation','OPTIONAL','Potential major gap; controlled mild spatial augmentation only after explicit approval, not paper copied strengths.'),('Mixup','OPTIONAL','Literature-supported regularization, no demonstrated local effect; not in frozen default.'),('CutMix','OPTIONAL','Unverified local effect; not combined with Mixup by default.'),('Dropout','ADOPT','Common head .2 unchanged.'),('Weight decay','ADOPT','AdamW1e-4 retained.'),('Label smoothing','OPTIONAL','Could conflict with class weights; not default.'),('Warmup/cosine/OneCycle','OPTIONAL','Different optimization package; not stack with plateau scheduler.'),('ReduceLROnPlateau','ADOPT','Accuracy mode; same factor .5, patience2; longer early-stop allowance.'),('Gradient clipping','ADOPT','Unscaled global norm1 for numerical protection in future strict recipe, not changed in screening.'),('EMA','OPTIONAL','Extra state/selection complexity; no current evidence.'),('Increased resolution','REJECT for default','Current224 matched; TTA failed and S14deferred, no demonstrated gain.'),('Weighted soft voting','ADOPT if selected','Exact frozen weights only, otherwise equal reference.'),('CBAM/attention','REJECT as forced final add-on','PreserveS05 mixed/near-neutral ablation; native attention retained.'),('Grad-CAM/XAI','ADOPT later','Correct/incorrect development and final test examples after final freeze.')]
 recipe='# FINAL HIGH-PERFORMANCE RECIPE (prepared, NOT launched)\n\nFrozen architecture: '+json.dumps(frozen['models'])+'\nFrozen member weights: '+str(frozen['ensemble_weights'])+'\n\n'+md_table([dict(Technique=t,Decision=d,Reason=r) for t,d,r in techniques],['Technique','Decision','Reason'])+'\n## Coherent default\nFresh ImageNet weights; strict lesion-disjoint train/val;224px/ImageNet normalization; horizontal flip.5; fullfine-tune; effectivebatch32; AMPtraining with finite guards and FP32validation from start; weightedCE; AdamW discriminativeLR3e-5/1e-4; dropout.2; decay1e-4; gradientclip1; plateau scheduler; cap50/minimumreview15/patience12. Record this new recipe version separately from screening. Optional techniques excluded from default, require a deliberate amendment. Ensemblemember weights frozen; no strict-validation weight search. Checkpointselection uses frozen ensembleaccuracy for final ensemble, with independentmacroF1 and member states saved.\n\nNo guide/paper result claimed for this unexecuted recipe. Strict adaptation can change performance. No next backbone.\n'
 atomic_text(ROOT/'research/FINAL_HIGH_PERFORMANCE_RECIPE.md',recipe)
 trend=[]
 for rid in OLD+[r['experiment_id'] for r in rows if r.get('experiment_id') not in OLD and 'Accuracy' in r]:
  h=pd.read_csv(ROOT/'results/structured_experiments'/rid/'history.csv')
  trend.append(dict(model=pool[rid]['record']['model'],epochs=len(h),accuracy_peak_epoch=int(h.loc[h.val_accuracy.idxmax(),'epoch']),f1_peak_epoch=int(h.loc[h.val_macro_f1.idxmax(),'epoch']),best_accuracy=float(h.val_accuracy.max()),final_accuracy=float(h.iloc[-1].val_accuracy),last5_accuracy=h.tail(5).val_accuracy.tolist()))
 write_json(OUT/'learning_curve_50_epoch_assessment.json',trend)
 plan='# FINAL MAXIMUM-50-EPOCH TRAINING PLAN (NOT authorized/launched)\n\n'+md_table(trend,['model','epochs','accuracy_peak_epoch','f1_peak_epoch','best_accuracy','final_accuracy'])+'\nMost current models peaked before the cap and often declined;20epochs was not a proven binding limit.50is an allowance for a new strict training phase, not a promised improvement or requirement to chooseepoch50.\n\n1. Fresh strict train/validation from ImageNet; never resume exploratory checkpoints.\n2. Frozen architectures and weights; train modelmembers independently with cap50 and earlystopping patience12 after minimum15. Record all member predictions/states each epoch so a coherent fixed-ensemble checkpoint can be selected on strict validation without weight tuning.\n3. Save bestaccuracy, independentbestmacroF1, latest/fulloptimizer/scheduler/scaler/RNG and compositeepoch/memberidentities. Epoch23canremainselected if later performance deteriorates. Store every serious run curves, class metrics/matrices, runtime and predictions.\n4. FP32validation, finite guards, explicit numerical amendments only; safe atomic resumes.\n5. Freeze preprocessing/weights/calibration/selection and all methodology beforeONElockedtest evaluation. Exploratory and strict results must remain separate.\n6. Final Grad-CAM/architecture-specific CAM and paper artifacts afterselection. No test loading, strict training or long run executed by current batch.\n'
 atomic_text(ROOT/'research/FINAL_50_EPOCH_TRAINING_PLAN.md',plan)
