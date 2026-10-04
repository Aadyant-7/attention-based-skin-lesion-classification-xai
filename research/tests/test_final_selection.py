"""CPU tests for bounded selection, gates and recovery; no training."""
import unittest,numpy as np
from research.final_selection import qualified_sets,templates,materially_better,TINY,SMALL,V2,B0
from research.run_final_architecture_selection import recoverable
class FinalSelectionTests(unittest.TestCase):
 def pool(self):
  def e(i,acc,f1,wrong):
   ok=np.ones(100,dtype=bool);ok[wrong]=False
   return dict(id=i,metrics=dict(accuracy=acc,macro_f1=f1),ok=ok)
  return {TINY:e(TINY,.9175,.86,list(range(30))),SMALL:e(SMALL,.9222,.85,list(range(20,50))),V2:e(V2,.895,.82,list(range(40,70))),B0:e(B0,.86,.77,list(range(10,60))), 'new1':e('new1',.93,.88,list(range(50,80))), 'new2':e('new2',.91,.84,list(range(70,100)))}
 def test_candidate_bounds_and_exclusion(self):
  pool=self.pool();sets=qualified_sets(pool,['new1','new2'])
  self.assertLessEqual(sum(len(s)==2 for s in sets),2);self.assertLessEqual(sum(len(s)==3 for s in sets),3);self.assertLessEqual(sum(len(s)==4 for s in sets),1)
  self.assertEqual(len(sets),len({frozenset(s) for s in sets}))
  pool['new1']['metrics']['accuracy']=.80;pool['new2']['metrics']['macro_f1']=.70
  self.assertEqual(qualified_sets(pool,['new1','new2']),[[SMALL,V2,B0]])
 def test_two_weight_templates_only(self):
  pool=self.pool()
  for s in [[TINY,SMALL],[TINY,SMALL,V2],[TINY,SMALL,V2,B0]]:
   w=templates(s,pool);self.assertEqual(len(w),2)
   for a in w:self.assertAlmostEqual(sum(a),1);self.assertTrue(all(x>0 for x in a))
 def test_material_gate_rejects_decimal_gain_and_class_damage(self):
  ref=dict(accuracy=.9248,macro_f1=.8775,per_class={c:dict(recall=.8) for c in ['akiec','mel']})
  self.assertFalse(materially_better(dict(ref,accuracy=.9261),ref))
  self.assertTrue(materially_better(dict(ref,accuracy=.935),ref))
  self.assertFalse(materially_better(dict(ref,accuracy=.94,macro_f1=.85),ref))
 def test_numerical_and_config_failures_never_retry(self):
  self.assertFalse(recoverable(-1,'FloatingPointError: Nonfinite validation loss'))
  self.assertFalse(recoverable(-1,'CUDA out of memory'))
  self.assertFalse(recoverable(1,'ValueError: Resume config or runner changed'))
  self.assertTrue(recoverable(130,'KeyboardInterrupt'))
 def test_automatic_postprocessing_fallback_in_isolated_directory(self):
  import tempfile,json,pandas as pd
  from unittest.mock import patch
  import research.final_selection as fs
  from research.common import write_json,write_csv,sha256
  from research.train import metric_report
  realroot=fs.ROOT;real_evidence=fs.evidence
  with tempfile.TemporaryDirectory() as td:
   root=__import__('pathlib').Path(td);v=pd.DataFrame(dict(image_id=['v'+str(i) for i in range(7)],diagnosis=list(fs.CLASSES),label=list(range(7))))
   train=v.copy();train.image_id=['t'+str(i) for i in range(7)];train['split']='train';vv=v.copy();vv['split']='val'
   split=root/'data/splits/exploratory/image_level_dev_v1.csv';split.parent.mkdir(parents=True);pd.concat([train,vv]).to_csv(split,index=False)
   prob=np.eye(7)*.9+np.ones((7,7))*.1/7;m=metric_report(np.arange(7),prob,.1);pool={};registry=[]
   for rid in fs.OLD+[fs.S12,fs.REF,fs.DENSE_FUSION]:
    path=root/'results/structured_experiments'/rid;path.mkdir(parents=True)
    csv=path/'validation_predictions.csv';write_csv(csv,[dict(image_id='v'+str(i),true_class=c,predicted_class=c,**{'p_'+a:float(prob[i,j]) for j,a in enumerate(fs.CLASSES)}) for i,c in enumerate(fs.CLASSES)])
    if rid in fs.OLD:
     for name in ['config.json','training_summary.json','history.csv']:(path/name).write_bytes((realroot/'results/structured_experiments'/rid/name).read_bytes())
     c=json.loads((path/'config.json').read_text());model=c['model'];weights=c['weights'];attention=c['attention']
    else:model='ensemble';weights='';attention='none'
    r=dict(experiment_id=rid,model=model,pretrained_weights=weights,attention=attention,best_epoch=1,epochs=20,runtime_seconds=1,ensemble_members=json.dumps([fs.B0,fs.TINY,fs.V2]) if rid==fs.REF else '',ensemble_weights=json.dumps([1/3]*3) if rid==fs.REF else '',checkpoint='checkpoints/fake.pt',checkpoint_sha256='fake',metrics_path=str((path/'validation_metrics.json').relative_to(root)),confusion_matrix_path='fixture.csv',split_sha256=sha256(split),protocol='exploratory_image_level',status='completed',evaluation_split='validation')
    pool[rid]=dict(id=rid,record=r,metrics=m,prob=prob,csv=csv,ok=np.ones(7,dtype=bool));registry.append(r)
   def ev(rid):return pool[rid] if rid in pool else real_evidence(rid)
   def update(r):registry.append(r)
   write_json(root/'failed_config.json',dict(model='resnet101'))
   manifest=dict(candidates=[dict(id='newfailed',config='failed_config.json')],ensemble_policy='fixture bounded',material_gate='fixture1pp')
   status=dict(candidates={'newfailed':dict(status='failed',reason='synthetic failure')})
   with patch.object(fs,'ROOT',root),patch.object(fs,'OUT',root/'comparison'),patch.object(fs,'evidence',side_effect=ev),patch.object(fs,'val_frame',return_value=v.set_index('image_id')),patch.object(fs,'read_registry',return_value=registry),patch.object(fs,'upsert',side_effect=update),patch.object(fs,'metric_figures'),patch.object(fs,'comparison_figures'),patch.object(fs,'verify_figures'),patch('torch.cuda.is_available',side_effect=AssertionError('No GPU allowed')):
    frozen=fs.postprocess(manifest,status)
   self.assertEqual(frozen['selected_run'],fs.REF);self.assertTrue(frozen['architecture_search_closed'])
   for name in ['MODEL_VS_ACCURACY.md','FINAL_ARCHITECTURE_SELECTION.md','FINAL_HIGH_PERFORMANCE_RECIPE.md','FINAL_50_EPOCH_TRAINING_PLAN.md']:self.assertTrue((root/'research'/name).exists())
   proof=json.loads((root/'comparison/pipeline_verification.json').read_text());self.assertEqual(proof['equal_candidates'],1);self.assertEqual(proof['additional_weighted_candidates'],2)
if __name__=='__main__':unittest.main()
