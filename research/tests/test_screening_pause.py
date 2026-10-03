"""CPU-only staged screening contract; no images, weights or CUDA."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from research.train import save_screening_snapshot,run
from research.phase3.close_s02 import load
from research.common import ROOT
class ScreeningPauseTests(unittest.TestCase):
 def test_snapshot_preserves_recipe_and_does_not_complete(self):
  parent=ROOT/'results/structured_experiments/s06_convnext_tiny_none_exploratory_seed42'
  import pandas as pd
  c=load(parent/'config.json');m=load(parent/'validation_metrics.json')
  best=dict(epoch=8,metrics=m,predictions=pd.read_csv(parent/'validation_predictions.csv').to_dict('records'))
  with tempfile.TemporaryDirectory() as tmp:
   obj=SimpleNamespace(path=Path(tmp),config=c,history=pd.read_csv(parent/'history.csv').iloc[:8].to_dict('records'),row=dict(status='running'))
   with patch('research.registry.upsert') as registry,patch('research.plots.metric_figures'),patch('research.plots.training_figures') as curves:
    save_screening_snapshot(obj,best,best,8)
    self.assertEqual(obj.row['status'],'paused_screening');registry.assert_called_once()
    self.assertEqual(c,obj.config);self.assertEqual(len(curves.call_args.args[0]),8)
    self.assertTrue(load(Path(tmp)/'screening/epoch_08/screening_status.json')['continuation_requires_approval'])
    self.assertFalse((Path(tmp)/'validation_metrics.json').exists())
 def test_invalid_boundary_rejected_before_cuda(self):
  with self.assertRaises(ValueError):run({'max_epochs':20},stop_after_epoch=20)
if __name__=='__main__':unittest.main()
