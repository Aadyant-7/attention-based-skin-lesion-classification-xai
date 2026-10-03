"""Synthetic CPU safeguards for numerical recovery; never research training."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from research import train


class RecoveryChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config={'experiment_id':'s10_efficientnet_v2_s_none_exploratory_seed42'}
        self.model=torch.nn.Linear(2,7,bias=False).eval()
        self.batch=(torch.ones(2,2),torch.tensor([0,1]),['synthetic_a','synthetic_b'])

    def test_finite_amp_pass_retains_all_samples_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=train.evaluate_validation(self.model,[self.batch],torch.ones(7),self.config,Path(tmp),14,True)
            self.assertEqual(result[-1],'amp_fp16')
            self.assertEqual(result[2],self.batch[2])
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_fp16_overflow_repeats_complete_pass_fp32_only_when_authorized(self):
        with torch.no_grad():self.model.weight.fill_(40000)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            with self.assertRaises(train.NonfiniteValidation):
                train.evaluate_validation(self.model,[self.batch],torch.ones(7),self.config,path,14,False)
            result=train.evaluate_validation(self.model,[self.batch],torch.ones(7),self.config,path,14,True)
            self.assertEqual(result[-1],'fp32')
            self.assertEqual(result[2],self.batch[2])
            self.assertAlmostEqual(result[3],float(torch.log(torch.tensor(7.))),places=5)
            decisions=[json.loads(p.read_text()) for p in path.glob('*.json')]
            recovered=next(r for r in decisions if r['decision']=='repeat_entire_validation_fp32')
            self.assertGreater(recovered['logits_inf'],0)
            self.assertTrue(recovered['fp32_probe_finite'])
            self.assertTrue(recovered['amp_replay_first_nonfinite_module'])

    def test_nonfinite_parameters_cannot_be_repaired_or_sanitized(self):
        with torch.no_grad():self.model.weight[0,0]=float('nan')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            with self.assertRaises(train.NonfiniteValidation):
                train.evaluate_validation(self.model,[self.batch],torch.ones(7),self.config,path,14,True)
            event=json.loads(next(path.glob('*.json')).read_text())
            self.assertEqual(event['decision'],'stop_preserve_valid_checkpoint')
            self.assertTrue(event['model_nonfinite_tensors'])
            self.assertTrue(torch.isnan(self.model.weight[0,0]))

    def test_resume_migration_pins_sources_config_checkpoint_and_epoch(self):
        old={'research/train.py':'old','research/models.py':'unchanged'}
        new={'research/train.py':'new','research/models.py':'unchanged'}
        checkpoint={'history':[{}]*13,'code_hashes':old}
        fix=dict(kind='s10_validation_numerics_v1',experiment_id=self.config['experiment_id'],checkpoint_epoch=13,
            from_code_hashes=old,to_code_hashes=new,config_sha256='expected',checkpoint_sha256='expected')
        with patch.object(train,'code_hashes',return_value=new),patch.object(train,'sha256',return_value='expected'):
            train.validate_resume_fix(self.config,checkpoint,fix)
            for change in ({'checkpoint_epoch':14},{'checkpoint_sha256':'wrong'}, {'config_sha256':'wrong'},
                           {'to_code_hashes':dict(new,**{'research/models.py':'changed'})}):
                with self.assertRaises(ValueError):train.validate_resume_fix(self.config,checkpoint,dict(fix,**change))
            with self.assertRaises(ValueError):train.validate_resume_fix({'experiment_id':'s06_other'},checkpoint,fix)


if __name__=='__main__':unittest.main()
