"""CPU-only protocol, safety, and fixed-fusion checks."""
import unittest,json
from unittest.mock import patch
import numpy as np
import pandas as pd
from research.common import ROOT
from research.strict_protocol import partition_metadata
from research.strict_train import validate_config,checked_validation_batch,NonfiniteValidation
from research.final_strict_results import equal_probabilities

class StrictTests(unittest.TestCase):
    def test_test_labels_excluded_before_parsing(self):
        original=pd.read_csv;calls=[]
        def spy(path,*args,**kwargs):
            calls.append(kwargs);return original(path,*args,**kwargs)
        with patch('research.strict_protocol.pd.read_csv',side_effect=spy):dev,r=partition_metadata()
        self.assertEqual(calls[0]['usecols'],['image_id','lesion_id','split'])
        self.assertTrue(callable(calls[1]['skiprows']))
        self.assertEqual(len(dev),8512);self.assertNotIn('test',set(dev.split))
        self.assertEqual(r['lesion_overlaps'],dict(train_val=0,train_test=0,val_test=0))
    def test_frozen_config_rejects_recipe_drift(self):
        c=json.loads((ROOT/'research/configs/final_strict/s28_efficientnet_b0_final_strict_seed42.json').read_text())
        validate_config(c)
        for key,value in [('max_epochs',20),('patience',5),('validation_precision','amp_fp16'),('gradient_clip_norm',0)]:
            with self.subTest(key=key):
                modified=dict(c);modified[key]=value
                with self.assertRaises(ValueError):validate_config(modified)
    def test_equal_fusion_no_weight_search_and_invalid_members_rejected(self):
        a=np.array([[.7,.3],[.1,.9]]);b=np.array([[.1,.9],[.8,.2]]);c=np.array([[.4,.6],[.5,.5]])
        np.testing.assert_allclose(equal_probabilities([a,b,c]),(a+b+c)/3)
        with self.assertRaises(ValueError):equal_probabilities([a,b])
        with self.assertRaises(ValueError):equal_probabilities([a,b,np.full_like(c,np.nan)])
    def test_nonfinite_fp32_is_rejected(self):
        import torch
        class Bad(torch.nn.Module):
            def forward(self,x):return torch.full((len(x),7),float('nan'))
        with self.assertRaises(NonfiniteValidation):checked_validation_batch(Bad(),torch.ones(2,3,4,4),torch.tensor([0,1]),torch.ones(7),fp32=True)

    def test_fixed_closeout_in_isolated_directory(self):
        import tempfile,shutil
        from pathlib import Path
        import research.final_strict_results as module
        from research.strict_train import metric_report
        dev,_=partition_metadata();val=dev.query("split=='val'")
        p=np.full((len(val),7),.1/6);p[np.arange(len(val)),val.label]=.9
        m=metric_report(val.label.to_numpy(),p,.1)
        frames=[dict(image_id=r.image_id,true_class=r.diagnosis,predicted_class=r.diagnosis,**{f'p_{c}':float(p[i,j]) for j,c in enumerate(module.CLASSES)}) for i,r in enumerate(val.itertuples())]
        configs=[dict(experiment_id=f's{i}_fixture',model=name,weights='fixture') for i,name in zip([28,29,30],['efficientnet_b0','convnext_tiny','efficientnet_v2_s'])]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'results/final_strict/v1'
            for c in configs:
                folder=root/'results/structured_experiments'/c['experiment_id'];folder.mkdir(parents=True)
                pd.DataFrame(frames).to_csv(folder/'validation_predictions.csv',index=False)
                ck=root/'checkpoints/structured'/c['experiment_id'];ck.mkdir(parents=True);(ck/'best.pt').write_bytes(b'fixture')
            ref='results/structured_experiments/s13_s12_four_flip_tta_exploratory_seed42/validation_metrics_identity_fp32.json'
            (root/ref).parent.mkdir(parents=True);shutil.copyfile(ROOT/ref,root/ref)
            with patch.object(module,'ROOT',root),patch.object(module,'OUT',out),patch.object(module,'verify_model',return_value=(m,{'best_epoch':17})),patch.object(module,'relative',side_effect=lambda path:Path(path).relative_to(root).as_posix()),patch.object(module,'upsert') as registry:
                delta=module.closeout(configs,'s31_fixture')
                self.assertEqual(delta['correct'],1503);self.assertEqual(registry.call_count,1)
                self.assertTrue((root/'research/FINAL_STRICT_VALIDATION_RESULTS.md').exists())
                self.assertTrue((out/'ensemble/figures/confusion_matrix.pdf').exists())
                self.assertFalse(json.loads((out/'closeout_verification.json').read_text())['test_loader'])

if __name__=='__main__':unittest.main()
