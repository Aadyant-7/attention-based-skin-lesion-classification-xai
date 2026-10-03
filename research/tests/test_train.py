"""CPU-only runner safeguards; synthetic fixtures are not research experiments."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from research import train
from research.common import ROOT, CLASSES


class RunnerChecks(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'research/configs/phase3/s01_efficientnet_b0_none_strict_seed42.json').read_text())
        torch.set_num_threads(2)

    def test_whole_effective_batch_normalization_matches_gradients(self):
        torch.manual_seed(17)
        net=torch.nn.Linear(3,7);other=copy.deepcopy(net)
        x=torch.randn(6,3);y=torch.tensor([0,0,1,5,5,6]);w=torch.tensor([2.,1.,.5,.8,.7,.2,1.3])
        den=w[y].sum()
        loss=train.weighted_numerator(net(x),y,w)/den;loss.backward()
        for xx,yy in ((x[:3],y[:3]),(x[3:],y[3:])):
            (train.weighted_numerator(other(xx),yy,w)/den).backward()
        for p,q in zip(net.parameters(),other.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=1e-6)
        wrong=sum(torch.nn.functional.cross_entropy(net(xx),yy,weight=w) for xx,yy in ((x[:3],y[:3]),(x[3:],y[3:])))/2
        self.assertGreater(abs(loss.item()-wrong.item()),1e-4)

    def test_cpu_preflight_does_not_use_cuda_or_open_images(self):
        with patch.object(train.Image,'open',side_effect=AssertionError('Image read')),patch.object(torch.cuda,'is_available',side_effect=AssertionError('CUDA query')):
            train.validate_config(self.config)
            tr,va,w=train.development_data(self.config)
        self.assertEqual((len(tr),len(va)),(7009,1503))
        self.assertEqual(set(tr.split),{'train'});self.assertEqual(set(va.split),{'val'})
        self.assertAlmostEqual(w.mean().item(),1.,places=6)

    def test_manifest_hash_and_recipe_changes_rejected(self):
        cfg=copy.deepcopy(self.config);cfg['split_sha256']='0'*64
        with self.assertRaises(ValueError): train.development_data(cfg)
        for key,value in (('batch_size',8),('protocol','exploratory_image_level'),('augmentation','stronger')):
            cfg=copy.deepcopy(self.config);cfg[key]=value
            with self.assertRaises(ValueError): train.validate_config(cfg)

    def test_test_partition_rejected_before_image_loading(self):
        frame=pd.DataFrame([dict(split='test',path='should-not-open.jpg',label=0,image_id='locked')])
        with patch.object(train.Image,'open',side_effect=AssertionError('Image read')):
            for training in (True,False):
                with self.assertRaises(ValueError): train.DevelopmentImages(frame,self.config,training)

    def test_atomic_checkpoint_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'latest.pt';path.write_bytes(b'preserved')
            with patch.object(torch,'save',side_effect=RuntimeError('Interrupted save')):
                with self.assertRaises(RuntimeError): train.atomic_checkpoint(path,{'model':'fixture'})
            self.assertEqual(path.read_bytes(),b'preserved');self.assertEqual(list(path.parent.iterdir()),[path])

    def test_rng_roundtrip_cpu(self):
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        state=train.rng_state();expected=(random.random(),np.random.rand(),torch.rand(2))
        train.restore_rng(state)
        actual=(random.random(),np.random.rand(),torch.rand(2))
        self.assertEqual(expected[:2],actual[:2]);torch.testing.assert_close(expected[2],actual[2])

    def test_metric_report_agrees_with_saved_artifact_validator(self):
        from research.plots import validate_metrics
        y=np.arange(7);probs=np.eye(7);probs[[0,1]]=probs[[1,0]]
        m=train.metric_report(y,probs,.3)
        self.assertEqual(validate_metrics(m).sum(),7);self.assertAlmostEqual(m['accuracy'],5/7)

    def test_completed_run_exit_precedes_cuda_and_model_initialization(self):
        with patch.object(train,'completed_run',return_value=True),patch.object(torch.cuda,'is_available',side_effect=AssertionError('CUDA')),patch.object(train,'ResearchClassifier',side_effect=AssertionError('Model')):
            train.run(self.config)

    def test_resume_repairs_partial_history_without_touching_real_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'results/structured_experiments'/self.config['experiment_id'];path.mkdir(parents=True)
            (path/'config.json').write_text(json.dumps(self.config))
            (path/'history.csv').write_text('epoch\n999\n')
            split=root/self.config['split_manifest'];split.parent.mkdir(parents=True)
            pd.DataFrame([{'split':'val','diagnosis':c} for c in CLASSES]).to_csv(split,index=False)
            snapshot=dict(config=self.config,history=[dict(epoch=1,val_macro_f1=.1)],registry_row={'experiment_id':self.config['experiment_id']},
                          code_hashes={'test':'hash'},runtime_versions={'test':'runtime'})
            with patch.object(train,'ROOT',root),patch.object(train,'code_hashes',return_value={'test':'hash'}),patch.object(train,'runtime_versions',return_value={'test':'runtime'}),patch('research.registry.upsert') as upsert:
                obj=train.restore_experiment(self.config,snapshot)
                self.assertEqual(obj.history[0]['epoch'],1);self.assertEqual(obj.row['epochs'],1)
                self.assertEqual(pd.read_csv(path/'history.csv').epoch.tolist(),[1]);upsert.assert_called_once()
                broken=dict(snapshot,code_hashes={'test':'changed'})
                with self.assertRaises(ValueError): train.restore_experiment(self.config,broken)

    def test_convnext_native_norm_uses_backbone_lr(self):
        from research.models import ResearchClassifier
        model=ResearchClassifier('convnext_tiny',weights=None)
        groups=train.optimizer_groups(model,self.config)
        backbone_ids={id(p) for p in groups[0]['params']};head_ids={id(p) for p in groups[1]['params']}
        self.assertTrue({id(p) for p in model.head[1].parameters()}<=backbone_ids)
        self.assertTrue({id(p) for p in model.head[-1].parameters()}<=head_ids)
        self.assertFalse(backbone_ids&head_ids)
        self.assertEqual(len(backbone_ids|head_ids),len(list(model.parameters())))


if __name__=='__main__': unittest.main()
