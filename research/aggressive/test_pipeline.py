"""CPU gates: mathematical loss, no dropped samples, stage/BN safety, fixed fusion and recovery."""
import unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from .core import focal_per_sample,mixed_focal_sum,epoch_batches,micro_ranges,stopping_state,EnhancedModel,config
from .results import fusion

class EnhancedTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_focal_stability_and_gamma_zero(self):
        logits=torch.tensor([[10000.,-10000.,0.],[-10000.,10000.,0.]],requires_grad=True)
        y=torch.tensor([1,0]);w=torch.tensor([1.,2.,3.])
        loss=focal_per_sample(logits,y,w,2.2).sum();self.assertTrue(torch.isfinite(loss));loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        z=torch.randn(6,3);y=torch.tensor([0,1,2,0,1,2])
        torch.testing.assert_close(focal_per_sample(z,y,w,0).sum(),F.cross_entropy(z,y,weight=w,reduction='sum'))
    def test_mixup_blended_focal_and_gradients(self):
        z=torch.randn(4,7,requires_grad=True);a=torch.tensor([0,1,2,4]);b=torch.tensor([4,2,1,0]);w=torch.arange(1.,8.);lam=.3
        actual=mixed_focal_sum(z,a,b,lam,w,2.2)
        expected=lam*focal_per_sample(z,a,w,2.2).sum()+(1-lam)*focal_per_sample(z,b,w,2.2).sum()
        torch.testing.assert_close(actual,expected);actual.backward();self.assertTrue(torch.isfinite(z.grad).all())
    def test_all_training_samples_and_terminal_bn(self):
        batches=epoch_batches(7009,43);flat=[i for b in batches for i in b]
        self.assertEqual(len(flat),7009);self.assertEqual(set(flat),set(range(7009)));self.assertEqual(len(batches[-1]),33)
        for n in [32,33,17]:
            for micro in [16,32]:
                ranges=micro_ranges(n,micro);self.assertEqual(sum(b-a for a,b in ranges),n);self.assertTrue(all(2<=b-a<=micro for a,b in ranges))
    def test_patience_only_after25(self):
        c=dict(min_delta=.0003,minimum_epochs=25)
        reference=-1.;stale=0
        for epoch in range(1,25):reference,stale=stopping_state(reference,stale,.8,epoch,c)
        self.assertEqual(stale,0)
        reference,stale=stopping_state(reference,stale,.8001,25,c);self.assertEqual(stale,1)
        reference,stale=stopping_state(reference,stale,.801,26,c);self.assertEqual(stale,0)
    def test_real_b3_cbam_bn_and_staged_full_finetuning(self):
        spec=config()['models'][0];m=EnhancedModel(spec,False);m.train();self.assertEqual(m.stage(1),'head')
        self.assertFalse(any(p.requires_grad for p in m.features.parameters()));self.assertTrue(all(p.requires_grad for p in m.attention.parameters()))
        self.assertTrue(all(not child.training for child in m.features.modules() if isinstance(child,torch.nn.BatchNorm2d)))
        m.train();m.stage(3);self.assertTrue(any(p.requires_grad for p in m.features[-1].parameters()))
        m.train();m.stage(6);self.assertTrue(all(p.requires_grad for p in m.features.parameters()))
        out=m(torch.randn(2,3,64,64));self.assertEqual(tuple(out.shape),(2,7));self.assertTrue(torch.isfinite(out).all())
        m.stage(1);out.sum().backward();self.assertTrue(any(p.grad is not None for p in m.attention.parameters()))
    def test_predeclared_weights_no_grid(self):
        a=np.array([[.7,.3],[.1,.9]]);b=1-a;c=np.full_like(a,.5)
        np.testing.assert_allclose(fusion([a,b,c],[.4,.4,.2]),.4*a+.4*b+.2*c)
        with self.assertRaises(ValueError):fusion([a,b,c],[.5,.3,.2])
        with self.assertRaises(ValueError):fusion([a,b,np.full_like(c,np.nan)],[.4,.4,.2])
    def test_compatible_checkpoint_rejects_source_and_nonfinite_state(self):
        from research import run_aggressive_enhanced_pipeline as master
        c=config();spec=c['models'][0]
        checkpoint=dict(config=c,spec=spec,source_hashes=master.hashes(),runtime_versions=master.runtime_versions(),history=[],model={'a':torch.ones(1)},optimizer={})
        with patch.object(master.torch,'load',return_value=checkpoint):self.assertTrue(master.compatible(spec,c))
        checkpoint['model']['a']=torch.tensor([float('nan')])
        with patch.object(master.torch,'load',return_value=checkpoint):self.assertFalse(master.compatible(spec,c))

    def test_xai_gradients_attention_and_occlusion_in_isolated_cpu_fixture(self):
        import pandas as pd
        from PIL import Image
        from torch import nn
        from src.cbam import CBAM
        from . import xai
        class DenseToy(nn.Module):
            def __init__(self):super().__init__();self.denseblock4=nn.Conv2d(3,2,1)
            def forward(self,x):return self.denseblock4(x)
        class Toy(nn.Module):
            def __init__(self,spec,pretrained=False):
                super().__init__();torch.manual_seed(42)
                self.features=nn.Sequential(DenseToy()) if spec['model']=='densenet201' else nn.Sequential(nn.Conv2d(3,2,1),nn.ReLU())
                self.attention=CBAM(2) if spec['attention']=='cbam' else nn.Identity()
                self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(2,7))
            def forward(self,x):return self.head(self.attention(self.features(x)))
            def cuda(self):return self
        c=config()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);image=out/'fixture.png';Image.fromarray(np.full((224,224,3),128,dtype=np.uint8)).save(image)
            val=pd.DataFrame([dict(image_id='fixture',label=4,diagnosis='mel',split='val',path=str(image))])
            frame=pd.DataFrame([dict(image_id='fixture',true_class='mel',predicted_class='nv',disagreement=True)])
            def checkpoint(path,**_):
                spec=next(s for s in c['models'] if s['id']==Path(path).parent.name)
                return dict(model=Toy(spec).state_dict())
            with patch.object(xai,'OUT',out),patch.object(xai,'CKPT',out),patch.object(xai,'EnhancedModel',Toy),patch.object(xai,'data',return_value=(None,val,None)),patch.object(xai.torch,'load',side_effect=checkpoint),patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self):
                result=xai.generate(c,frame)
            self.assertEqual(result['cases'],1)
            self.assertTrue((out/'xai/fixture/gradcam_panel.pdf').exists())
            self.assertTrue((out/'xai/fixture/cbam_channel_attention.csv').exists())
            self.assertTrue((out/'xai/fixture/occlusion_sensitivity.npy').exists())

    def test_full_cpu_report_artifacts_preserve_primary_fixed_weights(self):
        import shutil,pandas as pd
        from research.common import ROOT,CLASSES,write_json
        from .core import weighted_metrics,predictions
        from . import results,xai
        c=config();labels=np.tile(np.arange(7),2);ids=[f'fixture{i:02}' for i in range(14)]
        val=pd.DataFrame(dict(image_id=ids,label=labels,diagnosis=[CLASSES[i] for i in labels],split='val'))
        p=np.full((14,7),.1/6);p[np.arange(14),labels]=.9;m=weighted_metrics(labels,p,.1)
        frame=pd.DataFrame(predictions(ids,labels,p));w=torch.ones(7)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'results/aggressive_enhanced/v1';ck=root/'checkpoints/aggressive_enhanced/v1'
            out.mkdir(parents=True)
            for spec in c['models']:
                folder=out/spec['id'];folder.mkdir()
                write_json(folder/'training_summary.json',dict(stopping_epoch=25,precision='bf16'))
                pd.DataFrame([dict(epoch=1,backbone_lr=spec['lr'])]).to_csv(folder/'history.csv',index=False)
                checkpoint=ck/spec['id'];checkpoint.mkdir(parents=True);(checkpoint/'best.pt').write_bytes(b'fixture')
            write_json(out/'class_distribution.json',dict(training_counts={x:4 for x in CLASSES},class_weights={x:1 for x in CLASSES}))
            for path in ['results/final_locked_test/v1/ensemble/test_metrics.json','results/final_strict/v1/ensemble/validation_metrics.json','results/structured_experiments/s13_s12_four_flip_tta_exploratory_seed42/validation_metrics_identity_fp32.json']:
                (root/path).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,root/path)
            with patch.object(results,'ROOT',root),patch.object(results,'OUT',out),patch.object(results,'CKPT',ck),patch.object(results,'data',return_value=(None,val,w)),patch.object(results,'verify_saved',return_value=({'best_epoch':7},m,frame)),patch.object(results,'relative',side_effect=lambda p:Path(p).relative_to(root).as_posix()),patch.object(results,'upsert') as registry,patch.object(xai,'generate',return_value=dict(cases=0,fixture=True)):
                observed=results.closeout(c)
            self.assertEqual(observed['accuracy'],1.);self.assertEqual(registry.call_count,2)
            self.assertEqual(json.loads((out/'primary_weighted/frozen_members.json').read_text())['weights'],[.4,.4,.2])
            self.assertTrue((root/'research/aggressive/AGGRESSIVE_ENHANCED_FINAL_RESULTS.md').exists())
            self.assertTrue((out/'primary_weighted/figures/roc_auc.pdf').exists())

if __name__=='__main__':unittest.main()
