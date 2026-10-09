"""CPU-only checks of actual fusion-gradient semantics and duplicate-pass guard."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from research.final_cbam_reporting import finalize as f


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.backbone_name='convnext_tiny'
        self.features=torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.ReLU())
        self.attention=torch.nn.Identity(); self.head=torch.nn.Linear(4,7)

    def forward(self,x):return self.head(self.features(x).mean((2,3)))


class Checks(unittest.TestCase):
    def test_sequential_cam_matches_full_fusion_gradient(self):
        torch.manual_seed(42); models=[Tiny().eval(),Tiny().eval()];x=torch.randn(1,3,8,8,requires_grad=True)
        activations=[];handles=[]
        for model in models:
            handles.append(model.features[-1].register_forward_hook(lambda _m,_i,o:activations.append(o)))
        probabilities=[m(x).softmax(1) for m in models]
        # Other three fixed branches do not affect derivatives of these branches.
        score=sum(.2*p[0,2] for p in probabilities)
        gradients=torch.autograd.grad(score,activations)
        for handle in handles:handle.remove()
        for m,a,g in zip(models,activations,gradients):
            raw=torch.relu((g.mean((2,3),keepdim=True)*a).sum(1,keepdim=True))
            expected=torch.nn.functional.interpolate(raw,(224,224),mode='bilinear',align_corners=False)[0,0].detach().numpy()
            p,actual,heat,attention,metadata=f.cam_of_probability(m,x.detach().clone().requires_grad_(True),2)
            np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-5)
            self.assertEqual(p.shape,(7,));self.assertTrue(np.isfinite(heat).all());self.assertFalse(attention)

    def test_second_inference_blocked_before_cuda(self):
        with tempfile.TemporaryDirectory(dir=f.ROOT,prefix='.s81_probe_') as temporary:
            folder=Path(temporary);folder.resolve().relative_to(f.ROOT.resolve())
            (folder/'inference_started.json').write_text('{}')
            with patch.object(f,'OUT',folder),patch.object(f,'frozen',return_value={}),patch.object(f,'log_setup'),patch.object(f,'deterministic_cuda') as cuda:
                with self.assertRaises(FileExistsError):f.evaluate()
                cuda.assert_not_called()


if __name__=='__main__':unittest.main()
