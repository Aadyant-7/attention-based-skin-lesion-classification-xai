"""CPU checks of the fixed inference views and no-GPU preflight policy."""
import unittest
from unittest.mock import patch
import torch
from research.tta import flip,check
from research.common import ROOT
from research.phase3.close_s02 import load
class TTAViews(unittest.TestCase):
    def test_fixed_geometry_preserves_values_and_identity(self):
        x=torch.arange(24).reshape(1,2,3,4)
        self.assertIs(flip(x,'identity'),x)
        for view,dims in [('horizontal',[-1]),('vertical',[-2]),('both',[-2,-1])]:
            self.assertTrue(torch.equal(flip(x,view),x.flip(dims)))
            self.assertTrue(torch.equal(flip(flip(x,view),view),x))
    def test_preflight_never_queries_cuda(self):
        c=load(ROOT/'research/configs/phase3/s13_s12_four_flip_tta_exploratory_seed42.json')
        with patch('torch.cuda.is_available',side_effect=AssertionError('CUDA queried')):
            frame,val,parents=check(c)
        self.assertEqual(len(val),1503);self.assertEqual(len(parents),3)
    def test_unreviewed_views_or_precision_refused(self):
        c=load(ROOT/'research/configs/phase3/s13_s12_four_flip_tta_exploratory_seed42.json')
        for k,v in [('views',['identity','horizontal']),('precision','amp_fp16')]:
            with self.assertRaises(ValueError):check(dict(c,**{k:v}))
if __name__=='__main__':unittest.main()
