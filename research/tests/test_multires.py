"""CPU preflight and exact fixed resolution/member composition checks."""
import unittest
from unittest.mock import patch
import numpy as np
from research.multires import check,fixed_fusion,S06
from research.common import ROOT
from research.phase3.close_s02 import load
class MultiResolution(unittest.TestCase):
    def setUp(self):self.config=load(ROOT/'research/configs/phase3/s14_s12_convnext_224_320_exploratory_seed42.json')
    def test_exact_member_weight_preserved(self):
        ids=self.config['parent_run_ids'];identity={rid:np.eye(7)[[j,j]] for j,rid in enumerate(ids)};higher=np.eye(7)[[3,3]]
        found=fixed_fusion(self.config,identity,higher)
        expected=sum(identity[r]/3 for r in ids if r!=S06)+identity[S06]/6+higher/6
        np.testing.assert_allclose(found,expected,rtol=0,atol=1e-16)
    def test_metadata_check_does_not_query_cuda(self):
        with patch('torch.cuda.is_available',side_effect=AssertionError('CUDA queried')):
            _,val,parents,identity,_=check(self.config)
        self.assertEqual(len(val),1503);self.assertEqual(len(identity),3)
    def test_extra_sizes_flips_or_weights_refused(self):
        for key,value in [('resolutions',[224,320,384]),('views',['identity','horizontal']),('resolution_weights',[.7,.3]),('weights',[.2,.3,.5])]:
            with self.assertRaises((ValueError,AssertionError)):check(dict(self.config,**{key:value}))
if __name__=='__main__':unittest.main()
