"""CPU-only strong-backbone interfaces; no downloaded weights/images/optimizer."""
import json,unittest
from unittest.mock import patch
import torch
from research.common import ROOT
from research.models import ResearchClassifier
from research.train import validate_config,optimizer_groups
class StrongBackbones(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_densenet201_and_b3_full_gradient_interfaces(self):
        for model,rid,channels in [('densenet201','s15_densenet201_none_exploratory_seed42',1920),('efficientnet_b3','s16_efficientnet_b3_none_exploratory_seed42',1536)]:
            config=json.loads((ROOT/'research/configs/phase3'/f'{rid}.json').read_text());validate_config(config)
            with patch('torch.cuda.is_available',side_effect=AssertionError('CUDA queried')):
                net=ResearchClassifier(model,weights=None,attention='none',dropout=config['head_dropout'])
                net.eval()
                with torch.no_grad():out=net(torch.zeros(1,3,224,224))
                self.assertEqual(tuple(out.shape),(1,7));self.assertTrue(torch.isfinite(out).all());self.assertEqual(net.head[-1].in_features,channels)
                net.train();out=net(torch.rand(2,3,64,64));torch.nn.functional.cross_entropy(out,torch.tensor([0,4])).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
                self.assertTrue(all(p.requires_grad for p in net.parameters()));self.assertIsNotNone(next(net.features.parameters()).grad)
                groups=optimizer_groups(net,config);self.assertEqual(sum(len(g['params']) for g in groups),len(list(net.parameters())))
                self.assertEqual(groups[0]['lr'],3e-5);self.assertEqual(groups[1]['lr'],1e-4)
    def test_new_configs_retain_exact_registered_recipe(self):
        recipe=json.loads((ROOT/'research/phase3/recipe_screening_v1.json').read_text())
        for rid in ['s15_densenet201_none_exploratory_seed42','s16_efficientnet_b3_none_exploratory_seed42']:
            config=json.loads((ROOT/'research/configs/phase3'/f'{rid}.json').read_text())
            for key,value in recipe.items():self.assertEqual(config[key],value)
if __name__=='__main__':unittest.main()
