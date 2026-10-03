"""Synthetic CPU checks for the proposed matched added-CBAM experiment."""
import copy
import json
import unittest
import torch
from research.common import ROOT
from research.models import ResearchClassifier
from research.train import validate_config, optimizer_groups


class CBAMProposalChecks(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'research/configs/phase3/s05_efficientnet_b0_cbam_exploratory_seed42.json').read_text(encoding='utf-8'))
        torch.set_num_threads(2)

    def test_registered_pair_changes_only_declared_attention_metadata(self):
        validate_config(self.config)
        control=json.loads((ROOT/'research/configs/phase3/s03_efficientnet_b0_none_exploratory_seed42.json').read_text(encoding='utf-8'))
        differences={k for k in set(control)|set(self.config) if control.get(k)!=self.config.get(k)}
        self.assertEqual(differences,{'recipe_version','phase','attention','experiment_id','question'})
        for key,value in (('batch_size',8),('attention','none'),('protocol','strict_lesion_disjoint')):
            broken=copy.deepcopy(self.config);broken[key]=value
            with self.assertRaises(ValueError):validate_config(broken)

    def test_cbam_spatial_forward_gradients_and_optimizer_assignment(self):
        torch.manual_seed(7)
        model=ResearchClassifier('efficientnet_b0',weights=None,attention='cbam',dropout=.2).eval()
        shapes=[]
        hook=model.attention.register_forward_pre_hook(lambda module,args:shapes.append(tuple(args[0].shape)))
        logits=model(torch.randn(2,3,224,224));hook.remove()
        self.assertEqual(tuple(logits.shape),(2,7));self.assertEqual(shapes,[(2,1280,7,7)])
        torch.nn.functional.cross_entropy(logits,torch.tensor([0,4])).backward()
        gradients=[p.grad for p in model.attention.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().sum()>0 for g in gradients))
        groups=optimizer_groups(model,self.config)
        fresh={id(p) for p in model.attention.parameters()}|{id(p) for p in model.head[-1].parameters()}
        self.assertEqual(fresh,{id(p) for p in groups[1]['params']})
        self.assertEqual(groups[1]['lr'],self.config['learning_rate']['head'])
        self.assertEqual(sum(p.numel() for p in model.parameters()),4221413)


if __name__=='__main__':unittest.main()
