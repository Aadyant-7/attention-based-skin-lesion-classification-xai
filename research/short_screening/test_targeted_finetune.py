import unittest,copy
import numpy as np
from .targeted_finetune import batches,gate,recipe

class TargetedChecks(unittest.TestCase):
    def test_equal_compute_and_only_training_focus_indices(self):
        labels=np.tile(np.arange(7),100)
        control=batches(labels,1,False);target=batches(labels,1,True)
        self.assertEqual(len(control),len(target));self.assertTrue(all(len(b)==32 for b in control+target))
        a=np.concatenate(control);b=np.concatenate(target)
        self.assertTrue(np.all((b>=0)&(b<len(labels))))
        self.assertGreater(np.isin(labels[b],[2,4]).sum(),np.isin(labels[a],[2,4]).sum())
        self.assertEqual(control,batches(labels,1,False))
        self.assertEqual(target,batches(labels,1,True))
    def test_material_gate_cannot_pass_with_tiny_gain(self):
        base=dict(accuracy=.9,macro_f1=.85,per_class={c:dict(recall=r) for c,r in [('mel',.6),('bkl',.8),('nv',.98)]})
        candidate=copy.deepcopy(base);candidate.update(accuracy=.906,macro_f1=.86)
        candidate['per_class']['mel']['recall']=.63;candidate['per_class']['bkl']['recall']=.83
        self.assertTrue(gate(candidate,base,base)['passed'])
        candidate['accuracy']=.901;self.assertFalse(gate(candidate,base,base)['passed'])
        candidate['accuracy']=.906;candidate['per_class']['nv']['recall']=.96
        self.assertFalse(gate(candidate,base,base)['passed'])
    def test_hard_budget(self):
        c=recipe();self.assertEqual(c['control_epochs']+c['targeted_max_epochs'],20)
        self.assertEqual(c['pilot_epochs'],5);self.assertFalse(c['test_evaluation'])

if __name__=='__main__':unittest.main()
