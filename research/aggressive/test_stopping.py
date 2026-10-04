import unittest
from .stopping import stopping_status

def rows(n,accuracy=.852,f1=.8):
    return [dict(epoch=e,val_accuracy=accuracy,val_macro_f1=f1,
                 backbone_lr=1e-4*.7**((e-1)//10),lr_after=1e-4*.7**(e//10)) for e in range(1,n+1)]

class StoppingTests(unittest.TestCase):
    def test_minimum_window_and_lr_opportunity(self):
        self.assertFalse(stopping_status(rows(24))['stop'])
        self.assertTrue(stopping_status(rows(25))['stop'])
        h=rows(25)
        for r in h:r['lr_after']=r['backbone_lr']=1e-4
        self.assertFalse(stopping_status(h)['stop'])
    def test_tiny_gain_does_not_reset(self):
        h=rows(29);h[-1]['val_accuracy']=.8524
        s=stopping_status(h);self.assertEqual(s['last_meaningful_epoch'],1)
        self.assertTrue(s['stop'])
    def test_either_meaningful_metric_resets(self):
        for column,value in [('val_accuracy',.854),('val_macro_f1',.803)]:
            h=rows(29);h[-1][column]=value;s=stopping_status(h)
            self.assertEqual(s['stale'],0);self.assertFalse(s['stop'])
    def test_cumulative_improvement_from_meaningful_reference(self):
        h=rows(29)
        for i in range(1,29):h[i]['val_accuracy']=.852+min(i*.0001,.002)
        self.assertEqual(stopping_status(h)['last_accuracy_improvement_epoch'],21)
    def test_plateau_guard_and_lr_settle(self):
        h=rows(33)
        for r in h[22:]:r['val_accuracy']=.86
        s=stopping_status(h);self.assertTrue(s['stop']);self.assertEqual(s['reason'],'late_meaningful_plateau')
        self.assertFalse(stopping_status(h[:31])['stop'])
    def test_improving_low_accuracy_not_absolute_target(self):
        h=rows(45)
        for i,r in enumerate(h):r['val_accuracy']=.7+i*.0021
        self.assertFalse(stopping_status(h)['stop'])
        self.assertEqual(stopping_status(rows(50))['reason'],'epoch_cap')

if __name__=='__main__':unittest.main()
