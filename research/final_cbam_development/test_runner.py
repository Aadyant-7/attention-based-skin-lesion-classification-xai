"""Meaningful CPU gates for bookkeeping recovery, seeded loaders and artifact closure."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from research.common import ROOT, CLASSES, relative, sha256
from research.strict_train import atomic_checkpoint
from research.short_screening.lesion_bag_screen import report
from .protocol import CONFIG, verified_development, stopping
from .runtime import signature, loader
from .artifacts import COLS, fp32_parts, repair, closeout


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c,cls.sig=signature();cls.train,cls.val=verified_development(cls.c)
        cls.registry_sha=sha256(ROOT/'results/master_experiment_registry.csv')

    def test_epoch_loader_replay_and_forbidden_partition(self):
        c=dict(self.c,batch_size=2,workers=0)
        frame=self.train.iloc[:4]
        state=torch.get_rng_state();a=list(loader(frame,c,3,True))
        torch.set_rng_state(state);b=list(loader(frame,c,3,True))
        self.assertEqual([i for batch in a for i in batch[2]],[i for batch in b for i in batch[2]])
        self.assertTrue(all(torch.equal(x[0],y[0]) for x,y in zip(a,b)))
        with self.assertRaises(ValueError):loader(frame.assign(split='test'),c,3,True)
        # Worker process seed path is checked once without touching the full cohort.
        c['workers']=2
        a=list(loader(frame,c,3,True));b=list(loader(frame,c,3,True))
        self.assertTrue(all(torch.equal(x[0],y[0]) for x,y in zip(a,b)))

    def test_committed_epoch_repairs_outputs_without_retraining(self):
        parent=ROOT/'.cache/final_cbam_development/v1';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='artifact-probe-',dir=parent) as tmp:
            p=Path(tmp);folder=p/'results';ck=p/'checkpoints';ck.mkdir();folder.mkdir()
            c=copy.deepcopy(self.c);c.update(results=relative(folder),checkpoints=relative(ck))
            probabilities=fp32_parts(self.sig,self.val)[0];metrics=report(self.val.label.to_numpy(),probabilities)
            f=self.val[['image_id','lesion_id']].copy();f['true_class']=self.val.diagnosis
            f['predicted_class']=[CLASSES[i] for i in probabilities.argmax(1)];f[COLS]=probabilities
            best=dict(epoch=1,metrics=metrics,predictions=f.to_dict('records'),model={'disposable_probe':torch.zeros(1)})
            history=[dict(epoch=i,train_loss=.1,train_accuracy=.9,val_loss=metrics['loss'],
                val_accuracy=metrics['accuracy'],val_macro_f1=metrics['macro_f1'],scheduler_lr_reduced=i==8) for i in range(1,41)]
            payload=dict(config=c,signature=self.sig,epoch=40,history=history,best_accuracy=best,best_macro_f1=best,
                last_validation=dict(metrics=metrics,predictions=f.to_dict('records')),model=best['model'],meaningful_stopping=stopping(history,c),
                runtime_seconds=0,optimizer_updates=0,disposable_artifact_fixture=True)
            latest=ck/'latest.pt';atomic_checkpoint(latest,payload);before=sha256(latest)
            # Simulate a power cut after commit, before any CSV/registry write.
            committed=torch.load(latest,map_location='cpu',weights_only=False)
            repair(committed,folder,ck,c,publish=False)
            self.assertEqual(len(pd.read_csv(folder/'history.csv')),40)
            self.assertEqual(sha256(latest),before)
            repair(committed,folder,ck,c,publish=False)
            self.assertEqual(len(pd.read_csv(folder/'history.csv')),40)
            closeout(committed,folder,ck,c,self.val,publish=False)
            self.assertEqual(json.loads((folder/'progress.json').read_text())['status'],'completed')
            self.assertFalse(json.loads((folder/'fixed_ensemble/summary.json').read_text())['material_gate_passed'])
            for f in ['validation_predictions.csv','validation_probabilities.csv','figures/confusion_matrix.png',
                      'figures/confusion_matrix.pdf','figures/accuracy_curves.png','macro_f1_selected/figures/per_class_metrics.pdf',
                      'final_epoch/validation_metrics.json','fixed_ensemble/comparison_figures/model_comparison.pdf']:
                self.assertTrue((folder/f).is_file(),f)
            self.assertEqual(sha256(latest),before)
        self.assertEqual(sha256(ROOT/'results/master_experiment_registry.csv'),self.registry_sha)


if __name__=='__main__':unittest.main()
