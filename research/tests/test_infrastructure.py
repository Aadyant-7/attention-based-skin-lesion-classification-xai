"""Infrastructure checks only; no training, downloaded weights or image loading."""
import copy
import csv
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from research.common import ROOT, SPLITS
from research.plots import validate_metrics
from research import registry


class RunContractChecks(unittest.TestCase):
    def fixture(self):
        from research.experiment import Experiment
        from research.common import sha256
        obj=Experiment.__new__(Experiment)
        obj.config={'experiment_id':'s01_fixture','split_manifest':SPLITS['strict_lesion_disjoint'],
                    'split_sha256':sha256(ROOT/SPLITS['strict_lesion_disjoint'])}
        metrics=json.loads((ROOT/'results/runs/efficientnet_b0_cbam_weighted_v1/validation_metrics.json').read_text())
        obj.validation_support={c:v['support'] for c,v in metrics['per_class'].items()}
        obj.history=[{'val_macro_f1':metrics['macro_f1']}]
        return obj,metrics

    def test_historical_checkpoint_cannot_be_used_as_new_run_checkpoint(self):
        obj,metrics=self.fixture()
        with self.assertRaises(ValueError):
            obj.complete(metrics,ROOT/'checkpoints/efficientnet_b0_cbam_weighted_v1/best.pt',1,0)

    def test_same_size_wrong_validation_cohort_is_rejected(self):
        obj,metrics=self.fixture()
        exp=pd.read_csv(ROOT/SPLITS['exploratory_image_level']).query("split=='val'")
        obj.validation_support=exp.diagnosis.value_counts().to_dict()
        with self.assertRaises(ValueError):
            obj.complete(metrics,ROOT/'checkpoints/efficientnet_b0_cbam_weighted_v1/best.pt',1,0)


class EvidenceChecks(unittest.TestCase):
    def setUp(self):
        self.metrics=json.loads((ROOT/'results/runs/efficientnet_b0_cbam_weighted_v1/validation_metrics.json').read_text())

    def test_saved_matrix_and_reported_metrics_are_consistent(self):
        cm=validate_metrics(self.metrics)
        self.assertEqual(int(cm.sum()),1503)

    def test_inconsistent_accuracy_and_macro_f1_rejected(self):
        for key in ('accuracy','macro_f1'):
            m=copy.deepcopy(self.metrics)
            m[key]+=.01
            with self.assertRaises(ValueError):
                validate_metrics(m)

    def test_wrong_class_support_rejected(self):
        m=copy.deepcopy(self.metrics)
        m['per_class']['mel']['support']+=1
        with self.assertRaises(ValueError):
            validate_metrics(m)

    def test_protocol_manifests_keep_original_test_identity(self):
        strict=pd.read_csv(ROOT/SPLITS['strict_lesion_disjoint'])
        exp=pd.read_csv(ROOT/SPLITS['exploratory_image_level'])
        self.assertTrue(strict.query("split=='test'").equals(exp.query("split=='test'")))
        groups={s:set(strict.loc[strict.split==s,'lesion_id']) for s in ('train','val','test')}
        self.assertFalse(groups['train']&groups['val'])
        self.assertFalse(groups['train']&groups['test'])
        self.assertFalse(groups['val']&groups['test'])
        self.assertEqual(len(set(exp.query("split=='train'").lesion_id)&set(exp.query("split=='val'").lesion_id)),563)


class RegistryChecks(unittest.TestCase):
    def test_clone_refresh_does_not_drop_missing_local_history(self):
        from research.common import write_csv,sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'registry.csv'
            portable=root/'results/legacy/evidence/validation_metrics.json'
            portable.parent.mkdir(parents=True)
            portable.write_bytes((ROOT/'results/runs/efficientnet_b0_cbam_weighted_v1/validation_metrics.json').read_bytes())
            row={k:'' for k in registry.FIELDS}
            row.update(experiment_id='legacy:missing-original',era='legacy',metrics_path='results/runs/missing/validation_metrics.json',
                       portable_metrics_path=portable.relative_to(root).as_posix(),source_sha256=sha256(portable))
            write_csv(path,[row],registry.FIELDS)
            with patch.object(registry,'ROOT',root),patch.object(registry,'REGISTRY',path),patch.object(registry,'collect_legacy',return_value=[]):
                rows=registry.rebuild()
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['source_availability'],'portable_evidence')

    def test_partial_legacy_schema_and_parallel_writers_preserve_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'registry.csv'
            path.write_text('experiment_id,era\nlegacy:old,legacy\n',encoding='utf-8')
            with patch.object(registry,'REGISTRY',path):
                with ThreadPoolExecutor(max_workers=4) as workers:
                    list(workers.map(lambda n:registry.upsert({'experiment_id':f's{n:02d}_test','era':'structured','status':'initialized'}),range(8)))
                rows=registry.read_registry()
                self.assertEqual(len(rows),9)
                self.assertEqual(len({r['experiment_id'] for r in rows}),9)
                self.assertEqual(rows[0]['experiment_id'],'legacy:old')
                with self.assertRaises(ValueError):
                    registry.upsert({'experiment_id':'legacy:old','era':'structured'})

    def test_missing_identifier_header_fails_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'registry.csv'
            original='accuracy\n0.8\n'
            path.write_text(original)
            with patch.object(registry,'REGISTRY',path):
                with self.assertRaises(ValueError):
                    registry.upsert({'experiment_id':'s01_test','era':'structured'})
            self.assertEqual(path.read_text(),original)


class ModelInterfaceChecks(unittest.TestCase):
    def test_six_backbones_and_optional_attention_without_weights(self):
        import torch
        from research.models import ResearchClassifier, CHANNELS
        torch.set_num_threads(2)
        x=torch.zeros(2,3,32,32)
        for name in CHANNELS:
            for attention in ('none','cbam'):
                with self.subTest(model=name,attention=attention):
                    net=ResearchClassifier(name,weights=None,attention=attention).eval()
                    with torch.inference_mode():
                        logits=net(x)
                    self.assertEqual(tuple(logits.shape),(2,7))
                    self.assertTrue(torch.isfinite(logits).all().item())
                    del net

    def test_weight_alias_cannot_silently_change(self):
        from research.models import ResearchClassifier
        with self.assertRaises(ValueError):
            ResearchClassifier('resnet50',weights='DEFAULT')
        with self.assertRaises(ValueError):
            ResearchClassifier('resnet50',weights='EfficientNet_B0_Weights.IMAGENET1K_V1')


if __name__=='__main__':
    unittest.main()
