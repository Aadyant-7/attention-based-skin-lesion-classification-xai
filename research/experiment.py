"""Future structured-run artifact contract. This module never trains a model."""
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from .common import ROOT, SPLITS, sha256, relative, write_csv, write_json
from .registry import FIELDS, upsert
from .plots import validate_metrics, metric_figures, training_figures

REQUIRED = ('experiment_id', 'question', 'model', 'weights', 'attention', 'protocol', 'seed',
            'image_size', 'preprocessing', 'augmentation', 'imbalance', 'loss', 'optimizer',
            'learning_rate', 'batch_size', 'max_epochs', 'selection_metric')


class Experiment:
    def __init__(self, config):
        missing = [k for k in REQUIRED if k not in config]
        if missing:
            raise ValueError(f'Missing config fields: {missing}')
        if config['protocol'] not in SPLITS or config['selection_metric'] != 'macro_f1':
            raise ValueError('Use an explicit registered development protocol and macro_f1 selection')
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,100}', config['experiment_id']):
            raise ValueError('Unsafe experiment identifier')
        self.config = dict(config)
        self.path = ROOT/'results/structured_experiments'/config['experiment_id']
        self.path.mkdir(parents=True, exist_ok=False)  # no overwrite or implicit resume
        split = ROOT/SPLITS[config['protocol']]
        self.config['split_manifest'] = relative(split)
        self.config['split_sha256'] = sha256(split)
        self.history = []
        self.validation_support=pd.read_csv(split).query("split == 'val'").diagnosis.value_counts().to_dict()
        self.row = {k:'' for k in FIELDS}
        self.row.update(experiment_id=config['experiment_id'], era='structured', record_kind='training_run',
                        phase=config.get('phase','backbone_comparison'), protocol=config['protocol'],
                        evaluation_split='validation', split_manifest=relative(split), split_sha256=sha256(split),
                        method=config['question'], model=config['model'], pretrained_weights=config['weights'],
                        attention=config['attention'], status='initialized', decision='pending',
                        config_path=relative(self.path/'config.json'))
        for k in ('image_size','preprocessing','augmentation','imbalance','loss','optimizer','batch_size','seed'):
            self.row[k] = config[k] if isinstance(config[k],(str,int,float)) else json.dumps(config[k])
        if isinstance(config['learning_rate'],dict):
            self.row['backbone_lr']=config['learning_rate'].get('backbone','')
            self.row['head_lr']=config['learning_rate'].get('head','')
        write_json(self.path/'config.json',self.config)
        environment = {'created_utc':datetime.now(timezone.utc).isoformat(), 'python':sys.version,
                       'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                       'packages':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}
        write_json(self.path/'environment.json',environment)
        upsert(self.row)

    def log_epoch(self, row):
        required = ('epoch','train_loss','train_accuracy','val_loss','val_accuracy','val_macro_f1')
        if any(k not in row for k in required) or int(row['epoch']) != len(self.history)+1:
            raise ValueError('Epoch rows must contain required metrics and be sequential')
        self.history.append(dict(row))
        write_csv(self.path/'history.csv',self.history)
        self.row.update(status='running',epochs=len(self.history),history_path=relative(self.path/'history.csv'))
        upsert(self.row)

    def complete(self, metrics, checkpoint, best_epoch, runtime_seconds, notes=''):
        validate_metrics(metrics)
        if {c:int(v['support']) for c,v in metrics['per_class'].items()}!=self.validation_support:
            raise ValueError('Result supports do not match the declared validation split')
        if sha256(ROOT/self.config['split_manifest']) != self.config['split_sha256']:
            raise ValueError('Split changed during the experiment')
        if not self.history or not 1 <= best_epoch <= len(self.history):
            raise ValueError('Missing history / invalid best epoch')
        selected = self.history[best_epoch-1]
        maximum = max(float(r['val_macro_f1']) for r in self.history)
        if abs(float(selected['val_macro_f1'])-maximum)>1e-8 or abs(float(metrics['macro_f1'])-maximum)>1e-8:
            raise ValueError('Metrics must match the macro-F1-selected epoch')
        checkpoint = Path(checkpoint).resolve()
        checkpoint.relative_to(ROOT/'checkpoints/structured'/self.config['experiment_id'])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        write_json(self.path/'validation_metrics.json',metrics)
        figures=self.path/'figures'
        title=f"{self.config['model']} | {self.config['protocol']} validation"
        metric_figures(metrics,figures,title)
        training_figures(pd.DataFrame(self.history),figures,title)
        self.row.update(status='completed',best_epoch=best_epoch,runtime_seconds=runtime_seconds,
                        checkpoint=relative(checkpoint),checkpoint_available_local=True,
                        checkpoint_sha256=sha256(checkpoint),
                        metrics_path=relative(self.path/'validation_metrics.json'),source_sha256=sha256(self.path/'validation_metrics.json'),
                        confusion_matrix_path=relative(self.path/'validation_metrics.json'),plots_dir=relative(figures),
                        val_loss=metrics.get('loss',''),notes=notes)
        for k in ('accuracy','macro_precision','macro_recall','macro_f1'):
            self.row[k]=metrics[k]
        write_json(self.path/'record.json',self.row)
        upsert(self.row)

    def fail(self, reason):
        self.row.update(status='failed',notes=str(reason))
        write_json(self.path/'record.json',self.row)
        upsert(self.row)
