"""Authoritative evidence registry: legacy imports and new run records."""
import csv
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from .common import ROOT, SPLITS, sha256, relative, write_csv

REGISTRY = ROOT / 'results/master_experiment_registry.csv'
FIELDS = ['experiment_id', 'era', 'record_kind', 'phase', 'protocol', 'evaluation_split',
          'split_manifest', 'split_sha256', 'method', 'model', 'pretrained_weights', 'attention',
          'image_size', 'preprocessing', 'augmentation', 'imbalance', 'loss', 'optimizer',
          'backbone_lr', 'head_lr', 'batch_size', 'seed', 'epochs', 'best_epoch',
          'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'val_loss',
          'runtime_seconds', 'checkpoint', 'checkpoint_available_local', 'config_path',
          'ensemble_members','ensemble_weights',
          'history_path', 'metrics_path', 'source_selector', 'source_sha256',
          'portable_metrics_path','portable_history_path','portable_config_path','source_availability',
          'checkpoint_sha256','confusion_matrix_path', 'plots_dir', 'status', 'decision', 'notes']


@contextmanager
def locked(path=None):
    path=Path(path or REGISTRY)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix('.lock')
    deadline = time.monotonic() + 10
    while True:
        try:
            f = lock.open('x')
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Registry busy; inspect stale lock if no writer is active: {lock}')
            time.sleep(.1)
    try:
        yield
    finally:
        f.close()
        lock.unlink(missing_ok=True)


def read_registry():
    if not REGISTRY.exists():
        return []
    with REGISTRY.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'experiment_id' not in reader.fieldnames:
            raise ValueError('Registry lacks experiment_id')
        rows = list(reader)
    if any(None in r for r in rows) or len({r['experiment_id'] for r in rows}) != len(rows):
        raise ValueError('Malformed registry / duplicate experiment identifiers')
    return rows


def upsert(row):
    if not row.get('experiment_id') or row.get('era') != 'structured':
        raise ValueError('New registry rows require experiment_id and era=structured')
    with locked():
        rows = read_registry()
        previous = next((r for r in rows if r['experiment_id'] == row['experiment_id']), None)
        if previous and previous['era'] != 'structured':
            raise ValueError('Cannot overwrite legacy entry')
        rows = [r for r in rows if r['experiment_id'] != row['experiment_id']] + [row]
        write_csv(REGISTRY, sorted(rows, key=lambda r: r['experiment_id']), FIELDS)


def protocol_for(path):
    return 'exploratory_image_level' if 'results/exploratory/' in relative(path) else 'strict_lesion_disjoint'


def make_record(path, metrics, selector='', metadata=None, kind='inference_candidate'):
    metadata = metadata or {}
    protocol = protocol_for(path)
    split = SPLITS[protocol]
    source = relative(path)
    rid = 'legacy:' + source + (('#'+selector) if selector else '')
    row = {key: '' for key in FIELDS}
    row.update(experiment_id=rid, era='legacy', record_kind=kind, phase=source.split('/')[1],
               protocol=protocol, evaluation_split='validation', split_manifest=split,
               split_sha256=sha256(ROOT/split), metrics_path=source, source_selector=selector,
               source_sha256=sha256(path), method=metrics.get('method', metrics.get('run_name', metadata.get('run_name', path.stem))),
               source_availability='original',
               model=metrics.get('family',metrics.get('classifier',metrics.get('model',metrics.get('components','')))),
               status='historical_observed', decision='not_selected_for_new_final_model',
               notes='Imported from saved evidence; protocol mapped from audited producer/folder. Empty fields mean not recorded. Validation-selected; test not evaluated.')
    for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'runtime_seconds'):
        if key in metrics:
            value = float(metrics[key])
            if key != 'runtime_seconds' and not 0 <= value <= 1:
                raise ValueError((path, key, value))
            row[key] = value
    row['val_loss'] = metrics.get('loss', '')
    row['ensemble_members']=metrics.get('components','')
    row['ensemble_weights']=json.dumps(metrics['weights']) if isinstance(metrics.get('weights'),(dict,list)) else metrics.get('weights','')
    if 'confusion_matrix' in metrics or 'confusion_matrix_json' in metrics:
        row['confusion_matrix_path'] = source + (('#'+selector) if selector else '')
    for key in ('epochs', 'best_epoch', 'runtime_seconds', 'checkpoint'):
        row[key] = metadata.get(key, row[key])
    if row['checkpoint']:
        row['checkpoint_available_local'] = (ROOT / row['checkpoint']).is_file()
    return row


def collect_legacy():
    rows = []
    highlights=json.loads((ROOT/'research/legacy_highlights.json').read_text())
    for path in sorted((ROOT/'results').rglob('*.json')):
        if any(part in ('legacy', 'structured_experiments', 'figures', 'final', 'model_comparison', 'datasets', 'audit') for part in path.relative_to(ROOT/'results').parts):
            continue
        obj = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(obj, dict):
            continue
        if relative(path).startswith('results/exploratory/runs/') and path.name=='validation_metrics.json':
            if (ROOT/'results/exploratory'/f'{path.parent.name}_summary.json').exists():
                continue  # the authoritative summary records this same training result
        if 'accuracy' in obj and 'macro_f1' in obj:
            rows.append(make_record(path, obj))
        elif isinstance(obj.get('best_validation'), dict):
            rows.append(make_record(path, obj['best_validation'], 'best_validation', obj, 'training_run'))
    # CSV trial tables retain candidates with no individual full-metric JSON.
    for path in sorted((ROOT/'results').rglob('*.csv')):
        if any(p in ('legacy', 'structured_experiments', 'figures', 'datasets', 'model_comparison', 'final', 'audit') for p in path.relative_to(ROOT/'results').parts):
            continue
        if path.name in ('history.csv', 'experiments.csv', 'master_experiment_registry.csv', 'summary.csv') or '_history' in path.name:
            continue
        with path.open(encoding='utf-8', newline='') as f:
            candidates = list(csv.DictReader(f))
        for i, c in enumerate(candidates):
            if c.get('accuracy') and c.get('macro_f1'):
                rows.append(make_record(path, c, f'csv_row:{i+1}', kind='tabular_candidate'))
    with (ROOT/'results/experiments.csv').open(encoding='utf-8', newline='') as f:
        ledger = list(csv.DictReader(f))
    # Attach verified training metadata to existing metric records instead of cloning entries.
    for row in rows:
        source = row['metrics_path']
        if source.startswith('results/runs/') and source.endswith('/validation_metrics.json'):
            run_name = Path(source).parent.name
            record = next((r for r in ledger if r.get('run_name') == run_name), {})
            config_path = ROOT / Path(source).parent / 'config.json'
            config = json.loads(config_path.read_text()) if config_path.exists() else {}
            row.update(record_kind='training_run', method=run_name, model=record.get('variant', ''),
                       pretrained_weights='ImageNet-1K (torchvision EfficientNet-B0 default; recorded producer)',
                       attention=record.get('cbam',''), loss=record.get('loss',''), augmentation=record.get('augmentation',''),
                       epochs=record.get('epochs',''), best_epoch=record.get('best_epoch',''),
                       checkpoint=record.get('checkpoint',''), status=record.get('status','historical_observed'),
                       config_path=relative(config_path) if config_path.exists() else '')
            for k in ('image_size','imbalance','optimizer','backbone_lr','head_lr','batch_size','seed'):
                row[k] = config.get(k, record.get(k,''))
            row['preprocessing']=f"RGB; resize {row['image_size']} square; ImageNet normalization (verified src/data.py)"
            row['runtime_seconds'] = record.get('runtime_seconds','')  # leave repaired missing runtime empty
        elif row['record_kind'] == 'training_run' and source.startswith('results/exploratory/'):
            config_path = ROOT/'results/exploratory/runs'/row['method']/'config.json'
            if config_path.exists():
                config = json.loads(config_path.read_text())
                row['config_path'] = relative(config_path)
                for k in ('image_size','imbalance','backbone_lr','head_lr','batch_size','seed'):
                    row[k] = config.get(k,'')
                row.update(model='EfficientNet-B0', attention='CBAM', pretrained_weights='ImageNet-1K (recorded producer)',
                           augmentation=config.get('recipe','baseline_horizontal_flip'), loss='weighted CE', optimizer='AdamW')
        if row['record_kind'] == 'training_run':
            history = ROOT / Path(source).parent / 'history.csv'
            if source.startswith('results/exploratory/'):
                history = ROOT/'results/exploratory'/f"{row['method']}_history.csv"
            row['history_path'] = relative(history) if history.exists() else ''
            if row['checkpoint']:
                row['checkpoint_available_local'] = (ROOT/row['checkpoint']).is_file()
        if 'panderm_base_last2_pilot_v1/best_validation_metrics.json' in source:
            p = ROOT/Path(source).parent/'protocol.json'
            m = json.loads(p.read_text())
            row.update(record_kind='training_run', model=m.get('backbone',''), method=m.get('run',''),
                       epochs=m.get('completed_epochs',''), best_epoch=m.get('best_epoch',''), checkpoint=m.get('checkpoint',''),
                       history_path=relative(ROOT/Path(source).parent/'history.csv'))
            row['checkpoint_available_local'] = (ROOT/row['checkpoint']).is_file()
        neighboring = ROOT/Path(source).parent/'protocol.json'
        if neighboring.exists() and not row['config_path']:
            row['config_path'] = relative(neighboring)
        if neighboring.exists():
            metadata=json.loads(neighboring.read_text())
            row['pretrained_weights']=row['pretrained_weights'] or metadata.get('weights_source','')
            if not row['checkpoint'] and metadata.get('source_checkpoints'):
                row['checkpoint']=json.dumps(metadata['source_checkpoints'])
                row['checkpoint_available_local']=all((ROOT/p).is_file() for p in metadata['source_checkpoints'])
        # Explicit historical highlights; no label implying a new final selection.
        leaders = {
            'results/exploratory/panderm_base_image_level_tta_v1/b0_four_plus_four_svc_views_eight_equal.json': 'historical_exploratory_accuracy_leader',
            'results/exploratory/multires_ensemble/four_equal.json': 'historical_exploratory_b0_leader',
            'results/accuracy_exploration/panderm_base_strict_tta_v1/prior_four_with_svc_tta_equal.json': 'historical_strict_accuracy_leader',
            'results/accuracy_exploration/panderm_base_strict_tta_v1/weighted_b0_plus_svc_tta_two_equal.json': 'historical_strict_macro_f1_leader'}
        if source in leaders:
            row['decision'] = leaders[source]
        if source in highlights:
            m=highlights[source]
            if any(not (ROOT/p).is_file() for p in m['source_scripts']):
                raise FileNotFoundError('Missing producer for historical highlight')
            for k in ('model','image_size','attention'):
                row[k]=m[k]
            row['checkpoint']=json.dumps(m['checkpoints'])
            row['checkpoint_available_local']=all((ROOT/p).is_file() for p in m['checkpoints'])
            row['ensemble_members']=json.dumps(m['ensemble_members'])
            row['ensemble_weights']=json.dumps(m['ensemble_weights'])
            row['notes']+=' Component metadata verified against '+', '.join(m['source_scripts'])+'.'
        for field in ('metrics_path','history_path','config_path'):
            portable=ROOT/'results/legacy/evidence'/str(row[field])
            if row[field] and portable.is_file():
                row['portable_'+field]=relative(portable)
    return rows


def rebuild():
    with locked():
        current=read_registry()
        retained = [r for r in current if r['era'] != 'legacy']
        previous={r['experiment_id']:r for r in current}
        rows = collect_legacy() + retained
        current_ids={r['experiment_id'] for r in rows}
        for old in current:
            if old['era']=='legacy' and old['experiment_id'] not in current_ids and not (ROOT/old['metrics_path']).exists():
                portable=ROOT/old['portable_metrics_path'] if old.get('portable_metrics_path') else None
                old['source_availability']='portable_evidence' if portable and portable.is_file() else 'unavailable_local'
                if portable and portable.is_file() and sha256(portable)!=old['source_sha256']:
                    raise ValueError('Portable historical metric evidence changed')
                rows.append(old)  # never erase evidence rows merely because ignored assets are absent
        for r in rows:
            old=previous.get(r['experiment_id'],{})
            if r['era']=='legacy' and old.get('source_sha256')==r['source_sha256'] and old.get('plots_dir') and (ROOT/old['plots_dir']).is_dir():
                r['plots_dir']=old['plots_dir']
        if len({r['experiment_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate evidence identifiers')
        write_csv(REGISTRY, sorted(rows,key=lambda r:r['experiment_id']), FIELDS)
    return rows
