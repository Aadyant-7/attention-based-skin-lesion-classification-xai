"""Paper comparison from completed common-recipe validation artifacts; no models."""
import json
from .common import ROOT, write_csv
from .plots import validate_metrics, comparison_figures
from .registry import read_registry
from .train import validate_config


def main():
    groups={}
    for r in read_registry():
        if (r['era'],r['status'])!=('structured','completed'):
            continue
        if r['record_kind'] != 'training_run':
            continue  # CPU fusion has its own cost-labelled comparison; no fake training recipe.
        config=json.loads((ROOT/r['config_path']).read_text(encoding='utf-8'))
        validate_config(config) # only the reviewed v1 common recipe belongs here
        m=json.loads((ROOT/r['metrics_path']).read_text(encoding='utf-8'));validate_metrics(m)
        key=(config['recipe_version'],r['protocol'],str(r['seed']),config['selection_metric'])
        rows=groups.setdefault(key,[])
        rows.append(dict(display_name=r['model']+' | '+r['attention'],experiment_id=r['experiment_id'],protocol=r['protocol'],
             recipe_version=config['recipe_version'],selection_metric=config['selection_metric'],
             weights=r['pretrained_weights'],seed=r['seed'],epochs=r['epochs'],best_epoch=r['best_epoch'],
             accuracy=m['accuracy'],macro_precision=m['macro_precision'],macro_recall=m['macro_recall'],
             macro_f1=m['macro_f1'],melanoma_recall=m['per_class']['mel']['recall'],
             runtime_seconds=r['runtime_seconds'],checkpoint_sha256=r['checkpoint_sha256']))
    if not groups:
        print('No completed common-recipe runs; no comparison output created.');return
    for (recipe,protocol,seed,selection),rows in groups.items():
        other='macro_f1' if selection=='accuracy' else 'accuracy'
        rows.sort(key=lambda r:(-r[selection],-r[other]))
        out=ROOT/f'results/model_comparison/structured/{recipe}_seed{seed}'
        write_csv(out/'comparison.csv',rows)
        if len(rows)>=2:
            comparison_figures(rows,out,f'{recipe} | {protocol} validation | seed{seed}')
        print(f'{len(rows)} real completed runs; table={out / "comparison.csv"}')


if __name__=='__main__': main()
