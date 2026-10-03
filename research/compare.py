"""Paper comparison from completed common-recipe validation artifacts; no models."""
import json
from .common import ROOT, write_csv
from .plots import validate_metrics, comparison_figures
from .registry import read_registry
from .train import validate_config


def main():
    rows=[]
    for r in read_registry():
        if (r['era'],r['status'],r['phase'],r['protocol'],r['attention'],str(r['seed']))!=('structured','completed','backbone_comparison','strict_lesion_disjoint','none','42'):
            continue
        config=json.loads((ROOT/r['config_path']).read_text(encoding='utf-8'))
        validate_config(config) # only the reviewed v1 common recipe belongs here
        m=json.loads((ROOT/r['metrics_path']).read_text(encoding='utf-8'));validate_metrics(m)
        rows.append(dict(display_name=r['model'],experiment_id=r['experiment_id'],protocol=r['protocol'],
             weights=r['pretrained_weights'],seed=r['seed'],epochs=r['epochs'],best_epoch=r['best_epoch'],
             accuracy=m['accuracy'],macro_precision=m['macro_precision'],macro_recall=m['macro_recall'],
             macro_f1=m['macro_f1'],melanoma_recall=m['per_class']['mel']['recall'],
             runtime_seconds=r['runtime_seconds'],checkpoint_sha256=r['checkpoint_sha256']))
    rows.sort(key=lambda r:(-r['macro_f1'],-r['accuracy']))
    if not rows:
        print('No completed common-recipe runs; no comparison output created.');return
    out=ROOT/'results/model_comparison/structured/backbone_comparison_v1_seed42'
    write_csv(out/'comparison.csv',rows)
    if len(rows)>=2:
        comparison_figures(rows,out,'Structured common recipe v1 | strict validation | seed42')
    print(f'{len(rows)} real completed runs; table={out / "comparison.csv"}')


if __name__=='__main__': main()
