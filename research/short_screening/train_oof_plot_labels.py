"""Correct endpoint labels in new CPU group-OOF figures; never rescore a model."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.common import ROOT,CLASSES,sha256,write_json
from research.plots import save


def main():
    sources=[('s72_nonlinear_frozen_feature_head','train_group_metrics'),
             ('s73_matched_contrastive_head_cpu','metrics')]
    for folder,key in sources:
        out=ROOT/'results/short_screening'/folder
        summary=out/'summary.json';digest=sha256(summary)
        metrics=json.loads(summary.read_text())[key]
        names=list(metrics);y=np.arange(len(names))
        fig,ax=plt.subplots(figsize=(9,3.5))
        for offset,field,label in [(-.18,'accuracy','Accuracy'),(.18,'macro_f1','Macro-F1')]:
            bars=ax.barh(y+offset,[metrics[n][field] for n in names],.35,label=label)
            ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=9)
        ax.set(yticks=y,yticklabels=names,xlim=(0,1),xlabel='Train-only lesion-group out-of-fold score',
               title=folder.split('_')[0].upper()+': CPU feature heads | 7,009 training OOF images')
        ax.invert_yaxis();ax.legend(loc='upper left',bbox_to_anchor=(1.02,1));ax.grid(axis='x',alpha=.2)
        save(fig,out/'comparison_figures','model_comparison')
        for name,m in metrics.items():
            fig,ax=plt.subplots(figsize=(7,3))
            counts=np.asarray(m['confusion_matrix']).sum(1)
            ax.bar(CLASSES,counts,color='#38678c')
            ax.set(ylabel='Train-fold heldout images',title=f'{name} | training-only group-OOF class support')
            save(fig,out/name/'figures','class_support')
        assert sha256(summary)==digest
        write_json(out/'plot_label_amendment.json',dict(change='Label training-OOF comparison/class support explicitly; shared generic plotter uses validation wording',
                    metric_or_prediction_change=False,summary_sha256_unchanged=digest,
                    original_common_plotter_unchanged=True))


if __name__=='__main__':main()
