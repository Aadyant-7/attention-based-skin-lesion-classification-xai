"""Paper figures from saved metrics only: PNG (300 dpi) and vector PDF."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .common import CLASSES, write_csv


def save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / (name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def validate_metrics(m):
    for key in ('accuracy', 'macro_f1', 'macro_precision', 'macro_recall'):
        value = m.get(key)
        if value is None or not np.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f'Invalid or missing {key}')
    cm = np.asarray(m['confusion_matrix'])
    if cm.shape != (7, 7) or not np.isfinite(cm).all() or (cm < 0).any() or not np.equal(cm, np.floor(cm)).all():
        raise ValueError('Confusion matrix must have 7x7 nonnegative integer counts')
    if cm.sum() == 0:
        raise ValueError('Empty confusion matrix')
    if not np.isclose(np.trace(cm) / cm.sum(), float(m['accuracy']), atol=1e-8):
        raise ValueError('Accuracy disagrees with confusion matrix')
    if set(m['per_class']) != set(CLASSES):
        raise ValueError('Class order/membership not established')
    support = np.asarray([m['per_class'][c]['support'] for c in CLASSES])
    if not np.array_equal(cm.sum(axis=1), support):
        raise ValueError('Confusion matrix and per-class supports disagree')
    # Recompute balanced metrics; do not trust an arbitrary reported summary.
    diag = np.diag(cm)
    p = np.divide(diag, cm.sum(axis=0), out=np.zeros(7, dtype=float), where=cm.sum(axis=0) > 0)
    r = np.divide(diag, support, out=np.zeros(7, dtype=float), where=support > 0)
    f = np.divide(2*p*r, p+r, out=np.zeros(7), where=(p+r) > 0)
    for key, values in (('macro_precision', p), ('macro_recall', r), ('macro_f1', f)):
        if not np.isclose(float(m[key]), values.mean(), atol=1e-8):
            raise ValueError(f'{key} disagrees with confusion matrix')
    for key, values in (('precision',p),('recall',r),('f1',f)):
        if not np.allclose([m['per_class'][c][key] for c in CLASSES],values,atol=1e-8):
            raise ValueError(f'Per-class {key} disagrees with confusion matrix')
    return cm.astype(int)


def metric_figures(metrics, out, title):
    cm = validate_metrics(metrics)
    write_csv(Path(out)/'confusion_matrix.csv',[{'true_class':c,**dict(zip(CLASSES,cm[i].tolist()))} for i,c in enumerate(CLASSES)])
    normalized=np.divide(cm,cm.sum(axis=1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(axis=1,keepdims=True)>0)
    write_csv(Path(out)/'confusion_matrix_normalized.csv',[{'true_class':c,**dict(zip(CLASSES,normalized[i].tolist()))} for i,c in enumerate(CLASSES)])
    for normalized in (False, True):
        values = np.divide(cm,cm.sum(axis=1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(axis=1,keepdims=True)>0) if normalized else cm
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized else None)
        fig.colorbar(im, ax=ax, fraction=.045)
        ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted class', ylabel='True class', title=title + '\n' + ('Row-normalized confusion matrix' if normalized else 'Confusion matrix'))
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f'{values[i,j]:.2f}' if normalized else str(cm[i,j]), ha='center', va='center',
                        fontsize=8, color='white' if values[i,j] > values.max()*.55 else 'black')
        save(fig, out, 'confusion_matrix_normalized' if normalized else 'confusion_matrix')
    rows = [{'class': c, **metrics['per_class'][c]} for c in CLASSES]
    write_csv(Path(out) / 'per_class_metrics.csv', rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(7)
    for i, key in enumerate(('precision', 'recall', 'f1')):
        ax.bar(x + (i-1)*.25, [metrics['per_class'][c][key] for c in CLASSES], .25, label=key)
    ax.set(xticks=x, xticklabels=CLASSES, ylim=(0, 1), ylabel='Score', title=title + '\nPer-class performance')
    ax.legend(ncol=3)
    save(fig, out, 'per_class_metrics')
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(CLASSES, cm.sum(axis=1), color='#38678c')
    ax.set(ylabel='Validation images', title=title + '\nClass support')
    save(fig, out, 'class_support')


def training_figures(history, out, title, selection_metric='macro_f1'):
    if history is None or history.empty:
        return
    for name, columns, ylabel in (
        ('loss_curves', ('train_loss', 'val_loss'), 'Loss'),
        ('accuracy_curves', ('train_accuracy', 'val_accuracy'), 'Accuracy'),
        ('macro_f1_curve', ('val_macro_f1',), 'Validation macro F1')):
        available = [c for c in columns if c in history and history[c].notna().any()]
        if not available:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in available:
            ax.plot(history['epoch'], history[c], marker='.', label=c.replace('_', ' '))
        column='val_'+selection_metric
        if column in history and history[column].notna().any():
            selected=history.loc[history[column].idxmax(),'epoch']
            label='Macro-F1' if selection_metric=='macro_f1' else 'Accuracy'
            ax.axvline(selected,linestyle='--',color='#666666',alpha=.65,label=label+'-selected epoch')
        ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
        ax.legend()
        save(fig, out, name)


def comparison_figures(rows, out, title):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, max(3, len(rows)*.55)))
    y = np.arange(len(rows))
    bars=ax.barh(y-.18, [float(r['accuracy']) for r in rows], .35, label='Accuracy')
    ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=8)
    bars=ax.barh(y+.18, [float(r['macro_f1']) for r in rows], .35, label='Macro F1')
    ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=8)
    ax.set(yticks=y, yticklabels=[r['display_name'] for r in rows], xlim=(0, 1), title=title, xlabel='Validation score')
    ax.invert_yaxis()
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
    ax.grid(axis='x', alpha=.2)
    save(fig, out, 'model_comparison')
    write_csv(Path(out) / 'comparison.csv', rows)
