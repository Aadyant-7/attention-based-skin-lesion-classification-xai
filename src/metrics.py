"""Validation metrics; test evaluation is a separate explicit command."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from .data import CLASSES


def classification_metrics(y_true, y_pred):
    labels = list(range(len(CLASSES)))
    output = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        output.update({f"{avg}_precision": float(p), f"{avg}_recall": float(r), f"{avg}_f1": float(f)})
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    output["per_class"] = {name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])} for i, name in enumerate(CLASSES)}
    output["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return output
