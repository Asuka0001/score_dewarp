"""Supervision-aware auxiliary metrics; absent classes/labels are undefined."""
import numpy as np


def classification_metrics(truth, predicted):
    truth, predicted = np.asarray(truth, dtype=np.int64), np.asarray(predicted, dtype=np.int64)
    if truth.ndim != 1 or predicted.shape != truth.shape:
        raise ValueError("Classification inputs must be matching vectors")
    if ((truth < 0) | (truth > 2)).any() or ((predicted < 0) | (predicted > 3)).any():
        raise ValueError("Kinds must be 0..2; prediction 3 denotes unknown")
    confusion = np.zeros((3, 4), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    support = confusion.sum(1)
    recalls = np.divide(confusion.diagonal(), support, out=np.zeros(3, float), where=support > 0)
    return {"accuracy": float((truth == predicted).mean()) if len(truth) else None,
            "balanced_accuracy": float(recalls[support > 0].mean()) if (support > 0).any() else None,
            "recall": [float(r) if n else None for r, n in zip(recalls, support)],
            "confusion": confusion.tolist(), "samples": len(truth)}


def segmentation_metrics(confusion):
    confusion = np.asarray(confusion)
    intersection = confusion.diagonal()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.zeros(len(union), float), where=union > 0)
    return {"miou": float(iou[union > 0].mean()) if (union > 0).any() else None,
            "iou": [float(v) if n else None for v, n in zip(iou, union)],
            "confusion": confusion.tolist()}


def heatmap_metrics(tp, predicted, truth):
    result = []
    for hit, p, t in zip(tp, predicted, truth):
        result.append({"precision": float(hit / p) if p else None,
                       "recall": float(hit / t) if t else None,
                       "f1": float(2 * hit / (p + t)) if p + t else None,
                       "tp": int(hit), "predicted": int(p), "truth": int(t)})
    values = [r["f1"] for r in result if r["f1"] is not None]
    return {"channels": result, "macro_f1": float(np.mean(values)) if values else None}
