"""Metrics for fracture segmentation and grading (numpy and scipy only, so they run anywhere).

Segmentation (per femur, binary masks: fracture vs background):
  dice, iou, hd95 (95th-percentile symmetric surface distance, in pixels), nsd (normalised surface Dice at a
  tolerance in pixels). The images have no pixel spacing, so distances are in pixels of the 1024 x 1536 crop.
  Conventions (Metrics Reloaded style, stated in every paper):
    * both masks empty -> dice = iou = nsd = 1, hd95 = 0;
    * exactly one empty -> dice = iou = nsd = 0, hd95 = NaN (callers report it separately or apply a penalty).

Classification:
  roc_auc (Mann-Whitney, ties counted as 1/2), binary_metrics, multiclass_metrics (accuracy, macro-F1,
  balanced accuracy, per-class precision/recall/F1, Cohen's kappa, confusion matrix), expected calibration
  error, Brier score, Youden threshold, and the error breakdown used by the base article
  (detection / location / grade errors).
"""
from __future__ import annotations

import math

import numpy as np

from .classes import BY_CLS6

# --------------------------------------------------------------------------------------------- segmentation


def _as_bool(m: np.ndarray) -> np.ndarray:
    return np.asarray(m) > 0


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _as_bool(pred), _as_bool(gt)
    s = int(p.sum()) + int(g.sum())
    return 1.0 if s == 0 else 2.0 * int((p & g).sum()) / s


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _as_bool(pred), _as_bool(gt)
    u = int((p | g).sum())
    return 1.0 if u == 0 else int((p & g).sum()) / u


def _surface(m: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary mask (pixels inside the mask with at least one 4-neighbour outside)."""
    from scipy.ndimage import binary_erosion

    return m & ~binary_erosion(m, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool), border_value=0)


def _surface_distances(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distances from each surface pixel of pred to gt's surface, and vice versa (both masks non-empty)."""
    from scipy.ndimage import distance_transform_edt

    ys, xs = np.nonzero(pred | gt)
    y0, y1, x0, x1 = ys.min(), ys.max() + 2, xs.min(), xs.max() + 2  # crop: exact, both surfaces are inside
    y0, x0 = max(y0 - 1, 0), max(x0 - 1, 0)
    p, g = pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]
    sp, sg = _surface(p), _surface(g)
    dt_g = distance_transform_edt(~sg)  # distance of every pixel to the nearest gt surface pixel
    dt_p = distance_transform_edt(~sp)
    return dt_g[sp], dt_p[sg]


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _as_bool(pred), _as_bool(gt)
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float("nan")
    a, b = _surface_distances(p, g)
    return float(np.percentile(np.concatenate([a, b]), 95))


def nsd(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0) -> float:
    """Normalised surface Dice: share of both surfaces lying within ``tol`` pixels of the other surface."""
    p, g = _as_bool(pred), _as_bool(gt)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    a, b = _surface_distances(p, g)
    return float(((a <= tol).sum() + (b <= tol).sum()) / (len(a) + len(b)))


def seg_metrics(pred: np.ndarray, gt: np.ndarray, nsd_tol: float | tuple[float, ...] = 2.0) -> dict[str, float]:
    """Dice, IoU, HD95, NSD at one or several tolerances (keys nsd2, nsd10, ...) and both areas.

    The surface distances are computed once and shared by HD95 and every NSD tolerance; the values are the same as
    those of hd95() and nsd()."""
    p, g = _as_bool(pred), _as_bool(gt)
    tols = (float(nsd_tol),) if np.isscalar(nsd_tol) else tuple(float(t) for t in nsd_tol)
    if not p.any() and not g.any():
        hd, ns = 0.0, {t: 1.0 for t in tols}
    elif not p.any() or not g.any():
        hd, ns = float("nan"), {t: 0.0 for t in tols}
    else:
        a, b = _surface_distances(p, g)
        hd = float(np.percentile(np.concatenate([a, b]), 95))
        ns = {t: float(((a <= t).sum() + (b <= t).sum()) / (len(a) + len(b))) for t in tols}
    out = {"dice": dice(p, g), "iou": iou(p, g), "hd95": hd}
    out.update({f"nsd{t:g}": v for t, v in ns.items()})
    out.update(area_pred=int(p.sum()), area_gt=int(g.sum()))
    return out


# ------------------------------------------------------------------------------------------- classification


def roc_auc(y_true, score) -> float:
    """Area under the ROC curve = P(score of a random positive > random negative), ties count 1/2."""
    from scipy.stats import rankdata

    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(s)  # average ranks for ties
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def binary_metrics(y_true, y_pred) -> dict[str, float]:
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    tp, tn = int((y & p).sum()), int((~y & ~p).sum())
    fp, fn = int((~y & p).sum()), int((y & ~p).sum())

    def div(a, b):
        return a / b if b else float("nan")

    return {"sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp), "ppv": div(tp, tp + fp),
            "npv": div(tn, tn + fn), "accuracy": div(tp + tn, len(y)), "f1": div(2 * tp, 2 * tp + fp + fn),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def confusion_matrix(y_true, y_pred, n_classes: int = 6) -> np.ndarray:
    t, p = np.asarray(y_true, int), np.asarray(y_pred, int)
    return np.bincount(t * n_classes + p, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    # rows = true class, columns = predicted class


def multiclass_metrics(y_true, y_pred, n_classes: int = 6) -> dict:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    n = cm.sum()
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, 0.0)  # sklearn convention: 0 when never predicted
        f1 = np.where(precision + np.nan_to_num(recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    present = support > 0
    po = tp.sum() / n if n else float("nan")
    pe = float((support * predicted).sum() / n**2) if n else float("nan")
    return {
        "accuracy": float(po),
        "macro_f1": float(np.mean(f1[present])),
        "balanced_accuracy": float(np.nanmean(recall[present])),
        "kappa": float((po - pe) / (1 - pe)) if pe < 1 else float("nan"),
        "per_class": {BY_CLS6[c].code: {"precision": float(precision[c]), "recall": float(recall[c]),
                                         "f1": float(f1[c]), "support": int(support[c])}
                      for c in range(n_classes)},
        "confusion_matrix": cm.tolist(),
    }


def error_breakdown(y_true, y_pred) -> dict[str, int]:
    """Split 6-class errors like the base article: detection (fracture vs none), location, grade."""
    out = {"detection": 0, "location": 0, "grade": 0}
    for t, p in zip(np.asarray(y_true, int), np.asarray(y_pred, int)):
        if t == p:
            continue
        if t == 0 or p == 0:
            out["detection"] += 1
        elif BY_CLS6[t].location != BY_CLS6[p].location:
            out["location"] += 1
        else:
            out["grade"] += 1
    return out


def expected_calibration_error(probs: np.ndarray, y_true, n_bins: int = 15) -> float:
    """ECE of the top-class probability with equal-width bins (Guo et al. 2017)."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y_true, int)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(probs: np.ndarray, y_true) -> float:
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(probs.shape[1])[np.asarray(y_true, int)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def youden_threshold(y_true, score) -> float:
    """Observed score ``t`` that maximises sensitivity + specificity - 1 for the rule ``score >= t``.

    Ties go to the LOWER threshold (the higher sensitivity): missing a fracture costs more than a false alarm.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=float)
    n1, n0 = max(int(y.sum()), 1), max(int((~y).sum()), 1)
    best_t, best_j = 0.5, -math.inf
    for t in np.unique(s):  # ascending, so a strict ">" keeps the lowest threshold among ties
        p = s >= t
        j = (p & y).sum() / n1 + (~p & ~y).sum() / n0 - 1
        if j > best_j + 1e-12:
            best_t, best_j = float(t), j
    return best_t
