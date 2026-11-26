"""Per-case result files and the headline metrics computed from them, shared by every training script.

Every model writes one row per femur and model (CASE_COLS) for its held-out fold (OOF) and for the test set. The
same decision rule turns any model's row into a prediction, so all models are scored by the same code:

  * ``score``     continuous fracture score, used for the AUC and to fit the threshold (Youden on the OOF rows);
  * ``top_conf``  a femur is called fractured when top_conf >= tau (for every model so far top_conf == score);
  * ``top_cls6``  the class given to a femur that is called fractured (1..5);
  * ``conf_1..5`` per-class confidences (with 1 - score they form the 6-class probabilities used for calibration);
  * ``dice_top``, ``iou_top``, ``hd95_top``, ``nsd*_top``: the predicted mask against the released mask, for femurs
    with a fracture; a fracture that is not called counts as Dice 0 (missed). Classifiers leave these empty, and
    their Dice metrics are NaN.

Used by scripts/train_yolo_seg.py (S01-S03) and scripts/train_net.py (S04-S10). The analysis scripts
(run_stats.py, run_breakdown.py) read the same files.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from . import metrics as M
from .classes import CLASSES

NSD_TOLS = (2.0, 10.0, 20.0)  # pixels of the 1024 x 1536 image; 2 px was too strict to separate models (C01)
CASE_COLS = ["file", "split", "fold", "model", "cls6_true", "n_det", "score", "conf_1", "conf_2", "conf_3", "conf_4",
             "conf_5", "top_cls6", "top_conf", "area_gt", "area_top", "dice_top", "iou_top", "hd95_top", "nsd2_top",
             "dice_union025", "area_union025", "nsd10_top", "nsd20_top"]
_TEXT_COLS = ("file", "split", "model")


def seg_case(pred: np.ndarray, gt: np.ndarray, nd: int | None = 6) -> dict:
    """Per-case segmentation columns of one predicted mask against the released mask (fractured femurs only),
    rounded to ``nd`` decimals (None: not rounded). HD95 is never rounded."""
    s = M.seg_metrics(pred, gt, NSD_TOLS)
    r = (lambda v: round(v, nd)) if nd is not None else (lambda v: v)
    out = {"dice_top": r(s["dice"]), "iou_top": r(s["iou"]), "hd95_top": s["hd95"]}
    out.update({f"nsd{t:g}_top": r(s[f"nsd{t:g}"]) for t in NSD_TOLS})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CASE_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CASE_COLS})


def read_csv(path: Path) -> list[dict]:
    """Rows of a per-case file: numbers as floats, empty cells as None ("nan" as NaN); files written before a column
    existed simply lack it (read as None)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in CASE_COLS:
            if c in _TEXT_COLS:
                continue
            v = r.get(c, "")
            v = "" if v is None else v
            r[c] = float(v) if v not in ("", "nan") else (float("nan") if v == "nan" else None)
        r["cls6_true"], r["top_cls6"], r["fold"] = int(r["cls6_true"]), int(r["top_cls6"]), int(r["fold"])
    return rows


def _num(v) -> float:
    return float("nan") if v is None else float(v)


def evaluate(rows: list[dict], tau: float) -> dict:
    """All headline metrics for a set of per-case rows at fracture threshold ``tau``."""
    y = np.array([r["cls6_true"] for r in rows])
    score = np.array([r["score"] for r in rows])
    pred = np.array([r["top_cls6"] if r["top_conf"] >= tau else 0 for r in rows])
    out: dict = {"n": len(rows), "threshold": round(float(tau), 6), "auc_fracture": M.roc_auc(y > 0, score)}
    out.update({f"fracture_{k}": v for k, v in M.binary_metrics(y > 0, pred > 0).items()})
    mc = M.multiclass_metrics(y, pred)
    out.update(accuracy_6class=mc["accuracy"], macro_f1=mc["macro_f1"], balanced_accuracy=mc["balanced_accuracy"],
               kappa=mc["kappa"], garden12_recall=mc["per_class"]["2AB"]["recall"],
               garden12_detected=float(np.mean(pred[y == 1] > 0)) if (y == 1).any() else float("nan"),
               errors=M.error_breakdown(y, pred), per_class=mc["per_class"], confusion_matrix=mc["confusion_matrix"])
    fx = [r for r in rows if r["cls6_true"] > 0]
    det = [r for r in fx if r["top_conf"] >= tau]
    dice_fx = [_num(r.get("dice_top")) if r["top_conf"] >= tau else 0.0 for r in fx]

    def fx_mean(key: str) -> float:  # optional mask columns: absent or empty for models without masks
        return float(np.mean([_num(r.get(key)) if r["top_conf"] >= tau else 0.0 for r in fx])) if fx else float("nan")

    out.update(
        dice_fractured_mean=float(np.mean(dice_fx)) if fx else float("nan"),
        dice_fractured_median=float(np.median(dice_fx)) if fx else float("nan"),
        dice_detected_mean=float(np.mean([_num(r.get("dice_top")) for r in det])) if det else float("nan"),
        iou_fractured_mean=fx_mean("iou_top"),
        nsd2_fractured_mean=fx_mean("nsd2_top"),
        hd95_detected_median=_nanmedian([_num(r.get("hd95_top")) for r in det]),
        n_fractured=len(fx), n_fractured_missed=len(fx) - len(det),
        dice_per_class={CLASSES[c].code: float(np.mean([d for d, r in zip(dice_fx, fx) if r["cls6_true"] == c]))
                        for c in range(1, 6) if any(r["cls6_true"] == c for r in fx)},
    )
    for t in NSD_TOLS[1:]:  # newer files only (S01/S02 have NSD at 2 px only)
        key = f"nsd{t:g}_top"
        if fx and all(r.get(key) is not None for r in fx):
            out[f"nsd{t:g}_fractured_mean"] = fx_mean(key)
    probs = np.array([[1 - r["score"]] + [r[f"conf_{c}"] for c in range(1, 6)] for r in rows])
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-9, None)
    out.update(ece_derived=M.expected_calibration_error(probs, y), brier_derived=M.brier_score(probs, y))
    return out


def _nanmedian(x: list[float]) -> float:
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.median(a)) if a.size else float("nan")


def scalar_summary(ms: list[dict]) -> tuple[dict, dict]:
    """Mean and SD (ddof 1) of every scalar metric over several evaluate() results (e.g. the 5 fold models)."""
    keys = [k for k, v in ms[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    keys = [k for k in keys if all(k in m for m in ms)]

    def vals(k):
        v = np.array([m[k] for m in ms], dtype=float)
        return v[~np.isnan(v)]

    mean = {k: float(vals(k).mean()) if vals(k).size else float("nan") for k in keys}
    sd = {k: float(vals(k).std(ddof=1)) if vals(k).size > 1 else float("nan") for k in keys}
    return mean, sd


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not (isinstance(v, float) and math.isnan(v))
