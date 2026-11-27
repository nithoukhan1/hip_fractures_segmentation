#!/usr/bin/env python3
"""Per-class breakdown of filed runs, and the comparison with the base article.

For each run it reports, at the run's OOF (Youden) fracture threshold:
  * every headline metric as mean ± SD over the 5 held-out folds (OOF) and over the 5 fold models on the test set,
    next to the pooled OOF value and the 5-model test ensemble;
  * recall, precision, F1 and Dice for each of the 6 classes (Dice with missed fractures counted as 0, and over
    detected fractures only), the detection rate, and the predicted / true mask area;
  * errors split like the article (detection, location, grade), location accuracy and grade accuracy;
  * segmentation diagnostics: Dice under several definitions (the article does not state its own), HD95, NSD;
  * the fold model the article's selection rule would pick (best accuracy on its own held-out fold, i.e. no test
    data used), and its test results;
  * side by side with the article: its Table 2 entry for the same model size and image size (mean ± SD of 6 runs)
    and its final model (Fig. 3 confusion matrix), from src/femurseg/reference.py.

Reads only filed run folders (per-case CSVs, metrics.json, config.json), so it never changes a run. Writes
results/analysis/<name>/breakdown.json and breakdown.md. Needs numpy and scipy (as src/femurseg/metrics.py).

  python scripts/run_breakdown.py S01_yolov8s_seg_1024 S02_yolov8s_seg_640 --name C01_yolo_1024_vs_640
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from femurseg import metrics as M  # noqa: E402
from femurseg import reference as REF  # noqa: E402
from femurseg.classes import BY_CLS6, CLASSES  # noqa: E402

CODES = [c.code for c in CLASSES]
NAMES = {c.code: c.name for c in CLASSES}
DEFAULT_TAU = 0.25  # Ultralytics' default confidence; the union masks in the per-case files use it
SIZES = {"n": "nano", "s": "small", "m": "medium"}


# ---------------------------------------------------------------------------------------------- reading


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read_cases(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    rows = []
    for r in raw:
        d = {k: _num(v) for k, v in r.items() if k not in ("file", "split", "model")}
        d.update(file=r["file"], model=str(r["model"]), cls6_true=int(d["cls6_true"]), fold=int(d["fold"]),
                 top_cls6=int(d["top_cls6"]))
        rows.append(d)
    return rows


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r.get(key, float("nan")) for r in rows], dtype=float)


def _mean(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if x.size else float("nan")


def _median(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.median(x)) if x.size else float("nan")


# ---------------------------------------------------------------------------------------------- evaluation


def evaluate(rows: list[dict], tau: float) -> dict:
    """Headline, per-class and segmentation numbers for one set of per-case rows at fracture threshold tau.

    Decision rule of scripts/train_yolo_seg.py: a femur is called fractured when its top instance has confidence
    >= tau, and gets the class and mask of that instance."""
    y = np.array([r["cls6_true"] for r in rows])
    called = col(rows, "top_conf") >= tau
    pred = np.where(called, [r["top_cls6"] for r in rows], 0)
    mc = M.multiclass_metrics(y, pred)
    b = M.binary_metrics(y > 0, pred > 0)
    err = M.error_breakdown(y, pred)
    out = {"n": len(rows), "auc_fracture": M.roc_auc(y > 0, col(rows, "score")),
           "fracture_sensitivity": b["sensitivity"], "fracture_specificity": b["specificity"],
           "accuracy_6class": mc["accuracy"], "macro_f1": mc["macro_f1"],
           "balanced_accuracy": mc["balanced_accuracy"], "kappa": mc["kappa"],
           "errors": err, "n_errors": int(sum(err.values()))}

    fx = y > 0
    det = fx & called
    both = det & (pred > 0)
    loc_ok = np.array([BY_CLS6[t].location == BY_CLS6[p].location for t, p in zip(y[both], pred[both])], bool)
    out["location_accuracy"] = _mean(loc_ok)  # fractured femurs called fractured: right location?
    t_ok, p_ok = y[both][loc_ok], pred[both][loc_ok]
    graded = np.isin(t_ok, [1, 2, 3, 4])  # neck and pertrochanteric have grades; subtrochanteric has none
    out["grade_accuracy_given_location"] = _mean(t_ok[graded] == p_ok[graded])

    dice = col(rows, "dice_top")
    dice_fx = np.where(det, dice, 0.0)  # missed fracture = 0
    a_gt, a_top = col(rows, "area_gt"), col(rows, "area_top")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = a_top / a_gt
    out["per_class"] = {}
    for c, code in enumerate(CODES):
        pc = mc["per_class"][code]
        e = {"n": pc["support"], "recall": pc["recall"], "precision": pc["precision"], "f1": pc["f1"]}
        if c > 0:
            m = y == c
            e.update(dice=_mean(dice_fx[m]), dice_detected=_mean(dice[m & called]), detected=_mean(called[m]),
                     area_ratio_median=_median(ratio[m & called]), hd95_detected_median=_median(col(rows, "hd95_top")[m & called]))
        out["per_class"][code] = e
    out["confusion_matrix"] = mc["confusion_matrix"]

    inter = np.where(det, np.nan_to_num(dice) * (a_top + a_gt) / 2, 0.0)  # intersection recovered from Dice and areas
    has_ratio = bool(det.any()) and not np.isnan(ratio[det]).all()  # False for image classifiers (no masks)
    nf_score = np.where(called[~fx], 0.0, 1.0)  # no-fracture femur: 1 if nothing is called, else 0
    out["seg"] = {
        "dice_detected_mean": _mean(dice[det]),
        "dice_detected_sd": float(np.std(dice[det], ddof=1)) if det.sum() > 1 else float("nan"),
        "dice_fractured_mean": _mean(dice_fx[fx]),
        "dice_all_femurs_mean": _mean(np.r_[dice_fx[fx], nf_score]),
        "dice_global": float(2 * inter[fx].sum() / (np.where(det, a_top, 0)[fx].sum() + a_gt[fx].sum())),
        "area_ratio_median": _median(ratio[det]),
        "area_ratio_q25": float(np.nanpercentile(ratio[det], 25)) if has_ratio else float("nan"),
        "area_ratio_q75": float(np.nanpercentile(ratio[det], 75)) if has_ratio else float("nan"),
        "hd95_detected_median": _median(col(rows, "hd95_top")[det]),
        "nsd2_detected_mean": _mean(col(rows, "nsd2_top")[det]),
        "n_fractured": int(fx.sum()), "n_detected": int(det.sum()),
    }
    union = col(rows, "dice_union025")
    det25 = fx & (col(rows, "top_conf") >= DEFAULT_TAU)
    if det25.any() and not np.isnan(union[det25]).all():  # single models only (the ensemble has no union masks)
        out["seg"]["dice_detected_top_at025_mean"] = _mean(dice[det25])
        out["seg"]["dice_detected_union_at025_mean"] = _mean(union[det25])
    return out


def flatten(block: dict, prefix: str = "") -> dict[str, float]:
    """Numbers of an evaluate() block as flat keys (per_class.2AB.recall, seg.dice_global, errors.grade, ...)."""
    flat = {}
    for k, v in block.items():
        if k == "confusion_matrix":
            continue
        if isinstance(v, dict):
            flat.update(flatten(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[f"{prefix}{k}"] = float(v)
    return flat


def mean_sd(blocks: list[dict]) -> dict[str, dict[str, float]]:
    flats = [flatten(b) for b in blocks]
    keys = [k for k in flats[0] if all(k in f for f in flats)]
    out = {}
    for k in keys:
        v = np.array([f[k] for f in flats], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = {"mean": float(v.mean()) if v.size else float("nan"),
                  "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"), "n": int(v.size)}
    return out


def analyse_run(results: Path, run: str) -> dict:
    d = results / run
    met = json.loads((d / "metrics.json").read_text())
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    tau = float(met["thresholds"]["oof_youden"])
    oof = read_cases(d / "per_case_oof.csv")
    test = read_cases(d / "per_case_test.csv")
    ens = read_cases(d / "per_case_test_ensemble.csv")
    folds = sorted({r["fold"] for r in oof})
    models = sorted({r["model"] for r in test}, key=lambda m: int(m))
    per_fold = {k: evaluate([r for r in oof if r["fold"] == k], tau) for k in folds}
    per_model = {m: evaluate([r for r in test if r["model"] == m], tau) for m in models}
    # the article's rule: of the trained models, keep the one with the best validation accuracy (no test data used)
    best = max(folds, key=lambda k: (per_fold[k]["accuracy_6class"], -k))
    out = {
        "run": run, "threshold": tau, "model": cfg.get("model", ""), "imgsz": cfg.get("imgsz"),
        "net": cfg.get("net", ""), "size": cfg.get("size", ""),  # scripts/train_net.py runs (S04-S10)
        "oof": evaluate(oof, tau),
        "oof_folds": {str(k): v for k, v in per_fold.items()},
        "oof_folds_mean_sd": mean_sd(list(per_fold.values())),
        "test_models": per_model,
        "test_models_mean_sd": mean_sd(list(per_model.values())),
        "selected_model": {"fold": best, "rule": "highest 6-class accuracy on its own held-out fold",
                           "held_out_accuracy": per_fold[best]["accuracy_6class"], "test": per_model[str(best)]},
        "test_ensemble": evaluate(ens, tau),
        "oof_minus_test_models": {},
    }
    # check: the pooled numbers must equal the ones the training script wrote
    for scope in ("oof", "test_ensemble"):
        pairs = [(key, out[scope][key], met[scope][key]) for key in ("accuracy_6class", "macro_f1", "auc_fracture")]
        pairs.append(("dice_fractured_mean", out[scope]["seg"]["dice_fractured_mean"], met[scope]["dice_fractured_mean"]))
        for key, a, b in pairs:  # NaN on both sides (no masks: image classifiers) counts as equal
            assert (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9, f"{run} {scope} {key}: {a} vs metrics.json {b}"
    # mask areas by class: true masks of the training femurs (OOF rows) and of the test femurs, and the predicted
    # masks of fractures that were called (OOF; test pooled over the 5 fold models)
    first = [r for r in test if r["model"] == models[0]]
    called = lambda r: r["top_conf"] >= tau  # noqa: E731
    out["areas"] = {}
    for c, code in enumerate(CODES[1:], start=1):
        gt_tr = [r["area_gt"] for r in oof if r["cls6_true"] == c]
        gt_te = [r["area_gt"] for r in first if r["cls6_true"] == c]
        p_oof = [r["area_top"] for r in oof if r["cls6_true"] == c and called(r)]
        p_te = [r["area_top"] for r in test if r["cls6_true"] == c and called(r)]
        out["areas"][code] = {"true_train_median": _median(gt_tr), "true_test_median": _median(gt_te),
                              "pred_oof_median": _median(p_oof), "pred_test_median": _median(p_te),
                              "n_train": len(gt_tr), "n_test": len(gt_te)}
    ratios = [r["area_top"] / r["area_gt"] for r in test if r["cls6_true"] > 0 and called(r) and r["area_gt"] > 0]
    q = np.percentile(ratios, [25, 75]) if ratios else [float("nan")] * 2
    out["test_models_pooled_area"] = {"median": _median(ratios), "q25": float(q[0]), "q75": float(q[1]), "n": len(ratios)}
    size = SIZES.get(Path(str(cfg.get("model", ""))).name.split("-")[0][-1:], "")
    out["paper_setting"] = {"size": size, "imgsz": cfg.get("imgsz"),
                            "grid": REF.GRID.get((size, cfg.get("imgsz"))) if size else None}
    if out["net"]:  # not a YOLO run: compare with the article's final model, and with its own run of the same
        # classifier where Table 3 has one (DenseNet121, EfficientNet-B4)
        arch = out["net"].split(":")[1].split(".")[0] if out["net"].count(":") >= 1 else ""
        name = REF.TIMM_TO_TABLE3.get(arch) if out["net"].startswith("cls:") else None
        out["paper_setting"].update(final_model=True, table3=name, table3_row=REF.TABLE3.get(name) if name else None)
    return out


def paper_final() -> dict:
    cm = np.asarray(REF.FINAL_CONFUSION)
    y = np.concatenate([[t] * int(cm[t].sum()) for t in range(6)])
    p = np.concatenate([np.repeat(np.arange(6), cm[t]) for t in range(6)])
    mc = M.multiclass_metrics(y, p)
    return {"accuracy_6class": mc["accuracy"], "macro_f1": mc["macro_f1"], "balanced_accuracy": mc["balanced_accuracy"],
            "kappa": mc["kappa"], "per_class": mc["per_class"], "errors": M.error_breakdown(y, p),
            "confusion_matrix": cm.tolist(), **{k: v for k, v in REF.FINAL.items() if k not in ("errors",)}}


# ---------------------------------------------------------------------------------------------- report


def f3(x, nd=3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def pct(x, nd=1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}"


def pm(ms: dict, key: str, nd=3, scale=1.0) -> str:
    e = ms.get(key)
    if not e or math.isnan(e["mean"]):
        return "n/a"
    return f"{scale * e['mean']:.{nd}f} ± {scale * e['sd']:.{nd}f}"


def get(block: dict, key: str):
    v = block
    for part in key.split("."):
        v = v.get(part) if isinstance(v, dict) else None
        if v is None:
            return float("nan")
    return v


def cm_table(cm) -> list[str]:
    head = "| True \\ predicted | " + " | ".join(CODES) + " |"
    lines = [head, "|---" * (len(CODES) + 1) + "|"]
    for t, row in enumerate(cm):
        lines.append(f"| {CODES[t]} | " + " | ".join(str(int(v)) for v in row) + " |")
    return lines


HEADLINE = [("auc_fracture", "Fracture AUC", 3, 1.0), ("fracture_sensitivity", "Fracture sensitivity", 3, 1.0),
            ("fracture_specificity", "Fracture specificity", 3, 1.0), ("accuracy_6class", "6-class accuracy (%)", 1, 100.0),
            ("macro_f1", "Macro-F1", 3, 1.0), ("balanced_accuracy", "Balanced accuracy", 3, 1.0),
            ("kappa", "Cohen's kappa", 3, 1.0), ("location_accuracy", "Location accuracy (fractures called)", 3, 1.0),
            ("grade_accuracy_given_location", "Grade accuracy, location right", 3, 1.0),
            ("seg.dice_fractured_mean", "Dice, fractured femurs (missed = 0)", 3, 1.0),
            ("seg.dice_detected_mean", "Dice, detected fractures only", 3, 1.0),
            ("seg.area_ratio_median", "Predicted / true area, median", 2, 1.0),
            ("seg.hd95_detected_median", "HD95 (px), median, detected", 1, 1.0),
            ("errors.detection", "Errors: detection", 1, 1.0), ("errors.location", "Errors: location", 1, 1.0),
            ("errors.grade", "Errors: grade", 1, 1.0), ("n_errors", "Errors: total", 1, 1.0)]


def run_section(r: dict, final: dict) -> list[str]:
    o, of, tm, en, sel = r["oof"], r["oof_folds_mean_sd"], r["test_models_mean_sd"], r["test_ensemble"], r["selected_model"]
    nf, nt = o["n"], en["n"]
    what = (f"Network `{r['net']}` at {r['size']} px" if r.get("net")
            else f"Model `{Path(str(r['model'])).name}` at {r['imgsz']} px")
    L = [f"## {r['run']}", "",
         f"{what}; fracture threshold {r['threshold']:.4f} (Youden on the "
         f"pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own "
         f"held-out fold, {pct(sel['held_out_accuracy'])}%) is fold {sel['fold']}.", "",
         "### Headline metrics", "",
         f"| Metric | OOF pooled (n={nf:,}) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n={nt}) |",
         "|---|---|---|---|---|---|"]
    for key, label, nd, sc in HEADLINE:
        fmt = (lambda v: f"{sc * v:.{nd}f}" if not (isinstance(v, float) and math.isnan(v)) else "n/a")
        L.append(f"| {label} | {fmt(get(o, key))} | {pm(of, key, nd, sc)} | {pm(tm, key, nd, sc)} | "
                 f"{fmt(get(sel['test'], key))} | {fmt(get(en, key))} |")
    L += ["", "Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture "
          "found but wrong location; grade = right location, wrong grade.", "",
          "### Recall per class", "",
          "| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |",
          "|---|---|---|---|---|---|---|"]
    for code in CODES:
        k = f"per_class.{code}.recall"
        L.append(f"| {code} {NAMES[code]} | {o['per_class'][code]['n']} / {en['per_class'][code]['n']} | "
                 f"{f3(get(o, k))} | {pm(of, k)} | {pm(tm, k)} | {f3(get(sel['test'], k))} | {f3(get(en, k))} |")
    L += ["", "### Dice per class (fractured femurs; a missed fracture counts as 0)", "",
          "| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |",
          "|---|---|---|---|---|---|---|---|"]
    for code in CODES[1:]:
        k = f"per_class.{code}."
        L.append(f"| {code} {NAMES[code]} | {f3(get(o, k + 'dice'))} | {pm(of, k + 'dice')} | {pm(tm, k + 'dice')} | "
                 f"{f3(get(en, k + 'dice'))} | {f3(get(o, k + 'detected'))} / {pm(tm, k + 'detected')} | "
                 f"{f3(get(o, k + 'dice_detected'))} / {pm(tm, k + 'dice_detected')} | "
                 f"{f3(get(o, k + 'area_ratio_median'), 2)} / {pm(tm, k + 'area_ratio_median', 2)} |")
    L += ["", "### Confusion matrices (rows = true class)", "", f"OOF pooled ({nf:,} femurs):", ""] + cm_table(o["confusion_matrix"])
    L += ["", f"Test, 5-model ensemble ({nt} femurs):", ""] + cm_table(en["confusion_matrix"]) + [""]
    return L


def paper_section(runs: list[dict], final: dict) -> list[str]:
    L = ["## Comparison with the base article", "",
         f"Source: {REF.CITATION}. Table 2 gives mean ± SD of six training runs per setting on the same 210 test "
         "femurs; the final model is the run with the best validation accuracy (Table 3, confusion matrix in Fig. 3). "
         "Our fold models are trained on 4/5 of the 1,043 training femurs (about 834), the article's on all 1,043 "
         "plus a 201-femur validation set; our counterpart of its selected model is the fold model with the best "
         "accuracy on its own held-out fold.", ""]
    for r in runs:
        g, tm, sel, en = (r["paper_setting"]["grid"] or {}), r["test_models_mean_sd"], r["selected_model"]["test"], r["test_ensemble"]
        s = r["paper_setting"]
        title = (f"### {r['run']} vs the article's final model (YOLOv8s-seg @1024)" if s.get("final_model")
                 else f"### {r['run']} vs the article's YOLOv8-seg {s['size']} @ {s['imgsz']}")
        L += [title, "",
              "| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |",
              "|---|---|---|---|---|---|"]
        is_final = (s["size"], s["imgsz"]) == ("small", 1024) or bool(s.get("final_model"))
        fin = (lambda v, nd=3, sc=1.0: f"{sc * v:.{nd}f}") if is_final else (lambda v, nd=3, sc=1.0: "(other setting)")
        gm = (lambda k, nd=3, sc=1.0: f"{sc * g[k][0]:.{nd}f} ± {sc * g[k][1]:.{nd}f}" if k in g else "n/a")
        L.append(f"| Fracture AUC | {gm('auc')} | {pm(tm, 'auc_fracture')} | {fin(final['auc'])} | "
                 f"{f3(sel['auc_fracture'])} | {f3(en['auc_fracture'])} |")
        L.append(f"| 6-class accuracy (%) | {gm('accuracy_6class', 1, 100)} | {pm(tm, 'accuracy_6class', 1, 100)} | "
                 f"{fin(final['accuracy_6class'], 1, 100)} | {pct(sel['accuracy_6class'])} | {pct(en['accuracy_6class'])} |")
        L.append(f"| Fracture sensitivity | n/a | {pm(tm, 'fracture_sensitivity')} | {fin(final['fracture_sensitivity'])} | "
                 f"{f3(sel['fracture_sensitivity'])} | {f3(en['fracture_sensitivity'])} |")
        L.append(f"| Fracture specificity | n/a | {pm(tm, 'fracture_specificity')} | {fin(final['fracture_specificity'])} | "
                 f"{f3(sel['fracture_specificity'])} | {f3(en['fracture_specificity'])} |")
        L.append(f"| Macro-F1 (article: from Fig. 3) | n/a | {pm(tm, 'macro_f1')} | {fin(final['macro_f1'])} | "
                 f"{f3(sel['macro_f1'])} | {f3(en['macro_f1'])} |")
        L.append(f"| Dice (article: definition not stated; ours: detected fractures only) | {gm('dice')} | "
                 f"{pm(tm, 'seg.dice_detected_mean')} | {fin(final['dice'], 2)} | {f3(sel['seg']['dice_detected_mean'])} | "
                 f"{f3(en['seg']['dice_detected_mean'])} |")
        for code in CODES:
            k = f"per_class.{code}.recall"
            L.append(f"| Recall {code} {NAMES[code]} | n/a | {pm(tm, k)} | {fin(final['per_class'][code]['recall'])} | "
                     f"{f3(get(sel, k))} | {f3(get(en, k))} |")
        for kind in ("detection", "location", "grade"):
            L.append(f"| Errors: {kind} | n/a | {pm(tm, 'errors.' + kind, 1)} | {fin(final['errors'][kind], 0)} | "
                     f"{get(sel, 'errors.' + kind)} | {get(en, 'errors.' + kind)} |")
        L.append(f"| Errors: total | n/a | {pm(tm, 'n_errors', 1)} | {fin(sum(final['errors'].values()), 0)} | "
                 f"{sel['n_errors']} | {en['n_errors']} |")
        L.append("")
        t3 = s.get("table3_row")
        if t3:
            L += [f"The article's own {s['table3']} (Table 3, one run on the same test set): fracture AUC "
                  f"{t3['auc']:.3f} ({t3['auc_ci'][0]:.3f}–{t3['auc_ci'][1]:.3f}), 6-class accuracy "
                  f"{100 * t3['accuracy_6class']:.1f}% ({100 * t3['accuracy_ci'][0]:.1f}–{100 * t3['accuracy_ci'][1]:.1f}).", ""]
    L += ["Article's final model, confusion matrix (Fig. 3):", ""] + cm_table(final["confusion_matrix"]) + [""]
    return L


def dice_section(runs: list[dict], final: dict) -> list[str]:
    L = ["## Dice definitions", "",
         "The article reports \"the average Dice between annotated fracture regions and the regions automatically "
         f"detected\": {final['dice']:.2f} ({final['dice_ci'][0]:.2f}–{final['dice_ci'][1]:.2f}) for its final model. "
         "That interval is about mean ± 1.96 SD of single femurs, so it probably averages detected fractures only. "
         "The same predictions give quite different numbers under other common definitions (test set, mean ± SD of "
         "the 5 fold models; the ensemble has no union masks):", "",
         "| Definition | " + " | ".join(r["run"] for r in runs) + " |", "|---" * (len(runs) + 1) + "|"]
    rows = [("Detected fractures only, top instance (our main definition)", "seg.dice_detected_mean"),
            ("Same at Ultralytics' default threshold 0.25", "seg.dice_detected_top_at025_mean"),
            ("Same at 0.25, union of all instances ≥ 0.25", "seg.dice_detected_union_at025_mean"),
            ("All fractured femurs, missed = 0", "seg.dice_fractured_mean"),
            ("All 210 femurs, no-fracture femur = 1 if nothing called else 0", "seg.dice_all_femurs_mean"),
            ("Global (pooled pixels over fractured femurs)", "seg.dice_global")]
    for label, key in rows:
        L.append(f"| {label} | " + " | ".join(pm(r["test_models_mean_sd"], key) for r in runs) + " |")
    def spread(seg: dict) -> str:
        m, sd = seg["dice_detected_mean"], seg["dice_detected_sd"]
        return "n/a" if math.isnan(m) or math.isnan(sd) else f"{m - 1.96 * sd:.2f}–{min(1.0, m + 1.96 * sd):.2f} (ensemble)"

    L.append("| Per-femur spread, detected only: mean ± 1.96 SD (compare the article's 0.56–0.98) | " + " | ".join(
        spread(r["test_ensemble"]["seg"]) for r in runs) + " |")
    L.append("")
    L += ["## OOF versus test segmentation", "",
          "| Run | Dice detected, OOF pooled | Dice detected, test models | Predicted / true area, OOF (median, IQR) | Predicted / true area, test models (median, IQR, pooled over models) |",
          "|---|---|---|---|---|"]
    for r in runs:
        o = r["oof"]["seg"]
        tq = r["test_models_pooled_area"]
        iqr = (lambda m, a, b: "n/a" if math.isnan(m) else f"{m:.2f} ({a:.2f}–{b:.2f})")
        L.append(f"| {r['run']} | {f3(o['dice_detected_mean'])} | {pm(r['test_models_mean_sd'], 'seg.dice_detected_mean')} | "
                 f"{iqr(o['area_ratio_median'], o['area_ratio_q25'], o['area_ratio_q75'])} | "
                 f"{iqr(tq['median'], tq['q25'], tq['q75'])} |")
    L += ["", "Median mask area in pixels by class: true masks of the training femurs and of the test femurs (the "
          "article says its test labels were reviewed by a committee), and the masks the models predict for called "
          "fractures (OOF; test pooled over the 5 fold models).", "",
          "| Class | True, train | True, test | Test / train | " + " | ".join(
              f"Predicted OOF, {r['run']} | Predicted test, {r['run']}" for r in runs) + " |",
          "|---" * (4 + 2 * len(runs)) + "|"]
    for code in CODES[1:]:
        a0 = runs[0]["areas"][code]
        L.append(f"| {code} {NAMES[code]} | {a0['true_train_median']:,.0f} | {a0['true_test_median']:,.0f} | "
                 f"{a0['true_test_median'] / a0['true_train_median']:.2f} | " + " | ".join(
                     f"{px(r['areas'][code]['pred_oof_median'])} | {px(r['areas'][code]['pred_test_median'])}"
                     for r in runs) + " |")
    L.append("")
    return L


def px(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.0f}"


def compare_section(runs: list[dict]) -> list[str]:
    if len(runs) < 2:
        return []
    L = ["## Runs side by side (test, 5 fold models mean ± SD; OOF pooled in brackets)", "",
         "| Metric | " + " | ".join(r["run"] for r in runs) + " |", "|---" * (len(runs) + 1) + "|"]
    keys = [("accuracy_6class", "6-class accuracy (%)", 1, 100.0), ("macro_f1", "Macro-F1", 3, 1.0),
            ("seg.dice_fractured_mean", "Dice, fractured (missed = 0)", 3, 1.0)]
    keys += [(f"per_class.{c}.recall", f"Recall {c}", 3, 1.0) for c in CODES]
    keys += [(f"per_class.{c}.dice", f"Dice {c}", 3, 1.0) for c in CODES[1:]]
    for key, label, nd, sc in keys:
        L.append(f"| {label} | " + " | ".join(
            f"{pm(r['test_models_mean_sd'], key, nd, sc)} ({f3(sc * get(r['oof'], key), nd)})" for r in runs) + " |")
    L += ["", "Paired tests between the runs (same femurs) are in `stats.md` of the same folder, if it was made with "
          "`scripts/run_stats.py ... --compare`.", ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="filed run names under results/")
    ap.add_argument("--name", help="output folder name under results/analysis/ (default: the run names joined)")
    ap.add_argument("--results", default=str(REPO / "results"))
    a = ap.parse_args()
    results = Path(a.results)
    runs = []
    for run in a.runs:
        runs.append(analyse_run(results, run))
    final = paper_final()
    out_dir = results / "analysis" / (a.name or "_".join(a.runs))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "breakdown.json").write_text(json.dumps({"runs": runs, "paper_final": final}, indent=1, default=float))
    L = [f"# Breakdown: {', '.join(a.runs)}", "",
         "Made by `scripts/run_breakdown.py` from the filed run folders. Decisions use each run's OOF threshold. "
         "Mean ± SD is over the 5 held-out folds (OOF) or the 5 fold models (test); SD uses n − 1.", ""]
    for r in runs:
        L += run_section(r, final)
    L += compare_section(runs) + paper_section(runs, final) + dice_section(runs, final)
    (out_dir / "breakdown.md").write_text("\n".join(L))
    print(f"written: {out_dir / 'breakdown.json'} and breakdown.md")


if __name__ == "__main__":
    main()
