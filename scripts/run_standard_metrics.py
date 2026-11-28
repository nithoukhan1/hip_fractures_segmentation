#!/usr/bin/env python3
"""The standard metrics of Analysis/11 section 2.2 for filed runs, in one set of tables.

Classification (6 classes, and fracture yes/no):
  accuracy; precision, recall and F1 per class and as the macro average over the 6 classes; fracture sensitivity
  (= recall of "fractured") and specificity; AUC of fracture yes/no (with DeLong's interval) and the one-vs-rest
  macro AUC of the 6 classes; the confusion matrix.
Segmentation (fractured femurs):
  Dice and IoU, where a fracture the model does not call counts 0; HD95 in pixels of the 1024 x 1536 image, median
  over the fractures the model calls (HD95 is undefined for a missed fracture, so the number missed is given beside).
95% intervals: percentile bootstrap over whole x-rays (2,000 resamples) for accuracy, macro-F1 and Dice; DeLong for
the fracture AUC. Decisions use each run's own out-of-fold threshold, exactly as scripts/run_stats.py does, so the
numbers equal those of the earlier analyses.

Reads only the filed run folders and writes results/analysis/<name>/ (standard_metrics.json and .md).

    python scripts/run_standard_metrics.py                      # every model run of the paper
    python scripts/run_standard_metrics.py S10_m0_convnext_unet S01_yolov8s_seg_1024 --name check
    python scripts/run_standard_metrics.py S10_m0_convnext_unet A1_m0_deletion N_segzoom_fold0 --folds 0 --name x
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from femurseg import metrics as M  # noqa: E402
from femurseg import stats as S  # noqa: E402
from femurseg.classes import BY_CLS6  # noqa: E402
from femurseg.data import DEFAULT_FOLDS_CSV, load_records  # noqa: E402
from femurseg.evaluation import case_probs  # noqa: E402
from run_stats import SCOPES, _num, decisions, read_cases  # noqa: E402

# folder -> the name used in the documents (GLOSSARY.md), in the order of the tables
NAMES = {
    "S01_yolov8s_seg_1024": "YOLOv8s-seg (published method, our retraining)",
    "S02_yolov8s_seg_640": "YOLOv8s-seg at 640 px",
    "S03_yolo11s_seg_1024": "YOLO11s-seg",
    "S04_densenet121": "DenseNet-121",
    "S05_efficientnet_b4": "EfficientNet-B4",
    "S06_convnext_tiny": "ConvNeXt-T classifier",
    "S07_unet_r34": "ResNet-34 U-Net",
    "S08_deeplabv3p_r34": "ResNet-34 DeepLabV3+",
    "S09_segformer_b2": "SegFormer-B2",
    "S10_m0_convnext_unet": "ConvNeXt U-Net",
    "A1_m0_deletion": "ConvNeXt U-Net, second run",
    "A2_region": "Region-pooling model",
    "A3_geometry": "Geometry model",
    "A4_hierarchy": "Hierarchy model",
    "A5_shared_grade": "Shared-trunk model",
    "A6_slgnet": "SLG-Net",
    "R1_aasn": "AASN-style model",
    "R2_random_ref": "Random-reference model",
    "R3_nearest_ref": "Nearest-reference model",
    "R4_refrac": "RefFrac",
    "R5_concat": "Concatenation model",
    "R6a_no_align": "No-alignment model",
    "R6b_affine": "Affine-alignment model",
    "R7_partner": "Own-hip model",
}
# the fold-0 screening runs of the new method candidates (Analysis/11 section 7); not in the default all-runs table
SCREENING = {
    "N_segzoom_fold0": "Segment-and-zoom network, seed 2026",
    "N_segzoom_fold0_seed2": "Segment-and-zoom network, seed 2027",
    "N_res1152_fold0": "ConvNeXt U-Net at 1152 × 768, seed 2026",
    "N_res1152_fold0_seed2": "ConvNeXt U-Net at 1152 × 768, seed 2027",
    "N_convnext_s_fold0": "ConvNeXt U-Net with ConvNeXt-S, seed 2026",
    "N_convnext_s_fold0_seed2": "ConvNeXt U-Net with ConvNeXt-S, seed 2027",
}
GROUPS = [("Baselines", [k for k in NAMES if k.startswith("S")] + ["A1_m0_deletion"]),
          ("SLG-Net and its ablation chain", [k for k in NAMES if k.startswith("A") and k != "A1_m0_deletion"]),
          ("RefFrac and its variants", [k for k in NAMES if k.startswith("R")]),
          ("New method candidates (fold-0 screening)", list(SCREENING))]
NAMES = {**NAMES, **SCREENING}
SCOPE_TITLE = {"oof": "Out of fold (1,043 femurs)", "test_ensemble": "Test set (210 femurs, ensemble of the 5 fold models)"}


def ovr_macro_auc(probs: np.ndarray, y: np.ndarray) -> float:
    """One-vs-rest AUC of each class present in y (with at least one femur of another class), averaged."""
    aucs = [M.roc_auc(y == c, probs[:, c]) for c in range(probs.shape[1]) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def macro_pr(y, p) -> tuple[float, float]:
    mc = M.multiclass_metrics(y, p)["per_class"]
    present = [v for v in mc.values() if v["support"] > 0]
    return float(np.mean([v["precision"] for v in present])), float(np.mean([v["recall"] for v in present]))


def scope_metrics(rows: list[dict], tau: float, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    dec = decisions(rows, tau)
    y, p, s, d = dec["y"], dec["pred"], dec["score"], dec["dice"]
    mc = M.multiclass_metrics(y, p)
    b = M.binary_metrics(y > 0, p > 0)
    prec, rec = macro_pr(y, p)
    probs, _ = case_probs(rows)

    def boot(f):
        r = S.bootstrap_ci(f, y, p, s, d, n_boot=n_boot, groups=groups, seed=seed)
        return {"estimate": r["estimate"], "lo": r["lo"], "hi": r["hi"]}

    out = {"n": int(len(y)), "threshold": float(tau),
           "accuracy": boot(lambda y, p, s, d: M.multiclass_metrics(y, p)["accuracy"]),
           "correct": int((y == p).sum()),
           "macro_precision": prec, "macro_recall": rec,
           "macro_f1": boot(lambda y, p, s, d: M.multiclass_metrics(y, p)["macro_f1"]),
           "fracture_sensitivity": b["sensitivity"], "fracture_specificity": b["specificity"],
           "auc_fracture": S.delong_ci(y > 0, s), "auc_6class_ovr_macro": ovr_macro_auc(probs, y),
           "per_class": mc["per_class"], "confusion_matrix": mc["confusion_matrix"]}
    fx = y > 0
    if not np.isnan(d[fx]).all():  # a model with masks
        det = fx & (p > 0)
        iou = np.array([(_num(r.get("iou_top")) if pp > 0 else 0.0) for r, t, pp in zip(rows, y, p) if t > 0])
        hd = np.array([_num(r.get("hd95_top")) for r, m in zip(rows, det) if m])
        hd = hd[np.isfinite(hd)]
        out.update(dice=boot(lambda y, p, s, d: float(np.nanmean(d[y > 0]))), iou=float(np.nanmean(iou)),
                   hd95_median=float(np.median(hd)) if hd.size else float("nan"),
                   hd95_iqr=[float(np.percentile(hd, 25)), float(np.percentile(hd, 75))] if hd.size else None,
                   n_fractured=int(fx.sum()), n_missed=int((fx & (p == 0)).sum()))
    return out


def ci(v: dict, nd: int = 3) -> str:
    return f"{v['estimate']:.{nd}f} ({v['lo']:.{nd}f}–{v['hi']:.{nd}f})"


def f3(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def write_md(path: Path, report: dict) -> None:
    res = report["runs"]
    lines = ["# Standard metrics of every model", "",
             "Classification: accuracy, macro precision, macro recall and macro-F1 over the 6 classes, fracture "
             "sensitivity and specificity, fracture AUC and the one-vs-rest macro AUC of the 6 classes. Segmentation "
             "(fractured femurs): Dice and IoU, where a missed fracture counts 0, and HD95 in pixels of the "
             "1024 × 1536 image (median over the fractures the model calls; the number missed is shown beside it). "
             f"95% intervals in brackets: bootstrap over whole x-rays ({report['n_boot']} resamples) for accuracy, "
             "macro-F1 and Dice, DeLong for the fracture AUC. Each run uses its own out-of-fold fracture threshold. "
             "Script: `scripts/run_standard_metrics.py`.", ""]
    folds = report.get("folds", "all")
    if folds != "all":
        lines += [f"**Only the held-out femurs of fold(s) {', '.join(map(str, folds))}**, as in the fold-0 screening "
                  "(Analysis/11 section 7): each run's fracture threshold is fitted again on exactly these femurs, and "
                  "the test set is not used.", ""]
    titles = dict(SCOPE_TITLE)
    if folds != "all":
        n = next((v["oof"]["n"] for v in res.values() if "oof" in v), 0)
        titles["oof"] = f"Out of fold, fold(s) {', '.join(map(str, folds))} ({n:,} femurs)"
    for sc in SCOPES:
        if not any(sc in v for v in res.values()):
            continue
        lines += [f"## {titles[sc]}", ""]
        for title, runs in GROUPS:
            runs = [r for r in runs if r in res and sc in res[r]]
            if not runs:
                continue
            lines += [f"### {title}: classification", "",
                      "| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Sensitivity | Specificity "
                      "| Fracture AUC | 6-class AUC |", "|---|---|---|---|---|---|---|---|---|"]
            for r in runs:
                m = res[r][sc]
                lines.append(f"| {NAMES.get(r, r)} | {m['correct']}/{m['n']} = {ci(m['accuracy'])} "
                             f"| {f3(m['macro_precision'])} | {f3(m['macro_recall'])} | {ci(m['macro_f1'])} "
                             f"| {f3(m['fracture_sensitivity'])} | {f3(m['fracture_specificity'])} "
                             f"| {m['auc_fracture']['auc']:.3f} ({m['auc_fracture']['lo']:.3f}–"
                             f"{min(1.0, m['auc_fracture']['hi']):.3f}) | {f3(m['auc_6class_ovr_macro'])} |")
            seg = [r for r in runs if "dice" in res[r][sc]]
            if seg:
                lines += ["", f"### {title}: segmentation", "",
                          "| Model | Dice | IoU | HD95, px: median (IQR) | Fractures missed |", "|---|---|---|---|---|"]
                for r in seg:
                    m = res[r][sc]
                    iqr = m["hd95_iqr"]
                    hd = f"{m['hd95_median']:.0f} ({iqr[0]:.0f}–{iqr[1]:.0f})" if iqr else "–"
                    lines.append(f"| {NAMES.get(r, r)} | {ci(m['dice'])} | {f3(m['iou'])} | {hd} "
                                 f"| {m['n_missed']} of {m['n_fractured']} |")
            lines.append("")
        lines += [f"### Per class: precision / recall / F1 ({titles[sc].split(' (')[0].lower()})", "",
                  "| Model | " + " | ".join(BY_CLS6[c].name for c in range(6)) + " |", "|---|" + "---|" * 6]
        for title, runs in GROUPS:
            for r in [r for r in runs if r in res and sc in res[r]]:
                pc = res[r][sc]["per_class"]
                lines.append(f"| {NAMES.get(r, r)} | " + " | ".join(
                    f"{v['precision']:.2f} / {v['recall']:.2f} / {v['f1']:.2f}"
                    for v in (pc[BY_CLS6[c].code] for c in range(6))) + " |")
        lines.append("")
    lines += ["## Confusion matrices", "", "Rows = true class, columns = predicted class, in the order "
              + ", ".join(BY_CLS6[c].name for c in range(6)) + ". All matrices are in `standard_metrics.json`.", ""]
    for r in [r for r in ("S10_m0_convnext_unet", "S01_yolov8s_seg_1024") if r in res]:
        for sc in SCOPES:
            if sc in res[r]:
                lines += [f"**{NAMES[r]}, {titles[sc].split(' (')[0].lower()}**", "",
                          "| True \\ predicted | " + " | ".join(BY_CLS6[c].name for c in range(6)) + " |",
                          "|---|" + "---|" * 6]
                lines += [f"| {BY_CLS6[i].name} | " + " | ".join(str(v) for v in row) + " |"
                          for i, row in enumerate(res[r][sc]["confusion_matrix"])]
                lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folders (default: every model run of the paper that is filed)")
    ap.add_argument("--name", default="standard_metrics_all_runs", help="output folder under results/analysis/")
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--folds-csv", default=str(DEFAULT_FOLDS_CSV))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--folds", default="", help="only the out-of-fold femurs of these folds (e.g. 0), the threshold "
                    "fitted again on them and no test set, exactly as scripts/run_stats.py --folds (fold-0 screening)")
    a = ap.parse_args(argv)
    folds = [int(v) for v in a.folds.split(",") if v.strip()]
    results = Path(a.results)
    runs = a.runs or [r for r in NAMES if r not in SCREENING and (results / r / "metrics.json").exists()]
    group_of = {(r.split, r.file): r.group for r in load_records(a.folds_csv)}
    report = {"n_boot": a.n_boot, "seed": a.seed, "folds": folds or "all", "names": {r: NAMES.get(r, r) for r in runs}, "runs": {}}
    for run in runs:
        d = results / run
        tau = (json.loads((d / "metrics.json").read_text()).get("thresholds") or {}).get("oof_youden")
        if folds:
            oof = [r for r in read_cases(d / SCOPES["oof"]) if int(float(r["fold"])) in folds]
            if not oof:
                raise SystemExit(f"{run}: no out-of-fold femurs in folds {folds}")
            tau = M.youden_threshold([int(r["cls6_true"]) > 0 for r in oof], [_num(r["score"]) for r in oof])
        if tau is None:
            raise SystemExit(f"{run}: metrics.json has no out-of-fold threshold")
        report["runs"][run] = {"threshold": tau}
        for sc, fn in SCOPES.items():
            if not (d / fn).exists() or (folds and sc != "oof"):
                continue
            rows = read_cases(d / fn)
            if folds:
                rows = [r for r in rows if int(float(r["fold"])) in folds]
            split = "test" if sc.startswith("test") else "train"
            groups = np.array([group_of[(split, r["file"])] for r in rows])
            report["runs"][run][sc] = scope_metrics(rows, tau, groups, a.n_boot, a.seed)
        print(f"{run}: done", flush=True)
    out = results / "analysis" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "standard_metrics.json").write_text(json.dumps(report, indent=2, default=float))
    write_md(out / "standard_metrics.md", report)
    print(f"written: {out}/standard_metrics.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
