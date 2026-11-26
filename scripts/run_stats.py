#!/usr/bin/env python3
"""95% confidence intervals for filed runs, and paired comparisons between runs.

Reads only the filed run folders (per-case CSVs and metrics.json), so it can be re-run at any time; it never edits
a run folder. Output goes to results/analysis/<name>/ (stats.json and stats.md).

Intervals: percentile bootstrap with 2,000 resamples of whole x-rays (the two femurs of an x-ray are not
independent), DeLong for AUC, and the exact Clopper-Pearson interval for 6-class accuracy (as the base article).
Paired comparisons (same femurs): DeLong for AUC, exact McNemar (and mid-p) for right/wrong decisions, a
randomisation test that swaps the two runs' predictions per x-ray for macro-F1, and Wilcoxon signed-rank (Pratt)
for per-femur Dice; Holm-adjusted p-values over the comparisons of each scope.

Examples
  python scripts/run_stats.py S01_yolov8s_seg_1024
  python scripts/run_stats.py S01_yolov8s_seg_1024 S02_yolov8s_seg_640 --compare --name C01_yolo_1024_vs_640
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from femurseg import metrics as M  # noqa: E402
from femurseg import stats as S  # noqa: E402
from femurseg.data import DEFAULT_FOLDS_CSV, load_records  # noqa: E402

SCOPES = {"oof": "per_case_oof.csv", "test_ensemble": "per_case_test_ensemble.csv"}
METRICS = ("auc_fracture", "fracture_sensitivity", "fracture_specificity", "accuracy_6class", "macro_f1",
           "balanced_accuracy", "kappa", "dice_fractured_mean", "dice_detected_mean", "garden12_recall")


# ---------------------------------------------------------------------------------------------- per-case data


def read_cases(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(v: str) -> float:
    return float(v) if v not in ("", None) else float("nan")


def decisions(rows: list[dict], tau: float) -> dict[str, np.ndarray]:
    """Per-femur arrays: true class, fracture score, predicted class, Dice (fractured femurs; missed = 0).

    Uses a ``pred_cls6`` column when the run wrote one; otherwise the YOLO rule of scripts/train_yolo_seg.py
    (class of the top instance if its confidence >= tau, else no fracture)."""
    y = np.array([int(r["cls6_true"]) for r in rows])
    score = np.array([_num(r["score"]) for r in rows])
    if rows and "pred_cls6" in rows[0]:
        pred = np.array([int(r["pred_cls6"]) for r in rows])
    else:
        pred = np.array([int(float(r["top_cls6"])) if _num(r["top_conf"]) >= tau else 0 for r in rows])
    dice = np.array([(_num(r["dice_top"]) if p > 0 else 0.0) if t > 0 else np.nan
                     for r, t, p in zip(rows, y, pred)])
    return {"file": np.array([r["file"] for r in rows]), "y": y, "score": score, "pred": pred, "dice": dice}


# ---------------------------------------------------------------------------------------------------- metrics


def metric_fn(name: str):
    """A function f(y, pred, score, dice) -> float for each metric name."""
    def auc(y, p, s, d):
        return M.roc_auc(y > 0, s)

    def binm(key):
        return lambda y, p, s, d: M.binary_metrics(y > 0, p > 0)[key]

    def multi(key):
        return lambda y, p, s, d: M.multiclass_metrics(y, p)[key]

    def dice_fx(y, p, s, d):
        return float(np.nanmean(d)) if (y > 0).any() else float("nan")

    def dice_det(y, p, s, d):
        m = (y > 0) & (p > 0)
        return float(np.mean(d[m])) if m.any() else float("nan")

    def g12(y, p, s, d):
        return float(np.mean(p[y == 1] == 1)) if (y == 1).any() else float("nan")

    return {"auc_fracture": auc, "fracture_sensitivity": binm("sensitivity"), "fracture_specificity": binm("specificity"),
            "accuracy_6class": multi("accuracy"), "macro_f1": multi("macro_f1"),
            "balanced_accuracy": multi("balanced_accuracy"), "kappa": multi("kappa"),
            "dice_fractured_mean": dice_fx, "dice_detected_mean": dice_det, "garden12_recall": g12}[name]


def intervals(dec: dict, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    y, p, s, d = dec["y"], dec["pred"], dec["score"], dec["dice"]
    out = {}
    for name in METRICS:
        f = metric_fn(name)
        r = S.bootstrap_ci(f, y, p, s, d, n_boot=n_boot, groups=groups, seed=seed)
        out[name] = {"estimate": r["estimate"], "lo": r["lo"], "hi": r["hi"], "method": "cluster bootstrap (x-ray)"}
    out["auc_fracture_delong"] = {k: v for k, v in S.delong_ci(y > 0, s).items()} | {"method": "DeLong"}
    k = int((p == y).sum())
    lo, hi = S.proportion_ci(k, len(y), method="exact")
    out["accuracy_6class_exact"] = {"estimate": k / len(y), "lo": lo, "hi": hi, "method": "Clopper-Pearson",
                                    "correct": k, "n": len(y)}
    out["n_femurs"], out["n_xrays"] = int(len(y)), int(len(np.unique(groups)))
    return out


def compare(da: dict, db: dict, groups: np.ndarray, n_boot: int, n_perm: int, seed: int) -> dict:
    """Paired comparison of run A and run B on the same femurs (rows already aligned by file)."""
    y = da["y"]
    res = {"auc_fracture": S.delong_test(y > 0, da["score"], db["score"]) | {"test": "DeLong"}}
    res["accuracy_6class"] = S.mcnemar_exact(da["pred"] == y, db["pred"] == y) | {
        "test": "McNemar", "diff": float(np.mean(da["pred"] == y) - np.mean(db["pred"] == y))}
    ok_a, ok_b = (da["pred"] > 0) == (y > 0), (db["pred"] > 0) == (y > 0)
    res["fracture_detection"] = S.mcnemar_exact(ok_a, ok_b) | {
        "test": "McNemar (binary fracture decision)", "diff": float(ok_a.mean() - ok_b.mean())}
    mf1 = lambda yy, pp: M.multiclass_metrics(yy, pp)["macro_f1"]  # noqa: E731
    perm = S.permutation_test(mf1, y, da["pred"], db["pred"], n_perm=n_perm, groups=groups, seed=seed)
    boot = S.bootstrap_diff(mf1, y, da["pred"], db["pred"], n_boot=n_boot, groups=groups, seed=seed)
    res["macro_f1"] = {"test": "randomisation (swap per x-ray)", "diff": perm["diff"], "p": perm["p"],
                       "diff_lo": boot["lo"], "diff_hi": boot["hi"]}
    fx = y > 0
    w = S.wilcoxon_paired(da["dice"][fx], db["dice"][fx])
    res["dice_fractured"] = w | {"test": "Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0"}
    names = list(res)
    adj = S.holm([res[n]["p"] for n in names])
    for n, a in zip(names, adj):
        res[n]["p_holm"] = float(a)
    return res


# ------------------------------------------------------------------------------------------------------- main


def load_run(results: Path, run: str) -> tuple[dict, dict]:
    d = results / run
    met = json.loads((d / "metrics.json").read_text())
    tau = (met.get("thresholds") or {}).get("oof_youden")
    if tau is None:
        raise SystemExit(f"{run}: metrics.json has no OOF threshold")
    scopes = {sc: decisions(read_cases(d / fn), tau) for sc, fn in SCOPES.items() if (d / fn).exists()}
    return {"tau": tau, "metrics": met}, scopes


def fmt(v: dict, key="estimate", nd=3) -> str:
    x = v.get(key)
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run names (folders in --results)")
    ap.add_argument("--compare", action="store_true", help="also compare every pair of runs")
    ap.add_argument("--name", default="", help="output folder name under results/analysis/ (default: first run)")
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--folds-csv", default=str(DEFAULT_FOLDS_CSV))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)
    results = Path(a.results)
    group_of = {(r.split, r.file): r.group for r in load_records(a.folds_csv)}
    out_dir = results / "analysis" / (a.name or a.runs[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"runs": a.runs, "n_boot": a.n_boot, "n_perm": a.n_perm, "seed": a.seed, "intervals": {},
                    "comparisons": {}}
    loaded = {run: load_run(results, run) for run in a.runs}
    for run, (info, scopes) in loaded.items():
        report["intervals"][run] = {"threshold": info["tau"]}
        for sc, dec in scopes.items():
            split = "test" if sc.startswith("test") else "train"
            groups = np.array([group_of[(split, f)] for f in dec["file"]])
            report["intervals"][run][sc] = intervals(dec, groups, a.n_boot, a.seed)
    if a.compare:
        for i, ra in enumerate(a.runs):
            for rb in a.runs[i + 1:]:
                key = f"{ra} vs {rb}"
                report["comparisons"][key] = {}
                for sc in SCOPES:
                    if sc not in loaded[ra][1] or sc not in loaded[rb][1]:
                        continue
                    da, db = loaded[ra][1][sc], loaded[rb][1][sc]
                    order_b = {f: i for i, f in enumerate(db["file"])}
                    if set(order_b) != set(da["file"]):
                        raise SystemExit(f"{key} ({sc}): the two runs did not predict the same femurs")
                    idx = np.array([order_b[f] for f in da["file"]])
                    db = {k: v[idx] for k, v in db.items()}
                    split = "test" if sc.startswith("test") else "train"
                    groups = np.array([group_of[(split, f)] for f in da["file"]])
                    report["comparisons"][key][sc] = compare(da, db, groups, a.n_boot, a.n_perm, a.seed)

    (out_dir / "stats.json").write_text(json.dumps(report, indent=2, default=float))
    lines = [f"# Statistics: {', '.join(a.runs)}", "",
             f"95% intervals: cluster bootstrap over x-rays ({a.n_boot} resamples, seed {a.seed}); AUC also DeLong; "
             "6-class accuracy also exact (Clopper-Pearson). Decisions use each run's OOF threshold.", ""]
    for run, blocks in report["intervals"].items():
        for sc in SCOPES:
            if sc not in blocks:
                continue
            b = blocks[sc]
            lines += [f"## {run}: {sc} ({b['n_femurs']} femurs, {b['n_xrays']} x-rays)", "",
                      "| Metric | Estimate | 95% CI |", "|---|---|---|"]
            for name in METRICS:
                lines.append(f"| {name} | {fmt(b[name])} | {fmt(b[name], 'lo')}–{fmt(b[name], 'hi')} |")
            lines.append(f"| auc_fracture (DeLong) | {fmt(b['auc_fracture_delong'], 'auc')} | "
                         f"{fmt(b['auc_fracture_delong'], 'lo')}–{fmt(b['auc_fracture_delong'], 'hi')} |")
            e = b["accuracy_6class_exact"]
            lines += [f"| accuracy_6class (exact) | {e['correct']}/{e['n']} = {fmt(e)} | {fmt(e, 'lo')}–{fmt(e, 'hi')} |", ""]
    for key, scopes in report["comparisons"].items():
        for sc, res in scopes.items():
            lines += [f"## {key}: {sc}", "", "| Endpoint | Test | Difference (A − B) | p | p (Holm) |", "|---|---|---|---|---|"]
            for name, r in res.items():
                diff = r.get("diff", r.get("mean_diff"))
                lines.append(f"| {name} | {r['test']} | {diff:+.3f} | {r['p']:.4f} | {r['p_holm']:.4f} |")
            lines.append("")
    (out_dir / "stats.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"written: {out_dir / 'stats.json'} and stats.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
