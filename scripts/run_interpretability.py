#!/usr/bin/env python3
"""Interpretability tests of Paper A (Analysis/06 §5, analysis C08), computed from filed run folders only.

1. Oracle test (runs with per_case_<part>_oracle.csv): the heads get the released mask instead of the predicted
   one; the network's own mask is unchanged. Base and oracle are compared on the same femurs, out of fold and on the
   test ensemble: correct decisions (exact McNemar), macro-F1 (randomisation test that swaps whole x-rays), Garden I-II
   recall, and on the fractured femurs the wrong decisions the oracle fixes and the correct ones it loses.
2. Region deletion vs control (runs with per_case_<part>_deletion.csv and _deletion_control.csv), on the fractured
   femurs:
   - the change of P(fracture) and of P(true class) under deletion and under the control, compared femur by femur
     with a Wilcoxon signed-rank test (Pratt);
   - the share of correct decisions lost under each, compared with an exact McNemar test on the femurs that are
     correct at base, overall and per class;
   - where the control was placed (folds/fold<k>/<oof|test>_deletion_info.csv).
3. Risk-coverage (every run): the confidence of a decision is the probability the run gives to the class it decided
   (1 - P(fracture) for "no fracture"; raw probabilities, no temperature). Femurs are set aside from the least
   confident up. Reported:
   - the share of all errors that falls among the 10% least confident femurs, with a 95% cluster-bootstrap interval
     over x-rays (flagging at random catches 10% on average);
   - the error rate of the 90% that are kept;
   - the area under the risk-coverage curve (AURC; lower is better), next to its best possible value for the same
     errors.
4. Geometry consistency (with --geometry RUN=CSV, the per-femur geometry of the run's predicted regions written by
   scripts/map_geometry.py): do the predicted region's size and position agree with the decided location and grade,
   as the released masks agree with the released labels (Report 03 §2)? The released masks' geometry comes from
   EDA/per_image_stats.csv.
   - Location: two cut points on the centroid height (neck above, pertrochanteric between, subtrochanteric below)
     are fitted on the released masks of the 680 OOF fractures; the share of femurs called fractured (with a
     region) whose region the rule places at the decided location, next to the rule's accuracy on the released
     masks against the released locations.
   - Grade: among the femurs decided as neck fractures, the AUC of the region's height for the decided Garden grade
     (III-IV vs I-II); among those decided as pertrochanteric, the AUC of the region's area for the decided Evans
     grade (III-V vs I-II); both with DeLong intervals, next to the same AUCs of the released masks for the released
     grades, and of the predicted regions for the released grades.

Decisions use each run's OOF threshold (metrics.json thresholds.oof_youden), as everywhere else. The changes and
shares of tests 1 and 2 are checked against the run's own metrics.json ("tests"), which train_net.py wrote. Writes
results/analysis/<name>/interpretability.json and interpretability.md; it never edits a run folder.

  python scripts/run_interpretability.py S10_m0_convnext_unet A2_region A3_geometry A4_hierarchy A5_shared_grade \\
      A6_slgnet --name C08_paperA_interpretability
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
sys.path.insert(0, str(REPO / "scripts"))
from femurseg import metrics as M  # noqa: E402
from femurseg import stats as S  # noqa: E402
from femurseg.data import DEFAULT_FOLDS_CSV, load_records  # noqa: E402
from femurseg.evaluation import case_probs  # noqa: E402
from run_stats import decisions, read_cases  # noqa: E402

# part -> (per-case file stem, split of the femurs in folds_v1.csv, prefix of the deletion info files)
PARTS = {"oof": ("per_case_oof", "train", "oof"), "test_ensemble": ("per_case_test_ensemble", "test", "test")}
CLASSES = ("NF", "2AB", "2CD", "3AB", "3CDE", "4")
FLAG_SHARE = 0.10
COVERAGES = (1.0, 0.95, 0.90, 0.80)


def load(d: Path, part: str, mode: str = "base") -> list[dict] | None:
    """The per-case rows of one part and test mode, sorted by file (None when the run did not write them)."""
    f = d / f"{PARTS[part][0]}{'' if mode == 'base' else '_' + mode}.csv"
    return sorted(read_cases(f), key=lambda r: r["file"]) if f.exists() else None


def written_probs(rows: list[dict]) -> np.ndarray:
    """[1 - score, conf_1..conf_5] exactly as written (train_net.py's own test effects use these)."""
    return np.array([[1 - float(r["score"])] + [float(r[f"conf_{c}"]) for c in range(1, 6)] for r in rows])


def macro_f1(y, pred) -> float:
    return M.multiclass_metrics(y, pred)["macro_f1"]


def garden12(y, pred) -> float:
    return float(np.mean(pred[y == 1] == 1)) if (y == 1).any() else float("nan")


def aligned(*decs: dict) -> None:
    if any(not np.array_equal(d["file"], decs[0]["file"]) for d in decs[1:]):
        raise SystemExit("the test modes of a run did not predict the same femurs")


# ----------------------------------------------------------------------------------------------------- tests


def oracle_block(base: dict, orc: dict, groups: np.ndarray, n_perm: int, seed: int) -> dict:
    y = base["y"]
    ok_b, ok_o, fx = base["pred"] == y, orc["pred"] == y, y > 0
    perm = S.permutation_test(macro_f1, y, orc["pred"], base["pred"], n_perm=n_perm, groups=groups, seed=seed)
    return {"n": int(len(y)), "correct_base": int(ok_b.sum()), "correct_oracle": int(ok_o.sum()),
            "mcnemar_oracle_vs_base": S.mcnemar_exact(ok_o, ok_b),
            "macro_f1_base": macro_f1(y, base["pred"]), "macro_f1_oracle": macro_f1(y, orc["pred"]),
            "macro_f1_p": perm["p"],
            "garden12_recall_base": garden12(y, base["pred"]), "garden12_recall_oracle": garden12(y, orc["pred"]),
            "n_fractured": int(fx.sum()), "fractured_correct_base": int((fx & ok_b).sum()),
            "fractured_correct_lost": int((fx & ok_b & ~ok_o).sum()),
            "fractured_wrong_base": int((fx & ~ok_b).sum()), "fractured_wrong_fixed": int((fx & ~ok_b & ok_o).sum())}


def deletion_block(rows: dict[str, list[dict]], decs: dict[str, dict]) -> dict:
    """rows / decs keyed by mode: base, deletion, deletion_control (aligned by file)."""
    y = decs["base"]["y"]
    fx, idx = y > 0, np.arange(len(y))
    p = {m: written_probs(r) for m, r in rows.items()}
    change = {"fracture_prob": {m: (1 - p[m][:, 0]) - (1 - p["base"][:, 0]) for m in ("deletion", "deletion_control")},
              "true_class_prob": {m: p[m][idx, y] - p["base"][idx, y] for m in ("deletion", "deletion_control")}}
    out: dict = {"n_fractured": int(fx.sum())}
    for name, ch in change.items():
        a, b = ch["deletion"][fx], ch["deletion_control"][fx]
        out[name] = {"deletion_mean": float(a.mean()), "deletion_median": float(np.median(a)),
                     "control_mean": float(b.mean()), "control_median": float(np.median(b)),
                     "wilcoxon_deletion_vs_control": S.wilcoxon_paired(a, b)}
    ok = fx & (decs["base"]["pred"] == y)
    lost = {m: ok & (decs[m]["pred"] != y) for m in ("deletion", "deletion_control")}
    wrong = fx & ~ok
    out.update(correct_base=int(ok.sum()), lost_deletion=int(lost["deletion"].sum()),
               lost_control=int(lost["deletion_control"].sum()),
               mcnemar_kept_deletion_vs_control=S.mcnemar_exact(~lost["deletion"][ok], ~lost["deletion_control"][ok]),
               wrong_base=int(wrong.sum()),
               fixed_deletion=int((wrong & (decs["deletion"]["pred"] == y)).sum()),
               fixed_control=int((wrong & (decs["deletion_control"]["pred"] == y)).sum()))
    out["per_class"] = {CLASSES[c]: {"correct_base": int((ok & (y == c)).sum()),
                                     "lost_deletion": int((lost["deletion"] & (y == c)).sum()),
                                     "lost_control": int((lost["deletion_control"] & (y == c)).sum())}
                        for c in range(1, 6)}
    return out


def placement(d: Path, prefix: str) -> dict:
    """Where the control went, over the fractured femurs (test: every fold model's own deletion of each femur)."""
    counts: dict[str, int] = {}
    for f in sorted((d / "folds").glob(f"fold*/{prefix}_deletion_info.csv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                if int(float(r["cls6_true"])) > 0:
                    counts[r["control"]] = counts.get(r["control"], 0) + 1
    return dict(sorted(counts.items()))


def flagged_share(y, pred, conf) -> float:
    """Share of all errors among the FLAG_SHARE least confident femurs (ties kept in file order)."""
    err = pred != y
    if not err.any():
        return float("nan")
    k = math.ceil(FLAG_SHARE * len(y))
    flag = np.zeros(len(y), bool)
    flag[np.argsort(conf, kind="stable")[:k]] = True
    return float(err[flag].sum() / err.sum())


def risk_coverage(dec: dict, rows: list[dict], groups: np.ndarray, n_boot: int, seed: int) -> dict:
    probs, y = case_probs(rows)
    pred = dec["pred"]
    conf = probs[np.arange(len(y)), pred]
    err = pred != y
    n, e = len(y), int(err.sum())
    order = np.argsort(-conf, kind="stable")  # most confident first
    risk = np.cumsum(err[order]) / np.arange(1, n + 1)
    best = np.maximum(0, np.arange(1, n + 1) - (n - e)) / np.arange(1, n + 1)
    k = math.ceil(FLAG_SHARE * n)
    kept = np.ones(n, bool)
    kept[np.argsort(conf, kind="stable")[:k]] = False
    boot = S.bootstrap_ci(flagged_share, y, pred, conf, n_boot=n_boot, groups=groups, seed=seed)
    return {"n": n, "errors": e, "error_rate": e / n, "flagged": k, "errors_flagged": int(err[~kept].sum()),
            "share_of_errors_flagged": {k2: boot[k2] for k2 in ("estimate", "lo", "hi")},
            "error_rate_kept": float(err[kept].mean()), "aurc": float(risk.mean()), "aurc_best": float(best.mean()),
            "risk_at_coverage": {f"{c:.2f}": float(err[order[:math.ceil(c * n)]].mean()) for c in COVERAGES}}


LOCATION = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}  # cls6 -> neck, pertrochanteric, subtrochanteric
LOCATION_NAMES = ("neck", "pertrochanteric", "subtrochanteric")


def fit_location_cuts(cy: np.ndarray, loc: np.ndarray) -> tuple[float, float]:
    """The two cut points on centroid height (neck above the first, pertrochanteric between, subtrochanteric below
    the second) that classify the given regions best; ties go to the first pair found."""
    order = np.argsort(cy, kind="stable")
    c, lo = cy[order], loc[order]
    n = len(c)
    cum = {k: np.concatenate([[0], np.cumsum(lo == k)]) for k in range(3)}
    i, j = np.arange(n + 1)[:, None], np.arange(n + 1)[None, :]
    correct = cum[0][:, None] + (cum[1][None, :] - cum[1][:, None]) + (cum[2][n] - cum[2][None, :])
    ii, jj = np.unravel_index(np.argmax(np.where(j >= i, correct, -1)), correct.shape)

    def cut(k: int) -> float:
        return -np.inf if k == 0 else (np.inf if k == n else float((c[k - 1] + c[k]) / 2))
    return cut(ii), cut(jj)


def locate(cy: np.ndarray, cuts: tuple[float, float]) -> np.ndarray:
    return np.where(cy < cuts[0], 0, np.where(cy < cuts[1], 1, 2))


def auc_block(positive: np.ndarray, score: np.ndarray) -> dict:
    """AUC with its DeLong interval; n and positives counted (undefined with one class only)."""
    out = {"n": int(len(score)), "positives": int(positive.sum())}
    if out["positives"] in (0, out["n"]):
        return out | {"auc": float("nan"), "lo": float("nan"), "hi": float("nan")}
    ci = S.delong_ci(positive, score)
    return out | {"auc": float(ci["auc"]), "lo": float(ci["lo"]), "hi": float(ci["hi"])}


def read_geometry(path: Path) -> dict[tuple[str, str], dict]:
    with open(path, newline="") as fh:
        return {(r["part"], r["file"]): r for r in csv.DictReader(fh)}


def released_geometry(eda_csv: Path) -> dict[tuple[str, str], dict]:
    """(split, file) -> area fraction, centroid height and bounding-box height of the released mask (empty: none)."""
    out = {}
    with open(eda_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("mask_cy"):
                out[(r["split"], r["file"])] = {"area_frac": float(r["mask_area_frac"]), "cy": float(r["mask_cy"]),
                                                "bh": float(r["mask_bh"])}
    return out


def geometry_block(dec: dict, rows: list[dict], geo: dict, released: dict, part: str, cuts) -> dict:
    """Consistency of one run's decisions with its own predicted regions (and with the released labels)."""
    split = PARTS[part][1]
    files, y, pred = dec["file"], dec["y"], dec["pred"]
    g = [geo.get((part, f)) for f in files]
    if any(v is None for v in g):
        raise SystemExit(f"geometry file has no {part} row for some femurs")
    has = np.array([v["cy"] != "" for v in g])
    val = {k: np.array([float(v[k]) if v[k] != "" else np.nan for v in g]) for k in ("cy", "bh", "area_frac")}
    area_px = np.array([float(v["area_px"]) for v in g])
    area_top = np.array([float(r["area_top"]) if r.get("area_top") not in ("", None) else 0.0 for r in rows])
    rel = np.abs(area_px - area_top)[area_top > 0] / area_top[area_top > 0]
    called = (pred > 0) & has
    ploc = np.array([LOCATION.get(int(p), -1) for p in pred])
    agree = locate(val["cy"][called], cuts) == ploc[called]
    neck, pert = called & np.isin(pred, (1, 2)), called & np.isin(pred, (3, 4))
    tneck, tpert = has & np.isin(y, (1, 2)), has & np.isin(y, (3, 4))
    out = {"femurs_called_with_region": int(called.sum()),
           "region_check": {"same_empty_or_not": bool(np.array_equal(area_px > 0, area_top > 0)),
                            "max_relative_area_difference": float(rel.max()) if rel.size else 0.0},
           "location_agreement": {"agree": int(agree.sum()), "n": int(called.sum())},
           "garden_height_auc_decided": auc_block(pred[neck] == 2, val["bh"][neck]),
           "evans_area_auc_decided": auc_block(pred[pert] == 4, val["area_frac"][pert]),
           "garden_height_auc_released_grade": auc_block(y[tneck] == 2, val["bh"][tneck]),
           "evans_area_auc_released_grade": auc_block(y[tpert] == 4, val["area_frac"][tpert])}
    # the released masks against the released labels, on the same femurs (the data's own consistency)
    rg = [released.get((split, f)) for f in files]
    rhas = np.array([v is not None for v in rg]) & (y > 0)
    rv = {k: np.array([v[k] if v is not None else np.nan for v in rg]) for k in ("cy", "bh", "area_frac")}
    tloc = np.array([LOCATION.get(int(t), -1) for t in y])
    rn, rp = rhas & np.isin(y, (1, 2)), rhas & np.isin(y, (3, 4))
    out["released"] = {"location_agreement": {"agree": int((locate(rv["cy"][rhas], cuts) == tloc[rhas]).sum()),
                                              "n": int(rhas.sum())},
                       "garden_height_auc": auc_block(y[rn] == 2, rv["bh"][rn]),
                       "evans_area_auc": auc_block(y[rp] == 4, rv["area_frac"][rp])}
    return out


def effects_match(ref: dict | None, mine: dict, kind: str) -> bool | None:
    """True when the run's metrics.json effects equal the ones recomputed here (None: nothing to compare)."""
    if not ref:
        return None
    if kind == "oracle":
        vals = [(ref["n_correct_base"], mine["fractured_correct_base"]),
                (ref["share_correct_lost"], mine["fractured_correct_lost"] / max(mine["fractured_correct_base"], 1)),
                (ref["share_wrong_fixed"], mine["fractured_wrong_fixed"] / max(mine["fractured_wrong_base"], 1))]
    else:
        m = "deletion" if kind == "deletion" else "control"
        lost = mine["lost_deletion"] if kind == "deletion" else mine["lost_control"]
        vals = [(ref["n_correct_base"], mine["correct_base"]),
                (ref["mean_change_fracture_prob"], mine["fracture_prob"][f"{m}_mean"]),
                (ref["mean_change_true_class_prob"], mine["true_class_prob"][f"{m}_mean"]),
                (ref["share_correct_lost"], lost / max(mine["correct_base"], 1))]
    return all(abs(float(a) - float(b)) < 1e-6 for a, b in vals)


# ------------------------------------------------------------------------------------------------------ report


def f3(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def pfmt(p: float) -> str:
    return "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def write_md(rep: dict) -> str:
    part_name = {"oof": "OOF", "test_ensemble": "test ensemble"}
    out = [f"# Interpretability tests: {', '.join(rep['runs'])}", "",
           "Decisions at each run's OOF threshold. Femurs: OOF 1,043 (680 fractured); test ensemble 210 (145 fractured).",
           f"Intervals: cluster bootstrap over x-rays ({rep['n_boot']} resamples); randomisation test for macro-F1 "
           f"({rep['n_perm']} swaps of whole x-rays); seed {rep['seed']}.", ""]
    if rep["oracle"]:
        out += ["## 1. Oracle test: the heads get the released mask", "",
                "| Run | Part | Correct: base → oracle (McNemar p) | Macro-F1: base → oracle (p) | Garden I-II recall: "
                "base → oracle | Fractured, wrong at base: fixed | Fractured, correct at base: lost | Same as metrics.json |",
                "|---|---|---|---|---|---|---|---|"]
        for run, parts in rep["oracle"].items():
            for part, b in parts.items():
                out.append(f"| {run} | {part_name[part]} | {b['correct_base']} → {b['correct_oracle']} "
                           f"({pfmt(b['mcnemar_oracle_vs_base']['p'])}) | {f3(b['macro_f1_base'])} → "
                           f"{f3(b['macro_f1_oracle'])} ({pfmt(b['macro_f1_p'])}) | {f3(b['garden12_recall_base'])} → "
                           f"{f3(b['garden12_recall_oracle'])} | {b['fractured_wrong_fixed']} of {b['fractured_wrong_base']} "
                           f"| {b['fractured_correct_lost']} of {b['fractured_correct_base']} | {b['matches_metrics_json']} |")
        out.append("")
    for run, parts in rep["deletion"].items():
        out += [f"## 2. Region deletion vs control: {run}", "",
                "| Part | Fractured femurs | ΔP(fracture): deletion / control, mean (median); Wilcoxon p | "
                "ΔP(true class): deletion / control, mean (median); Wilcoxon p | Correct decisions lost: deletion / "
                "control (McNemar p) | Wrong ones fixed: deletion / control | Same as metrics.json |",
                "|---|---|---|---|---|---|---|"]
        for part, b in parts.items():
            cell = {}
            for k in ("fracture_prob", "true_class_prob"):
                c = b[k]
                cell[k] = (f"{c['deletion_mean']:+.3f} ({c['deletion_median']:+.3f}) / {c['control_mean']:+.3f} "
                           f"({c['control_median']:+.3f}); {pfmt(c['wilcoxon_deletion_vs_control']['p'])}")
            cb = max(b["correct_base"], 1)
            out.append(f"| {part_name[part]} | {b['n_fractured']} | {cell['fracture_prob']} | {cell['true_class_prob']} | "
                       f"{b['lost_deletion']} ({100 * b['lost_deletion'] / cb:.1f}%) / {b['lost_control']} "
                       f"({100 * b['lost_control'] / cb:.1f}%) of {b['correct_base']} "
                       f"({pfmt(b['mcnemar_kept_deletion_vs_control']['p'])}) | {b['fixed_deletion']} / "
                       f"{b['fixed_control']} of {b['wrong_base']} | {b['matches_metrics_json']} |")
        out += ["", "Correct decisions lost, per true class (deletion / control, of those correct at base):", "",
                "| Part | " + " | ".join(CLASSES[1:]) + " |", "|---|" + "---|" * 5]
        for part, b in parts.items():
            out.append(f"| {part_name[part]} | " + " | ".join(
                f"{v['lost_deletion']} / {v['lost_control']} of {v['correct_base']}" for v in b["per_class"].values()) + " |")
        out += ["", "Where the control went (fractured femurs; test: each of the 5 fold models' own deletion): "
                + "; ".join(f"{part_name[part]}: " + ", ".join(f"{k} {v}" for k, v in b["control_placement"].items())
                            for part, b in parts.items()) + ".", ""]
    out += ["## 3. Risk-coverage: setting aside the least confident decisions", "",
            f"| Run | Part | Errors | Errors among the {int(FLAG_SHARE * 100)}% least confident (95% CI) | Error rate: all → "
            f"the {100 - int(FLAG_SHARE * 100)}% kept | AURC (best possible) |", "|---|---|---|---|---|---|"]
    for run, parts in rep["risk_coverage"].items():
        for part, b in parts.items():
            s = b["share_of_errors_flagged"]
            out.append(f"| {run} | {part_name[part]} | {b['errors']} of {b['n']} | {b['errors_flagged']} = "
                       f"{100 * s['estimate']:.1f}% ({100 * s['lo']:.1f}–{100 * s['hi']:.1f}%) | "
                       f"{100 * b['error_rate']:.1f}% → {100 * b['error_rate_kept']:.1f}% | {f3(b['aurc'])} "
                       f"({f3(b['aurc_best'])}) |")
    out += ["", "Confidence = the probability of the decided class (raw, before temperature scaling). Flagging at "
            "random would catch 10% of the errors on average.", ""]
    if rep.get("geometry"):
        geo = rep["geometry"]
        t1, t2 = geo["location_cuts"]

        def auc(b: dict) -> str:
            return "n/a" if math.isnan(b["auc"]) else f"{b['auc']:.2f} ({b['lo']:.2f}–{b['hi']:.2f}), n = {b['n']}"

        def share(b: dict) -> str:
            return f"{b['agree']} of {b['n']} ({100 * b['agree'] / max(b['n'], 1):.1f}%)"
        out += ["## 4. Geometry consistency: does the region agree with the decided location and grade?", "",
                f"Location rule: centroid height (relative to the image) below {t1:.3f} = neck, below {t2:.3f} = "
                f"pertrochanteric, otherwise subtrochanteric; fitted on {geo['cuts_fitted_on']}. Garden: region "
                "height; Evans: region area (Report 03 §2). AUCs with DeLong 95% intervals.", "",
                "| Run | Part | Location: rule on the region = decided location | Garden: height vs decided grade, AUC "
                "| Evans: area vs decided grade, AUC | Garden: height vs released grade | Evans: area vs released "
                "grade | Region check (areas vs per-case file) |", "|---|---|---|---|---|---|---|---|"]
        shown = set()
        for run, parts in geo["runs"].items():
            for part, b in parts.items():
                rc = b["region_check"]
                out.append(f"| {run} | {part_name[part]} | {share(b['location_agreement'])} | "
                           f"{auc(b['garden_height_auc_decided'])} | {auc(b['evans_area_auc_decided'])} | "
                           f"{auc(b['garden_height_auc_released_grade'])} | {auc(b['evans_area_auc_released_grade'])} | "
                           f"{'same' if rc['same_empty_or_not'] else 'DIFFERENT'} empty/non-empty; max area "
                           f"difference {100 * rc['max_relative_area_difference']:.2f}% |")
                shown.add(part)
        first = next(iter(geo["runs"].values()))
        for part in sorted(shown):
            r = first[part]["released"]
            out.append(f"| released masks, released labels | {part_name[part]} | {share(r['location_agreement'])} "
                       f"(released location) | – | – | {auc(r['garden_height_auc'])} | {auc(r['evans_area_auc'])} | – |")
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run names (folders in --results)")
    ap.add_argument("--name", required=True, help="output folder name under results/analysis/")
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--folds-csv", default=str(DEFAULT_FOLDS_CSV))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--geometry", action="append", default=[], metavar="RUN=CSV",
                    help="per-femur geometry of a run's predicted regions (scripts/map_geometry.py); repeatable")
    ap.add_argument("--eda", default=str(REPO / "EDA" / "per_image_stats.csv"),
                    help="the released masks' geometry (EDA/eda_femoral.py)")
    a = ap.parse_args(argv)
    results = Path(a.results)
    records = load_records(a.folds_csv)
    group_of = {(r.split, r.file): r.group for r in records}
    geometry_of = {}
    for item in a.geometry:
        run, sep, path = item.partition("=")
        if not sep or run not in a.runs or not Path(path).exists():
            raise SystemExit(f"--geometry {item!r}: must be RUN=CSV with a run of the list and an existing file")
        geometry_of[run] = read_geometry(Path(path))
    rep: dict = {"runs": a.runs, "n_boot": a.n_boot, "n_perm": a.n_perm, "seed": a.seed, "flag_share": FLAG_SHARE,
                 "oracle": {}, "deletion": {}, "risk_coverage": {}}
    base_of: dict[tuple[str, str], tuple[dict, list[dict]]] = {}
    for run in a.runs:
        d = results / run
        met = json.loads((d / "metrics.json").read_text())
        tau = (met.get("thresholds") or {}).get("oof_youden")
        if tau is None:
            raise SystemExit(f"{run}: metrics.json has no OOF threshold")
        tests = met.get("tests") or {}
        for part, (_, split, prefix) in PARTS.items():
            rows = {m: load(d, part, m) for m in ("base", "oracle", "deletion", "deletion_control")}
            if rows["base"] is None:
                continue
            decs = {m: decisions(r, tau) for m, r in rows.items() if r is not None}
            aligned(*decs.values())
            groups = np.array([group_of[(split, f)] for f in decs["base"]["file"]])
            base_of[(run, part)] = (decs["base"], rows["base"])
            rep["risk_coverage"].setdefault(run, {})[part] = risk_coverage(decs["base"], rows["base"], groups,
                                                                          a.n_boot, a.seed)
            if "oracle" in decs:
                b = oracle_block(decs["base"], decs["oracle"], groups, a.n_perm, a.seed)
                b["matches_metrics_json"] = effects_match((tests.get("oracle") or {}).get(f"effects_{part}"), b, "oracle")
                rep["oracle"].setdefault(run, {})[part] = b
            if "deletion" in decs and "deletion_control" in decs:
                sub = {m: rows[m] for m in ("base", "deletion", "deletion_control")}
                b = deletion_block(sub, {m: decs[m] for m in sub})
                b["control_placement"] = placement(d, prefix)
                checks = [effects_match((tests.get(m) or {}).get(f"effects_{part}"), b, k)
                          for m, k in (("deletion", "deletion"), ("deletion_control", "control"))]
                b["matches_metrics_json"] = None if None in checks else all(checks)
                rep["deletion"].setdefault(run, {})[part] = b
    if geometry_of:
        released = released_geometry(Path(a.eda))
        fit = [(released[(r.split, r.file)]["cy"], LOCATION[r.cls6]) for r in records
               if r.split == "train" and r.cls6 > 0 and (r.split, r.file) in released]
        cuts = fit_location_cuts(np.array([c for c, _ in fit]), np.array([k for _, k in fit]))
        rep["geometry"] = {"location_cuts": list(cuts), "cuts_fitted_on": f"{len(fit)} OOF fractures (released masks)",
                           "runs": {run: {part: geometry_block(*base_of[(run, part)], geo, released, part, cuts)
                                          for part in PARTS if (run, part) in base_of}
                                    for run, geo in geometry_of.items()}}
    out_dir = results / "analysis" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "interpretability.json").write_text(json.dumps(rep, indent=2, default=float))
    md = write_md(rep)
    (out_dir / "interpretability.md").write_text(md)
    print(md)
    print(f"written: {out_dir / 'interpretability.json'} and interpretability.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
