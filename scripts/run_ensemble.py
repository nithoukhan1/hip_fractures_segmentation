#!/usr/bin/env python3
"""Probability-averaging ensembles of filed runs, from their saved per-femur predictions (no training, no GPU).

For each ensemble, the six class probabilities [P(no fracture), P(2AB) … P(4)] of its members are averaged per femur:
out of fold from per_case_oof.csv, on the test set from per_case_test_ensemble.csv (each member's own 5-model
ensemble). The fracture score is 1 − P(no fracture); its threshold is fitted (Youden) on the ensemble's OOF femurs
only; a femur at or above it gets the most probable fracture class, otherwise "no fracture" (the rule of every run).
The mask (and so Dice) comes from the first member, so an ensemble is "segment with that model, classify with the
ensemble".

Every comparison uses scripts/run_stats.py's paired tests (DeLong, McNemar, randomisation test for macro-F1,
Wilcoxon for Dice) plus Garden I-II recall (exact McNemar on the Garden I-II femurs), with Holm across all the
comparisons, endpoint by endpoint and scope by scope. Writes results/analysis/<name>/ensembles.json and .md.

  python scripts/run_ensemble.py --name C17_ensembles
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
from femurseg.data import DEFAULT_FOLDS_CSV, load_records  # noqa: E402
from run_refstats import g12_compare  # noqa: E402
from run_stats import METRICS, compare, decisions, fmt, intervals, read_cases  # noqa: E402

M0, M0B, SLG, REF = "S10_m0_convnext_unet", "A1_m0_deletion", "A6_slgnet", "R4_refrac"
# Fixed before any ensemble was computed (Analysis/10 §3). E0 is the control: two copies of M0 (two trainings).
ENSEMBLES = {"E0_m0x2": [M0, M0B], "E1_m0_slg": [M0, SLG], "E2_m0_ref": [M0, REF], "E3_m0_slg_ref": [M0, SLG, REF],
             "E4_all": [M0, M0B, SLG, REF]}
PAIRS = [("E0_m0x2", M0), ("E1_m0_slg", M0), ("E2_m0_ref", M0), ("E3_m0_slg_ref", M0), ("E4_all", M0),
         ("E1_m0_slg", "E0_m0x2"), ("E2_m0_ref", "E0_m0x2"), ("E3_m0_slg_ref", "E0_m0x2"), ("E4_all", "E0_m0x2")]
PARTS = {"oof": ("per_case_oof.csv", "train"), "test_ensemble": ("per_case_test_ensemble.csv", "test")}


def probs(rows: list[dict]) -> np.ndarray:
    return np.array([[1 - float(r["score"])] + [float(r[f"conf_{c}"]) for c in range(1, 6)] for r in rows])


def ensemble_decisions(members: list[list[dict]], tau: float) -> dict:
    """members: each member's rows of one part, sorted by file. Mask and Dice from the first member."""
    files = [r["file"] for r in members[0]]
    if any([r["file"] for r in m] != files for m in members[1:]):
        raise SystemExit("ensemble members did not predict the same femurs")
    p = np.mean([probs(m) for m in members], axis=0)
    score = 1 - p[:, 0]
    y = np.array([int(r["cls6_true"]) for r in members[0]])
    pred = np.where(score >= tau, 1 + np.argmax(p[:, 1:], axis=1), 0)
    dice0 = np.array([float(r["dice_top"]) if r.get("dice_top") not in ("", None) else 0.0 for r in members[0]])
    dice = np.where(y > 0, np.where(pred > 0, dice0, 0.0), np.nan)
    return {"file": np.array(files), "y": y, "score": score, "pred": pred, "dice": dice}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--folds-csv", default=str(DEFAULT_FOLDS_CSV))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)
    results = Path(a.results)
    group_of = {(r.split, r.file): r.group for r in load_records(a.folds_csv)}
    runs = sorted({m for ms in ENSEMBLES.values() for m in ms})
    rows = {(run, part): sorted(read_cases(results / run / fn), key=lambda r: r["file"])
            for run in runs for part, (fn, _) in PARTS.items()}
    decs: dict[str, dict[str, dict]] = {}
    taus: dict[str, float] = {}
    for run in runs:
        tau = json.loads((results / run / "metrics.json").read_text())["thresholds"]["oof_youden"]
        taus[run] = tau
        decs[run] = {part: decisions(rows[(run, part)], tau) for part in PARTS}
    for name, members in ENSEMBLES.items():
        oof = ensemble_decisions([rows[(m, "oof")] for m in members], 0.0)
        taus[name] = M.youden_threshold(oof["y"] > 0, oof["score"])
        decs[name] = {part: ensemble_decisions([rows[(m, part)] for m in members], taus[name]) for part in PARTS}
    rep: dict = {"ensembles": ENSEMBLES, "thresholds": taus, "intervals": {}, "comparisons": {}, "g12": {}}
    groups = {part: np.array([group_of[(split, f)] for f in decs[M0][part]["file"]])
              for part, (_, split) in PARTS.items()}
    for name in [M0, M0B, SLG, REF, *ENSEMBLES]:
        rep["intervals"][name] = {part: intervals(decs[name][part], groups[part], a.n_boot, a.seed) for part in PARTS}
    for ea, eb in PAIRS:
        key = f"{ea} vs {eb}"
        rep["comparisons"][key], rep["g12"][key] = {}, {}
        for part in PARTS:
            da, db = decs[ea][part], decs[eb][part]
            assert np.array_equal(da["file"], db["file"])
            rep["comparisons"][key][part] = compare(da, db, groups[part], a.n_boot, a.n_perm, a.seed)
            rep["g12"][key][part] = g12_compare(da, db, groups[part], a.n_boot, a.seed)
    # Holm across all comparisons, per endpoint and scope
    for part in PARTS:
        endpoints = {n for k in rep["comparisons"] for n in rep["comparisons"][k][part]}
        for n in endpoints:
            cell = [rep["comparisons"][k][part][n] for k in rep["comparisons"] if n in rep["comparisons"][k][part]]
            for r, adj in zip(cell, S.holm([r["p"] for r in cell])):
                r["p_holm_family"] = float(adj)
        cell = [rep["g12"][k][part] for k in rep["g12"]]
        for r, adj in zip(cell, S.holm([r["p"] for r in cell])):
            r["p_holm_family"] = float(adj)
    out = results / "analysis" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "ensembles.json").write_text(json.dumps(rep, indent=1, default=float))
    L = ["# Ensembles from saved predictions", "",
         "Members' 6-class probabilities averaged per femur; threshold (Youden) fitted on each ensemble's OOF femurs; "
         "mask and Dice from the first member (M0). Holm (family) across all 9 comparisons, per endpoint and scope.", "",
         "| Model | Members | Threshold |", "|---|---|---|"]
    L += [f"| {n} | {' + '.join(ENSEMBLES.get(n, [n]))} | {taus[n]:.5f} |" for n in rep["intervals"]]
    for part in PARTS:
        L += ["", f"## {part}", "", "| Model | Correct | macro-F1 (95% CI) | Dice fractured | Garden I-II | AUC |",
              "|---|---|---|---|---|---|"]
        for n, iv in rep["intervals"].items():
            b = iv[part]
            e = b["accuracy_6class_exact"]
            L.append(f"| {n} | {e['correct']}/{e['n']} | {fmt(b['macro_f1'])} ({fmt(b['macro_f1'], 'lo')}–"
                     f"{fmt(b['macro_f1'], 'hi')}) | {fmt(b['dice_fractured_mean'])} | {fmt(b['garden12_recall'])} | "
                     f"{fmt(b['auc_fracture'])} |")
        L += ["", "| A vs B | macro-F1 diff; p; Holm | accuracy diff; p; Holm | Dice diff; p; Holm | Garden I-II diff "
              "(only A / only B); p; Holm |", "|---|---|---|---|---|"]
        for key in rep["comparisons"]:
            c, g = rep["comparisons"][key][part], rep["g12"][key][part]
            cell = lambda n: (f"{c[n].get('diff', c[n].get('mean_diff')):+.3f}; {c[n]['p']:.3f}; "  # noqa: E731
                              f"{c[n]['p_holm_family']:.3f}" if n in c else "–")
            L.append(f"| {key} | {cell('macro_f1')} | {cell('accuracy_6class')} | {cell('dice_fractured')} | "
                     f"{g['diff']:+.3f} ({g['only_a']} / {g['only_b']}); {g['p']:.3f}; {g['p_holm_family']:.3f} |")
    (out / "ensembles.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
