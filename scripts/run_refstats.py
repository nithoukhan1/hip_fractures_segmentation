#!/usr/bin/env python3
"""Paper B analyses that scripts/run_stats.py does not cover (Analysis/08 §2.5, §2.6, §5; analyses C13–C16).

Reads only filed run folders and the fixed pair table of step B-1 (results/R00_reference_bank_r2/pairs.csv); it
never edits a run folder. Every interval is a cluster bootstrap over x-rays (2,000 resamples, seed 2026). Output goes
to results/analysis/<name>/<command>.json and .md.

Commands
  g12        Garden I-II recall, the second primary endpoint of Paper B, for the pairs given (difference A − B):
             exact McNemar on the Garden I-II femurs (right = predicted 2AB), a bootstrap interval of the recall
             difference, and Holm across the pairs, scope by scope. --folds refits each run's threshold on those
             OOF femurs only (the folds 1–4 check).
  subgroups  C13's subgroups, Dice on fractured femurs (missed = 0): the smallest third of released masks (each
             part's own lower tertile of mask area, since the test masks are smaller, C01) and fractured femurs with
             a healthy partner vs single femurs. Wilcoxon (Pratt) per pair, Holm across the pairs per subgroup and
             scope.
  reftests   The reference tests every R run wrote (self, random; R7 also blind, retrieved, fractured), each against
             the mode it replaces: Dice, Garden I-II recall, macro-F1, the change of P(fracture) on fractured and on
             healthy femurs, healthy femurs that turn fractured, and fractures that are lost. Wilcoxon for Dice,
             Holm across the runs per mode and scope.
  partner    C14: R7 (the patient's own other hip) against R3 and R4, RQ-B2 and RQ-B3 of Analysis/08 §2.5.

Thresholds. A mode used as an operating point (R7 blind and retrieved) gets its own Youden threshold, fitted on its
own OOF predictions (§2.5, rule 4 of §3). A swap test is read at the threshold of the mode it replaces (R1–R6 self
and random: the run's base threshold; R7 random and fractured: blind's), so that it shows what the swap does to the
same decision rule.

Examples
  python scripts/run_refstats.py g12 S10_m0_convnext_unet R2_random_ref R3_nearest_ref R4_refrac \\
      --pairs R2_random_ref:S10_m0_convnext_unet,R3_nearest_ref:R2_random_ref --name C13_paperB_reference
  python scripts/run_refstats.py subgroups S10_m0_convnext_unet R2_random_ref --pairs R2_random_ref:S10_m0_convnext_unet \\
      --name C13_paperB_reference
  python scripts/run_refstats.py reftests R1_aasn R2_random_ref R7_partner --name C13_paperB_reference
  python scripts/run_refstats.py partner --name C14_paperB_partner
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
sys.path.insert(0, str(REPO / "scripts"))
from femurseg import metrics as M  # noqa: E402
from femurseg import stats as S  # noqa: E402
from femurseg.data import DEFAULT_FOLDS_CSV, load_records  # noqa: E402
from run_stats import decisions, parse_pairs, read_cases  # noqa: E402

PARTS = {"oof": ("per_case_oof", "train"), "test_ensemble": ("per_case_test_ensemble", "test")}
PAIRS_CSV = REPO / "results" / "R00_reference_bank_r2" / "pairs.csv"
FX_PARTNER, HEALTHY_FX_PARTNER, SINGLE = "fractured, healthy partner", "healthy, fractured partner", "single"
# mode -> the mode it replaces (its comparison and, for a swap test, its threshold)
SWAP_BASE = {"self": "base", "random": "base", "retrieved": "blind", "blind": "base", "fractured": "blind"}
R7, R7_SWAP_BASE = "R7_partner", {"random": "blind", "fractured": "blind", "retrieved": "blind", "blind": "base"}
OWN_THRESHOLD_MODES = ("blind", "retrieved")  # R7's label-free operating modes (§2.5)


# ---------------------------------------------------------------------------------------------- loading


def load_rows(results: Path, run: str, part: str, mode: str = "base", folds: list[int] | None = None):
    f = results / run / f"{PARTS[part][0]}{'' if mode == 'base' else '_' + mode}.csv"
    if not f.exists():
        return None
    rows = sorted(read_cases(f), key=lambda r: r["file"])
    if folds:
        rows = [r for r in rows if int(float(r["fold"])) in folds]
    return rows


def youden(rows: list[dict]) -> float:
    return M.youden_threshold([int(r["cls6_true"]) > 0 for r in rows], [float(r["score"]) for r in rows])


def base_tau(results: Path, run: str, folds: list[int] | None = None) -> float:
    if folds:
        return youden(load_rows(results, run, "oof", "base", folds))
    return json.loads((results / run / "metrics.json").read_text())["thresholds"]["oof_youden"]


def mode_tau(results: Path, run: str, mode: str, cache: dict) -> float:
    """The threshold a mode is read at (see the module docstring)."""
    key = (run, mode)
    if key not in cache:
        if mode == "base":
            cache[key] = base_tau(results, run)
        elif run == R7 and mode in OWN_THRESHOLD_MODES:
            cache[key] = youden(load_rows(results, run, "oof", mode))
        else:
            cache[key] = mode_tau(results, run, (R7_SWAP_BASE if run == R7 else SWAP_BASE)[mode], cache)
    return cache[key]


def with_probs(rows: list[dict], tau: float) -> dict:
    d = decisions(rows, tau)
    d["p_fracture"] = d["score"]
    d["area_gt"] = np.array([float(r["area_gt"]) if r.get("area_gt") not in ("", None) else np.nan for r in rows])
    return d


def align(da: dict, db: dict) -> dict:
    """db reordered to da's femurs."""
    order = {f: i for i, f in enumerate(db["file"])}
    if set(order) != set(da["file"]):
        raise SystemExit("the two predictions do not cover the same femurs")
    idx = np.array([order[f] for f in da["file"]])
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in db.items()}


def pair_types(path: Path = PAIRS_CSV) -> dict[tuple[str, str], str]:
    with open(path, newline="") as fh:
        return {(r["split"], r["file"]): r["pair_type"] for r in csv.DictReader(fh)}


# ------------------------------------------------------------------------------------------- statistics


def recall_g12(y, pred) -> float:
    m = np.asarray(y) == 1
    return float(np.mean(np.asarray(pred)[m] == 1)) if m.any() else float("nan")


def nanmean_stat(y, v) -> float:
    v = np.asarray(v, float)
    return float(np.nanmean(v)) if (~np.isnan(v)).any() else float("nan")


def g12_compare(da: dict, db: dict, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    """Garden I-II recall of A minus B on the same femurs: exact McNemar and a cluster-bootstrap interval."""
    m = da["y"] == 1
    ca, cb = da["pred"][m] == 1, db["pred"][m] == 1
    mc = S.mcnemar_exact(ca, cb)
    bd = S.bootstrap_diff(recall_g12, da["y"][m], da["pred"][m], db["pred"][m], n_boot=n_boot, groups=groups[m],
                          seed=seed)
    return {"n": int(m.sum()), "recall_a": float(ca.mean()), "recall_b": float(cb.mean()), "diff": bd["diff"],
            "lo": bd["lo"], "hi": bd["hi"], "only_a": mc["b"], "only_b": mc["c"], "p": mc["p"],
            "p_midp": mc["p_midp"]}


def mean_ci(v: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    keep = ~np.isnan(v)
    if not keep.any():
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    ci = S.bootstrap_ci(lambda x: float(np.mean(x)), v[keep], n_boot=n_boot, groups=groups[keep], seed=seed)
    return {"n": int(keep.sum()), "mean": float(np.mean(v[keep])), "lo": ci["lo"], "hi": ci["hi"]}


def paired_values(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    """Mean of a − b with a cluster-bootstrap interval, and Wilcoxon signed-rank (Pratt)."""
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b, g = a[keep], b[keep], groups[keep]
    if not keep.any():
        return {"n": 0, "mean_a": float("nan"), "mean_b": float("nan"), "diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p": float("nan")}
    bd = S.bootstrap_diff(nanmean_stat, np.zeros(len(a)), a, b, n_boot=n_boot, groups=g, seed=seed)
    w = S.wilcoxon_paired(a, b)
    return {"n": int(len(a)), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": bd["diff"],
            "lo": bd["lo"], "hi": bd["hi"], "p": w["p"]}


def add_holm(rows: list[dict], key: str = "p") -> None:
    ps = [r[key] for r in rows]
    ok = [i for i, p in enumerate(ps) if p == p]
    adj = S.holm([ps[i] for i in ok]) if ok else []
    for r in rows:
        r["p_holm"] = float("nan")
    for i, v in zip(ok, adj):
        rows[i]["p_holm"] = float(v)


def subgroup_masks(da: dict, part: str, ptype: dict) -> dict[str, np.ndarray]:
    split = PARTS[part][1]
    fx = da["y"] > 0
    cut = np.quantile(da["area_gt"][fx], 1 / 3)
    pt = np.array([ptype.get((split, f), "") for f in da["file"]])
    return {"smallest third of masks": fx & (da["area_gt"] <= cut),
            "larger two thirds": fx & (da["area_gt"] > cut),
            "fractured, healthy partner": fx & (pt == FX_PARTNER),
            "fractured, single": fx & (pt == SINGLE)}


# ----------------------------------------------------------------------------------------------- commands


def cmd_g12(a, results: Path, group_of: dict) -> dict:
    folds = [int(v) for v in a.folds.split(",") if v.strip()]
    pairs = parse_pairs(a.runs, a.pairs)
    parts = ["oof"] if folds else list(PARTS)
    out = {"command": "g12", "runs": a.runs, "pairs": [f"{x}:{y}" for x, y in pairs], "folds": folds or "all",
           "thresholds": {r: base_tau(results, r, folds) for r in a.runs}, "scopes": {}}
    for part in parts:
        decs = {r: with_probs(load_rows(results, r, part, "base", folds), out["thresholds"][r]) for r in a.runs}
        rows = []
        for ra, rb in pairs:
            da, db = decs[ra], align(decs[ra], decs[rb])
            groups = np.array([group_of[(PARTS[part][1], f)] for f in da["file"]])
            rows.append({"a": ra, "b": rb, **g12_compare(da, db, groups, a.n_boot, a.seed)})
        add_holm(rows)
        out["scopes"][part] = {"recall": {r: recall_g12(d["y"], d["pred"]) for r, d in decs.items()},
                               "n": int((next(iter(decs.values()))["y"] == 1).sum()), "pairs": rows}
    return out


def cmd_subgroups(a, results: Path, group_of: dict) -> dict:
    pairs = parse_pairs(a.runs, a.pairs)
    ptype = pair_types()
    out = {"command": "subgroups", "runs": a.runs, "pairs": [f"{x}:{y}" for x, y in pairs], "scopes": {}}
    for part in PARTS:
        decs = {r: with_probs(load_rows(results, r, part), base_tau(results, r)) for r in a.runs}
        ref = decs[a.runs[0]]
        decs = {r: align(ref, d) for r, d in decs.items()}
        groups = np.array([group_of[(PARTS[part][1], f)] for f in ref["file"]])
        masks = subgroup_masks(ref, part, ptype)
        fx = ref["y"] > 0
        sc = {"area_cut": float(np.quantile(ref["area_gt"][fx], 1 / 3)), "subgroups": {}}
        for name, m in masks.items():
            per_run = {r: mean_ci(np.where(m, d["dice"], np.nan), groups, a.n_boot, a.seed) for r, d in decs.items()}
            rows = [{"a": ra, "b": rb, **paired_values(decs[ra]["dice"][m], decs[rb]["dice"][m], groups[m],
                                                        a.n_boot, a.seed)} for ra, rb in pairs]
            add_holm(rows)
            sc["subgroups"][name] = {"n": int(m.sum()), "dice": per_run, "pairs": rows}
        out["scopes"][part] = sc
    return out


def mode_effects(base: dict, mode: dict, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    y = base["y"]
    fx, hl = y > 0, y == 0
    dp = mode["p_fracture"] - base["p_fracture"]
    dice = paired_values(mode["dice"][fx], base["dice"][fx], groups[fx], n_boot, seed)
    return {"n": int(len(y)), "threshold_base": base.get("tau"), "threshold_mode": mode.get("tau"),
            "dice_base": dice["mean_b"], "dice_mode": dice["mean_a"], "dice_diff": dice["diff"],
            "dice_lo": dice["lo"], "dice_hi": dice["hi"], "p": dice["p"],
            "g12_base": recall_g12(y, base["pred"]), "g12_mode": recall_g12(y, mode["pred"]),
            "macro_f1_base": M.multiclass_metrics(y, base["pred"])["macro_f1"],
            "macro_f1_mode": M.multiclass_metrics(y, mode["pred"])["macro_f1"],
            "dp_fractured_mean": float(dp[fx].mean()), "dp_healthy_mean": float(dp[hl].mean()),
            "healthy_turned_fractured": int((hl & (base["pred"] == 0) & (mode["pred"] > 0)).sum()),
            "healthy_correct_base": int((hl & (base["pred"] == 0)).sum()),
            "fractures_lost": int((fx & (base["pred"] > 0) & (mode["pred"] == 0)).sum()),
            "fractures_found_base": int((fx & (base["pred"] > 0)).sum())}


def load_mode(results: Path, run: str, part: str, mode: str, cache: dict) -> dict | None:
    rows = load_rows(results, run, part, mode)
    if rows is None:
        return None
    tau = mode_tau(results, run, mode, cache)
    d = with_probs(rows, tau)
    d["tau"] = tau
    return d


def cmd_reftests(a, results: Path, group_of: dict) -> dict:
    cache: dict = {}
    out = {"command": "reftests", "runs": a.runs, "scopes": {}}
    for part in PARTS:
        by_mode: dict[str, list[dict]] = {}
        for run in a.runs:
            swap = R7_SWAP_BASE if run == R7 else SWAP_BASE
            modes = ("blind", "retrieved", "random", "fractured") if run == R7 else ("self", "random")
            for mode in modes:
                dm = load_mode(results, run, part, mode, cache)
                db = load_mode(results, run, part, swap[mode], cache)
                if dm is None or db is None:
                    continue
                dm = align(db, dm)
                groups = np.array([group_of[(PARTS[part][1], f)] for f in db["file"]])
                by_mode.setdefault(mode, []).append({"run": run, "against": swap[mode],
                                                     **mode_effects(db, dm, groups, a.n_boot, a.seed)})
        for rows in by_mode.values():
            add_holm(rows)
        out["scopes"][part] = by_mode
    return out


def cmd_partner(a, results: Path, group_of: dict) -> dict:
    cache: dict = {}
    ptype = pair_types()
    out = {"command": "partner", "thresholds": {}, "scopes": {}}
    for part in PARTS:
        split = PARTS[part][1]
        r7 = {m: load_mode(results, R7, part, m, cache) for m in ("base", "blind", "retrieved", "random", "fractured")}
        ref = r7["blind"]
        r7 = {m: align(ref, d) for m, d in r7.items()}
        others = {run: {m: align(ref, load_mode(results, run, part, m, cache)) for m in ("base", "random")}
                  for run in ("R3_nearest_ref", "R4_refrac")}
        groups = np.array([group_of[(split, f)] for f in ref["file"]])
        pt = np.array([ptype.get((split, f), "") for f in ref["file"]])
        y = ref["y"]
        fxp, hfp = (y > 0) & (pt == FX_PARTNER), (y == 0) & (pt == HEALTHY_FX_PARTNER)
        sc: dict = {"n_fractured_healthy_partner": int(fxp.sum()), "n_healthy_fractured_partner": int(hfp.sum())}

        # overall, per mode (label-free modes at their own thresholds)
        sc["modes"] = {m: {"threshold": d["tau"], "correct": int((d["pred"] == y).sum()), "n": int(len(y)),
                           "macro_f1": M.multiclass_metrics(y, d["pred"])["macro_f1"],
                           "dice": nanmean_stat(None, d["dice"][y > 0]), "g12": recall_g12(y, d["pred"]),
                           "fp": int(((y == 0) & (d["pred"] > 0)).sum()), "fn": int(((y > 0) & (d["pred"] == 0)).sum())}
                       for m, d in r7.items()}

        # RQ-B2: own hip vs retrieved, fractured femurs with a healthy partner
        rq2 = [{"a": "R7 blind (partner)", "b": "R7 retrieved",
                **paired_values(r7["blind"]["dice"][fxp], r7["retrieved"]["dice"][fxp], groups[fxp], a.n_boot, a.seed)},
               {"a": "R7 blind (partner)", "b": "R3 (retrieved, trained so)",
                **paired_values(r7["blind"]["dice"][fxp], others["R3_nearest_ref"]["base"]["dice"][fxp], groups[fxp],
                                a.n_boot, a.seed)}]
        add_holm(rq2)
        g12_rows = [{"a": "R7 blind", "b": "R7 retrieved",
                     **g12_compare(r7["blind"], r7["retrieved"], groups, a.n_boot, a.seed)},
                    {"a": "R7 blind", "b": "R3", **g12_compare(r7["blind"], others["R3_nearest_ref"]["base"], groups,
                                                               a.n_boot, a.seed)}]
        add_holm(g12_rows)
        sc["rq_b2_dice"], sc["rq_b2_g12_all_femurs"] = rq2, g12_rows

        # RQ-B3.1: healthy femurs whose partner is fractured, blind (fractured partner) vs retrieved
        pb, pr = r7["blind"]["p_fracture"][hfp], r7["retrieved"]["p_fracture"][hfp]
        fpb, fpr = r7["blind"]["pred"][hfp] > 0, r7["retrieved"]["pred"][hfp] > 0
        sc["rq_b3_1"] = {"n": int(hfp.sum()), "p_fracture_blind_mean": float(pb.mean()),
                         "p_fracture_retrieved_mean": float(pr.mean()),
                         "p_fracture_blind_median": float(np.median(pb)), "p_fracture_retrieved_median": float(np.median(pr)),
                         "wilcoxon_p": S.wilcoxon_paired(pb, pr)["p"],
                         "fp_blind": int(fpb.sum()), "fp_retrieved": int(fpr.sum()),
                         "mcnemar_fp": S.mcnemar_exact(~fpb, ~fpr),
                         "fp_base_leak": int((r7["base"]["pred"][hfp] > 0).sum())}

        # RQ-B3.2: swap to a random healthy femur, fractured femurs with a healthy partner; R7 vs R3 and R4
        swap = {}
        for name, before, after in (("R7 (partner → random)", r7["blind"], r7["random"]),
                                    ("R3 (nearest → random)", others["R3_nearest_ref"]["base"],
                                     others["R3_nearest_ref"]["random"]),
                                    ("R4 (3 nearest → random)", others["R4_refrac"]["base"], others["R4_refrac"]["random"])):
            swap[name] = {"dp": after["p_fracture"][fxp] - before["p_fracture"][fxp],
                          "ddice": after["dice"][fxp] - before["dice"][fxp],
                          "lost": int(((before["pred"] > 0) & (after["pred"] == 0))[fxp].sum())}
        sc["rq_b3_2"] = {"per_run": {k: {"dp_mean": float(v["dp"].mean()), "dp_median": float(np.median(v["dp"])),
                                         "ddice_mean": float(np.nanmean(v["ddice"])), "fractures_lost": v["lost"]}
                                     for k, v in swap.items()}}
        names = list(swap)
        cmp_rows = []
        for other in names[1:]:
            for metric in ("dp", "ddice"):
                cmp_rows.append({"a": names[0], "b": other, "metric": metric,
                                 **paired_values(swap[names[0]][metric], swap[other][metric], groups[fxp], a.n_boot,
                                                 a.seed)})
        for metric in ("dp", "ddice"):
            add_holm([r for r in cmp_rows if r["metric"] == metric])
        sc["rq_b3_2"]["comparisons"] = cmp_rows

        # RQ-B3.3: a fractured reference, against blind
        fx, hl = y > 0, y == 0
        dp = r7["fractured"]["p_fracture"] - r7["blind"]["p_fracture"]
        sc["rq_b3_3"] = {"dp_fractured_mean": float(dp[fx].mean()), "dp_healthy_mean": float(dp[hl].mean()),
                         "dice": paired_values(r7["fractured"]["dice"][fx], r7["blind"]["dice"][fx], groups[fx],
                                               a.n_boot, a.seed),
                         "fractures_lost": int((fx & (r7["blind"]["pred"] > 0) & (r7["fractured"]["pred"] == 0)).sum()),
                         "healthy_turned_fractured": int((hl & (r7["blind"]["pred"] == 0) &
                                                          (r7["fractured"]["pred"] > 0)).sum())}

        # the leak: base (label-chosen partner) vs blind, per pair type
        sc["leak"] = {}
        for name, m in (("fractured, healthy partner", fxp), ("healthy, fractured partner", hfp),
                        ("single", pt == SINGLE)):
            sc["leak"][name] = {"n": int(m.sum()),
                                "correct_base": int((r7["base"]["pred"][m] == y[m]).sum()),
                                "correct_blind": int((r7["blind"]["pred"][m] == y[m]).sum()),
                                "p_fracture_base": float(r7["base"]["p_fracture"][m].mean()),
                                "p_fracture_blind": float(r7["blind"]["p_fracture"][m].mean())}
        out["scopes"][part] = sc
    out["thresholds"] = {f"{r} {m}": t for (r, m), t in cache.items()}
    return out


# ------------------------------------------------------------------------------------------------ report


def f3(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pf(p) -> str:
    return "n/a" if p is None or p != p else ("<.001" if p < 0.001 else f"{p:.3f}")


def ci(r: dict, d="diff") -> str:
    return f"{r[d]:+.3f} ({r['lo']:+.3f} to {r['hi']:+.3f})"


def md_g12(o: dict) -> str:
    L = ["# Garden I-II recall (a primary endpoint of Paper B)", "",
         f"Runs: {', '.join(o['runs'])}. Folds: {o['folds']}. Decisions at each run's OOF threshold"
         f"{' refitted on these folds' if o['folds'] != 'all' else ''}. Right = predicted 2AB. Exact McNemar "
         "(mid-p also shown); difference with a cluster-bootstrap 95% interval; Holm across the pairs, per scope.", ""]
    for part, sc in o["scopes"].items():
        L += [f"## {part} ({sc['n']} Garden I-II femurs)", "",
              "| Run | Recall |", "|---|---|"] + [f"| {r} | {f3(v)} |" for r, v in sc["recall"].items()] + [""]
        L += ["| A vs B | Recall A | Recall B | Difference (95% CI) | Only A right | Only B right | p | mid-p | p (Holm) |",
              "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['a']} vs {r['b']} | {f3(r['recall_a'])} | {f3(r['recall_b'])} | {ci(r)} | {r['only_a']} | "
              f"{r['only_b']} | {pf(r['p'])} | {pf(r['p_midp'])} | {pf(r['p_holm'])} |" for r in sc["pairs"]]
        L.append("")
    return "\n".join(L)


def md_subgroups(o: dict) -> str:
    L = ["# Subgroups: Dice on fractured femurs (missed = 0)", "",
         "Smallest third: each part's own lower tertile of released mask area among fractured femurs. Paired: the "
         "fractured femur of a two-femur x-ray whose other femur is healthy. Wilcoxon (Pratt); Holm across the pairs, "
         "per subgroup and scope.", ""]
    for part, sc in o["scopes"].items():
        L += [f"## {part} (area cut {sc['area_cut']:.0f} px)", ""]
        runs = list(next(iter(sc["subgroups"].values()))["dice"])
        L += ["| Subgroup | n | " + " | ".join(runs) + " |", "|---|---|" + "---|" * len(runs)]
        for name, g in sc["subgroups"].items():
            L.append(f"| {name} | {g['n']} | " + " | ".join(
                f"{f3(g['dice'][r]['mean'])} ({f3(g['dice'][r]['lo'])}–{f3(g['dice'][r]['hi'])})" for r in runs) + " |")
        L += ["", "| Subgroup | A vs B | Difference (95% CI) | p | p (Holm) |", "|---|---|---|---|---|"]
        for name, g in sc["subgroups"].items():
            L += [f"| {name} | {r['a']} vs {r['b']} | {ci(r)} | {pf(r['p'])} | {pf(r['p_holm'])} |" for r in g["pairs"]]
        L.append("")
    return "\n".join(L)


def md_reftests(o: dict) -> str:
    L = ["# Reference tests: each mode against the mode it replaces", "",
         "Dice on fractured femurs (missed = 0), Wilcoxon (Pratt), Holm across the runs per mode and scope. "
         "ΔP = mean change of P(fracture). Healthy → fractured: of the healthy femurs right in the replaced mode. "
         "Lost: of the fractures found in the replaced mode.", ""]
    for part, by_mode in o["scopes"].items():
        L += [f"## {part}", "",
              "| Mode | Run | Against | Dice before → after (diff, 95% CI) | p (Holm) | Garden I-II | macro-F1 | "
              "ΔP fractured | ΔP healthy | Healthy → fractured | Fractures lost |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for mode, rows in by_mode.items():
            for r in rows:
                L.append(f"| {mode} | {r['run']} | {r['against']} | {f3(r['dice_base'])} → {f3(r['dice_mode'])} "
                         f"({r['dice_diff']:+.3f}, {r['dice_lo']:+.3f} to {r['dice_hi']:+.3f}) | {pf(r['p_holm'])} | "
                         f"{f3(r['g12_base'])} → {f3(r['g12_mode'])} | {f3(r['macro_f1_base'])} → "
                         f"{f3(r['macro_f1_mode'])} | {r['dp_fractured_mean']:+.3f} | {r['dp_healthy_mean']:+.3f} | "
                         f"{r['healthy_turned_fractured']}/{r['healthy_correct_base']} | "
                         f"{r['fractures_lost']}/{r['fractures_found_base']} |")
        L.append("")
    return "\n".join(L)


def md_partner(o: dict) -> str:
    L = ["# C14: R7, the patient's own other hip (RQ-B2, RQ-B3; Analysis/08 §2.5)", "",
         "Thresholds: base = R7's own OOF threshold (training policy, uses the partner's label: the leak); blind and "
         "retrieved = each refitted on its own OOF predictions; random and fractured = blind's (swap tests). R3 and "
         "R4: their base thresholds.", "",
         "| Mode | Threshold |", "|---|---|"] + [f"| {k} | {v:.6f} |" for k, v in o["thresholds"].items()] + [""]
    for part, sc in o["scopes"].items():
        L += [f"## {part}", "", "### R7 per mode", "",
              "| Mode | Correct | macro-F1 | Dice | Garden I-II | False positives | Missed fractures |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {m} | {v['correct']}/{v['n']} | {f3(v['macro_f1'])} | {f3(v['dice'])} | {f3(v['g12'])} | {v['fp']} | "
              f"{v['fn']} |" for m, v in sc["modes"].items()]
        L += ["", f"### RQ-B2: own hip vs retrieved ({sc['n_fractured_healthy_partner']} fractured femurs with a "
                  "healthy partner), Dice", "",
              "| A vs B | Dice A | Dice B | Difference (95% CI) | p | p (Holm) |", "|---|---|---|---|---|---|"]
        L += [f"| {r['a']} vs {r['b']} | {f3(r['mean_a'])} | {f3(r['mean_b'])} | {ci(r)} | {pf(r['p'])} | "
              f"{pf(r['p_holm'])} |" for r in sc["rq_b2_dice"]]
        L += ["", "Garden I-II recall, all femurs:", "",
              "| A vs B | Recall A | Recall B | Difference (95% CI) | p | p (Holm) |", "|---|---|---|---|---|---|"]
        L += [f"| {r['a']} vs {r['b']} | {f3(r['recall_a'])} | {f3(r['recall_b'])} | {ci(r)} | {pf(r['p'])} | "
              f"{pf(r['p_holm'])} |" for r in sc["rq_b2_g12_all_femurs"]]
        b31 = sc["rq_b3_1"]
        L += ["", f"### RQ-B3.1: healthy femurs whose partner is fractured ({b31['n']})", "",
              f"- P(fracture), mean (median): blind {b31['p_fracture_blind_mean']:.4f} "
              f"({b31['p_fracture_blind_median']:.2e}), retrieved {b31['p_fracture_retrieved_mean']:.4f} "
              f"({b31['p_fracture_retrieved_median']:.2e}); Wilcoxon p {pf(b31['wilcoxon_p'])}.",
              f"- False positives: blind {b31['fp_blind']}, retrieved {b31['fp_retrieved']} (McNemar p "
              f"{pf(b31['mcnemar_fp']['p'])}); base (leak) {b31['fp_base_leak']}.", "",
              f"### RQ-B3.2: swap to a random healthy femur ({sc['n_fractured_healthy_partner']} fractured femurs "
              "with a healthy partner)", "",
              "| Run (swap) | ΔP(fracture) mean | median | ΔDice mean | Fractures lost |", "|---|---|---|---|---|"]
        L += [f"| {k} | {v['dp_mean']:+.4f} | {v['dp_median']:+.2e} | {v['ddice_mean']:+.4f} | {v['fractures_lost']} |"
              for k, v in sc["rq_b3_2"]["per_run"].items()]
        L += ["", "| A vs B | Change of | Difference of the changes (95% CI) | p | p (Holm) |", "|---|---|---|---|---|"]
        L += [f"| {r['a']} vs {r['b']} | {'P(fracture)' if r['metric'] == 'dp' else 'Dice'} | {ci(r)} | {pf(r['p'])} | "
              f"{pf(r['p_holm'])} |" for r in sc["rq_b3_2"]["comparisons"]]
        b33 = sc["rq_b3_3"]
        L += ["", "### RQ-B3.3: a fractured reference (against blind)", "",
              f"- ΔP(fracture): fractured femurs {b33['dp_fractured_mean']:+.3f}, healthy femurs "
              f"{b33['dp_healthy_mean']:+.4f}.",
              f"- Dice {f3(b33['dice']['mean_b'])} → {f3(b33['dice']['mean_a'])} ({ci(b33['dice'])}, p "
              f"{pf(b33['dice']['p'])}).",
              f"- Fractures lost {b33['fractures_lost']}; healthy femurs turned fractured "
              f"{b33['healthy_turned_fractured']}.", "",
              "### The leak: base (partner chosen by its label) vs blind", "",
              "| Femurs | n | Correct base | Correct blind | P(fracture) base | P(fracture) blind |",
              "|---|---|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['correct_base']} | {v['correct_blind']} | {v['p_fracture_base']:.4f} | "
              f"{v['p_fracture_blind']:.4f} |" for k, v in sc["leak"].items()]
        L.append("")
    return "\n".join(L)


COMMANDS = {"g12": (cmd_g12, md_g12), "subgroups": (cmd_subgroups, md_subgroups),
            "reftests": (cmd_reftests, md_reftests), "partner": (cmd_partner, md_partner)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("runs", nargs="*", help="run names (not used by partner)")
    ap.add_argument("--pairs", default="", help="'A:B,C:D' (difference A − B); g12 and subgroups")
    ap.add_argument("--folds", default="", help="g12 only: OOF femurs of these folds, thresholds refitted")
    ap.add_argument("--name", required=True, help="output folder under results/analysis/")
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--folds-csv", default=str(DEFAULT_FOLDS_CSV))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)
    if a.command in ("g12", "subgroups") and not a.pairs:
        raise SystemExit(f"{a.command} needs --pairs")
    results = Path(a.results)
    group_of = {(r.split, r.file): r.group for r in load_records(a.folds_csv)}
    run, md = COMMANDS[a.command]
    rep = run(a, results, group_of)
    rep.update(n_boot=a.n_boot, seed=a.seed)
    out = results / "analysis" / a.name
    out.mkdir(parents=True, exist_ok=True)
    stem = a.command + (f"_folds_{a.folds.replace(',', '-')}" if a.folds else "")
    (out / f"{stem}.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / f"{stem}.md").write_text(md(rep) + "\n")
    print(f"written: {out / stem}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
