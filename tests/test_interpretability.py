"""Checks of scripts/run_interpretability.py and scripts/map_geometry.py (analysis C08) on small hand-made cases.

Run with ``python tests/test_interpretability.py`` or ``pytest tests``. Needs numpy, scipy and OpenCV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import map_geometry as MG  # noqa: E402
import run_interpretability as RI  # noqa: E402


def test_location_cuts_separate_three_ordered_groups():
    cy = np.array([0.30, 0.32, 0.35, 0.36, 0.41, 0.42, 0.45, 0.47, 0.55, 0.60])
    loc = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    t1, t2 = RI.fit_location_cuts(cy, loc)
    assert 0.36 < t1 < 0.41 and 0.47 < t2 < 0.55, (t1, t2)
    assert (RI.locate(cy, (t1, t2)) == loc).all()
    # one neck femur placed among the pertrochanteric ones cannot be fixed by any cut: 9 of 10 stay right
    loc2 = loc.copy()
    loc2[5] = 0
    assert (RI.locate(cy, RI.fit_location_cuts(cy, loc2)) == loc2).sum() == 9


def test_flagged_share_counts_the_least_confident_tenth():
    y = np.zeros(20, int)
    pred = np.zeros(20, int)
    pred[[0, 1, 7]] = 1  # three errors
    conf = np.linspace(0.99, 0.60, 20)
    conf[[0, 1]] = [0.10, 0.20]  # two of them are the two least confident femurs (10% of 20)
    assert abs(RI.flagged_share(y, pred, conf) - 2 / 3) < 1e-12
    assert np.isnan(RI.flagged_share(y, y, conf))  # no errors: undefined


def test_oracle_and_deletion_counts():
    y = np.array([0, 1, 2, 3, 4, 5, 2, 3])
    base = {"y": y, "pred": np.array([0, 1, 2, 4, 4, 0, 2, 3]), "file": np.arange(8)}  # wrong: femurs 3 and 5
    orc = {"y": y, "pred": np.array([0, 1, 2, 3, 4, 0, 1, 3]), "file": np.arange(8)}  # fixes 3, breaks 6
    b = RI.oracle_block(base, orc, np.arange(8), n_perm=50, seed=1)
    assert (b["fractured_wrong_base"], b["fractured_wrong_fixed"]) == (2, 1)
    assert (b["fractured_correct_base"], b["fractured_correct_lost"]) == (5, 1)
    assert (b["correct_base"], b["correct_oracle"]) == (6, 6)

    def rows(scores, conf_true):
        out = []
        for t, s, c in zip(y, scores, conf_true):
            conf = {f"conf_{k}": (c if k == t else (1 - s - c if t > 0 else s) / 4) for k in range(1, 6)}
            out.append({"score": s, **conf})
        return out
    s = np.full(8, 0.9)
    r = {"base": rows(s, np.full(8, 0.8)), "deletion": rows(s - 0.1, np.full(8, 0.2)),
         "deletion_control": rows(s, np.full(8, 0.79))}
    decs = {"base": base, "deletion": {"y": y, "pred": np.array([0, 2, 2, 2, 2, 2, 2, 2])},
            "deletion_control": base}
    d = RI.deletion_block(r, decs)
    assert d["n_fractured"] == 7 and d["correct_base"] == 5
    assert d["lost_deletion"] == 3 and d["lost_control"] == 0  # femurs 1, 4 and 7 (true 2CD: 2 and 6 stay right)
    assert abs(d["true_class_prob"]["deletion_mean"] + 0.6) < 1e-9 and abs(d["fracture_prob"]["deletion_mean"] + 0.1) < 1e-9
    assert d["per_class"]["2CD"] == {"correct_base": 2, "lost_deletion": 0, "lost_control": 0}


def test_map_geometry_matches_the_eda_measures():
    prob = np.zeros((768, 512), np.float32)
    prob[192:384, 128:256] = 1.0  # rows 1/4..1/2 and columns 1/4..1/2 of the 768 x 512 map
    g = MG.geometry(prob)
    assert abs(g["area_frac"] - 0.0625) < 0.002 and g["components"] == 1
    assert abs(g["cy"] - 0.375) < 0.002 and abs(g["cx"] - 0.375) < 0.002
    assert abs(g["bh"] - 0.25) < 0.003 and abs(g["bw"] - 0.25) < 0.003
    empty = MG.geometry(np.zeros((768, 512), np.float32))
    assert empty["area_px"] == 0 and empty["cy"] == "" and empty["components"] == 0


def _run(y, base, deletion, control):
    """Minimal rows and decisions of one run with deletion tests (probabilities: 0.9 on the decided class)."""
    def rows(pred):
        out = []
        for c in pred:
            p = [0.02] * 6
            p[c] = 0.9
            out.append({"score": str(1 - p[0]), **{f"conf_{k}": str(p[k]) for k in range(1, 6)}})
        return out
    files = np.array([f"f{i}" for i in range(len(y))])
    decs = {m: {"file": files, "y": np.asarray(y), "pred": np.asarray(p)}
            for m, p in (("base", base), ("deletion", deletion), ("deletion_control", control))}
    return {"rows": {m: rows(decs[m]["pred"]) for m in decs}, "decs": decs}


def test_deletion_compare_counts_only_femurs_both_get_right():
    y = [1, 2, 3, 4, 5, 2, 0]
    a = _run(y, base=[1, 2, 3, 4, 5, 3, 0], deletion=[2, 2, 2, 2, 5, 3, 0], control=[1, 2, 3, 4, 5, 3, 0])
    b = _run(y, base=[1, 2, 3, 4, 2, 2, 0], deletion=[1, 2, 2, 4, 2, 2, 0], control=[1, 2, 3, 4, 2, 2, 0])
    r = RI.deletion_compare(a, b, np.arange(7), n_boot=200, seed=0)
    # both right at base: femurs 0-3 (femur 4: B wrong; femur 5: A wrong; femur 6 healthy)
    assert r["n_both_correct"] == 4
    assert (r["lost_a"], r["lost_b"]) == (3, 1)  # A loses 0, 2, 3; B loses 2
    assert (r["only_a_lost"], r["only_b_lost"]) == (2, 0)
    assert abs(r["diff"] - 0.5) < 1e-12 and r["control_lost_a"] == 0


def test_section13_verdict_follows_the_rule():
    oof = {"diff": 0.16, "p_midp": 0.001}
    assert RI.section13_verdict(oof, {"diff": 0.13}, {"A": 0.013, "B": 0.014}).startswith("A depends")
    assert RI.section13_verdict(oof, {"diff": -0.02}, {"A": 0.01, "B": 0.01}).startswith("similar")
    assert RI.section13_verdict(oof, {"diff": 0.13}, {"A": 0.06, "B": 0.01}).startswith("inconclusive")
    assert RI.section13_verdict({"diff": 0.03, "p_midp": 0.3}, {"diff": 0.1}, {"A": 0, "B": 0}).startswith("similar")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} interpretability tests passed")
