"""Checks of scripts/run_refstats.py (analyses C13–C16) on small hand-made cases.

Run with ``python tests/test_refstats.py`` or ``pytest tests``. Needs numpy and scipy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import run_refstats as RR  # noqa: E402


def _dec(y, pred, files=None, **extra):
    y, pred = np.asarray(y), np.asarray(pred)
    files = np.array(files if files is not None else [f"f{i}" for i in range(len(y))])
    return {"file": files, "y": y, "pred": pred, **{k: np.asarray(v, float) for k, v in extra.items()}}


def test_g12_compare_counts_only_garden12_femurs():
    y = [1, 1, 1, 1, 2, 0]
    a = _dec(y, [1, 1, 1, 2, 1, 0])  # 3/4 Garden I-II right; the 2CD femur called 2AB does not count
    b = _dec(y, [1, 2, 2, 2, 2, 0])  # 1/4
    r = RR.g12_compare(a, b, np.arange(6), n_boot=200, seed=0)
    assert r["n"] == 4 and r["recall_a"] == 0.75 and r["recall_b"] == 0.25
    assert abs(r["diff"] - 0.5) < 1e-12 and (r["only_a"], r["only_b"]) == (2, 0)


def test_align_reorders_arrays_and_keeps_scalars():
    a = _dec([0, 1], [0, 1], files=["x", "y"])
    b = _dec([1, 0], [1, 0], files=["y", "x"])
    b["tau"] = 0.3
    c = RR.align(a, b)
    assert list(c["file"]) == ["x", "y"] and list(c["y"]) == [0, 1] and c["tau"] == 0.3


def test_subgroup_masks_use_each_parts_own_tertile_and_pair_types():
    y = np.array([1, 2, 3, 4, 5, 2, 0])
    area = np.array([10, 20, 30, 40, 50, 60, 0], float)
    d = _dec(y, y, files=[f"f{i}" for i in range(7)], area_gt=area)
    ptype = {("train", "f0"): RR.FX_PARTNER, ("train", "f1"): RR.SINGLE, ("train", "f6"): RR.HEALTHY_FX_PARTNER}
    m = RR.subgroup_masks(d, "oof", ptype)
    assert list(np.where(m["smallest third of masks"])[0]) == [0, 1]  # lower tertile of 6 fractured areas
    assert m["larger two thirds"].sum() == 4 and not m["larger two thirds"][6]
    assert list(np.where(m["fractured, healthy partner"])[0]) == [0]
    assert list(np.where(m["fractured, single"])[0]) == [1]


def test_paired_values_drops_nan_and_measures_the_mean_difference():
    a = np.array([0.8, 0.7, np.nan, 0.5])
    b = np.array([0.6, 0.7, 0.9, 0.4])
    r = RR.paired_values(a, b, np.arange(4), n_boot=200, seed=0)
    assert r["n"] == 3 and abs(r["diff"] - (0.2 + 0.0 + 0.1) / 3) < 1e-12


def test_add_holm_skips_undefined_p():
    rows = [{"p": 0.01}, {"p": float("nan")}, {"p": 0.04}]
    RR.add_holm(rows)
    assert abs(rows[0]["p_holm"] - 0.02) < 1e-12 and abs(rows[2]["p_holm"] - 0.04) < 1e-12
    assert rows[1]["p_holm"] != rows[1]["p_holm"]


def test_threshold_policy_swap_tests_inherit_the_mode_they_replace(tmp_path=None):
    import tempfile
    root = Path(tmp_path or tempfile.mkdtemp())
    run = root / RR.R7
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"thresholds": {"oof_youden": 0.5}}')
    head = "file,split,fold,model,cls6_true,score,top_cls6,top_conf,area_gt,dice_top\n"
    rows = ["a,train,0,0,1,0.9,1,0.9,10,0.8", "b,train,0,0,0,0.2,1,0.2,0,", "c,train,1,0,2,0.3,2,0.3,10,0.7",
            "d,train,1,0,0,0.1,1,0.1,0,"]
    (run / "per_case_oof_blind.csv").write_text(head + "\n".join(rows) + "\n")
    cache: dict = {}
    t_blind = RR.mode_tau(root, RR.R7, "blind", cache)
    assert 0.2 < t_blind <= 0.3  # own Youden threshold: separates c (0.3, fractured) from b (0.2)
    assert RR.mode_tau(root, RR.R7, "random", cache) == t_blind
    assert RR.mode_tau(root, RR.R7, "fractured", cache) == t_blind
    assert RR.mode_tau(root, RR.R7, "base", cache) == 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
