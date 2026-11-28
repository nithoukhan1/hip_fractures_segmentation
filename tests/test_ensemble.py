"""Checks of scripts/run_ensemble.py (analysis C17) on small hand-made cases.

Run with ``python tests/test_ensemble.py`` or ``pytest tests``. Needs numpy and scipy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import run_ensemble as RE  # noqa: E402


def _row(file, y, score, conf, dice=""):
    r = {"file": file, "cls6_true": str(y), "score": str(score), "dice_top": str(dice)}
    r.update({f"conf_{c}": str(v) for c, v in zip(range(1, 6), conf)})
    return r


def test_ensemble_averages_probabilities_and_keeps_first_members_mask():
    a = [_row("f0", 1, 0.9, [0.6, 0.3, 0, 0, 0], 0.8), _row("f1", 0, 0.1, [0.05, 0.05, 0, 0, 0])]
    b = [_row("f0", 1, 0.7, [0.1, 0.6, 0, 0, 0], 0.5), _row("f1", 0, 0.3, [0.1, 0.2, 0, 0, 0])]
    d = RE.ensemble_decisions([a, b], tau=0.5)
    assert np.allclose(d["score"], [0.8, 0.2])
    # averaged fracture classes for f0: class 1 = .35, class 2 = .45, so class 2 wins although A alone says 1
    assert list(d["pred"]) == [2, 0]
    assert d["dice"][0] == 0.8 and np.isnan(d["dice"][1])


def test_missed_fracture_gets_dice_zero_and_mismatched_members_fail():
    a = [_row("f0", 3, 0.2, [0, 0, 0.2, 0, 0], 0.9)]
    d = RE.ensemble_decisions([a, a], tau=0.5)
    assert d["pred"][0] == 0 and d["dice"][0] == 0.0
    try:
        RE.ensemble_decisions([a, [_row("g0", 3, 0.2, [0, 0, 0.2, 0, 0])]], tau=0.5)
    except SystemExit:
        pass
    else:
        raise AssertionError("members with different femurs must stop the run")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
