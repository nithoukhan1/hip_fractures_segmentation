"""Checks of scripts/run_breakdown.py on hand-made per-case rows (needs numpy and scipy).

Run with ``python tests/test_breakdown.py`` or pytest.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_breakdown", REPO / "scripts" / "run_breakdown.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)  # also puts src/ on sys.path
from femurseg import reference as R  # noqa: E402

NAN = float("nan")


def row(t: int, p: int, dice: float = NAN, a_gt: float = 0.0, a_top: float = 0.0, model: str = "0") -> dict:
    """One femur: true class t, predicted class p (0 = no fracture called; threshold 0.5 below)."""
    return {"file": "", "model": model, "fold": 0, "cls6_true": t, "top_cls6": p if p else (t or 1),
            "top_conf": 0.9 if p else 0.1, "score": 0.9 if p else 0.1, "dice_top": dice, "area_gt": a_gt,
            "area_top": a_top, "hd95_top": NAN, "nsd2_top": NAN, "dice_union025": NAN}


def test_article_confusion_matrix_round_trip():
    cm = R.FINAL_CONFUSION
    rows = [row(t, p, 0.8, 100, 100) if t else row(t, p) for t in range(6) for p in range(6) for _ in range(cm[t][p])]
    e = B.evaluate(rows, 0.5)
    assert e["n"] == 210 and abs(e["accuracy_6class"] - 181 / 210) < 1e-12
    assert e["errors"] == R.FINAL["errors"] and e["n_errors"] == 29
    assert e["confusion_matrix"] == [list(r) for r in cm]
    for c, code in enumerate(R.CLASS_ORDER):
        assert abs(e["per_class"][code]["recall"] - cm[c][c] / sum(cm[c])) < 1e-12
    # fractured femurs called fractured: 138; right location: 132; graded (not subtrochanteric) of those: 123, 108 right
    assert abs(e["location_accuracy"] - 132 / 138) < 1e-12
    assert abs(e["grade_accuracy_given_location"] - 108 / 123) < 1e-12


def test_dice_definitions():
    rows = [row(2, 2, 0.8, 100, 100),   # intersection 80
            row(2, 2, 0.5, 100, 300),   # intersection 100
            row(1, 0, 0.9, 50, 50),     # missed: counts as 0, its predicted area is ignored
            row(0, 0), row(0, 3)]       # no fracture: nothing called (1) and a false alarm (0)
    s = B.evaluate(rows, 0.5)["seg"]
    assert abs(s["dice_detected_mean"] - 0.65) < 1e-12
    assert abs(s["dice_fractured_mean"] - (0.8 + 0.5 + 0) / 3) < 1e-12
    assert abs(s["dice_all_femurs_mean"] - (0.8 + 0.5 + 0 + 1 + 0) / 5) < 1e-12
    assert abs(s["dice_global"] - 2 * (80 + 100) / ((100 + 300) + (100 + 100 + 50))) < 1e-12
    assert s["area_ratio_median"] == 2.0 and (s["n_fractured"], s["n_detected"]) == (3, 2)
    pc = B.evaluate(rows, 0.5)["per_class"]
    assert pc["2AB"]["dice"] == 0.0 and pc["2AB"]["detected"] == 0.0 and math.isnan(pc["2AB"]["dice_detected"])
    assert abs(pc["2CD"]["dice"] - 0.65) < 1e-12


def test_mean_sd():
    a = B.evaluate([row(2, 2, 0.8, 100, 100), row(0, 0)], 0.5)
    b = B.evaluate([row(2, 2, 0.6, 100, 100), row(0, 3)], 0.5)
    ms = B.mean_sd([a, b])
    assert abs(ms["seg.dice_detected_mean"]["mean"] - 0.7) < 1e-12
    assert abs(ms["seg.dice_detected_mean"]["sd"] - np.std([0.8, 0.6], ddof=1)) < 1e-12
    assert ms["accuracy_6class"]["mean"] == 0.75 and ms["errors.detection"]["mean"] == 0.5


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} breakdown tests passed")
