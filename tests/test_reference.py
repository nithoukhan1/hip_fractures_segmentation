"""Checks that the numbers typed from the base article (src/femurseg/reference.py) agree with each other.

The confusion matrix of Fig. 3 must reproduce Table 1's test counts and every figure the article derives from it
(6-class accuracy, sensitivity, specificity, PPV, NPV, binary accuracy and the error counts in the text), and our own
metric code must give the article's exact accuracy interval. Run with ``python tests/test_reference.py`` or pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from femurseg import metrics as M  # noqa: E402
from femurseg import reference as R  # noqa: E402
from femurseg.classes import EXPECTED_COUNTS  # noqa: E402


def cases_from_confusion(cm) -> tuple[np.ndarray, np.ndarray]:
    cm = np.asarray(cm)
    y = np.concatenate([[t] * int(cm[t].sum()) for t in range(len(cm))])
    p = np.concatenate([np.repeat(np.arange(len(cm)), cm[t]) for t in range(len(cm))])
    return y, p


def test_confusion_matrix_matches_table1_and_text():
    cm = np.asarray(R.FINAL_CONFUSION)
    assert tuple(cm.sum(axis=1)) == tuple(R.TEST_COUNTS[c] for c in R.CLASS_ORDER)
    assert tuple(cm.sum(axis=1)) == tuple(EXPECTED_COUNTS["test"][c] for c in range(6))
    assert cm.sum() == 210 and np.trace(cm) == 181 and round(181 / 210, 4) == R.FINAL["accuracy_6class"]
    y, p = cases_from_confusion(cm)
    assert M.error_breakdown(y, p) == R.FINAL["errors"]
    b = M.binary_metrics(y > 0, p > 0)
    assert round(b["sensitivity"], 4) == R.FINAL["fracture_sensitivity"]
    assert round(b["specificity"], 4) == R.FINAL["fracture_specificity"]
    assert round(b["ppv"], 4) == R.FINAL["fracture_ppv"] and round(b["npv"], 4) == R.FINAL["fracture_npv"]
    assert round(b["accuracy"], 3) == R.FINAL["fracture_accuracy"]


def test_accuracy_interval_and_grid():
    from femurseg import stats as S

    lo, hi = S.proportion_ci(181, 210, method="exact")
    assert (round(lo, 4), round(hi, 4)) == R.FINAL["accuracy_6class_ci"]
    assert R.grid("small", 1024)["accuracy_6class"] == (0.8440, 0.0251) and len(R.GRID) == 12


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} reference tests passed")
