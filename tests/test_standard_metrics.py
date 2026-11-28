"""Checks of scripts/run_standard_metrics.py against scikit-learn (macro precision and recall, one-vs-rest AUC)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import run_standard_metrics as SM  # noqa: E402


def test_macro_precision_recall_and_ovr_auc_match_sklearn():
    from sklearn.metrics import precision_score, recall_score, roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 300)
    p = np.where(rng.random(300) < 0.7, y, rng.integers(0, 6, 300))
    prec, rec = SM.macro_pr(y, p)
    assert abs(prec - precision_score(y, p, average="macro", zero_division=0)) < 1e-12
    assert abs(rec - recall_score(y, p, average="macro")) < 1e-12
    probs = rng.dirichlet(np.ones(6), 300)
    probs[np.arange(300), y] += 0.3
    probs /= probs.sum(1, keepdims=True)
    want = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    assert abs(SM.ovr_macro_auc(probs, y) - want) < 1e-12


if __name__ == "__main__":
    test_macro_precision_recall_and_ovr_auc_match_sklearn()
    print("ok")
