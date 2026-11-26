"""Checks of src/femurseg/stats.py against published values, brute-force definitions and scipy.

Run with ``python tests/test_stats.py`` or ``pytest tests``. Needs numpy and scipy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from femurseg import metrics as M  # noqa: E402
from femurseg import stats as S  # noqa: E402

RNG = np.random.default_rng(7)


def test_proportion_ci_reproduces_the_base_article():
    # Paper: final YOLO model, 6-class accuracy 86.19% (80.77-90.55) on 210 test femurs = 181 correct, exact interval
    lo, hi = S.proportion_ci(181, 210, method="exact")
    assert round(181 / 210 * 100, 2) == 86.19 and round(lo * 100, 2) == 80.77 and round(hi * 100, 2) == 90.55
    lo, hi = S.proportion_ci(8, 10)  # Wilson interval, textbook value
    assert abs(lo - 0.4902) < 1e-4 and abs(hi - 0.9433) < 1e-4
    assert S.proportion_ci(0, 10, method="exact")[0] == 0.0 and S.proportion_ci(10, 10, method="exact")[1] == 1.0


def brute_delong(y, scores):
    """DeLong (1988) structural components, computed pair by pair."""
    y = np.asarray(y).astype(bool)
    comps10, comps01, aucs = [], [], []
    for s in scores:
        pos, neg = s[y], s[~y]
        psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        aucs.append(psi.mean())
        comps10.append(psi.mean(axis=1))
        comps01.append(psi.mean(axis=0))
    m, n = y.sum(), (~y).sum()
    cov = np.cov(np.array(comps10)) / m + np.cov(np.array(comps01)) / n
    return np.array(aucs), np.atleast_2d(cov)


def test_delong_matches_brute_force():
    for _ in range(10):
        y = RNG.random(80) < 0.4
        a = np.round(RNG.random(80) + 0.6 * y, 1)          # rounding creates ties
        b = np.round(0.5 * a + RNG.random(80) * 0.8, 1)    # correlated with a
        auc, cov = S._delong(y, np.vstack([a, b]))
        auc_bf, cov_bf = brute_delong(y, [a, b])
        assert np.allclose(auc, auc_bf) and np.allclose(cov, cov_bf)
        assert abs(auc[0] - M.roc_auc(y, a)) < 1e-12
        t = S.delong_test(y, a, b)
        var = cov_bf[0, 0] + cov_bf[1, 1] - 2 * cov_bf[0, 1]
        assert abs(t["se"] - np.sqrt(var)) < 1e-12
        c = S.delong_ci(y, a)
        assert c["lo"] <= c["auc"] <= c["hi"] and abs(c["se"] - np.sqrt(cov_bf[0, 0])) < 1e-12


def test_delong_identical_models():
    y = RNG.random(50) < 0.5
    s = RNG.random(50) + y
    t = S.delong_test(y, s, s)
    assert t["diff"] == 0 and t["p"] == 1.0


def test_mcnemar_known_values():
    a = np.array([1] * 1 + [0] * 9 + [1] * 20, bool)   # b = 1 (only A right), c = 9 (only B right), 20 both right
    b = np.array([0] * 1 + [1] * 9 + [1] * 20, bool)
    r = S.mcnemar_exact(a, b)
    assert (r["b"], r["c"]) == (1, 9)
    assert abs(r["p"] - 22 / 1024) < 1e-12              # 2 * P(X <= 1), X ~ Bin(10, 1/2)
    assert abs(r["p_midp"] - (22 / 1024 - 10 / 1024)) < 1e-12
    assert S.mcnemar_exact(a, a)["p"] == 1.0


def test_holm_known_values():
    adj = S.holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(S.holm([0.5, 0.9]), [1.0, 1.0])  # monotone: never below an earlier adjusted p


def test_wilcoxon_wrapper():
    from scipy.stats import wilcoxon

    x = RNG.random(40)
    y = x + RNG.normal(0.05, 0.1, 40)
    y[:5] = x[:5]  # some ties
    r = S.wilcoxon_paired(x, y)
    assert abs(r["p"] - wilcoxon(x, y, zero_method="pratt").pvalue) < 1e-12 and r["n_nonzero"] == 35
    assert S.wilcoxon_paired(x, x)["p"] == 1.0
    x2 = x.copy()
    x2[0] = np.nan
    assert S.wilcoxon_paired(x2, y)["n"] == 39


def test_bootstrap_by_cluster_equals_bootstrap_of_clusters():
    # Every x-ray holds two identical femurs: resampling x-rays must give the same interval as resampling the
    # unique values (same random draws), and a narrower one than wrongly resampling single femurs would suggest.
    v = RNG.random(60)
    twice = np.repeat(v, 2)
    groups = np.repeat(np.arange(60), 2)
    a = S.bootstrap_ci(np.mean, v, n_boot=500, seed=1)
    b = S.bootstrap_ci(np.mean, twice, n_boot=500, seed=1, groups=groups)
    assert abs(a["lo"] - b["lo"]) < 1e-12 and abs(a["hi"] - b["hi"]) < 1e-12 and b["clusters"] == 60
    naive = S.bootstrap_ci(np.mean, twice, n_boot=500, seed=1)
    assert (naive["hi"] - naive["lo"]) < (b["hi"] - b["lo"])


def test_bootstrap_mean_interval_is_sensible():
    x = RNG.normal(0, 1, 400)
    r = S.bootstrap_ci(np.mean, x, n_boot=2000)
    width = r["hi"] - r["lo"]
    assert r["lo"] < x.mean() < r["hi"] and 0.8 * 0.196 < width < 1.25 * 0.196  # 2 * 1.96 / sqrt(400) = 0.196
    y = np.r_[np.ones(2), np.zeros(8)].astype(bool)  # AUC is undefined when a resample has no positive case
    r = S.bootstrap_ci(M.roc_auc, y, RNG.random(10), n_boot=300)  # P(no positive) = 0.8 ** 10, about 11%
    assert r["n_undefined"] > 0 and not np.isnan(r["lo"])


def macro_f1(y, p):
    return M.multiclass_metrics(y, p)["macro_f1"]


def test_paired_difference_tests():
    y = RNG.integers(0, 6, 300)
    good = np.where(RNG.random(300) < 0.85, y, RNG.integers(0, 6, 300))
    bad = np.where(RNG.random(300) < 0.45, y, RNG.integers(0, 6, 300))
    groups = np.arange(300) // 2
    same = S.bootstrap_diff(macro_f1, y, good, good, n_boot=200)
    assert same["diff"] == 0 and same["lo"] == 0 == same["hi"] and same["p"] == 1.0
    d = S.bootstrap_diff(macro_f1, y, good, bad, n_boot=500, groups=groups)
    assert d["diff"] > 0.2 and d["lo"] > 0 and d["p"] < 0.01
    perm_same = S.permutation_test(macro_f1, y, good, good, n_perm=200)
    assert perm_same["p"] == 1.0
    perm = S.permutation_test(macro_f1, y, good, bad, n_perm=500, groups=groups)
    assert perm["p"] < 0.01 and perm["clusters"] == 150
    # 2-D predictions (per-class probabilities) are swapped row by row
    pa, pb = np.eye(6)[good], np.eye(6)[bad]
    acc = lambda yy, pp: float(np.mean(pp.argmax(axis=1) == yy))  # noqa: E731
    assert S.permutation_test(acc, y, pa, pb, n_perm=300)["p"] < 0.01


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} statistics tests passed")
