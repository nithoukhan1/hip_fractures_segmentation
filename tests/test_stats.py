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


def test_calibration_reads_a_networks_written_zero_as_rounding():
    """scripts/run_stats.py: a network run writes probabilities with 8 decimals, so a true class written as 0 still
    gets a temperature; a YOLO run with such a femur stays 'not applicable' (no probability for an undetected class)."""
    import csv
    import json
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_stats as RS

    rng = np.random.default_rng(3)
    z = 4 * rng.normal(size=(60, 6))
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = rng.integers(0, 6, 60)
    p[0] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # femur 0: true class 0 written as 0
    y[0] = 0
    rows = [{"file": f"{i:05d}_x.png", "fold": 0, "model": 0, "cls6_true": int(c), "score": round(1 - q[0], 8),
             **{f"conf_{k}": round(q[k], 8) for k in range(1, 6)}} for i, (q, c) in enumerate(zip(p, y))]
    with tempfile.TemporaryDirectory() as tmp:
        for run, cfg in (("NET", {"net": "cls:convnext_tiny"}), ("YOLO", {"model": "yolov8s-seg.pt"})):
            d = Path(tmp) / run
            d.mkdir()
            (d / "config.json").write_text(json.dumps(cfg))
            with open(d / "per_case_oof.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        net, yolo = RS.calibration(Path(tmp), "NET"), RS.calibration(Path(tmp), "YOLO")
    assert net["temperature"] is not None and "rounded_zeros" in net and "not_applicable" not in net, net
    assert net["oof"]["nll_before"] >= -np.log(M.PROB_FLOOR) / 60 and net["oof"]["nll_after"] is not None
    assert yolo["temperature"] is None and "not_applicable" in yolo and yolo["oof"]["nll_before"] is None, yolo


def test_chosen_pairs_and_family_holm():
    """scripts/run_stats.py --pairs and --family-holm (analysis C07): only the pairs given are compared, and each
    endpoint is Holm-adjusted across the pairs where it was tested, leaving each comparison's own p_holm alone."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_stats as RS

    assert RS.parse_pairs(["a", "b", "c"]) == [("a", "b"), ("a", "c"), ("b", "c")]
    assert RS.parse_pairs(["a", "b", "c"], "b:a, c:b,") == [("b", "a"), ("c", "b")]
    for bad in ("a:a", "a:d", "ab", "b:a,b:a"):
        try:
            RS.parse_pairs(["a", "b", "c"], bad)
        except SystemExit:
            continue
        raise AssertionError(f"--pairs {bad!r} was accepted")
    comp = {k: {"oof": {"macro_f1": {"p": p, "p_holm": 0.9}} | ({"dice_fractured": {"p": d, "p_holm": 0.9}}
                                                                  if d is not None else {})}
            for k, p, d in (("b vs a", 0.01, None), ("c vs b", 0.04, 0.02), ("c vs a", 0.03, 0.5))}
    RS.family_holm(comp)
    got = [comp[k]["oof"]["macro_f1"]["p_holm_family"] for k in ("b vs a", "c vs b", "c vs a")]
    assert np.allclose(got, [0.03, 0.06, 0.06]), got  # 3 tests: .01 x 3, .03 x 2, then .04 raised to .06
    assert np.allclose([comp[k]["oof"]["dice_fractured"]["p_holm_family"] for k in ("c vs b", "c vs a")], [0.04, 0.5])
    assert all(r["p_holm"] == 0.9 for c in comp.values() for r in c["oof"].values())


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} statistics tests passed")
