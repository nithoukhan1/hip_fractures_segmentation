"""Confidence intervals and paired tests for comparing models on the same femurs (numpy and scipy only).

Every model in this project predicts the same cases (the 1,043 out-of-fold femurs and the 210 test femurs), so
comparisons are paired. The two femurs of one x-ray are not independent, so the resampling functions accept
``groups`` (for example ``FemurRecord.group``) and then resample or swap whole x-rays.

  proportion_ci             Wilson or exact Clopper-Pearson interval for a proportion (accuracy, sensitivity, ...)
  delong_ci, delong_test    AUC with its DeLong interval; DeLong test for two correlated AUCs (Sun & Xu 2014)
  bootstrap_ci              percentile bootstrap interval of any statistic, optionally by cluster (x-ray)
  bootstrap_diff            paired bootstrap interval (and p-value) of the difference of a statistic between models
  permutation_test          paired randomisation test of that difference (swaps the two models' predictions per
                            case or per x-ray); use it for metrics without a closed-form test, e.g. macro-F1
  mcnemar_exact             exact and mid-p McNemar test on paired correct / wrong decisions
  wilcoxon_paired           Wilcoxon signed-rank test on paired per-case values (e.g. Dice), Pratt's zero handling
  holm                      Holm-Bonferroni adjusted p-values for a family of comparisons
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

SEED = 2026


# ------------------------------------------------------------------------------------------------ proportions


def proportion_ci(k: int, n: int, alpha: float = 0.05, method: str = "wilson") -> tuple[float, float]:
    """Two-sided (1 - alpha) interval for k successes out of n. ``method`` is "wilson" or "exact" (Clopper-Pearson,
    as R's binom.test and caret, which the base article used for its accuracy intervals)."""
    from scipy.stats import beta, norm

    if n <= 0:
        return float("nan"), float("nan")
    if method == "exact":
        lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
        return lo, hi
    if method != "wilson":
        raise ValueError(f"unknown method {method}")
    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


# ----------------------------------------------------------------------------------------------------- DeLong


def _delong(y_true, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUCs of k score vectors and their DeLong covariance matrix (fast midrank algorithm of Sun & Xu 2014)."""
    from scipy.stats import rankdata

    y = np.asarray(y_true).astype(bool)
    s = np.atleast_2d(np.asarray(scores, dtype=float))
    pos, neg = s[:, y], s[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        raise ValueError("DeLong needs at least two positive and two negative cases")
    tx, ty = rankdata(pos, axis=1), rankdata(neg, axis=1)
    tz = rankdata(np.hstack([pos, neg]), axis=1)
    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v10 = (tz[:, :m] - tx) / n          # per positive case: share of negatives scored below it (ties 1/2)
    v01 = 1 - (tz[:, m:] - ty) / m      # per negative case: share of positives scored above it (ties 1/2)
    cov = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
    return auc, cov


def delong_ci(y_true, score, alpha: float = 0.05) -> dict:
    """AUC with its DeLong standard error and (1 - alpha) interval (as pROC's ci.auc with method "delong")."""
    from scipy.stats import norm

    auc, cov = _delong(y_true, np.asarray(score, dtype=float)[None, :])
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    z = float(norm.ppf(1 - alpha / 2))
    a = float(auc[0])
    return {"auc": a, "se": se, "lo": max(0.0, a - z * se), "hi": min(1.0, a + z * se)}


def delong_test(y_true, score_a, score_b, alpha: float = 0.05) -> dict:
    """Two-sided DeLong test of AUC(a) = AUC(b) for two models scored on the same cases."""
    from scipy.stats import norm

    auc, cov = _delong(y_true, np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)]))
    diff = float(auc[0] - auc[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else 0.0
    p = float(2 * norm.sf(abs(z))) if se > 0 else 1.0
    q = float(norm.ppf(1 - alpha / 2))
    return {"auc_a": float(auc[0]), "auc_b": float(auc[1]), "diff": diff, "se": se, "z": z, "p": p,
            "diff_lo": diff - q * se, "diff_hi": diff + q * se}


# -------------------------------------------------------------------------------------------- resampling


def _cluster_index(n: int, groups) -> list[np.ndarray]:
    """Row indices of each cluster (each row its own cluster when ``groups`` is None)."""
    if groups is None:
        return [np.array([i]) for i in range(n)]
    g = np.asarray(groups)
    if len(g) != n:
        raise ValueError("groups must have one entry per case")
    _, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def _resample_rows(clusters: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    pick = rng.integers(0, len(clusters), len(clusters))
    return np.concatenate([clusters[i] for i in pick])


def bootstrap_ci(stat: Callable[..., float], *arrays, n_boot: int = 2000, alpha: float = 0.05, groups=None,
                 seed: int = SEED) -> dict:
    """Percentile bootstrap interval of ``stat(*arrays)``; rows are resampled together (by cluster if ``groups``).

    Resamples where the statistic is undefined (NaN, e.g. an AUC without positives) are dropped and counted.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays must have the same number of cases")
    clusters = _cluster_index(n, groups)
    rng = np.random.default_rng(seed)
    est = float(stat(*arrays))
    vals = []
    for _ in range(n_boot):
        idx = _resample_rows(clusters, rng)
        vals.append(float(stat(*[a[idx] for a in arrays])))
    vals = np.asarray(vals)
    ok = vals[~np.isnan(vals)]
    lo, hi = (np.quantile(ok, [alpha / 2, 1 - alpha / 2]) if len(ok) else (np.nan, np.nan))
    return {"estimate": est, "lo": float(lo), "hi": float(hi), "se": float(ok.std(ddof=1)) if len(ok) > 1 else float("nan"),
            "n_boot": n_boot, "n_undefined": int(len(vals) - len(ok)), "clusters": len(clusters)}


def bootstrap_diff(stat: Callable[..., float], y, pred_a, pred_b, n_boot: int = 2000, alpha: float = 0.05,
                   groups=None, seed: int = SEED) -> dict:
    """Paired bootstrap of ``stat(y, pred_a) - stat(y, pred_b)`` (same resampled cases for both models).

    The p-value is twice the smaller bootstrap tail beyond 0, so p < alpha matches an interval that excludes 0.
    """
    y, pa, pb = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    clusters = _cluster_index(len(y), groups)
    rng = np.random.default_rng(seed)
    est = float(stat(y, pa) - stat(y, pb))
    vals = []
    for _ in range(n_boot):
        idx = _resample_rows(clusters, rng)
        vals.append(float(stat(y[idx], pa[idx]) - stat(y[idx], pb[idx])))
    vals = np.asarray(vals)
    ok = vals[~np.isnan(vals)]
    if not len(ok):
        return {"diff": est, "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n_boot": n_boot,
                "n_undefined": len(vals), "clusters": len(clusters)}
    lo, hi = np.quantile(ok, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((ok <= 0).mean(), (ok >= 0).mean()))
    return {"diff": est, "lo": float(lo), "hi": float(hi), "p": float(p), "n_boot": n_boot,
            "n_undefined": int(len(vals) - len(ok)), "clusters": len(clusters)}


def permutation_test(stat: Callable[..., float], y, pred_a, pred_b, n_perm: int = 10000, groups=None,
                     seed: int = SEED) -> dict:
    """Paired randomisation test of ``stat(y, pred_a) - stat(y, pred_b)`` (two-sided).

    Under the null hypothesis the two models are exchangeable, so each case (or each x-ray, with ``groups``) has its
    two predictions swapped with probability 1/2. p = (1 + #|diff*| >= |diff|) / (1 + n_perm).
    """
    y, pa, pb = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    if pa.shape != pb.shape:
        raise ValueError("the two models' predictions must have the same shape")
    clusters = _cluster_index(len(y), groups)
    cid = np.empty(len(y), int)
    for c, rows in enumerate(clusters):
        cid[rows] = c
    rng = np.random.default_rng(seed)
    obs = float(stat(y, pa) - stat(y, pb))
    hits = 0
    for _ in range(n_perm):
        swap = (rng.random(len(clusters)) < 0.5)[cid]
        sw = swap.reshape((-1,) + (1,) * (pa.ndim - 1))
        a, b = np.where(sw, pb, pa), np.where(sw, pa, pb)
        hits += abs(stat(y, a) - stat(y, b)) >= abs(obs) - 1e-12
    return {"diff": obs, "p": (1 + hits) / (1 + n_perm), "n_perm": n_perm, "clusters": len(clusters)}


# ------------------------------------------------------------------------------------------ classical tests


def mcnemar_exact(correct_a, correct_b) -> dict:
    """McNemar test on paired right / wrong decisions. b = only A right, c = only B right.

    Returns the exact binomial p-value and the mid-p value (recommended by Fagerland et al. 2013, BMC Med Res Methodol).
    """
    from scipy.stats import binom

    a, b_ = np.asarray(correct_a).astype(bool), np.asarray(correct_b).astype(bool)
    b, c = int((a & ~b_).sum()), int((~a & b_).sum())
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0, "p_midp": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * float(binom.cdf(k, n, 0.5)))
    p_mid = min(1.0, p - float(binom.pmf(k, n, 0.5)))
    return {"b": b, "c": c, "p": p, "p_midp": p_mid}


def wilcoxon_paired(x, y, zero_method: str = "pratt") -> dict:
    """Wilcoxon signed-rank test of paired values (pairs with a NaN are dropped). Pratt's method keeps zero
    differences in the ranking, which suits Dice (many femurs where both models score 0 or both 1)."""
    from scipy.stats import wilcoxon

    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    d = x[keep] - y[keep]
    out = {"n": int(keep.sum()), "n_nonzero": int((d != 0).sum()), "mean_diff": float(d.mean()) if len(d) else float("nan"),
           "median_diff": float(np.median(d)) if len(d) else float("nan")}
    if out["n_nonzero"] == 0:
        return {**out, "statistic": float("nan"), "p": 1.0}
    r = wilcoxon(x[keep], y[keep], zero_method=zero_method, alternative="two-sided")
    return {**out, "statistic": float(r.statistic), "p": float(r.pvalue)}


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = adj
    return out
