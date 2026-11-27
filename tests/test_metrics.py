"""Checks of src/femurseg/metrics.py against brute-force definitions (and scikit-learn when it is installed).

Run with ``python tests/test_metrics.py`` or ``pytest tests``. Needs numpy and scipy only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from femurseg import metrics as M  # noqa: E402

RNG = np.random.default_rng(2026)


# ------------------------------------------------------------------------------------- brute-force helpers


def brute_surface(m: np.ndarray) -> np.ndarray:
    """Mask pixels with a 4-neighbour outside the mask (the image border counts as outside)."""
    h, w = m.shape
    out = np.zeros_like(m)
    for y, x in zip(*np.nonzero(m)):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if not (0 <= yy < h and 0 <= xx < w) or not m[yy, xx]:
                out[y, x] = True
                break
    return out


def brute_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For every True pixel of a, the Euclidean distance to the nearest True pixel of b."""
    pa, pb = np.argwhere(a).astype(float), np.argwhere(b).astype(float)
    return np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)).min(axis=1)


def random_mask(h: int = 40, w: int = 50) -> np.ndarray:
    """A union of 1-3 random ellipses, sometimes cut by the image border."""
    yy, xx = np.mgrid[:h, :w]
    m = np.zeros((h, w), bool)
    for _ in range(RNG.integers(1, 4)):
        cy, cx = RNG.uniform(-5, h + 5), RNG.uniform(-5, w + 5)
        ry, rx = RNG.uniform(2, 14), RNG.uniform(2, 14)
        m |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    return m


def random_pair():
    while True:
        a, b = random_mask(), random_mask()
        if a.any() and b.any():
            return a, b


# ------------------------------------------------------------------------------------------- segmentation


def test_dice_iou_simple():
    a = np.zeros((4, 4), int)
    b = np.zeros((4, 4), int)
    a[0, :2] = 1          # 2 pixels
    b[0, 1:4] = 3         # 3 pixels, any value > 0 counts
    assert M.dice(a, b) == 2 * 1 / 5
    assert M.iou(a, b) == 1 / 4
    z = np.zeros((4, 4))
    assert M.dice(z, z) == 1.0 and M.iou(z, z) == 1.0
    assert M.dice(a, z) == 0.0 and M.iou(z, a) == 0.0


def test_empty_conventions():
    z, a = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
    a[3:6, 3:6] = True
    assert M.hd95(z, z) == 0.0 and M.nsd(z, z) == 1.0
    assert np.isnan(M.hd95(a, z)) and np.isnan(M.hd95(z, a))
    assert M.nsd(a, z) == 0.0 and M.nsd(z, a) == 0.0
    assert M.hd95(a, a) == 0.0 and M.nsd(a, a) == 1.0


def test_surface_matches_brute_force():
    for _ in range(30):
        m = random_mask()
        assert np.array_equal(M._surface(m), brute_surface(m))


def test_hd95_and_nsd_match_brute_force():
    for _ in range(40):
        a, b = random_pair()
        sa, sb = brute_surface(a), brute_surface(b)
        d = np.concatenate([brute_dist(sa, sb), brute_dist(sb, sa)])
        assert abs(M.hd95(a, b) - np.percentile(d, 95)) < 1e-9
        for tol in (1.0, 2.0, 3.5):
            assert abs(M.nsd(a, b, tol) - (d <= tol).mean()) < 1e-12


def test_hd95_known_shift():
    a = np.zeros((30, 30), bool)
    a[10:20, 10:20] = True
    b = np.roll(a, 3, axis=1)  # same square moved 3 pixels right
    assert M.hd95(a, b) == 3.0
    assert M.nsd(a, b, tol=3.0) == 1.0
    assert M.hd95(a, b) == M.hd95(b, a)


def test_seg_metrics_keys():
    a, b = random_pair()
    s = M.seg_metrics(a, b)
    assert set(s) == {"dice", "iou", "hd95", "nsd2", "area_pred", "area_gt"}
    assert s["area_pred"] == a.sum() and s["area_gt"] == b.sum()


def test_seg_metrics_several_tolerances():
    z = np.zeros((30, 30), bool)
    for _ in range(10):
        a, b = random_pair()
        s = M.seg_metrics(a, b, (2, 10, 20))
        assert {"nsd2", "nsd10", "nsd20"} <= set(s)
        assert s["hd95"] == M.hd95(a, b) and s["dice"] == M.dice(a, b)
        for t in (2, 10, 20):
            assert s[f"nsd{t}"] == M.nsd(a, b, t)
    s = M.seg_metrics(z, z, (2, 10))
    assert (s["hd95"], s["nsd2"], s["nsd10"]) == (0.0, 1.0, 1.0)
    a, _ = random_pair()
    s = M.seg_metrics(z, a[:30, :30] | True, (2, 10))  # nothing predicted, fracture present
    assert np.isnan(s["hd95"]) and (s["dice"], s["nsd2"], s["nsd10"]) == (0.0, 0.0, 0.0)


# ----------------------------------------------------------------------------------------- classification


def brute_auc(y, s):
    pos, neg = s[y], s[~y]
    return np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])


def test_roc_auc_matches_pairwise_definition():
    for _ in range(20):
        y = RNG.random(60) < 0.4
        s = np.round(RNG.random(60) + 0.5 * y, 1)  # rounding creates ties
        if y.all() or (~y).all():
            continue
        assert abs(M.roc_auc(y, s) - brute_auc(y, s)) < 1e-12
    assert np.isnan(M.roc_auc([1, 1], [0.2, 0.3]))


def test_binary_metrics_known_case():
    y = [1, 1, 1, 0, 0, 0, 0]
    p = [1, 1, 0, 1, 0, 0, 0]
    b = M.binary_metrics(y, p)
    assert (b["tp"], b["fn"], b["fp"], b["tn"]) == (2, 1, 1, 3)
    assert b["sensitivity"] == 2 / 3 and b["specificity"] == 3 / 4
    assert b["ppv"] == 2 / 3 and b["npv"] == 3 / 4 and b["accuracy"] == 5 / 7 and b["f1"] == 4 / 6


def test_multiclass_by_hand():
    y = np.array([0, 0, 1, 2, 2, 3, 4, 5, 5, 5])
    p = np.array([0, 1, 1, 2, 3, 3, 0, 5, 5, 4])
    m = M.multiclass_metrics(y, p)
    cm = np.array(m["confusion_matrix"])
    assert cm.sum() == 10 and cm[5, 4] == 1 and cm[4, 0] == 1
    assert m["accuracy"] == 0.6
    recalls = [1 / 2, 1, 1 / 2, 1, 0, 2 / 3]
    precisions = [1 / 2, 1 / 2, 1, 1 / 2, 0, 1]
    f1 = [2 * a * b / (a + b) if a + b else 0 for a, b in zip(precisions, recalls)]
    assert abs(m["balanced_accuracy"] - np.mean(recalls)) < 1e-12
    assert abs(m["macro_f1"] - np.mean(f1)) < 1e-12
    pe = sum(cm[i].sum() * cm[:, i].sum() for i in range(6)) / 100
    assert abs(m["kappa"] - (0.6 - pe) / (1 - pe)) < 1e-12
    assert m["per_class"]["2AB"]["recall"] == 1.0 and m["per_class"]["NF"]["support"] == 2


def test_multiclass_against_sklearn_if_installed():
    try:
        from sklearn import metrics as S
    except ImportError:
        print("  (scikit-learn not installed: comparison skipped)")
        return
    for _ in range(20):
        y = RNG.integers(0, 6, 80)
        p = np.where(RNG.random(80) < 0.6, y, RNG.integers(0, 6, 80))
        m = M.multiclass_metrics(y, p)
        present = sorted(set(y.tolist()))
        assert abs(m["accuracy"] - S.accuracy_score(y, p)) < 1e-12
        assert abs(m["balanced_accuracy"] - S.balanced_accuracy_score(y, p)) < 1e-12
        assert abs(m["kappa"] - S.cohen_kappa_score(y, p)) < 1e-12
        assert abs(m["macro_f1"] - S.f1_score(y, p, labels=present, average="macro", zero_division=0)) < 1e-12
        yb, sb = y > 0, RNG.random(80) + 0.3 * (y > 0)
        assert abs(M.roc_auc(yb, sb) - S.roc_auc_score(yb, sb)) < 1e-12


def test_error_breakdown():
    # (true, predicted): NF->2AB detection, 2CD->NF detection, 2AB->3AB location, 3AB->3CDE grade, 2CD->2AB grade
    y = [0, 2, 1, 3, 2, 4]
    p = [1, 0, 3, 4, 1, 4]
    assert M.error_breakdown(y, p) == {"detection": 2, "location": 1, "grade": 2}


def test_calibration_known_values():
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    y = [0, 1]
    assert abs(M.expected_calibration_error(probs, y) - 0.35) < 1e-12
    assert abs(M.brier_score(probs, y) - 0.37) < 1e-12


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1.5, (20000, 6))
    soft = lambda v: np.exp(v - v.max(1, keepdims=True)) / np.exp(v - v.max(1, keepdims=True)).sum(1, keepdims=True)  # noqa: E731
    p_true = soft(z)
    y = (p_true.cumsum(1) < rng.random((len(z), 1))).sum(1)  # one draw per row from its true probabilities
    over = soft(3 * z)  # over-confident: logits 3 x too large
    t = M.fit_temperature(over, y)
    assert abs(t - 3) < 0.15, t
    cal = M.apply_temperature(over, t)
    assert np.allclose(M.apply_temperature(over, 3.0), p_true, atol=1e-4)  # the inverse at t = 3 (up to the 1e-12 floor)
    assert np.allclose(M.apply_temperature(over, 1.0), over) and (cal.argmax(1) == over.argmax(1)).all()
    assert M.expected_calibration_error(cal, y) < M.expected_calibration_error(over, y) / 3
    assert M.nll(cal, y) < M.nll(over, y) and M.brier_score(cal, y) < M.brier_score(over, y)
    assert abs(M.fit_temperature(p_true, y) - 1) < 0.05  # already calibrated: t stays near 1


def test_case_probs_from_numbers_or_csv_strings():
    from femurseg.evaluation import case_probs

    rows = [{"score": 0.8, "conf_1": 0.1, "conf_2": 0.6, "conf_3": 0.0, "conf_4": 0.0, "conf_5": 0.1, "cls6_true": 2},
            {"score": 0.1, "conf_1": 0.0, "conf_2": 0.0, "conf_3": 0.05, "conf_4": 0.0, "conf_5": 0.0, "cls6_true": 0}]
    p, y = case_probs(rows)
    p2, y2 = case_probs([{k: str(v) for k, v in r.items()} for r in rows])
    assert np.allclose(p, p2) and (y == y2).all() and np.allclose(p.sum(1), 1)
    assert np.allclose(p[0], [0.2, 0.1, 0.6, 0, 0, 0.1]) and list(y) == [2, 0]
    assert np.allclose(p[1], np.array([0.9, 0, 0, 0.05, 0, 0]) / 0.95)  # YOLO-style rows need normalising


def test_youden_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    # J = 0.5 at t = 0.35 (sens 1, spec .5) and at t = 0.8 (sens .5, spec 1): ties go to the lower threshold
    assert M.youden_threshold(y, s) == 0.35
    assert M.youden_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9]) == 0.7  # perfect separation
    for _ in range(20):  # the chosen threshold reaches the best J of all observed scores
        y = RNG.random(50) < 0.5
        s = RNG.random(50) + 0.4 * y
        t = M.youden_threshold(y, s)

        def j(th):
            p = s >= th
            return (p & y).sum() / y.sum() + (~p & ~y).sum() / (~y).sum() - 1

        assert abs(j(t) - max(j(u) for u in np.unique(s))) < 1e-12


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} metric tests passed")
