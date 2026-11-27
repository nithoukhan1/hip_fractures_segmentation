"""Checks of Paper B's pair index and reference table (src/femurseg/refbank.py): the counts of the real fold file,
the bank of each fold model, the exclusion of a femur's own group, the pelvis side and the orientation rule, the
seeded random references and the sanity checks. numpy only, no images.

Run with ``python tests/test_refbank.py`` or pytest.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from femurseg import refbank as RB  # noqa: E402
from femurseg.data import FemurRecord, load_records  # noqa: E402

D = 16
MIRROR = np.random.default_rng(7).permutation(D)  # what mirroring does to the synthetic features


def _records(n_xrays_per_fold: int = 8, n_test_xrays: int = 6) -> list[FemurRecord]:
    """Synthetic fold file: per fold, x-rays with a fractured + healthy pair (even ones) or one femur (odd ones)."""
    recs, i = [], 0
    for split, fold, n in [("train", k, n_xrays_per_fold) for k in range(5)] + [("test", -1, n_test_xrays)]:
        for x in range(n):
            xray = f"{split[:2]}{fold + 1}{x:03d}"
            if x % 2 == 0:  # pair: a fractured femur (class 1..5) and its healthy partner
                a, b = f"{i:05d}_{xray}.png", f"{i + 1:05d}_{xray}.png"
                c = 1 + (x // 2) % 5
                recs += [FemurRecord(a, split, fold, f"{i:05d}", xray, f"{split}:{xray}", c, 2, b, 0, "", f"m{i}"),
                         FemurRecord(b, split, fold, f"{i + 1:05d}", xray, f"{split}:{xray}", 0, 2, a, c, "",
                                     f"m{i + 1}")]
                i += 2
            else:  # one femur: healthy for x % 4 == 1, fractured otherwise
                c = 0 if x % 4 == 1 else 3
                recs.append(FemurRecord(f"{i:05d}_{xray}.png", split, fold, f"{i:05d}", xray, f"{split}:{xray}", c, 1,
                                        "", -1, "", f"m{i}"))
                i += 1
    return recs


def _features(recs: list[FemurRecord], seed: int = 0) -> dict:
    """[as released, mirrored] features: the two femurs of an x-ray share an anatomy vector, one of them on the other
    side (mirrored), so the partner matches best once mirrored, as in real x-rays."""
    rng = np.random.default_rng(seed)
    anatomy = {r.xray_id: rng.normal(size=D) for r in recs}
    feats = {}
    for r in recs:
        side = int(r.femur_id) % 2  # the two femurs of a pair have consecutive ids: one left, one right
        base = anatomy[r.xray_id] + 0.3 * rng.normal(size=D)
        released = base[MIRROR] if side else base
        mirrored = base if side else base[MIRROR]
        feats[r.key] = np.stack([released, mirrored])
    return feats


def test_real_fold_file_counts_and_banks():
    recs = load_records()
    assert [len(RB.bank_records(recs, m)) for m in range(5)] == [291, 290, 290, 291, 290]
    assert sum(len(RB.bank_records(recs, m, healthy=False)) for m in range(5)) == 4 * 680
    c = RB.counts(recs)
    assert c["train"]["by_pair_type"] == {"fractured, healthy partner": 334, "healthy, fractured partner": 334,
                                          "healthy, healthy partner": 14, "single": 361}, c["train"]
    assert c["test"]["by_pair_type"] == {"fractured, healthy partner": 64, "healthy, fractured partner": 64,
                                         "single": 82}, c["test"]
    assert (c["test"]["healthy"], c["test"]["fractured"]) == (65, 145)
    pairs = RB.pair_rows(recs)
    assert sum(p["partner_usable"] for p in pairs if p["split"] == "test" and p["cls6"] > 0) == 64
    dup = [r for r in recs if r.group == "train:00261"]  # the duplicate image: other x-ray id, same group
    assert sorted(r.file for r in dup) == ["00397_00261.png", "00398_00261.png", "00399_00262.png"]


def _sides(recs: list[FemurRecord]) -> dict:
    """Pelvis sides that match _features: side bit 0 = 'left' (features as released = the anatomy vector)."""
    return {r.key: "left" if int(r.femur_id) % 2 == 0 else "right" for r in recs}


def test_pelvis_side_reads_the_brighter_upper_half():
    img = np.zeros((96, 64), np.uint8)
    img[:30, :20] = 200   # pelvis: bright bone in the upper left
    img[40:, 25:40] = 150  # the shaft, in the lower half (ignored)
    assert RB.pelvis_side(img) == "left" and RB.pelvis_side(img[:, ::-1]) == "right"
    assert RB.pelvis_side(np.stack([img] * 3, axis=2)) == "left"  # colour images too
    assert RB.pelvis_side(img[:, :63]) == "left"  # odd width: the middle column is ignored
    o = RB.orientation(["left", "right"], ["left", "right", "right"])
    assert o.tolist() == [[False, True, True], [True, False, False]]


def test_given_orientation_overrides_the_feature_choice():
    rng = np.random.default_rng(2)
    bank = rng.normal(size=(3, 2, D))
    target = bank[1, 1] + 0.01 * rng.normal(size=D)  # matches bank femur 1 mirrored
    cos_f, mir_f = RB.similarities(target[None], bank)
    assert mir_f[0, 1]
    given = np.array([[False, False, True]])
    cos_g, mir_g = RB.similarities(target[None], bank, mirrored=given)
    assert (mir_g == given).all() and cos_g[0, 1] < cos_f[0, 1]  # bank femur 1 as released: a worse match
    s = np.array([[RB.unit(bank[m, int(given[0, m])]) @ RB.unit(target) for m in range(3)]])
    assert np.allclose(cos_g, s)


def test_nearest_skips_own_group_and_keeps_the_better_orientation():
    rng = np.random.default_rng(1)
    bank = rng.normal(size=(4, 2, D))
    target = bank[2, 1] + 0.01 * rng.normal(size=D)  # bank femur 2, mirrored
    bank[0, 0] = target                               # a perfect match, but from the target's own group
    cos, mirrored = RB.similarities(target[None], bank)
    order = RB.nearest(cos, ["g_t"], ["g_t", "g1", "g2", "g3"], k=3)
    assert order[0, 0] == 2 and mirrored[0, 2] and 0 not in order[0], order
    assert np.all(np.diff(cos[0, order[0]]) <= 0)  # most similar first
    try:
        RB.nearest(cos, ["g_t"], ["g_t", "g1", "g2", "g3"], k=4)
    except ValueError:
        pass
    else:
        raise AssertionError("asked for more references than the bank holds outside the own group")


def test_reference_table_follows_the_rules():
    recs = _records()
    feats, sides = _features(recs), _sides(recs)
    by_file = {r.file: r for r in recs}
    table = RB.reference_table(recs, feats, sides)
    assert len(table) == 5 * len(recs)
    assert table == RB.reference_table(recs, feats, sides)  # deterministic
    for t in table:
        m = t["fold_model"]
        target = by_file[t["file"]]
        assert t["role"] == ("test" if target.split == "test" else "held_out" if target.fold == m else "train")
        refs = [t[f"ref{j}"] for j in range(1, RB.TOP_K + 1)]
        assert len(set(refs)) == RB.TOP_K
        for f in refs + [t["random_healthy"]]:
            r = by_file[f]
            assert r.split == "train" and r.fold != m and r.cls6 == 0 and r.group != target.group, (t, f)
        rf = by_file[t["random_fractured"]]
        assert rf.split == "train" and rf.fold != m and rf.cls6 > 0 and rf.group != target.group
        for col in [f"ref{j}" for j in range(1, RB.TOP_K + 1)] + ["random_healthy", "random_fractured"]:
            assert t[f"{col}_mirrored"] == int(sides[by_file[t[col]].key] != sides[target.key]), (t, col)
        cos = [t[f"ref{j}_cos"] for j in range(1, RB.TOP_K + 1)]
        assert cos == sorted(cos, reverse=True) and cos[0] >= t["random_healthy_cos"] - 1e-9
    # the random references are spread over the bank, not always the same femur
    assert len({t["random_healthy"] for t in table if t["fold_model"] == 0}) > 3


def test_centre_uses_only_the_training_femurs_of_the_fold_model():
    recs = _records()
    feats = _features(recs)
    mu0 = RB.center_of(recs, feats, 0)
    train0 = [r for r in recs if r.split == "train" and r.fold != 0]
    assert np.allclose(mu0, np.mean([feats[r.key] for r in train0], axis=(0, 1)))
    changed = {k: (v + 100.0 if k[0] == "test" or any(r.key == k and r.fold == 0 for r in recs) else v)
               for k, v in feats.items()}  # held-out fold 0 and the test set: never in fold model 0's centre
    assert np.allclose(RB.center_of(recs, changed, 0), mu0)
    sides = _sides(recs)
    rows0 = [t for t in RB.reference_table(recs, feats, sides) if t["fold_model"] == 0 and t["role"] == "train"]
    rows0c = [t for t in RB.reference_table(recs, changed, sides) if t["fold_model"] == 0 and t["role"] == "train"]
    assert rows0 == rows0c  # the references of fold model 0's training femurs do not depend on them either


def test_seeded_pick_is_reproducible():
    recs = _records()
    target = recs[0]
    pool = RB.bank_records(recs, 0)
    a = RB.seeded_pick(target, 0, "random_healthy", pool)
    assert a == RB.seeded_pick(target, 0, "random_healthy", list(pool))
    eligible = [c for c in pool if c.group != target.group]
    h = int(hashlib.md5(f"{target.file}|0|random_healthy".encode()).hexdigest()[:12], 16)
    assert a == eligible[h % len(eligible)]


def test_sanity_checks():
    recs = _records()
    feats, sides = _features(recs), _sides(recs)
    pairs, table = RB.pair_rows(recs, feats, sides), RB.reference_table(recs, feats, sides)
    assert all(p["partner_mirrored_by_rule"] == 1 for p in pairs if p["partner"])  # partners: opposite sides
    assert {p["pelvis_side"] for p in pairs} == {"left", "right"}
    out = RB.sanity(recs, feats, table, pairs)
    m1 = out["1_mirror_check"]
    assert m1["femurs_with_partner"] == sum(1 for r in recs if r.pair_file) and m1["share"] == 1.0, m1
    assert m1["features_share"] > 0.9, m1  # the synthetic features do see left and right
    assert out["2_retrieval_vs_chance"]["top3_minus_random_mean"] > 0
    assert out["3_duplicates"]["pixel_identical_references"] == 0
    assert out["5_partner_rank"]["femurs"] > 0 and out["5_partner_rank"]["share_rank_1"] > 0.5, out["5_partner_rank"]
    assert [b["healthy_femurs"] for b in out["6_bank"]] == [len(RB.bank_records(recs, m)) for m in range(5)]
    # a reference with the same pixels as its femur stops the build
    t = next(x for x in table if x["role"] == "held_out")
    ref = next(r for r in recs if r.file == t["ref1"])
    clone = [FemurRecord(**{**r.__dict__, "pixel_md5": ref.pixel_md5}) if r.file == t["file"] else r for r in recs]
    try:
        RB.sanity(clone, feats, table, pairs)
    except ValueError as e:
        assert "same pixels" in str(e)
    else:
        raise AssertionError("a pixel-identical reference was not caught")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} reference-bank tests passed")
