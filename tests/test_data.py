"""Checks of the fixed folds (splits/folds_v1.csv) through src/femurseg/data.py.

Run with ``python tests/test_data.py`` or ``pytest tests``. Needs numpy only (no images).
With a data folder, ``python tests/test_data.py --data "<path to Femoral Fractures>"`` also checks that every
femur is found on disk and that a few images, masks and polygons agree.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from femurseg.classes import EXPECTED_COUNTS  # noqa: E402
from femurseg.data import ImageStore, fold_split, load_records  # noqa: E402

RECS = load_records()


def test_sizes_and_class_counts():
    assert len(RECS) == 1253
    for split, n in (("train", 1043), ("test", 210)):
        rs = [r for r in RECS if r.split == split]
        assert len(rs) == n
        assert dict(Counter(r.cls6 for r in rs)) == EXPECTED_COUNTS[split]
    assert {r.fold for r in RECS if r.split == "test"} == {-1}
    sizes = Counter(r.fold for r in RECS if r.split == "train")
    assert sorted(sizes) == [0, 1, 2, 3, 4] and all(208 <= v <= 209 for v in sizes.values()), sizes


def test_groups_and_pairs_stay_in_one_fold():
    folds_of_group = defaultdict(set)
    for r in RECS:
        folds_of_group[(r.split, r.group)].add(r.fold)
    assert all(len(v) == 1 for v in folds_of_group.values())
    by_file = {(r.split, r.file): r for r in RECS}
    for r in RECS:
        if r.pair_file:
            mate = by_file[(r.split, r.pair_file)]
            assert mate.fold == r.fold and mate.pair_file == r.file and mate.cls6 == r.pair_cls6
        if r.duplicate_of:
            assert by_file[(r.split, r.duplicate_of)].fold == r.fold


def test_fold_split_is_a_partition():
    train_files = {r.file for r in RECS if r.split == "train"}
    held_all = set()
    for k in range(5):
        train, held = fold_split(RECS, k)
        tf, hf = {r.file for r in train}, {r.file for r in held}
        assert not tf & hf and tf | hf == train_files
        assert all(r.split == "train" for r in train + held)
        held_all |= hf
    assert held_all == train_files


def test_record_properties():
    r = next(r for r in RECS if r.cls6 == 1 and r.split == "train")
    assert r.has_fracture and r.code == "2AB" and r.location == "neck" and r.key == ("train", r.file)
    nf = next(r for r in RECS if r.cls6 == 0)
    assert not nf.has_fracture and nf.code == "NF" and nf.location == "none"


def check_files_on_disk(data_root: str, n_sample: int = 12) -> None:
    """Every femur of the fold file is on disk; images, masks and polygons of a sample agree."""
    import cv2

    store = ImageStore(data_root, RECS)  # raises if any femur is missing
    rng = np.random.default_rng(0)
    sample = [RECS[i] for i in rng.choice(len(RECS), n_sample, replace=False)]
    sample += [next(r for r in RECS if r.cls6 == c) for c in range(6)]
    for r in sample:
        img, msk = store.image(r), store.mask(r)
        assert img.dtype == np.uint8 and img.shape == (1536, 1024), (r.file, img.shape)
        assert msk.shape == img.shape and set(np.unique(msk)) <= {0, r.cls6}, (r.file, np.unique(msk))
        poly = store.polygon(r)
        if not r.has_fracture:
            assert poly is None and not msk.any()
            continue
        yolo_id, pts = poly
        assert yolo_id == r.cls6 - 1
        raster = np.zeros_like(msk)
        cv2.fillPoly(raster, [np.round(pts * [1024, 1536]).astype(np.int32)], 1)
        d = 2 * ((raster > 0) & (msk > 0)).sum() / ((raster > 0).sum() + (msk > 0).sum())
        assert d > 0.98, (r.file, d)
    print(f"ok  files on disk: all {len(RECS)} femurs found; {len(sample)} images, masks and polygons agree")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="optional data folder (local 'Femoral Fractures' or /kaggle/input)")
    a = ap.parse_args()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    if a.data:
        check_files_on_disk(a.data)
    print(f"{len(tests)} data tests passed")
