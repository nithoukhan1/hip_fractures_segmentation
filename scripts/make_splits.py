#!/usr/bin/env python3
"""Make the fixed cross-validation folds for the HUSJA-UA femoral fracture dataset.

Why: the official release has train (1043 femurs) and test (210) but an EMPTY val folder.
Rules (see Analysis/03_Idea_Ranking_and_Research_Plan.md, section 7):
  1. The official test set is never touched (fold = -1).
  2. Train is split into K folds, grouped by x-ray ID so both femurs of one radiograph stay
     together. Pixel-identical images (one known pair) are merged into the same group.
  3. Folds are stratified by the 6-class label. Fold 0 is the default validation fold.

Output: one CSV row per femur (train and test), plus a JSON summary with per-fold counts and a
fingerprint of the file list, so the Kaggle copy of the data can be checked against the local one.

Needs only numpy and Pillow (or OpenCV). Works the same locally and on Kaggle.

Examples:
  python scripts/make_splits.py --data-root "E:/PhD/.../Femoral Fractures" --out splits/folds_v1.csv
  python scripts/make_splits.py --data-root /kaggle/input --out /kaggle/working/folds_check.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from femurseg.classes import BY_CLS6, EXPECTED_COUNTS, yolo_to_cls6  # noqa: E402
from femurseg.paths import find_data_root  # noqa: E402


def read_gray(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im.convert("L"))
    except ImportError:  # pragma: no cover - fallback when Pillow is missing
        import cv2

        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def read_yolo_class(label_path: Path) -> int:
    """Return the YOLO class id of a label file, or -1 if there is no fracture label."""
    if not label_path.exists():
        return -1
    ids = {int(float(line.split()[0])) for line in label_path.read_text().splitlines() if line.strip()}
    if not ids:
        return -1
    if len(ids) > 1:
        raise ValueError(f"{label_path} has more than one class: {sorted(ids)}")
    return ids.pop()


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def stratified_group_kfold(labels: list[int], groups: list[str], n_splits: int, seed: int) -> list[int]:
    """Greedy stratified group k-fold (same rule as scikit-learn's StratifiedGroupKFold, shuffle=True).

    Groups are visited from the most class-skewed to the least; each goes to the fold that keeps
    the per-class proportions most even across folds (ties go to the smaller fold).
    """
    classes = sorted(set(labels))
    cidx = {c: i for i, c in enumerate(classes)}
    gcount: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(classes)))
    for y, g in zip(labels, groups):
        gcount[g][cidx[y]] += 1
    order = sorted(gcount)
    np.random.RandomState(seed).shuffle(order)
    order.sort(key=lambda g: -float(np.std(gcount[g])))  # stable sort keeps the shuffled order for ties
    totals = np.sum([gcount[g] for g in order], axis=0)
    fold_counts = np.zeros((n_splits, len(classes)))
    fold_sizes = np.zeros(n_splits)
    assign: dict[str, int] = {}
    for g in order:
        best = None
        for k in range(n_splits):
            fold_counts[k] += gcount[g]
            score = float(np.mean(np.std(fold_counts / totals, axis=0)))
            fold_counts[k] -= gcount[g]
            key = (round(score, 12), fold_sizes[k], k)
            if best is None or key < best:
                best = key
        k = best[2]
        assign[g] = k
        fold_counts[k] += gcount[g]
        fold_sizes[k] += gcount[g].sum()
    return [assign[g] for g in groups]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="dataset folder, or a parent such as /kaggle/input")
    ap.add_argument("--out", default="splits/folds_v1.csv")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    root = find_data_root(args.data_root)
    print(f"data root: {root}")
    rows = []
    for split in ("train", "test"):
        for img in sorted((root / split / "images").glob("*.png")):
            stem = img.stem
            femur_id, xray_id = stem.split("_", 1)
            cls6 = yolo_to_cls6(read_yolo_class(root / split / "labels" / f"{stem}.txt"))
            c = BY_CLS6[cls6]
            rows.append(dict(
                file=img.name, split=split, femur_id=femur_id, xray_id=xray_id, cls6=cls6,
                yolo_id=c.yolo_id, code=c.code, class_name=c.name, location=c.location, grade=c.grade,
                pixel_md5=hashlib.md5(np.ascontiguousarray(read_gray(img)).tobytes()).hexdigest(),
            ))
    print(f"read {len(rows)} images")

    # Groups: x-ray ID, then merge pixel-identical images (keeps duplicate pairs in one fold).
    uf = UnionFind()
    by_md5: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        uf.find(f"{r['split']}:{r['xray_id']}")
        by_md5[r["pixel_md5"]].append(r)
    dup_note: dict[str, str] = {}
    for same in by_md5.values():
        if len(same) > 1:
            splits = {r["split"] for r in same}
            if len(splits) > 1:
                print(f"WARNING: identical image in train and test: {[r['file'] for r in same]}")
            for r in same[1:]:
                uf.union(f"{same[0]['split']}:{same[0]['xray_id']}", f"{r['split']}:{r['xray_id']}")
            for r in same:
                dup_note[r["file"]] = ";".join(o["file"] for o in same if o is not r)
    for r in rows:
        r["group"] = uf.find(f"{r['split']}:{r['xray_id']}")
        r["duplicate_of"] = dup_note.get(r["file"], "")

    # Contralateral partner: the other femur of the same x-ray, when the x-ray has two.
    by_xray: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_xray[(r["split"], r["xray_id"])].append(r)
    for members in by_xray.values():
        for r in members:
            others = [o for o in members if o is not r]
            r["n_femurs_in_xray"] = len(members)
            r["pair_file"] = others[0]["file"] if len(others) == 1 else ""
            r["pair_cls6"] = others[0]["cls6"] if len(others) == 1 else ""

    train = [r for r in rows if r["split"] == "train"]
    folds = stratified_group_kfold([r["cls6"] for r in train], [r["group"] for r in train], args.n_splits, args.seed)
    for r, k in zip(train, folds):
        r["fold"] = k
    for r in rows:
        r.setdefault("fold", -1)

    # Checks: groups never cross folds; counts match the paper's Table 1.
    group_folds: dict[str, set] = defaultdict(set)
    for r in train:
        group_folds[r["group"]].add(r["fold"])
    assert all(len(v) == 1 for v in group_folds.values()), "a group was split across folds"
    for split, expected in EXPECTED_COUNTS.items():
        got = Counter(r["cls6"] for r in rows if r["split"] == split)
        if dict(got) != expected:
            print(f"WARNING: {split} class counts {dict(sorted(got.items()))} differ from expected {expected}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["file", "split", "fold", "femur_id", "xray_id", "group", "cls6", "yolo_id", "code", "class_name",
            "location", "grade", "n_femurs_in_xray", "pair_file", "pair_cls6", "duplicate_of", "pixel_md5"]
    rows.sort(key=lambda r: (r["split"] != "train", r["file"]))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r[c] for c in cols} for r in rows)

    fingerprint = hashlib.sha1("\n".join(f"{r['split']}/{r['file']}:{r['cls6']}:{r['pixel_md5']}"
                                         for r in rows).encode()).hexdigest()
    table = {f"fold{k}": {BY_CLS6[c].name: sum(1 for r in train if r["fold"] == k and r["cls6"] == c)
                          for c in range(6)} for k in range(args.n_splits)}
    for k in range(args.n_splits):
        table[f"fold{k}"]["TOTAL"] = sum(1 for r in train if r["fold"] == k)
        table[f"fold{k}"]["x-rays"] = len({r["xray_id"] for r in train if r["fold"] == k})
    table["test"] = {BY_CLS6[c].name: sum(1 for r in rows if r["split"] == "test" and r["cls6"] == c) for c in range(6)}
    table["test"]["TOTAL"] = sum(1 for r in rows if r["split"] == "test")
    summary = dict(data_root=str(root), n_splits=args.n_splits, seed=args.seed, n_rows=len(rows),
                   fingerprint_sha1=fingerprint, counts=table,
                   duplicates=sorted({tuple(sorted([r["file"], r["duplicate_of"]])) for r in rows if r["duplicate_of"]}))
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=2))

    names = [BY_CLS6[c].name for c in range(6)] + ["TOTAL", "x-rays"]
    heads = [BY_CLS6[c].code for c in range(6)] + ["TOTAL", "x-rays"]
    print("\nclass codes: " + ", ".join(f"{BY_CLS6[c].code} = {BY_CLS6[c].name}" for c in range(6)))
    print("fold".ljust(8) + "".join(h.rjust(8) for h in heads))
    for k in range(args.n_splits):
        print(f"fold{k}".ljust(8) + "".join(str(table[f'fold{k}'][n]).rjust(8) for n in names))
    print("test".ljust(8) + "".join(str(table["test"].get(n, "")).rjust(8) for n in names))
    print(f"\nfingerprint {fingerprint}\nwrote {out} and {out.with_name(out.stem + '_summary.json')}")


if __name__ == "__main__":
    main()
