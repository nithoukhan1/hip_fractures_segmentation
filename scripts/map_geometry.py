#!/usr/bin/env python3
"""Geometry of each femur's predicted fracture region, from a run's saved probability maps (analysis C08, test 3 of
Analysis/06 §5).

Reads prob_maps/<oof|test>/fold<k>/<file> (768 x 512 PNG, value = 255 x fracture probability) of a run's Kaggle work
folder, either from the folder itself or straight from the downloaded output zip (paths ending in
work/<RUN>/prob_maps/...). The region is made as scripts/train_net.py makes it: the map is resized to the full
1024 x 1536 image (linear interpolation) and thresholded at 0.5. For the test set the maps of the 5 fold models are
averaged first (the test ensemble).

Per femur it writes the measures that EDA/eda_femoral.py wrote for the released masks (EDA/per_image_stats.csv):
area fraction, centroid x and y, bounding-box width and height (all relative to the image size), and the number of
connected parts (8-connectivity). scripts/run_interpretability.py --geometry reads the file.

Needs only numpy and OpenCV, so it can run where the large zip is (for example on the person's PC):

  python scripts/map_geometry.py results/_inbox/filed/A6_slgnet.zip --run A6_slgnet \\
      --out results/analysis/C08_paperA_interpretability/geometry_A6_slgnet.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

FULL_H, FULL_W = 1536, 1024
FIELDS = ["part", "file", "fold", "n_maps", "area_px", "area_frac", "cx", "cy", "bw", "bh", "components"]
PATTERN = re.compile(r"(?:^|/)prob_maps/(oof|test)/fold(\d+)/([^/]+\.png)$")


class Maps:
    """The base-mode probability maps of one run, from a work folder or a zip, keyed by (part, fold, file)."""

    def __init__(self, path: Path, run: str):
        self.zip = zipfile.ZipFile(path) if path.is_file() else None
        self.root = path
        if self.zip is not None:
            names = [n for n in self.zip.namelist() if f"{run}/prob_maps/" in n]
        else:
            names = [p.relative_to(path).as_posix() for p in path.rglob("*.png")]
        self.items: dict[tuple[str, int, str], str] = {}
        for n in names:
            m = PATTERN.search(n)
            if m:
                self.items[(m.group(1), int(m.group(2)), m.group(3))] = n
        if not self.items:
            raise SystemExit(f"no prob_maps/<oof|test>/fold<k>/*.png of {run} found in {path}")

    def read(self, key: tuple[str, int, str]) -> np.ndarray:
        import cv2

        name = self.items[key]
        buf = self.zip.read(name) if self.zip is not None else (self.root / name).read_bytes()
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"cannot decode {name}")
        return img.astype(np.float32) / 255


def geometry(prob: np.ndarray) -> dict:
    """The region (full size, >= 0.5) and its measures, as EDA/eda_femoral.py measures a released mask."""
    import cv2

    fg = cv2.resize(prob, (FULL_W, FULL_H), interpolation=cv2.INTER_LINEAR) >= 0.5
    out = {"area_px": int(fg.sum()), "area_frac": round(float(fg.mean()), 8)}
    if not fg.any():
        return out | {"cx": "", "cy": "", "bw": "", "bh": "", "components": 0}
    ys, xs = np.nonzero(fg)
    n_parts = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)[0] - 1
    return out | {"cx": round(float(xs.mean() / FULL_W), 6), "cy": round(float(ys.mean() / FULL_H), 6),
                  "bw": round(float((xs.max() - xs.min() + 1) / FULL_W), 6),
                  "bh": round(float((ys.max() - ys.min() + 1) / FULL_H), 6), "components": int(n_parts)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="the run's work folder (work/<RUN>) or its downloaded Kaggle output zip")
    ap.add_argument("--run", required=True, help="run name, e.g. A6_slgnet")
    ap.add_argument("--out", required=True, help="CSV to write")
    a = ap.parse_args(argv)
    maps = Maps(Path(a.source), a.run)
    rows = []
    for (part, fold, f) in sorted(k for k in maps.items if k[0] == "oof"):
        rows.append({"part": "oof", "file": f, "fold": fold, "n_maps": 1, **geometry(maps.read((part, fold, f)))})
    test: dict[str, list[int]] = {}
    for (part, fold, f) in maps.items:
        if part == "test":
            test.setdefault(f, []).append(fold)
    for f in sorted(test):
        mean = np.mean([maps.read(("test", k, f)) for k in sorted(test[f])], axis=0)
        rows.append({"part": "test_ensemble", "file": f, "fold": -1, "n_maps": len(test[f]), **geometry(mean)})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n_oof = sum(r["part"] == "oof" for r in rows)
    print(f"{a.run}: {n_oof} OOF femurs and {len(rows) - n_oof} test femurs "
          f"(maps per test femur: {sorted({len(v) for v in test.values()})}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
