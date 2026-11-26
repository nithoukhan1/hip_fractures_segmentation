"""Data access shared by every model: fold records, file paths and an in-memory image store.

Typical use (locally or on Kaggle):

    from femurseg.data import ImageStore, fold_split, load_records
    recs = load_records()                         # 1,253 rows of splits/folds_v1.csv
    train, held_out = fold_split(recs, fold=0)    # held_out = the fold this model never sees
    test = [r for r in recs if r.split == "test"]
    store = ImageStore("/kaggle/input", recs)     # finds every file; Kaggle's val/ is pooled into train
    store.preload()                               # reads all images and masks once (about 4 GB as uint8)
    img = store.image(rec)                        # (1536, 1024) uint8, grayscale
    msk = store.mask(rec)                         # (1536, 1024) uint8, 0 = background, 1..5 = cls6

Why preload: on Kaggle, reading one PNG from /kaggle/input takes about 95 ms (S00_env_check), so reading
every image again in every epoch would make the CPU, not the GPU, the bottleneck.

Data facts this module relies on (checked on the local copy, 2026-11-26):
  * images are 8-bit grayscale PNGs of 1024 x 1536 (width x height);
  * masks have the same size; pixel value = cls6 of the femur inside the fracture region, 0 elsewhere;
    femurs without fracture have an all-zero mask and no label file;
  * each label file holds exactly one YOLO-seg polygon (normalised x y pairs), whose rasterisation agrees
    with the mask (Dice about 0.999).
"""
from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classes import BY_CLS6
from .paths import build_index, find_data_root

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FOLDS_CSV = REPO / "splits" / "folds_v1.csv"


@dataclass(frozen=True)
class FemurRecord:
    """One femur (one row of splits/folds_v1.csv)."""

    file: str
    split: str            # "train" (Kaggle's val/ included) or "test"
    fold: int             # 0..4 for train, -1 for test
    femur_id: str
    xray_id: str
    group: str            # x-rays that must stay in the same fold
    cls6: int             # 0 = no fracture, 1..5 = fracture class (see classes.py)
    n_femurs_in_xray: int
    pair_file: str        # the other femur of the same x-ray, "" if none
    pair_cls6: int        # -1 if no pair
    duplicate_of: str
    pixel_md5: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.split, self.file)

    @property
    def has_fracture(self) -> bool:
        return self.cls6 > 0

    @property
    def code(self) -> str:
        return BY_CLS6[self.cls6].code

    @property
    def location(self) -> str:
        return BY_CLS6[self.cls6].location


def load_records(csv_path: str | Path = DEFAULT_FOLDS_CSV) -> list[FemurRecord]:
    """Read the fixed fold file. Rows keep the file's order (train first, then test, by file name)."""
    recs = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            recs.append(FemurRecord(
                file=r["file"], split=r["split"], fold=int(r["fold"]), femur_id=r["femur_id"],
                xray_id=r["xray_id"], group=r["group"], cls6=int(r["cls6"]),
                n_femurs_in_xray=int(r["n_femurs_in_xray"]), pair_file=r["pair_file"],
                pair_cls6=int(r["pair_cls6"]) if r["pair_cls6"] != "" else -1,
                duplicate_of=r["duplicate_of"], pixel_md5=r["pixel_md5"]))
    return recs


def fold_split(records: list[FemurRecord], fold: int) -> tuple[list[FemurRecord], list[FemurRecord]]:
    """Return (training femurs, held-out femurs) for the model of ``fold``. Test femurs are never included."""
    train = [r for r in records if r.split == "train" and r.fold != fold]
    held_out = [r for r in records if r.split == "train" and r.fold == fold]
    if not held_out:
        raise ValueError(f"fold {fold} has no femurs")
    return train, held_out


def read_png(path: Path) -> np.ndarray:
    """Read an 8-bit PNG as a 2-D uint8 array (OpenCV when available: faster than Pillow)."""
    try:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise OSError(f"cannot read {path}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr
    except ImportError:
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im.convert("L") if im.mode not in ("L", "P") else im).copy()


def read_polygon(label_path: Path | None) -> tuple[int, np.ndarray] | None:
    """Return (yolo_id, polygon as an (N, 2) array of normalised x, y) or None for a femur without fracture."""
    if label_path is None or not Path(label_path).exists():
        return None
    lines = [ln.split() for ln in Path(label_path).read_text().splitlines() if ln.strip()]
    if not lines:
        return None
    if len(lines) != 1:
        raise ValueError(f"{label_path}: expected one polygon, found {len(lines)}")
    vals = lines[0]
    return int(float(vals[0])), np.asarray(vals[1:], dtype=np.float64).reshape(-1, 2)


class ImageStore:
    """Finds every femur's files and keeps images and masks in memory once they are read."""

    def __init__(self, data_root: str | Path, records: list[FemurRecord] | None = None):
        self.root = find_data_root(data_root)
        self.index = build_index(self.root)
        self._img: dict[tuple[str, str], np.ndarray] = {}
        self._msk: dict[tuple[str, str], np.ndarray] = {}
        if records is not None:
            missing = [r.key for r in records if r.key not in self.index]
            if missing:
                raise FileNotFoundError(f"{len(missing)} femurs of the fold file are not on disk, e.g. {missing[:3]}")

    def path(self, rec: FemurRecord, kind: str = "image") -> Path | None:
        """Path of the femur's image, label or mask (label and mask can be None)."""
        return self.index[rec.key][kind]

    def image(self, rec: FemurRecord) -> np.ndarray:
        if rec.key not in self._img:
            self._img[rec.key] = read_png(self.path(rec, "image"))
        return self._img[rec.key]

    def mask(self, rec: FemurRecord) -> np.ndarray:
        """Released ground-truth mask: 0 = background, cls6 inside the fracture. All zeros if there is none."""
        if rec.key not in self._msk:
            p = self.path(rec, "mask")
            self._msk[rec.key] = read_png(p) if p is not None else np.zeros_like(self.image(rec))
        return self._msk[rec.key]

    def polygon(self, rec: FemurRecord) -> tuple[int, np.ndarray] | None:
        return read_polygon(self.path(rec, "label"))

    def preload(self, records: list[FemurRecord], masks: bool = True, workers: int = 8) -> float:
        """Read all images (and masks) of ``records`` with a thread pool; return the seconds it took."""
        t0 = time.time()

        def load(rec: FemurRecord) -> None:
            self.image(rec)
            if masks:
                self.mask(rec)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(load, records))
        return time.time() - t0

    def memory_gb(self) -> float:
        return sum(a.nbytes for d in (self._img, self._msk) for a in d.values()) / 1e9
