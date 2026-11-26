"""Images and masks for the PyTorch models (S04 onwards): a resized cache on local disk, a Dataset and augmentations.

Why a cache: reading one PNG from /kaggle/input takes about 0.1 s, and every model resizes the 1024 x 1536 images
to the same input size (768 x 512 = exactly half). The controller therefore reads every femur ONCE, resizes it and
writes three arrays to a local folder (default /tmp/femurseg_cache, never the Kaggle output):

  images.npy     (N, H, W) uint8   resized image (area interpolation)
  masks.npy      (N, H, W) uint8   resized mask: 0 = background, cls6 inside the fracture
  gt_bits.npy    (N, B)    uint8   released full-resolution fracture mask, bit-packed (for the metrics)
  meta.json      file list, sizes, and the fingerprint of the inputs

The fold processes open these arrays memory-mapped (shared, nothing is copied), so two folds on two GPUs and their
data-loader workers all read the same pages.

Augmentations (training only; Report 03 §6): horizontal flip, small affine (scale 0.9-1.1, rotation +-10 degrees,
shift +-5%), brightness/contrast, gamma, and mild noise or blur. No vertical flip (anatomy). The image stays uint8
until it reaches the GPU, where normalize() turns it into the 3-channel ImageNet-normalised float tensor.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data import FemurRecord, read_png
from .paths import build_index, find_data_root

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------------------------------------ cache


def resize_pair(img: np.ndarray, msk: np.ndarray, size_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize an image (area interpolation) and its mask. The mask is resized as a fracture/background map with the
    same interpolation (pixel kept if at least half of it is fracture), then given the femur's class value, so it is
    centred exactly like the image (a nearest-neighbour resize would shift it by half a pixel)."""
    import cv2

    h, w = size_hw
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if msk.any():
        frac = cv2.resize((msk > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5
        value = int(np.bincount(msk[msk > 0]).argmax())  # one class per femur in this dataset
        msk_small = frac.astype(np.uint8) * value
    else:
        msk_small = np.zeros((h, w), np.uint8)
    return small, msk_small


def cache_key(root: Path, records: list[FemurRecord], size_hw: tuple[int, int]) -> str:
    text = json.dumps([str(root), list(size_hw), [(r.split, r.file, r.cls6) for r in records]])
    return hashlib.md5(text.encode()).hexdigest()[:12]


def build_cache(data_root: str | Path, records: list[FemurRecord], size_hw: tuple[int, int], cache_dir: str | Path,
                threads: int = 8) -> Path:
    """Write the resized cache for ``records`` (unless it already exists) and return its folder."""
    root = find_data_root(data_root)
    folder = Path(cache_dir) / f"{size_hw[0]}x{size_hw[1]}_{cache_key(root, records, size_hw)}"
    if (folder / "meta.json").exists():
        return folder
    t0 = time.time()
    index = build_index(root)
    missing = [r.key for r in records if r.key not in index]
    if missing:
        raise FileNotFoundError(f"{len(missing)} femurs of the fold file are not on disk, e.g. {missing[:3]}")
    tmp = folder.with_name(folder.name + ".part")
    tmp.mkdir(parents=True, exist_ok=True)
    first = read_png(index[records[0].key]["image"])
    full_hw = first.shape
    n, (h, w) = len(records), size_hw
    images = np.lib.format.open_memmap(tmp / "images.npy", "w+", np.uint8, (n, h, w))
    masks = np.lib.format.open_memmap(tmp / "masks.npy", "w+", np.uint8, (n, h, w))
    bits = np.lib.format.open_memmap(tmp / "gt_bits.npy", "w+", np.uint8, (n, (full_hw[0] * full_hw[1] + 7) // 8))

    def one(i: int) -> int:
        r = records[i]
        img = read_png(index[r.key]["image"])
        p = index[r.key]["mask"]
        msk = read_png(p) if p is not None else np.zeros_like(img)
        if img.shape != full_hw or msk.shape != full_hw:
            raise ValueError(f"{r.file}: image {img.shape} / mask {msk.shape}, expected {full_hw} like the others")
        if msk.any() and int(msk.max()) != r.cls6:
            raise ValueError(f"{r.file}: mask value {int(msk.max())} but class {r.cls6} in the fold file")
        images[i], masks[i] = resize_pair(img, msk, size_hw)
        bits[i] = np.packbits((msk > 0).ravel())
        return int((msk > 0).sum())

    with ThreadPoolExecutor(max_workers=threads) as ex:
        areas = list(ex.map(one, range(n)))
    for a in (images, masks, bits):
        a.flush()
    meta = {"files": [[r.split, r.file] for r in records], "cls6": [r.cls6 for r in records], "area_gt": areas,
            "size_hw": list(size_hw), "full_hw": list(full_hw), "data_root": str(root),
            "seconds": round(time.time() - t0, 1)}
    (tmp / "meta.json").write_text(json.dumps(meta))
    tmp.rename(folder)  # complete caches only: a crash leaves a .part folder, which is rebuilt next time
    return folder


class Cache:
    """Read-only, memory-mapped view of a cache folder."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        meta = json.loads((self.folder / "meta.json").read_text())
        self.meta = meta
        self.size_hw = tuple(meta["size_hw"])
        self.full_hw = tuple(meta["full_hw"])
        self.images = np.load(self.folder / "images.npy", mmap_mode="r")
        self.masks = np.load(self.folder / "masks.npy", mmap_mode="r")
        self.bits = np.load(self.folder / "gt_bits.npy", mmap_mode="r")
        self.row = {(s, f): i for i, (s, f) in enumerate(meta["files"])}

    def rows(self, records: list[FemurRecord]) -> list[int]:
        return [self.row[r.key] for r in records]

    def gt_full(self, row: int) -> np.ndarray:
        """Released fracture mask at full resolution, as booleans."""
        h, w = self.full_hw
        return np.unpackbits(self.bits[row], count=h * w).reshape(h, w).astype(bool)


# ---------------------------------------------------------------------------------------------- dataset


def train_transforms(seed: int | None = None):
    import albumentations as A
    import cv2

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), border_mode=cv2.BORDER_CONSTANT,
                 fill=0, fill_mask=0, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.OneOf([A.GaussNoise(std_range=(0.02, 0.06), p=1.0), A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=0.3),
    ], seed=seed)


class FemurDataset:
    """(image uint8 (1, H, W), mask uint8 (H, W), cls6) for the given cache rows; augmented when ``augment``."""

    def __init__(self, cache: Cache, rows: list[int], augment: bool, seed: int | None = None):
        self.cache, self.rows = cache, list(rows)
        self.aug = train_transforms(seed) if augment else None
        self.cls6 = [cache.meta["cls6"][i] for i in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import torch

        row = self.rows[i]
        img, msk = np.array(self.cache.images[row]), np.array(self.cache.masks[row])
        if self.aug is not None:
            t = self.aug(image=img, mask=msk)
            img, msk = t["image"], t["mask"]
        return torch.from_numpy(np.ascontiguousarray(img))[None], torch.from_numpy(np.ascontiguousarray(msk)), self.cls6[i]


def worker_init_fn(worker_id: int) -> None:
    """Give every data-loader worker its own augmentation random stream (torch seeds each worker differently)."""
    import random

    import cv2
    import torch

    cv2.setNumThreads(1)  # one thread per worker: the workers already run in parallel on the few CPU cores
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    info = torch.utils.data.get_worker_info()
    if info is not None and getattr(info.dataset, "aug", None) is not None:
        info.dataset.aug.set_random_seed(seed)


def normalize(x, device):
    """uint8 (B, 1, H, W) -> float (B, 3, H, W), ImageNet-normalised, on ``device`` (grey copied to 3 channels)."""
    import torch

    x = x.to(device, non_blocking=True).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)
