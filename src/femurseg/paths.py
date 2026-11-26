"""Find the dataset and its files the same way on a local PC and on Kaggle.

Folder rule (decided 2026-11-26, see PROJECT_TRACKER.md):
  The official release has train/ (1,043 femurs), an EMPTY val/ and test/ (210).
  The Kaggle copy (utopianstar/dataset) moved 104 of the 1,043 training femurs into val/,
  without keeping the two femurs of an x-ray together. We always pool val/ (and valid/) back
  into train, so every copy of the data gives the same 1,043 training femurs and the same folds
  (splits/folds_v1.csv). The split names used everywhere in the code are therefore only
  "train" and "test".
"""
from __future__ import annotations

import os
from pathlib import Path

# CSV split name -> dataset folders that hold its files.
SPLIT_FOLDERS: dict[str, tuple[str, ...]] = {"train": ("train", "val", "valid"), "test": ("test",)}


def find_data_root(start: str | os.PathLike, max_depth: int = 6) -> Path:
    """Return the folder that holds ``train/images`` and ``test/images``, searching below ``start``.

    Locally ``start`` can be the "Femoral Fractures" folder itself. On Kaggle pass ``/kaggle/input``
    and the attached dataset is found wherever Kaggle mounted it.
    """
    start = Path(start)
    if (start / "train" / "images").is_dir() and (start / "test" / "images").is_dir():
        return start
    for dirpath, dirnames, _ in os.walk(start, followlinks=True):  # Kaggle may mount inputs via symlinks
        here = Path(dirpath)
        if len(here.relative_to(start).parts) >= max_depth:
            dirnames[:] = []
            continue
        if (here / "train" / "images").is_dir() and (here / "test" / "images").is_dir():
            return here
        dirnames[:] = [d for d in dirnames if d not in ("images", "labels", "masks")]
    raise FileNotFoundError(f"no folder with train/images and test/images found under {start}")


def build_index(root: str | os.PathLike) -> dict[tuple[str, str], dict]:
    """Map (split, file name) to where that femur's files are on disk.

    ``split`` is "train" or "test" (val/ and valid/ are pooled into "train", see the module note).
    Each value holds ``folder`` (the dataset folder it was found in), ``image``, ``label`` (None for a
    femur without fracture, which has no label file) and ``mask`` (None if missing).
    A file name found in two folders of the same split is an error, so nothing is counted twice.
    """
    root = Path(root)
    index: dict[tuple[str, str], dict] = {}
    for split, folders in SPLIT_FOLDERS.items():
        for folder in folders:
            img_dir = root / folder / "images"
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.glob("*.png")):
                key = (split, img.name)
                if key in index:
                    raise ValueError(f"{img.name} is in both {index[key]['folder']}/ and {folder}/ of split {split}")
                label = root / folder / "labels" / f"{img.stem}.txt"
                mask = root / folder / "masks" / img.name
                index[key] = {"folder": folder, "image": img,
                              "label": label if label.exists() else None,
                              "mask": mask if mask.exists() else None}
    return index
