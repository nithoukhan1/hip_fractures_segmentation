"""Find the dataset folder the same way on a local PC and on Kaggle."""
from __future__ import annotations

import os
from pathlib import Path


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
