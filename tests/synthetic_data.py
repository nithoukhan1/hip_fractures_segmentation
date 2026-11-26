"""Tiny synthetic dataset with the Kaggle folder layout, shared by the end-to-end smoke tests.

make_dataset() writes fake femurs (noise background, a bright ellipse = the fracture, class-dependent brightness and
place) into train/, val/ (1 in 4 training femurs, to exercise the pooling) and test/, with images, YOLO labels and
masks like the real data, and a matching fold file (5 folds of 12 femurs, 12 test femurs, every class in every fold).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

W, H = 256, 384  # synthetic image size (width x height), same 2:3 shape as the real 1024 x 1536 images
COLS = ["file", "split", "fold", "femur_id", "xray_id", "group", "cls6", "yolo_id", "code", "class_name", "location",
        "grade", "n_femurs_in_xray", "pair_file", "pair_cls6", "duplicate_of", "pixel_md5"]


def make_dataset(root: Path, n_per_fold: int = 12, n_test: int = 12, seed: int = 0) -> tuple[Path, Path]:
    """Fake femurs: noise background, a bright ellipse = the fracture (class-dependent brightness and place)."""
    import cv2

    rng = np.random.default_rng(seed)
    data = root / "input" / "datasets" / "someone" / "dataset" / "femoral" / "data"  # nested like Kaggle
    rows = []
    specs = [("train", k, n_per_fold) for k in range(5)] + [("test", -1, n_test)]
    i = 0
    for split, fold, n in specs:
        for j in range(n):
            cls6 = j % 6  # every fold holds every class
            name = f"{i:05d}_00000.png"
            folder = "test" if split == "test" else ("val" if j % 4 == 3 else "train")  # 1 in 4 train femurs in val/
            for sub in ("images", "labels", "masks"):
                (data / folder / sub).mkdir(parents=True, exist_ok=True)
            img = rng.normal(60, 12, (H, W)).clip(0, 255).astype(np.uint8)
            msk = np.zeros((H, W), np.uint8)
            if cls6 > 0:
                cx, cy = rng.integers(70, W - 70), 80 + 40 * cls6 + rng.integers(-15, 15)
                ax, ay = rng.integers(18, 35), rng.integers(12, 25)
                cv2.ellipse(msk, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, int(cls6), -1)
                img[msk > 0] = np.clip(img[msk > 0].astype(int) + 60 + 25 * cls6, 0, 255)
                cnt = max(cv2.findContours((msk > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                          key=cv2.contourArea)[:, 0, :] / [W, H]
                (data / folder / "labels" / f"{Path(name).stem}.txt").write_text(
                    f"{cls6 - 1} " + " ".join(f"{v:.6f}" for v in cnt.ravel()) + "\n")
            cv2.imwrite(str(data / folder / "images" / name), img)
            cv2.imwrite(str(data / folder / "masks" / name), msk)
            rows.append({"file": name, "split": split, "fold": fold, "femur_id": f"{i:05d}", "xray_id": f"{i:05d}",
                         "group": f"{split}:{i:05d}", "cls6": cls6, "yolo_id": cls6 - 1, "code": "", "class_name": "",
                         "location": "", "grade": "", "n_femurs_in_xray": 1, "pair_file": "", "pair_cls6": "",
                         "duplicate_of": "", "pixel_md5": ""})
            i += 1
        i += 100  # keep file names of different folds apart
    csv_path = root / "folds_smoke.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return root / "input", csv_path
