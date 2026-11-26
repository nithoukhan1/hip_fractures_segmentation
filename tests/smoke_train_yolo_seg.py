"""End-to-end smoke test of scripts/train_yolo_seg.py on a tiny synthetic dataset (CPU, a few minutes).

It builds a fake dataset with the Kaggle folder layout (train/, val/ and test/, with part of the training femurs in
val/ to exercise the pooling), a matching fold file, then runs 5 folds two at a time on the CPU for 2 epochs and
checks every output. It also checks --export-only and the time guard. Needs torch and ultralytics, so run it in a
Python environment that has them (Kaggle, or a local virtual environment):

    python tests/smoke_train_yolo_seg.py --work /tmp/smoke [--model yolov8n-seg.pt]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
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


def run(args_list: list[str]) -> int:
    cmd = [sys.executable, str(REPO / "scripts" / "train_yolo_seg.py"), *args_list]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="empty scratch folder (deleted and rebuilt)")
    ap.add_argument("--model", default="yolov8n-seg.yaml", help="yaml = untrained (offline); .pt = COCO weights")
    a = ap.parse_args()
    root = Path(a.work).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    data_root, folds_csv = make_dataset(root)
    common = ["--model", a.model, "--imgsz", "256", "--batch", "4", "--data-root", str(data_root),
              "--folds-csv", str(folds_csv), "--cache", "ram"]

    # 1) export only
    out = root / "results" / "SMOKE_export"
    assert run([*common, "--out", str(out), "--export-only", "--folds", "0,4"]) == 0
    for k in (0, 4):
        ydir = root / "work" / "SMOKE_export" / "yolo_data" / f"fold{k}"
        n_train = len(list((ydir / "images" / "train").iterdir()))
        n_val = len(list((ydir / "images" / "val").iterdir()))
        assert (n_train, n_val) == (48, 12), (n_train, n_val)
        n_poly = sum(1 for p in (ydir / "labels" / "train").glob("*.txt") if p.stat().st_size > 0)
        assert n_poly == 40, n_poly  # 5 of every 6 synthetic femurs have a fracture
    print("ok  export-only")

    # 2) time guard: nothing fits in a 0.001 h budget -> fold skipped, exit code 3
    out = root / "results" / "SMOKE_guard"
    assert run([*common, "--out", str(out), "--folds", "0", "--gpus", "cpu", "--epochs", "1",
                "--time-budget-h", "0.001", "--est-fold-min", "5"]) == 3
    res = json.loads((out / "metrics.json").read_text())
    assert res["folds_skipped_time_guard"] == [0] and res["folds_completed"] == []
    print("ok  time guard")

    # 3) full run: 5 folds, two at a time on the CPU, 2 epochs each
    out = root / "results" / "SMOKE"
    rc = run([*common, "--out", str(out), "--epochs", "2", "--gpus", "cpu,cpu", "--est-fold-min", "3"])
    assert rc == 0, f"exit code {rc}; see {root / 'work' / 'SMOKE' / 'logs'}"
    res = json.loads((out / "metrics.json").read_text())
    assert res["folds_completed"] == [0, 1, 2, 3, 4] and not res["folds_failed"], res
    for block in (res, res["at_default_threshold_0.25"]):
        assert block["oof"]["n"] == 60 and set(block["oof_per_fold"]) == {"0", "1", "2", "3", "4"}
        assert block["test_ensemble"]["n"] == 12 and "auc_fracture" in block["test_models_mean"]
        for key in ("auc_fracture", "accuracy_6class", "macro_f1", "dice_fractured_mean", "kappa", "ece_derived"):
            assert key in block["oof"], key
    with open(out / "per_case_oof.csv") as f:
        oof = list(csv.DictReader(f))
    with open(out / "per_case_test.csv") as f:
        test = list(csv.DictReader(f))
    with open(out / "per_case_test_ensemble.csv") as f:
        ens = list(csv.DictReader(f))
    assert len(oof) == 60 and len({r["file"] for r in oof}) == 60
    assert len(test) == 60 and len(ens) == 12
    assert all(r["fold"] == r["model"] for r in oof), "an OOF femur was predicted by a model that trained on it"
    for k in range(5):
        fd = out / "folds" / f"fold{k}"
        for name in ("oof.csv", "test.csv", "fold_meta.json", "train_curve.csv", "ultralytics_args.yaml", "log_tail.txt"):
            assert (fd / name).exists(), fd / name
        assert not (root / "work" / "SMOKE" / "yolo_data" / f"fold{k}").exists(), "YOLO data folder not removed"
        meta = json.loads((fd / "fold_meta.json").read_text())
        assert meta["n_train"] == 48 and meta["n_held_out"] == 12 and Path(meta["weights"]).exists()
        uargs = (fd / "ultralytics_args.yaml").read_text()
        assert "val: false" in uargs and "epochs: 2" in uargs, "training settings not passed through"
        workers = max(1, min(8, (os.cpu_count() or 2) // 2))  # the controller's rule for two parallel folds
        assert f"workers: {workers}" in uargs, "worker count not passed to the fold process"
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["start_weights"] and "summary.md" in {p.name for p in out.iterdir()}
    summary = (out / "summary.md").read_text()
    assert summary.count("| Fracture AUC |") == 2 and "n/a ± n/a" not in summary, "summary table incomplete"
    print((out / "summary.md").read_text())
    print("ok  full run: 5 folds, OOF, test, ensemble, summary")

    # 4) statistics on the filed outputs: intervals and a paired comparison of the run with itself
    cmd = [sys.executable, str(REPO / "scripts" / "run_stats.py"), "SMOKE", "SMOKE", "--compare", "--name", "SMOKE_stats",
           "--results", str(root / "results"), "--folds-csv", str(folds_csv), "--n-boot", "200", "--n-perm", "200"]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0, "run_stats.py failed"
    st = json.loads((root / "results" / "analysis" / "SMOKE_stats" / "stats.json").read_text())
    ci = st["intervals"]["SMOKE"]["test_ensemble"]["accuracy_6class"]
    assert ci["lo"] <= ci["estimate"] <= ci["hi"]
    same = st["comparisons"]["SMOKE vs SMOKE"]["oof"]
    assert all(r["p"] == 1.0 for r in same.values()), "a run compared with itself must give p = 1"
    print("ok  statistics: intervals and paired comparison")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
