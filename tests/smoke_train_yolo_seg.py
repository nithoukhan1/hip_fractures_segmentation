"""End-to-end smoke test of scripts/train_yolo_seg.py on a tiny synthetic dataset (CPU, a few minutes).

It builds a fake dataset with the Kaggle folder layout (train/, val/ and test/, with part of the training femurs in
val/ to exercise the pooling), a matching fold file, then runs 5 folds two at a time on the CPU for 2 epochs and
checks every output. It also checks --export-only, the time guard, and the analysis scripts run_stats.py and
run_breakdown.py on the outputs. Needs torch and ultralytics, so run it in a Python environment that has them
(Kaggle, or a local virtual environment):

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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_data import make_dataset  # noqa: E402


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
    main_table = summary.split("## Ultralytics' default threshold")[0]  # an untrained model may call nothing at 0.25
    assert summary.count("| Fracture AUC |") == 2 and "n/a ± n/a" not in main_table, "summary table incomplete"
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

    # 5) per-class breakdown and comparison with the article (it also re-checks the pooled numbers of metrics.json)
    cmd = [sys.executable, str(REPO / "scripts" / "run_breakdown.py"), "SMOKE", "--name", "SMOKE_breakdown",
           "--results", str(root / "results")]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0, "run_breakdown.py failed"
    bd = json.loads((root / "results" / "analysis" / "SMOKE_breakdown" / "breakdown.json").read_text())["runs"][0]
    assert bd["oof"]["n"] == 60 and len(bd["oof_folds"]) == 5 and len(bd["test_models"]) == 5
    assert bd["selected_model"]["fold"] in range(5) and set(bd["areas"]) == {"2AB", "2CD", "3AB", "3CDE", "4"}
    print("ok  breakdown: per class, over folds and fold models, errors, areas")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
