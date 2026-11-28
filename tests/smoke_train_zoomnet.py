"""End-to-end smoke test of the segment-and-zoom network in scripts/train_net.py (CPU, a few minutes).

On the synthetic dataset of tests/synthetic_data.py (384 x 256 images, the 2:3 shape of the release), without
downloads, it checks:
  1. the pre-flight of zoom:<ConvNeXt-T> at the full synthetic size;
  2. a 2-fold run with the test set, teacher forcing and the oracle and deletion tests: every output file, the zoom
     crop tables (per_case_<oof|test>_zoom.csv) and their pointing game in metrics.json and summary.md;
  3. a fold-0 development run (--no-test), as the screening runs on Kaggle use it.

    python tests/smoke_train_zoomnet.py --work /tmp/smoke_zoom
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_data import H, W, make_dataset  # noqa: E402

NET = "zoom:convnext_tiny.fb_in22k_ft_in1k"


def run(args_list: list[str]) -> int:
    cmd = [sys.executable, str(REPO / "scripts" / "train_net.py"), *args_list]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/smoke_zoom")
    a = ap.parse_args()
    root = Path(a.work).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    data_root, folds_csv = make_dataset(root, n_per_fold=6, n_test=6)
    common = ["--no-pretrained", "--size", f"{H}x{W}", "--batch", "4", "--micro-batch", "2", "--preflight-steps", "2",
              "--zoom-window", "128", "--zoom-input", "64", "--data-root", str(data_root), "--folds-csv",
              str(folds_csv), "--cache-dir", str(root / "cache")]
    results = root / "results"

    out = results / "ZOOM_CHECK"
    assert run([*common, "--check-nets", NET, "--gpus", "cpu", "--out", str(out)]) == 0
    rep = json.loads((out / "net_check.json").read_text())
    assert rep["nets"][0]["ok"] and rep["nets"][0]["est_fold_min"] > 0, rep
    print("ok  pre-flight")

    out = results / "ZOOM_RUN"
    rc = run([*common, "--net", NET, "--epochs", "2", "--tf-epochs", "1", "--folds", "0,1", "--gpus", "cpu,cpu",
              "--extra-tests", "oracle,deletion", "--out", str(out)])
    assert rc == 0, f"exit code {rc}; see {root / 'work' / 'ZOOM_RUN' / 'logs'}"
    res = json.loads((out / "metrics.json").read_text())
    assert res["folds_completed"] == [0, 1] and not res["folds_failed"], res
    assert "test_ensemble" in res and set(res["tests"]) == {"oracle", "deletion", "deletion_control"}
    z = res["zoom"]
    assert set(z) == {"oof", "test"} and z["oof"]["n_fractured"] == 10 and 0 <= z["oof"]["hit_rate"] <= 1, z
    with open(out / "per_case_oof_zoom.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12 and all(64 <= float(r["zoom_cy"]) <= H - 64 and 64 <= float(r["zoom_cx"]) <= W - 64
                                   for r in rows)
    meta = json.loads((out / "folds" / "fold0" / "fold_meta.json").read_text())
    assert meta["window"] == 128 and meta["zoom_input"] == 64 and meta["net"] == NET
    curve = (out / "folds" / "fold0" / "train_curve.csv").read_text()
    assert "ce_global" in curve and "tf_prob" in curve
    summary = (out / "summary.md").read_text()
    assert "Segment-and-zoom" in summary and "Pointing game of the zoom crop" in summary
    for name in ("per_case_oof.csv", "per_case_test_ensemble.csv", "per_case_oof_oracle.csv",
                 "per_case_oof_deletion.csv", "per_case_test_zoom.csv"):
        assert (out / name).exists(), name
    print("ok  2-fold run with the test set and the tests")

    out = results / "ZOOM_DEV"
    assert run([*common, "--net", NET, "--epochs", "1", "--folds", "0", "--no-test", "--gpus", "cpu",
                "--out", str(out)]) == 0
    res = json.loads((out / "metrics.json").read_text())
    assert res["folds_completed"] == [0] and "test_ensemble" not in res and set(res["zoom"]) == {"oof"}
    print("ok  fold-0 development run")
    print("all zoom smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
