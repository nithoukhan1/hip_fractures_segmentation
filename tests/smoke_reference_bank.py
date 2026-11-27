"""End-to-end smoke test of scripts/build_reference_bank.py on the tiny synthetic dataset (CPU, under a minute).

It builds the fake dataset of tests/synthetic_data.py, pairs some femurs into two-femur x-rays (a healthy femur with
a fractured one, like the real data), runs the script with random weights (--no-pretrained, no download) through
kaggle/run.py, and checks every output: pairs.csv, references.csv (5 fold models x every femur), sanity.json,
summary.md, the figure, config.json and features.npz. Needs torch, timm, opencv and matplotlib:

    python tests/smoke_reference_bank.py --work /tmp/smoke_bank
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_data import make_dataset  # noqa: E402


def pair_up(folds_csv: Path) -> int:
    """In every fold (and the test set), put each healthy femur in one x-ray with the next femur, which is fractured."""
    rows = list(csv.DictReader(open(folds_csv, newline="")))
    n = 0
    for i, r in enumerate(rows[:-1]):
        p = rows[i + 1]
        if r["cls6"] == "0" and p["cls6"] != "0" and r["fold"] == p["fold"] and r["split"] == p["split"]:
            for a, b in ((r, p), (p, r)):
                a.update(xray_id=r["xray_id"], group=r["group"], n_femurs_in_xray="2", pair_file=b["file"],
                         pair_cls6=b["cls6"])
            n += 1
    with open(folds_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="empty scratch folder (deleted and rebuilt)")
    root = Path(ap.parse_args().work).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    data_root, folds_csv = make_dataset(root)
    n_pairs = pair_up(folds_csv)
    n = sum(1 for _ in open(folds_csv)) - 1
    runs = {"R00_smoke": {"step": "4", "script": "scripts/build_reference_bank.py", "pip": [],
                          "args": ["--no-pretrained", "--size", "192x128", "--device", "cpu", "--data-root",
                                   str(data_root), "--folds-csv", str(folds_csv), "--cache-dir", str(root / "cache")]}}
    (root / "runs.json").write_text(json.dumps(runs))
    results = root / "kaggle_working" / "results"
    cmd = [sys.executable, str(REPO / "kaggle" / "run.py"), "R00_smoke", "--results", str(results), "--runs",
           str(root / "runs.json")]
    print("$", " ".join(cmd), flush=True)
    assert subprocess.run(cmd).returncode == 0
    out = results / "R00_smoke"
    for f in ("pairs.csv", "references.csv", "sanity.json", "summary.md", "nn_fold0.png", "config.json",
              "run_meta.json"):
        assert (out / f).exists() and (out / f).stat().st_size > 0, f
    pairs = list(csv.DictReader(open(out / "pairs.csv")))
    table = list(csv.DictReader(open(out / "references.csv")))
    assert len(pairs) == n and sum(p["partner"] != "" for p in pairs) == 2 * n_pairs > 0
    assert len(table) == 5 * n and {t["role"] for t in table} == {"train", "held_out", "test"}
    folds = {r["file"]: r for r in csv.DictReader(open(folds_csv))}
    for t in table:  # every reference: a healthy training femur of another fold and another group
        for col in ("ref1", "ref2", "ref3", "ref4", "ref5", "random_healthy"):
            r = folds[t[col]]
            assert r["split"] == "train" and r["cls6"] == "0" and r["fold"] != t["fold_model"] and r["group"] != t["group"]
    checks = json.loads((out / "sanity.json").read_text())["checks"]
    assert checks["1_mirror_check"]["femurs_with_partner"] == 2 * n_pairs
    assert "Mirror check" in (out / "summary.md").read_text()
    cfg = json.loads((out / "config.json").read_text())
    assert not cfg["pretrained"] and cfg["feature_dim"] == 768 and cfg["n_femurs"] == n
    feats = np.load(root / "kaggle_working" / "work" / "R00_smoke" / "features.npz")
    assert feats["features"].shape == (n, 2, 768) and list(feats["files"]) == list(folds)
    assert json.loads((out / "run_meta.json").read_text())["exit_code"] == 0
    print(f"ok  reference bank: {n} femurs ({n_pairs} pairs), {len(table)} reference rows, checks, figure, features")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
