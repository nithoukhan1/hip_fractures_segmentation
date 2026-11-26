"""End-to-end smoke test of scripts/train_net.py on a tiny synthetic dataset (CPU, about 6 minutes).

It builds the fake dataset of tests/synthetic_data.py, then checks, without any download (random initialisation):
  1. --check-nets (the S00_net_check pre-flight) for a classifier, a segmenter and M0;
  2. the time guard (a fold that cannot fit the budget is skipped, exit code 3);
  3. full 5-fold runs, two folds at a time on the CPU, of a classifier, a segmenter and M0 (M0 with gradient
     accumulation): every output file, out-of-fold integrity, probability maps, the ensemble, and that metrics.json
     equals a re-evaluation of the per-case files;
  4. run_stats.py (intervals and paired comparisons; no Dice test against a classifier) and run_breakdown.py.
Needs torch, timm, segmentation-models-pytorch, albumentations, scipy:

    python tests/smoke_train_net.py --work /tmp/smoke_net
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from femurseg.evaluation import evaluate, read_csv  # noqa: E402
from synthetic_data import make_dataset  # noqa: E402

CONVNEXT = "convnext_tiny.fb_in22k_ft_in1k"
NETS = {"SMOKE_CLS": f"cls:{CONVNEXT}", "SMOKE_SEG": "seg:Unet:resnet34", "SMOKE_M0": f"m0:{CONVNEXT}"}


def run(args_list: list[str], script: str = "train_net.py", quiet: bool = False) -> int:
    cmd = [sys.executable, str(REPO / "scripts" / script), *args_list]
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=quiet, text=True)
    if quiet and r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
    return r.returncode


def same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="empty scratch folder (deleted and rebuilt)")
    a = ap.parse_args()
    root = Path(a.work).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    data_root, folds_csv = make_dataset(root)
    common = ["--no-pretrained", "--size", "192x128", "--batch", "4", "--preflight-steps", "3", "--data-root",
              str(data_root), "--folds-csv", str(folds_csv), "--cache-dir", str(root / "cache")]
    results = root / "results"

    # 1) pre-flight of several networks (S00_net_check)
    out = results / "SMOKE_CHECK"
    assert run([*common, "--check-nets", ",".join(NETS.values()), "--gpus", "cpu,cpu", "--out", str(out)]) == 0
    rep = json.loads((out / "net_check.json").read_text())
    assert [n["net"] for n in rep["nets"]] == list(NETS.values()) and all(n["ok"] for n in rep["nets"])
    assert all(n["micro_batch"] == 4 and n["est_fold_min"] > 0 for n in rep["nets"])
    assert (out / "net_check.md").read_text().count("| yes |") == 3
    print("ok  net check: 3 networks built, stepped and timed")

    # 2) time guard
    out = results / "SMOKE_GUARD"
    assert run([*common, "--net", NETS["SMOKE_CLS"], "--folds", "0", "--gpus", "cpu", "--epochs", "1",
                "--time-budget-h", "0.0001", "--out", str(out)]) == 3
    res = json.loads((out / "metrics.json").read_text())
    assert res["folds_skipped_time_guard"] == [0] and res["folds_completed"] == []
    print("ok  time guard")

    # 3) full runs: 5 folds, two at a time on the CPU, 2 epochs
    for name, net in NETS.items():
        out = results / name
        extra = ["--micro-batch", "2"] if name == "SMOKE_M0" else []  # gradient accumulation 2 x 2
        rc = run([*common, "--net", net, "--epochs", "2", "--gpus", "cpu,cpu", "--out", str(out), *extra])
        assert rc == 0, f"{name}: exit code {rc}; see {root / 'work' / name / 'logs'}"
        has_mask = not net.startswith("cls:")
        res = json.loads((out / "metrics.json").read_text())
        assert res["folds_completed"] == [0, 1, 2, 3, 4] and not res["folds_failed"], res
        assert res["micro_batch"] == (2 if name == "SMOKE_M0" else 4)
        for block in (res, res["at_threshold_0.5"]):
            assert block["oof"]["n"] == 60 and set(block["oof_per_fold"]) == {"0", "1", "2", "3", "4"}
            assert block["test_ensemble"]["n"] == 12 and "auc_fracture" in block["test_models_mean"]
            assert math.isnan(block["oof"]["dice_fractured_mean"]) != has_mask
            if has_mask:
                assert "nsd10_fractured_mean" in block["oof"] and "nsd20_fractured_mean" in block["test_ensemble"]
        oof, test = read_csv(out / "per_case_oof.csv"), read_csv(out / "per_case_test.csv")
        ens = read_csv(out / "per_case_test_ensemble.csv")
        assert len(oof) == 60 and len({r["file"] for r in oof}) == 60 and len(test) == 60 and len(ens) == 12
        assert all(str(r["fold"]) == r["model"] for r in oof), "an OOF femur was predicted by a model that trained on it"
        assert all(abs(sum(r[f"conf_{c}"] for c in range(1, 6)) - r["score"]) < 1e-5 for r in oof + ens)
        assert all(r["top_conf"] == r["score"] for r in oof + test + ens)
        assert all((r["dice_top"] is not None) == (has_mask and r["cls6_true"] > 0) for r in oof + test + ens)
        tau = res["thresholds"]["oof_youden"]  # metrics.json = a re-evaluation of the files it wrote
        for scope, rows in (("oof", oof), ("test_ensemble", ens)):
            again = evaluate(rows, tau)
            for key in ("auc_fracture", "accuracy_6class", "macro_f1", "dice_fractured_mean", "ece_derived"):
                assert same(again[key], res[scope][key]), (name, scope, key)
        for k in range(5):
            fd = out / "folds" / f"fold{k}"
            for f in ("oof.csv", "test.csv", "fold_meta.json", "train_curve.csv", "log_tail.txt"):
                assert (fd / f).exists(), fd / f
            meta = json.loads((fd / "fold_meta.json").read_text())
            assert meta["n_train"] == 48 and meta["n_held_out"] == 12 and Path(meta["weights"]).exists()
            assert meta["accumulation"] == (2 if name == "SMOKE_M0" else 1)
            with open(fd / "train_curve.csv") as f:
                assert len(list(csv.DictReader(f))) == 2
            maps = root / "work" / name / "prob_maps"
            for part in ("oof", "test"):
                n_maps = len(list((maps / part / f"fold{k}").glob("*.png"))) if maps.exists() else 0
                assert n_maps == (12 if has_mask else 0), (name, part, k, n_maps)
        for f in ("config.json", "preflight.json", "summary.md"):
            assert (out / f).exists(), out / f
        summary = (out / "summary.md").read_text()
        assert summary.count("| Fracture AUC |") == 2 and ("NSD at 10 px" in summary) == has_mask
        print(f"ok  full run {name} ({net}): 5 folds, OOF, test, ensemble, maps, summary")

    # 4) statistics and breakdown on the filed outputs
    names = list(NETS)
    assert run([*names, "--compare", "--name", "SMOKE_stats", "--results", str(results), "--folds-csv", str(folds_csv),
                "--n-boot", "100", "--n-perm", "100"], script="run_stats.py", quiet=True) == 0
    st = json.loads((results / "analysis" / "SMOKE_stats" / "stats.json").read_text())
    assert "dice_fractured" not in st["comparisons"]["SMOKE_CLS vs SMOKE_SEG"]["oof"]
    assert "dice_fractured" in st["comparisons"]["SMOKE_SEG vs SMOKE_M0"]["oof"]
    assert math.isnan(st["intervals"]["SMOKE_CLS"]["oof"]["dice_fractured_mean"]["estimate"])
    print("ok  statistics: intervals and paired comparisons (no Dice test against a classifier)")
    assert run([*names, "--name", "SMOKE_breakdown", "--results", str(results)], script="run_breakdown.py",
               quiet=True) == 0
    bd = json.loads((results / "analysis" / "SMOKE_breakdown" / "breakdown.json").read_text())["runs"]
    assert [r["run"] for r in bd] == names and all(r["oof"]["n"] == 60 and len(r["test_models"]) == 5 for r in bd)
    assert math.isnan(bd[0]["oof"]["seg"]["dice_fractured_mean"]) and not math.isnan(bd[1]["oof"]["seg"]["dice_fractured_mean"])
    print("ok  breakdown: per class, folds, fold models, errors, areas")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
