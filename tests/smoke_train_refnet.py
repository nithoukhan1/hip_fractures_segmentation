"""End-to-end smoke test of RefFrac (Paper B) in scripts/train_net.py on the tiny synthetic dataset (CPU, about 8 min).

It builds the fake dataset of tests/synthetic_data.py with some femurs paired into two-femur x-rays, makes its
reference tables with scripts/build_reference_bank.py (random weights), and checks, without any download:
  1. --check-nets: every variant R1..R7 builds, steps and is timed with its references;
  2. a full 5-fold run of R4 (= RefFrac), two folds at a time on the CPU, with the tests self and random: every output
     file, out-of-fold integrity, that metrics.json equals a re-evaluation of the per-case files, the references
     each femur got (they must be the table's), the pointing game, and probability maps for every mode;
  3. a full 5-fold run of R7 with the tests blind, retrieved, random and fractured: the partner is used exactly
     where each policy says (training policy: only a healthy partner; blind: any partner);
  4. Paper B's fold-0 development path (Analysis/08 §4): two fold-0 runs (global window; margin loss) started as a
     parallel group through kaggle/run.py, filed by register_run.py (in a scratch copy of the repository), and
     compared with an M0 fold-0 run by run_stats.py --folds 0.
Needs torch, timm, segmentation-models-pytorch, albumentations, scipy, matplotlib:

    python tests/smoke_train_refnet.py --work /tmp/smoke_ref
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from femurseg import refnet as RN  # noqa: E402
from femurseg.evaluation import evaluate, read_csv  # noqa: E402
from smoke_reference_bank import pair_up  # noqa: E402
from synthetic_data import make_dataset  # noqa: E402

CONVNEXT = "convnext_tiny.fb_in22k_ft_in1k"


def run(args_list: list[str], script: str = "train_net.py", quiet: bool = False, folder: Path | None = None) -> int:
    cmd = [sys.executable, str((folder or REPO / "scripts") / script), *args_list]
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
    root = Path(ap.parse_args().work).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    data_root, folds_csv = make_dataset(root)
    n_pairs = pair_up(folds_csv)
    folds = {r["file"]: r for r in csv.DictReader(open(folds_csv, newline=""))}
    bank = root / "results" / "R00_smoke"
    assert run(["--no-pretrained", "--size", "192x128", "--device", "cpu", "--data-root", str(data_root),
                "--folds-csv", str(folds_csv), "--cache-dir", str(root / "cache"), "--out", str(bank)],
               script="build_reference_bank.py", quiet=True) == 0
    tables = {(int(r["fold_model"]), r["file"]): r for r in csv.DictReader(open(bank / "references.csv", newline=""))}
    print(f"ok  reference tables of {len(folds)} femurs ({n_pairs} pairs)")
    common = ["--no-pretrained", "--size", "192x128", "--batch", "4", "--preflight-steps", "3", "--data-root",
              str(data_root), "--folds-csv", str(folds_csv), "--cache-dir", str(root / "cache"),
              "--ref-table", str(bank / "references.csv"), "--pairs-table", str(bank / "pairs.csv")]
    results = root / "results"

    # 1) pre-flight of every variant
    nets = [f"ref:{CONVNEXT}:{v}" for v in RN.VARIANTS]
    out = results / "SMOKE_REF_CHECK"
    assert run([*common, "--check-nets", ",".join(nets), "--gpus", "cpu,cpu", "--out", str(out)]) == 0
    rep = json.loads((out / "net_check.json").read_text())
    assert [n["net"] for n in rep["nets"]] == nets and all(n["ok"] and n["micro_batch"] == 4 for n in rep["nets"])
    assert [n["references_per_femur"] for n in rep["nets"]] == [RN.VARIANTS[v]["k"] for v in RN.VARIANTS]
    print(f"ok  net check: {len(nets)} RefFrac variants built, stepped and timed with their references")

    # 2) R4 = RefFrac: 5 folds, two at a time, with the tests self and random
    out = results / "SMOKE_R4"
    assert run([*common, "--net", f"ref:{CONVNEXT}:R4", "--epochs", "2", "--gpus", "cpu,cpu", "--extra-tests",
                "self,random", "--out", str(out)]) == 0
    res = json.loads((out / "metrics.json").read_text())
    assert res["folds_completed"] == [0, 1, 2, 3, 4] and set(res["tests"]) == {"self", "random"}
    oof, ens = read_csv(out / "per_case_oof.csv"), read_csv(out / "per_case_test_ensemble.csv")
    assert len(oof) == 60 and len({r["file"] for r in oof}) == 60 and len(ens) == 12
    assert all(str(r["fold"]) == r["model"] for r in oof), "an OOF femur was predicted by a model that trained on it"
    tau = res["thresholds"]["oof_youden"]
    for scope, rows in (("oof", oof), ("test_ensemble", ens)):
        again = evaluate(rows, tau)
        for key in ("auc_fracture", "macro_f1", "dice_fractured_mean"):
            assert same(again[key], res[scope][key]), (scope, key)
    for mode in ("self", "random"):
        rows = read_csv(out / f"per_case_oof_{mode}.csv")
        assert sorted(r["file"] for r in rows) == sorted(r["file"] for r in oof)
        assert same(evaluate(rows, tau)["dice_fractured_mean"], res["tests"][mode]["oof"]["dice_fractured_mean"])
        assert "dice_fractured_mean" in res["tests"][mode]["test_ensemble"], mode  # maps saved in every mode
        assert res["tests"][mode]["effects_healthy_oof"]["n_fractured"] == sum(r["cls6_true"] == 0 for r in oof)
        for part in ("oof", "test"):
            for k in range(5):
                assert len(list((root / "work" / "SMOKE_R4" / "prob_maps" / f"{part}_{mode}" / f"fold{k}").glob("*.png"))) == 12
    refs = list(csv.DictReader(open(out / "per_case_oof_references.csv", newline="")))
    assert Counter(r["mode"] for r in refs) == {"base": 60, "self": 60, "random": 60}
    for r in refs:  # every reference is the table's, in the table's orientation
        row = tables[(int(r["model"]), r["file"])]
        want = {"base": [f"train/{row[f'ref{j}']}" + (" mirrored" if row[f"ref{j}_mirrored"] == "1" else "") for j in (1, 2, 3)],
                "random": [f"train/{row['random_healthy']}" + (" mirrored" if row["random_healthy_mirrored"] == "1" else "")],
                "self": [f"train/{r['file']}"]}[r["mode"]]
        assert r["refs"].split(";") == want, (r, want)
    pg = res["reference"]["oof"]["pointing_game"]
    assert pg["n"] == sum(r["cls6_true"] > 0 for r in oof) and 0 <= pg["hit_rate"] <= 1 and 0 < pg["chance"] < 1
    assert all((r["diff_hit"] != "") == (r["mode"] == "base" and r["cls6_true"] != "0") for r in refs)
    meta = json.loads((out / "folds" / "fold0" / "fold_meta.json").read_text())
    assert meta["variant"] == "R4" and meta["k"] == 3 and meta["policy"] == "nearest" and meta["align"] == "attention"
    assert meta["window_cells"] == {"1/16": 2, "1/32": 1} and meta["reference_table_md5"] and meta["params_m"] > 33
    summary = (out / "summary.md").read_text()
    assert "Reference tests (RefFrac" in summary and "Pointing game" in summary and "RefFrac R4: 3 reference(s)" in summary
    print("ok  R4 (RefFrac): 5 folds, tests self and random, references as in the table, pointing game, maps")

    # 3) R7: the patient's own other femur; the policies of the shortcut study
    out = results / "SMOKE_R7"
    assert run([*common, "--net", f"ref:{CONVNEXT}:R7", "--epochs", "1", "--gpus", "cpu,cpu", "--extra-tests",
                "blind,retrieved,random,fractured", "--out", str(out)]) == 0
    res = json.loads((out / "metrics.json").read_text())
    assert set(res["tests"]) == {"blind", "retrieved", "random", "fractured"}
    for part in ("oof", "test"):
        for r in csv.DictReader(open(out / f"per_case_{part}_references.csv", newline="")):
            f = folds[r["file"]]
            partner = f["pair_file"]
            uses_partner = r["refs"] == f"{f['split']}/{partner} mirrored"
            want = {"base": bool(partner) and f["pair_cls6"] == "0", "blind": bool(partner)}.get(r["mode"], False)
            assert uses_partner == want == (r["source"] == "partner"), (part, r)
    meta = json.loads((out / "folds" / "fold0" / "fold_meta.json").read_text())
    n_healthy_partner = sum(1 for f in folds.values() if f["split"] == "train" and f["fold"] != "0" and f["pair_file"]
                            and f["pair_cls6"] == "0")
    assert meta["train_sources"].get("partner") == n_healthy_partner > 0, meta["train_sources"]
    print("ok  R7: partner only when healthy (training policy), any partner when blind, none when retrieved")

    # 4) Paper B development: two fold-0 runs in one "session" (kaggle/run.py group), filed, compared on fold 0
    dev_path(root, common, folds_csv, results)
    print("SMOKE TEST PASSED")


def dev_path(root: Path, common: list[str], folds_csv: Path, results: Path) -> None:
    base = [*common, "--net", f"ref:{CONVNEXT}:R4", "--epochs", "1", "--folds", "0", "--no-test"]
    runs = {"R4_dev_a": {"step": "5", "script": "scripts/train_net.py", "pip": [],
                         "args": [*base, "--ref-window", "0", "--extra-tests", "self,random"]},
            "R4_dev_b": {"step": "5", "script": "scripts/train_net.py", "pip": [], "args": [*base, "--ref-margin", "0.5"]},
            "R_dev_group": {"step": "5", "parallel": ["R4_dev_a", "R4_dev_b"], "gpus": ["cpu", "cpu"]}}
    runs_file = root / "runs_smoke.json"
    runs_file.write_text(json.dumps(runs))
    kaggle_results = root / "kaggle_working" / "results"
    cmd = [sys.executable, str(REPO / "kaggle" / "run.py"), "R_dev_group", "--results", str(kaggle_results),
           "--runs", str(runs_file)]
    print("$", " ".join(cmd), flush=True)
    assert subprocess.run(cmd).returncode == 0
    group = kaggle_results / "R_dev_group"
    assert json.loads((group / "group_meta.json").read_text())["exit_codes"] == {"R4_dev_a": 0, "R4_dev_b": 0}
    for sub in ("R4_dev_a", "R4_dev_b"):
        m = json.loads((group / sub / "metrics.json").read_text())
        assert m["folds_completed"] == [0] and m["oof"]["n"] == 12 and "test_ensemble" not in m, sub
        assert "test" not in m["reference"], sub  # --no-test: no test-set references either
    meta_a = json.loads((group / "R4_dev_a" / "folds" / "fold0" / "fold_meta.json").read_text())
    meta_b = json.loads((group / "R4_dev_b" / "folds" / "fold0" / "fold_meta.json").read_text())
    assert meta_a["window"] == 0 and meta_a["window_cells"] == {"1/16": None, "1/32": None} and meta_a["margin"] == 0
    assert meta_b["margin"] == 0.5 and meta_b["window_cells"] == {"1/16": 2, "1/32": 1}
    curve_b = list(csv.DictReader(open(group / "R4_dev_b" / "folds" / "fold0" / "train_curve.csv")))
    assert "margin" in curve_b[0] and "margin" not in list(csv.DictReader(open(
        group / "R4_dev_a" / "folds" / "fold0" / "train_curve.csv")))[0]
    print("ok  parallel group: two fold-0 RefFrac runs in one session (global window; margin loss), no test set")

    fake = root / "fake_repo"
    (fake / "scripts").mkdir(parents=True)
    (fake / "results" / "_inbox").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "register_run.py", fake / "scripts")
    shutil.copy(REPO / "results" / "RUN_NOTES_TEMPLATE.md", fake / "results")
    (fake / "results" / "EXPERIMENTS.md").write_text("# Register\n\n| # | Date | RUN |\n|---|---|---|\n")
    zpath = fake / "results" / "_inbox" / "results_R_dev_group.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in sorted(group.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(kaggle_results.parent))
    assert run([str(zpath), "--purpose", "smoke"], script="register_run.py", quiet=True, folder=fake / "scripts") == 0
    reg = (fake / "results" / "EXPERIMENTS.md").read_text()
    assert "R4_dev_a" in reg and "R4_dev_b" in reg and "Paper B" in reg
    print("ok  register_run: the group zip filed as two Paper B runs")

    m0 = results / "SMOKE_M0_F0"  # M0's fold-0 model, the reference of the development rule
    assert run([*common, "--net", f"m0:{CONVNEXT}", "--epochs", "1", "--folds", "0", "--no-test", "--gpus", "cpu",
                "--out", str(m0)], quiet=True) == 0
    for sub in ("R4_dev_a", "R4_dev_b"):
        shutil.copytree(fake / "results" / sub, results / sub)
    assert run(["SMOKE_M0_F0", "R4_dev_a", "R4_dev_b", "--compare", "--folds", "0", "--name", "SMOKE_ref_dev",
                "--results", str(results), "--folds-csv", str(folds_csv), "--n-boot", "50", "--n-perm", "50"],
               script="run_stats.py", quiet=True) == 0
    st = json.loads((results / "analysis" / "SMOKE_ref_dev" / "stats.json").read_text())
    assert st["folds"] == [0] and "SMOKE_M0_F0 vs R4_dev_a" in st["comparisons"]
    assert "dice_fractured" in st["comparisons"]["SMOKE_M0_F0 vs R4_dev_b"]["oof"]
    print("ok  run_stats --folds 0: the development runs against M0 on the same 12 femurs, Dice compared")


if __name__ == "__main__":
    main()
