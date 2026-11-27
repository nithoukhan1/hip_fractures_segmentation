"""Checks of the Kaggle launcher's data check (kaggle/run.py) and of register_run.py's refusal to file a run that
never started (the other runs of a parallel group are still filed). No GPU, network or dataset needed.

Run with ``python tests/test_kaggle_run.py`` or pytest.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from femurseg.paths import describe_folder, find_data_root  # noqa: E402


def _launch(tmp: Path, runs: dict, run: str, extra: str = "") -> tuple[int, str, float]:
    runs_file = tmp / "runs.json"
    runs_file.write_text(json.dumps(runs))
    t0 = time.time()
    p = subprocess.run([sys.executable, str(REPO / "kaggle" / "run.py"), run, "--results", str(tmp / "results"),
                        "--runs", str(runs_file), "--extra", extra], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr, time.time() - t0


def _data_root(tmp: Path) -> Path:
    root = tmp / "input" / "datasets" / "utopianstar" / "dataset" / "femoral" / "data"
    for split in ("train", "test"):
        (root / split / "images").mkdir(parents=True)
    return root


def test_messages_say_what_the_input_folder_holds():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        assert "does not exist" in describe_folder(tmp / "missing")
        assert "is empty" in describe_folder(tmp) and "no dataset is attached" in describe_folder(tmp)
        (tmp / "datasets" / "someone" / "other").mkdir(parents=True)
        try:
            find_data_root(tmp)
        except FileNotFoundError as e:
            assert "datasets/someone/" in str(e), e
        else:
            raise AssertionError("found a data root in a folder without one")
        root = _data_root(tmp)  # Kaggle's layout below /kaggle/input (S00_env_check)
        assert find_data_root(tmp / "input") == root


def test_training_run_without_data_stops_before_pip():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "empty").mkdir()
        runs = {"R": {"step": "2", "script": "scripts/train_net.py", "pip": ["surely-not-a-real-package-0815"],
                      "args": ["--data-root", str(tmp / "empty"), "--net", "cls:x"]}}
        rc, log, secs = _launch(tmp, runs, "R")
        assert rc == 2, log  # had pip run first, the made-up package would have failed with exit code 1
        assert "NO DATASET FOUND" in log and "Add Input" in log and "is empty" in log, log
        meta = json.loads((tmp / "results" / "R" / "run_meta.json").read_text())
        assert meta["exit_code"] == 2 and meta["runtime_minutes"] == 0.0 and "no dataset found" in meta["error"]
        assert secs < 60, secs


def test_group_without_data_starts_nothing():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        sub = {"step": "3", "script": "scripts/train_net.py", "pip": [], "args": ["--data-root", str(tmp / "nope")]}
        runs = {"a": sub, "b": dict(sub), "G": {"step": "3", "parallel": ["a", "b"], "gpus": ["cpu", "cpu"]}}
        rc, log, _ = _launch(tmp, runs, "G")
        assert rc == 2 and "NO DATASET FOUND" in log and "does not exist" in log, log
        gm = json.loads((tmp / "results" / "G" / "group_meta.json").read_text())
        assert gm["exit_codes"] == {"a": 2, "b": 2} and "no dataset found" in gm["error"]
        assert not (tmp / "results" / "G" / "a").exists()  # no run was started


def test_training_run_with_data_starts_and_other_scripts_are_not_checked():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        root = _data_root(tmp)
        runs = {"R": {"step": "2", "script": "scripts/train_net.py", "pip": [], "args": ["--help"]},
                "E": {"step": "0", "script": "scripts/register_run.py", "pip": [], "args": ["--help"]}}
        rc, log, _ = _launch(tmp, runs, "R", extra=f"--data-root {root}")  # the root can come from --extra too
        assert rc == 0 and "NO DATASET" not in log and "usage:" in log, log
        assert json.loads((tmp / "results" / "R" / "run_meta.json").read_text())["exit_code"] == 0
        rc, log, _ = _launch(tmp, runs, "E")  # not a training script: no check, even without data
        assert rc == 0 and "NO DATASET" not in log, log


def _fake_repo(tmp: Path) -> Path:
    fake = tmp / "repo"
    (fake / "scripts").mkdir(parents=True)
    (fake / "results" / "_inbox").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "register_run.py", fake / "scripts")
    shutil.copy(REPO / "results" / "RUN_NOTES_TEMPLATE.md", fake / "results")
    (fake / "results" / "EXPERIMENTS.md").write_text("# Register\n\n| # | Date | RUN |\n|---|---|---|\n")
    return fake


def _register(fake: Path, name: str, members: dict) -> subprocess.CompletedProcess:
    z = fake / "results" / "_inbox" / f"results_{name}.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for member, content in members.items():
            zf.writestr(member, content if isinstance(content, str) else json.dumps(content))
    return subprocess.run([sys.executable, str(fake / "scripts" / "register_run.py"), str(z)],
                          capture_output=True, text=True)


def test_register_refuses_a_run_that_never_started():
    with tempfile.TemporaryDirectory() as t:
        fake = _fake_repo(Path(t))
        cases = {"old": {"run": "S09_x", "exit_code": 1, "runtime_minutes": 0.0},           # S09's zip of 2026-11-27
                 "new": {"run": "S09_x", "exit_code": 2, "runtime_minutes": 0.0, "error": "no dataset found: ..."}}
        for name, meta in cases.items():
            p = _register(fake, name, {"results/S09_x/run_meta.json": meta})
            assert p.returncode != 0 and "did not start" in p.stderr and "Nothing was filed" in p.stderr, p.stderr
            assert not (fake / "results" / "S09_x").exists(), name  # nothing filed, zip left in the inbox
            assert (fake / "results" / "_inbox" / f"results_{name}.zip").exists(), name
        p = _register(fake, "grp", {"results/G/group_meta.json": {"run": "G", "error": "no dataset found: ..."}})
        assert p.returncode != 0 and "group did not start" in p.stderr and "Nothing was filed" in p.stderr, p.stderr
        assert "| 1 |" not in (fake / "results" / "EXPERIMENTS.md").read_text()
        # a run that wrote nothing but ran for a long time is a real failure: filed, with its exit code
        p = _register(fake, "long", {"results/S09_y/run_meta.json": {"run": "S09_y", "exit_code": 1,
                                                                     "runtime_minutes": 95.0}})
        assert p.returncode == 0 and (fake / "results" / "S09_y" / "NOTES.md").exists(), p.stderr
        assert "| 1 |" in (fake / "results" / "EXPERIMENTS.md").read_text()


def test_register_files_the_runs_of_a_group_that_started():
    with tempfile.TemporaryDirectory() as t:
        fake = _fake_repo(Path(t))
        p = _register(fake, "A_dev_x", {
            "results/A_dev_x/group_meta.json": {"run": "A_dev_x", "exit_codes": {"A6_a": 0, "A6_b": 1}},
            "results/A_dev_x/A6_a/run_meta.json": {"run": "A6_a", "group": "A_dev_x", "step": "3", "exit_code": 0,
                                                   "runtime_minutes": 50.0},
            "results/A_dev_x/A6_a/metrics.json": {"oof": {"macro_f1": 0.8}},
            "results/A_dev_x/A6_b/run_meta.json": {"run": "A6_b", "group": "A_dev_x", "step": "3", "exit_code": 1,
                                                   "runtime_minutes": 0.2},
            "results/A_dev_x/A6_b/controller_log.txt": "Traceback (most recent call last): ..."})
        assert p.returncode == 0 and "not filed: run A6_b did not start" in p.stdout, p.stdout + p.stderr
        assert (fake / "results" / "A6_a" / "metrics.json").exists() and not (fake / "results" / "A6_b").exists()
        reg = (fake / "results" / "EXPERIMENTS.md").read_text()
        assert "| 1 |" in reg and "A6_a" in reg and "macro-F1 0.800" in reg and "A6_b" not in reg, reg


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} launcher tests passed")
