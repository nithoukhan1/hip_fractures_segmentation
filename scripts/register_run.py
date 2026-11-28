#!/usr/bin/env python3
"""File a finished Kaggle run into results/ and register it in results/EXPERIMENTS.md.

Usage (from the repository root, i.e. the local project folder):
  python scripts/register_run.py results/_inbox/results_S00_env_check.zip
  python scripts/register_run.py results/_inbox/results_S01_yolov8s_seg_1024.zip --version 4 --purpose "paper protocol"

What it does, step by step:
  1. Opens the zip downloaded from the Kaggle version's Output tab. It finds the run folder by its
     run_meta.json (written by kaggle/run.py). The zip of a parallel group (two runs in one session, see
     kaggle/run.py) holds one run folder per run; each is filed separately, with its own register line.
  2. Copies that folder to results/<RUN>/. It refuses to overwrite an existing run: a re-run must
     get a new name (for example S01_yolov8s_seg_1024_r2). A run that never started (no dataset attached, or it ended
     within a minute without any output) is not filed, so it can be run again under the same name; in a parallel
     group the other runs are still filed.
  3. Creates results/<RUN>/NOTES.md from results/RUN_NOTES_TEMPLATE.md, unless the run already has one.
  4. Appends one line to the register table at the end of results/EXPERIMENTS.md: date, run, track,
     step, git commit, Kaggle version, GPU, runtime, exit code and the key result.
  5. Moves the zip to results/_inbox/filed/, so the inbox only holds runs that are not filed yet.

Standard library only, so it runs anywhere (local PC, Kaggle, the Claude workspace).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
TRACKS = {"S": "Shared", "A": "Paper A", "R": "Paper B", "N": "New method"}


def key_result(run_dir: Path) -> tuple[str, str]:
    """Return (gpu, short key-result text) from the files a run wrote."""
    gpu, parts = "", []
    env = run_dir / "env_report.json"
    if env.exists():
        rep = json.loads(env.read_text())
        gpu = (rep.get("hardware") or {}).get("nvidia_smi", "") or ""
        gpu = "; ".join(sorted({line.split(",")[0].strip() for line in gpu.splitlines() if line.strip()}))
        n_gpu = len([line for line in ((rep.get("hardware") or {}).get("nvidia_smi") or "").splitlines() if line.strip()])
        gpu = f"{gpu} x{n_gpu}" if n_gpu > 1 else gpu
        sc, dt = rep.get("split_check") or {}, rep.get("data") or {}
        parts.append(f"same data: {sc.get('identical_files_classes_pixels')}")
        parts.append(f"same folds: {sc.get('identical_folds')}")
        parts.append(f"val empty: {dt.get('val_is_empty')}")
        if dt.get("val_pooled_into_train"):
            parts.append(f"val pooled into train: {dt['val_pooled_into_train']}")
        if "csv_files_found_on_disk" in sc:
            parts.append(f"all files found: {sc['csv_files_found_on_disk']}")
        speed = rep.get("gpu_speed") or {}
        if isinstance(speed, dict) and speed and "error" not in speed and "skipped" not in speed:
            parts.append("fp16 TFLOPs: " + ", ".join(str(v.get("fp16_matmul_tflops")) for v in speed.values()))
    check = run_dir / "net_check.json"
    if check.exists():  # S00_net_check: pre-flight of several networks
        rep = json.loads(check.read_text())
        nets = rep.get("nets") or []
        if not gpu:  # e.g. "Tesla T4 x2" when the networks ran on two GPUs of the same kind (config.json gpu_list)
            names = sorted({str(n.get("gpu")) for n in nets if n.get("gpu")})
            cfg = run_dir / "config.json"
            n_dev = len(json.loads(cfg.read_text()).get("gpu_list") or []) if cfg.exists() else 0
            gpu = ", ".join(names) + (f" x{n_dev}" if n_dev > 1 and len(names) == 1 else "")
        ok = [n for n in nets if n.get("ok")]
        parts.append(f"networks ok: {len(ok)}/{len(nets)}")
        parts += [f"{n['net']}: micro-batch {n['micro_batch']}x{n['accumulation']}, {n.get('peak_gb')} GB, "
                  f"~{n['est_fold_min']} min/fold" for n in ok]
        parts += [f"{n['net']}: FAILED {str(n.get('error'))[:80]}" for n in nets if not n.get("ok")]
    met = run_dir / "metrics.json"
    if met.exists():
        m = json.loads(met.read_text())
        gpu = gpu or str(m.get("gpu", ""))
        short = (("auc_fracture", "AUC"), ("accuracy_6class", "6-class acc"), ("macro_f1", "macro-F1"),
                 ("dice_fractured_mean", "Dice fractured"), ("garden12_recall", "Garden I-II recall"))
        for scope, label in (("oof", "OOF"), ("val", "val"), ("test_ensemble", "test ensemble"), (None, "")):
            block = m.get(scope) if scope else m
            vals = [f"{name} {block[k]:.3f}" for k, name in short
                    if isinstance(block, dict) and isinstance(block.get(k), (int, float)) and block[k] == block[k]]
            if vals:
                parts.append((f"{label}: " if label else "") + ", ".join(vals))
                if scope is None:
                    break
        if (m.get("thresholds") or {}).get("oof_youden") is not None:
            parts.append(f"threshold (OOF Youden) {m['thresholds']['oof_youden']:.3f}")
        if m.get("folds_skipped_time_guard") or m.get("folds_failed"):
            parts.append(f"folds skipped {m.get('folds_skipped_time_guard')}, failed {m.get('folds_failed')}")
    return gpu, "; ".join(parts) if parts else "see files"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", help="results_<RUN>.zip downloaded from Kaggle")
    ap.add_argument("--version", default="", help="Kaggle notebook version number")
    ap.add_argument("--purpose", default="", help="one line: why this run was made")
    args = ap.parse_args()
    zpath = Path(args.zip).resolve()

    with zipfile.ZipFile(zpath) as zf, tempfile.TemporaryDirectory() as tmp:
        metas = sorted(n for n in zf.namelist() if n.endswith("run_meta.json"))
        if not metas:  # a parallel group stopped by kaggle/run.py's data check holds only group_meta.json
            errors = [json.loads(zf.read(n)).get("error") for n in zf.namelist() if n.endswith("group_meta.json")]
            why = next((e for e in errors if e), None)
            sys.exit(f"{zpath.name}: the parallel group did not start ({why}). {NOT_FILED}" if why
                     else f"no run_meta.json in {zpath.name}")
        found, not_started = [], []
        for m in metas:
            prefix = m[: -len("run_meta.json")]
            meta = json.loads(zf.read(m))
            if len(metas) > 1 and not meta.get("group"):
                sys.exit(f"{zpath.name} holds {len(metas)} runs, but {m} is not part of a parallel group")
            if never_started(meta, [n for n in zf.namelist() if n.startswith(prefix) and not n.endswith("/")]):
                reason = meta.get("error") or (f"exit code {meta.get('exit_code')} after "
                                               f"{meta.get('runtime_minutes')} min, no outputs")
                not_started.append(f"run {meta.get('run') or Path(prefix).name} did not start ({reason})")
                continue
            found.append((prefix, meta))
        if not found:
            sys.exit(f"{zpath.name}: {'; '.join(not_started)}. {NOT_FILED}")
        names = [meta.get("run") or Path(prefix).name for prefix, meta in found]
        clash = [n for n in names if (RESULTS / n).exists()]
        if clash:
            sys.exit(f"results/{clash[0]} already exists. A re-run needs a new name; nothing was changed.")
        for (prefix, meta), run in zip(found, names):
            zf.extractall(tmp, members=[n for n in zf.namelist() if n.startswith(prefix)])
            shutil.copytree(Path(tmp) / prefix, RESULTS / run)

    for (prefix, meta), run in zip(found, names):
        file_run(run, meta, args)
    for msg in not_started:  # e.g. one run of a parallel group crashed at once while the other ran
        print(f"not filed: {msg}. Fix the cause and run it again under the same name.")

    filed = zpath.parent / "filed"
    try:
        filed.mkdir(exist_ok=True)
        zpath.replace(filed / zpath.name)
        moved = f"moved the zip to {filed.relative_to(REPO) if filed.is_relative_to(REPO) else filed}"
    except OSError as e:  # e.g. no permission to move; filing still succeeded
        moved = f"could not move the zip ({e}); you can move it to results/_inbox/filed/ by hand"
    print(moved)


NOT_FILED = "Nothing was filed: fix the cause and run it again under the same name."


def never_started(meta: dict, files: list[str]) -> bool:
    """True for a run the launcher stopped before training (no dataset: an "error" in run_meta.json), or one that
    ended within a minute without writing anything but its run_meta.json and log. A run that wrote nothing after
    longer is still filed, with its exit code, so the failure keeps its register line."""
    no_outputs = all(n.endswith(("run_meta.json", "controller_log.txt")) for n in files)
    return bool(meta.get("error")) or (no_outputs and float(meta.get("runtime_minutes") or 0) < 1.0)


def file_run(run: str, meta: dict, args) -> None:
    """Notes stub and register line for a run folder already copied to results/<run>/."""
    dest = RESULTS / run
    if not (dest / "NOTES.md").exists():
        tpl = (RESULTS / "RUN_NOTES_TEMPLATE.md").read_text()
        (dest / "NOTES.md").write_text(tpl.replace("<RUN>", run).replace("<PURPOSE>", args.purpose or "(fill in)"))

    gpu, result = key_result(dest)
    reg = RESULTS / "EXPERIMENTS.md"
    lines = reg.read_text().rstrip("\n").splitlines()
    n = sum(1 for line in lines if line.startswith("| ") and line.split("|")[1].strip().isdigit()) + 1
    track = TRACKS.get(run[:1], "?")
    row = (f"| {n} | {time.strftime('%Y-%m-%d')} | {run} | {track} | {meta.get('step', '')} | "
           f"{str(meta.get('git_commit', ''))[:7]} | {args.version} | {gpu} | {meta.get('runtime_minutes', '')} | "
           f"{meta.get('exit_code', '')} | {result} | {args.purpose} |")
    reg.write_text("\n".join(lines + [row]) + "\n")
    print(f"filed results/{run}/ ({sum(1 for _ in dest.rglob('*'))} files), register line #{n} added")
    print("register line:", row)


if __name__ == "__main__":
    main()
