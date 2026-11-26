#!/usr/bin/env python3
"""File a finished Kaggle run into results/ and register it in results/EXPERIMENTS.md.

Usage (from the repository root, i.e. the local project folder):
  python scripts/register_run.py results/_inbox/results_S00_env_check.zip
  python scripts/register_run.py results/_inbox/results_S01_yolov8s_seg_1024.zip --version 4 --purpose "paper protocol"

What it does, step by step:
  1. Opens the zip downloaded from the Kaggle version's Output tab. It finds the run folder by its
     run_meta.json (written by kaggle/run.py).
  2. Copies that folder to results/<RUN>/. It refuses to overwrite an existing run: a re-run must
     get a new name (for example S01_yolov8s_seg_1024_r2).
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
TRACKS = {"S": "Shared", "A": "Paper A", "R": "Paper B"}


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
        gpu = gpu or ", ".join(sorted({str(n.get("gpu")) for n in nets if n.get("gpu")}))
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
        metas = [n for n in zf.namelist() if n.endswith("run_meta.json")]
        if len(metas) != 1:
            sys.exit(f"expected exactly one run_meta.json in {zpath.name}, found {len(metas)}")
        prefix = metas[0][: -len("run_meta.json")]
        zf.extractall(tmp, members=[n for n in zf.namelist() if n.startswith(prefix)])
        src = Path(tmp) / prefix
        meta = json.loads((src / "run_meta.json").read_text())
        run = meta.get("run") or src.name
        dest = RESULTS / run
        if dest.exists():
            sys.exit(f"results/{run} already exists. A re-run needs a new name; nothing was changed.")
        shutil.copytree(src, dest)

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

    filed = zpath.parent / "filed"
    try:
        filed.mkdir(exist_ok=True)
        zpath.replace(filed / zpath.name)
        moved = f"moved the zip to {filed.relative_to(REPO) if filed.is_relative_to(REPO) else filed}"
    except OSError as e:  # e.g. no permission to move; filing still succeeded
        moved = f"could not move the zip ({e}); you can move it to results/_inbox/filed/ by hand"
    print(f"filed results/{run}/ ({sum(1 for _ in dest.rglob('*'))} files), register line #{n} added, {moved}")
    print("register line:", row)


if __name__ == "__main__":
    main()
