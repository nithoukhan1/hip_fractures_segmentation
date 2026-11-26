"""Run the folds of one configuration in parallel, one process per fold and one fold per GPU at a time.

Shared by scripts/train_yolo_seg.py and scripts/train_net.py. The controller (the script without --worker) calls
run_folds(); each fold runs as ``<script> --worker --fold k --device d ...`` in its own process, so a crash in one fold
does not stop the others, and its output goes to work/<RUN>/logs/fold<k>.log (the last 60 lines are copied to
results/<RUN>/folds/fold<k>/log_tail.txt).

Time guard: before starting a fold, the controller estimates its duration (the mean of the folds finished so far,
or the first guess) and does not start it if the session could pass the time budget; skipped folds are reported and
can be run later with --folds.
"""
from __future__ import annotations

import hashlib
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def run_folds(script: Path, worker_args: list[str], folds: list[int], gpus: list[str], out: Path, work: Path,
              t0: float, budget_h: float, est_fold_min: float) -> tuple[list[int], list[int], list[int]]:
    """Run every fold of ``folds``; ``gpus`` lists GPU ids or 'cpu' (one entry per fold running at the same time).
    Returns (done, skipped by the time guard, failed)."""
    todo: queue.Queue = queue.Queue()
    for k in folds:
        todo.put(k)
    done, skipped, failed, durations = [], [], [], []
    lock = threading.Lock()
    budget = budget_h * 3600

    def gpu_worker(gpu: str) -> None:
        while True:
            try:
                k = todo.get_nowait()
            except queue.Empty:
                return
            with lock:
                est = statistics.mean(durations) if durations else est_fold_min * 60
                if time.time() - t0 + 1.15 * est > budget:
                    skipped.append(k)
                    print(f"[guard] fold {k} not started: {(time.time() - t0) / 3600:.1f} h used, "
                          f"about {est / 3600:.1f} h needed, budget {budget_h} h", flush=True)
                    continue
            env = dict(os.environ)
            device = "cpu"
            if gpu != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = gpu
                device = "0"
            cmd = [sys.executable, str(script), "--worker", "--fold", str(k), "--device", device, *worker_args]
            log = work / "logs" / f"fold{k}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            ts = time.time()
            print(f"[fold {k}] started on {'GPU ' + gpu if gpu != 'cpu' else 'CPU'}; log {log}", flush=True)
            with open(log, "w") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
            tail = log.read_text(errors="replace").splitlines()[-60:]
            (out / "folds" / f"fold{k}").mkdir(parents=True, exist_ok=True)
            (out / "folds" / f"fold{k}" / "log_tail.txt").write_text("\n".join(tail) + "\n")
            with lock:
                if rc == 0:
                    done.append(k)
                    durations.append(time.time() - ts)
                else:
                    failed.append(k)
            print(f"[fold {k}] {'finished' if rc == 0 else f'FAILED (exit {rc})'} after "
                  f"{(time.time() - ts) / 60:.1f} min", flush=True)

    threads = [threading.Thread(target=gpu_worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return done, skipped, failed


def argv_without(argv, flags: tuple[str, ...] = ("--folds", "--gpus")) -> list[str]:
    """The controller's own command-line arguments minus ``flags`` (and their values), passed on to each worker."""
    src = list(sys.argv[1:] if argv is None else argv)
    out, skip = [], False
    for a in src:
        if skip:
            skip = False
            continue
        if a in flags:
            skip = True
            continue
        if any(a.startswith(f + "=") for f in flags):
            continue
        out.append(a)
    return out


def git_commit(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def file_md5(path: str | Path) -> str:
    p = Path(path)
    return hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else ""
