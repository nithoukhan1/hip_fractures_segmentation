#!/usr/bin/env python3
"""Start one named run on Kaggle. The runs are listed in kaggle/runs.json.

The launcher notebook clones the repository and calls, for example:
    python kaggle/run.py env_check
Each run writes into /kaggle/working/results/<run-name>/ and gets a run_meta.json with the git commit,
command, start time, runtime and exit code, so every result can be traced back to its code.

Parallel groups: a runs.json entry with "parallel": [run, run] starts those runs at the same time, one per GPU of
the session (GPU 0, 1, ...), for example two fold-0 development runs on Kaggle's two T4s. Their results go into
/kaggle/working/results/<group>/<run>/ (so the launcher's zip of the group holds all of them) and their weights into
/kaggle/working/work/<run>/; scripts/register_run.py files each run of the group separately. The runs of a group
must accept --gpus, --workers and --work (scripts/train_net.py does).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_group(name: str, spec: dict, runs: dict, results: Path, extra: str, runs_file: Path) -> int:
    """Start the runs of a parallel group, one per GPU, and wait for all of them."""
    subs = list(spec["parallel"])
    unknown = [s for s in subs if s not in runs or runs[s].get("parallel")]
    if unknown:
        sys.exit(f"group '{name}': {unknown} are not single runs in runs.json")
    out = results / name
    out.mkdir(parents=True, exist_ok=True)
    pip = sorted({p for s in subs for p in runs[s].get("pip", [])})
    if pip:  # once for the whole group, so the runs do not install at the same time
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pip], check=True)
    gpus = [str(g) for g in spec.get("gpus", range(len(subs)))]
    workers = max(1, (os.cpu_count() or 2) // len(subs))
    t0 = time.time()
    procs = []
    for sub, gpu in zip(subs, gpus):
        work = results.parent / "work" / sub
        sub_extra = f"--gpus {gpu} --workers {workers} --work {shlex.quote(str(work))} {extra}".strip()
        cmd = [sys.executable, str(Path(__file__).resolve()), sub, "--results", str(out), "--group", name,
               "--no-pip", "--runs", str(runs_file), "--extra", sub_extra]
        print(f"[{name}] starting {sub} on GPU {gpu}", flush=True)
        procs.append((sub, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))

    def stream(sub: str, proc: subprocess.Popen) -> None:
        (out / sub).mkdir(parents=True, exist_ok=True)
        with open(out / sub / "controller_log.txt", "w") as fh:  # filed with the run
            for line in proc.stdout:
                fh.write(line)
                print(f"[{sub}] {line}", end="", flush=True)

    threads = [threading.Thread(target=stream, args=(s, p)) for s, p in procs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    codes = {s: p.wait() for s, p in procs}
    meta = {"run": name, "step": spec.get("step"), "parallel": subs, "gpus": gpus[:len(subs)], "exit_codes": codes,
            "git_commit": git_commit(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "runtime_minutes": round((time.time() - t0) / 60, 2), "group_of_runs": True}
    (out / "group_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ngroup '{name}' finished in {meta['runtime_minutes']} min; exit codes {codes}; outputs in {out}")
    return max(codes.values()) if codes else 1


def git_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a run name from kaggle/runs.json")
    ap.add_argument("--extra", default="", help="extra arguments passed to the run's script")
    ap.add_argument("--results", default="/kaggle/working/results")
    ap.add_argument("--group", default="", help=argparse.SUPPRESS)        # set when started by a parallel group
    ap.add_argument("--no-pip", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--runs", default=str(REPO / "kaggle" / "runs.json"), help=argparse.SUPPRESS)  # tests only
    args = ap.parse_args()
    runs = json.loads(Path(args.runs).read_text())
    if args.run not in runs:
        sys.exit(f"unknown run '{args.run}'. Available: {', '.join(runs)}")
    spec = runs[args.run]
    if spec.get("parallel"):
        sys.exit(run_group(args.run, spec, runs, Path(args.results), args.extra, Path(args.runs).resolve()))
    out = Path(args.results) / args.run
    out.mkdir(parents=True, exist_ok=True)

    if spec.get("pip") and not args.no_pip:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *spec["pip"]], check=True)

    cmd = [sys.executable, str(REPO / spec["script"]), *spec.get("args", []), "--out", str(out), *shlex.split(args.extra)]
    meta = {"run": args.run, "step": spec.get("step"), "git_commit": git_commit(), "command": " ".join(cmd),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.group:
        meta["group"] = args.group
    print(json.dumps(meta, indent=2), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    meta.update(runtime_minutes=round((time.time() - t0) / 60, 2), exit_code=rc)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nrun '{args.run}' finished with exit code {rc} in {meta['runtime_minutes']} min; outputs in {out}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
