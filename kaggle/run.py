#!/usr/bin/env python3
"""Start one named run on Kaggle. The runs are listed in kaggle/runs.json.

The launcher notebook clones the repository and calls, for example:
    python kaggle/run.py env_check
Each run writes into /kaggle/working/results/<run-name>/ and gets a run_meta.json with the git commit,
command, start time, runtime and exit code, so every result can be traced back to its code.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    runs = json.loads((REPO / "kaggle" / "runs.json").read_text())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="one of: " + ", ".join(runs))
    ap.add_argument("--extra", default="", help="extra arguments passed to the run's script")
    ap.add_argument("--results", default="/kaggle/working/results")
    args = ap.parse_args()
    if args.run not in runs:
        sys.exit(f"unknown run '{args.run}'. Available: {', '.join(runs)}")
    spec = runs[args.run]
    out = Path(args.results) / args.run
    out.mkdir(parents=True, exist_ok=True)

    if spec.get("pip"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *spec["pip"]], check=True)

    cmd = [sys.executable, str(REPO / spec["script"]), *spec.get("args", []), "--out", str(out), *shlex.split(args.extra)]
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"run": args.run, "step": spec.get("step"), "git_commit": commit, "command": " ".join(cmd),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps(meta, indent=2), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    meta.update(runtime_minutes=round((time.time() - t0) / 60, 2), exit_code=rc)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nrun '{args.run}' finished with exit code {rc} in {meta['runtime_minutes']} min; outputs in {out}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
