#!/usr/bin/env python3
"""Step 0 on Kaggle: check the machine, the packages and the data before any training.

What it checks and writes to <out>/env_report.json:
  1. Hardware: GPU name(s) and memory, CPU count, RAM, free disk in /kaggle/working and /tmp.
  2. Software: Python, torch/CUDA, versions of the packages we will use, internet access (GitHub, PyPI).
  3. Data: where Kaggle mounted the dataset, image/label/mask counts per folder, whether val/ is empty
     (if not, its images are pooled into train: src/femurseg/paths.py), data.yml content, image size,
     and image read speed.
  4. Split check: re-runs scripts/make_splits.py on the Kaggle copy and compares it with the committed
     splits/folds_v1.csv (same files, same classes, same pixels, same folds), and checks that every
     file listed in splits/folds_v1.csv is found on disk.
  5. A 10-second GPU speed test (fp16 matrix multiply), used to calibrate the compute plan.

Every section is wrapped so one failure does not stop the rest. Usage (normally via kaggle/run.py):
  python kaggle/env_check.py --input /kaggle/input --out /kaggle/working/results/env_check
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from femurseg.classes import EXPECTED_COUNTS  # noqa: E402
from femurseg.paths import build_index, find_data_root  # noqa: E402


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def section(report: dict, name: str):
    """Decorator: run a check, store its result (or the error) under report[name]."""
    def wrap(fn):
        t0 = time.time()
        try:
            report[name] = fn()
        except Exception:  # noqa: BLE001
            report[name] = {"error": traceback.format_exc(limit=3)}
        print(f"[{time.time() - t0:5.1f}s] {name}: done")
        return fn
    return wrap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input", help="where Kaggle mounts attached datasets")
    ap.add_argument("--out", default="/kaggle/working/results/env_check")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "git_commit": sh(f"git -C '{REPO}' rev-parse HEAD"),
                 "git_ref": sh(f"git -C '{REPO}' describe --all --always")}

    @section(rep, "hardware")
    def _():
        mem = {}
        if Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, v = line.split(":", 1)
                mem[k] = v.strip()
        disk = {}
        for p in ("/kaggle/working", "/tmp"):
            if Path(p).exists():
                u = shutil.disk_usage(p)
                disk[p] = {"total_gb": round(u.total / 1e9, 1), "free_gb": round(u.free / 1e9, 1)}
        return {"cpu_count": os.cpu_count(), "ram_total": mem.get("MemTotal"), "ram_available": mem.get("MemAvailable"),
                "disk": disk, "platform": platform.platform(),
                "nvidia_smi": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")}

    @section(rep, "software")
    def _():
        versions = {}
        for dist in ("torch", "torchvision", "numpy", "pandas", "pillow", "opencv-python", "opencv-python-headless",
                     "scipy", "scikit-learn", "matplotlib", "PyYAML", "timm", "segmentation-models-pytorch",
                     "ultralytics", "albumentations", "monai"):
            try:
                versions[dist] = md.version(dist)
            except md.PackageNotFoundError:
                versions[dist] = None
        info = {"python": sys.version.split()[0], "versions": versions}
        try:
            import torch
            info["torch_cuda"] = {"available": torch.cuda.is_available(), "cuda": torch.version.cuda,
                                  "cudnn": torch.backends.cudnn.version(), "device_count": torch.cuda.device_count(),
                                  "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        except ImportError:
            info["torch_cuda"] = None
        net = {}
        for host in ("github.com", "pypi.org", "files.pythonhosted.org"):
            try:
                socket.create_connection((host, 443), timeout=5).close()
                net[host] = True
            except OSError:
                net[host] = False
        info["internet"] = net
        return info

    data: dict = {}

    @section(rep, "data")
    def _():
        root = find_data_root(args.input)
        data["root"] = root
        per_folder: dict[str, int] = {}
        for where in build_index(root).values():
            per_folder[where["folder"]] = per_folder.get(where["folder"], 0) + 1
        counts = {}
        for split in ("train", "val", "valid", "test"):
            d = root / split
            if d.exists():
                counts[split] = {sub: (len(list((d / sub).glob("*"))) if (d / sub).exists() else None)
                                 for sub in ("images", "labels", "masks")}
        yml = next((p for p in (root / "data.yml", root.parent / "data.yml", root / "data.yaml") if p.exists()), None)
        from PIL import Image
        imgs = sorted((root / "train" / "images").glob("*.png"))[:40]
        t0 = time.time()
        sizes = set()
        for p in imgs:
            with Image.open(p) as im:
                sizes.add((im.size, im.mode))
                im.load()
        read_ms = (time.time() - t0) / max(len(imgs), 1) * 1000
        return {"root": str(root), "listing_of_input": sh(f"find '{args.input}' -maxdepth 4 -type d | head -40"),
                "counts": counts, "val_is_empty": not any(v["images"] for k, v in counts.items() if k in ("val", "valid")),
                "val_pooled_into_train": sum(n for f, n in per_folder.items() if f in ("val", "valid")),
                "train_pool_size": sum(n for f, n in per_folder.items() if f != "test"),
                "data_yml": yml.read_text() if yml else None,
                "image_size_and_mode_(w,h)": sorted(str(s) for s in sizes),
                "pil_read_ms_per_image": round(read_ms, 1)}

    @section(rep, "split_check")
    def _():
        if "root" not in data:
            return {"skipped": "data root not found"}
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "make_splits.py"), "--data-root", str(data["root"]),
                            "--out", str(out / "folds_check.csv")], capture_output=True, text=True)
        (out / "make_splits_stdout.txt").write_text(r.stdout + "\n" + r.stderr)
        if r.returncode != 0:
            return {"error": r.stderr[-2000:]}
        mine = json.loads((REPO / "splits" / "folds_v1_summary.json").read_text())
        theirs = json.loads((out / "folds_check_summary.json").read_text())

        def load(p):
            with open(p, newline="") as f:
                return {(row["split"], row["file"]): (row["cls6"], row["fold"], row["pixel_md5"]) for row in csv.DictReader(f)}
        a, b = load(REPO / "splits" / "folds_v1.csv"), load(out / "folds_check.csv")
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        on_disk = set(build_index(data["root"]))
        missing, extra = sorted(set(a) - on_disk), sorted(on_disk - set(a))
        return {"fingerprint_local": mine["fingerprint_sha1"], "fingerprint_kaggle": theirs["fingerprint_sha1"],
                "identical_files_classes_pixels": mine["fingerprint_sha1"] == theirs["fingerprint_sha1"],
                "identical_folds": not diff, "n_differences": len(diff), "first_differences": [list(k) for k in diff[:10]],
                "csv_files_found_on_disk": not missing and not extra, "n_missing_on_disk": len(missing),
                "n_not_in_csv": len(extra), "first_missing": [list(k) for k in missing[:10]],
                "class_counts_match_paper": "WARNING" not in r.stdout}

    @section(rep, "gpu_speed")
    def _():
        import torch
        if not torch.cuda.is_available():
            return {"skipped": "no GPU"}
        res = {}
        for i in range(torch.cuda.device_count()):
            dev = torch.device(f"cuda:{i}")
            a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
            b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
            for _ in range(3):
                a @ b
            torch.cuda.synchronize(dev)
            n, t0 = 0, time.time()
            while time.time() - t0 < 5:
                a @ b
                n += 1
            torch.cuda.synchronize(dev)
            dt = time.time() - t0
            res[torch.cuda.get_device_name(i) + f" #{i}"] = {"fp16_matmul_tflops": round(2 * 4096 ** 3 * n / dt / 1e12, 1),
                                                             "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
        return res

    rep["expected_counts_from_paper"] = EXPECTED_COUNTS
    (out / "env_report.json").write_text(json.dumps(rep, indent=2, default=str))

    # Short human-readable summary at the end of the log.
    print("\n===== SUMMARY =====")
    hw, sw, dt, sc = rep.get("hardware", {}), rep.get("software", {}), rep.get("data", {}), rep.get("split_check", {})
    print("GPU          :", hw.get("nvidia_smi"))
    print("CPU / RAM    :", hw.get("cpu_count"), "cores /", hw.get("ram_total"))
    print("torch / CUDA :", (sw.get("versions") or {}).get("torch"), "/", (sw.get("torch_cuda") or {}).get("cuda"))
    print("internet     :", sw.get("internet"))
    print("data root    :", dt.get("root"))
    print("counts       :", dt.get("counts"))
    print("val empty    :", dt.get("val_is_empty"), "| val images pooled into train:", dt.get("val_pooled_into_train"),
          "| training pool:", dt.get("train_pool_size"))
    print("split check  : same data =", sc.get("identical_files_classes_pixels"), "| same folds =", sc.get("identical_folds"),
          "| all files found =", sc.get("csv_files_found_on_disk"))
    print("GPU speed    :", rep.get("gpu_speed"))
    print(f"report       : {out / 'env_report.json'}")


if __name__ == "__main__":
    main()
