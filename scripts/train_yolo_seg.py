#!/usr/bin/env python3
"""Baselines S01 / S02: the base article's YOLOv8-seg, retrained on our 5 folds.

Protocol (Analysis/01 §1.6 for the paper, Analysis/04 §3.4 for our honesty rules):
  * Ultralytics YOLOv8-seg from COCO weights, 100 epochs, Ultralytics default hyper-parameters (the paper does
    not report its own; ours are saved per fold in ultralytics_args.yaml). One model per fold.
  * Fixed training length and FINAL weights (last.pt). Training runs with val=False, so the held-out fold is
    never used to pick an epoch (Ultralytics only validates once, after the last epoch, for its log).
    The paper instead kept the best-validation-loss checkpoint; we report this difference.
  * Prediction rule of the paper: fracture = any instance; class = class of the highest-confidence instance;
    mask = that instance's mask. Per-class confidences are saved, so the fracture threshold is fitted
    afterwards on the out-of-fold (OOF) predictions only (Youden) and then applied unchanged to the test set.
    Results at Ultralytics' default threshold 0.25 are reported as well.
  * Test set: every fold model predicts it; we report mean +/- SD over the 5 models and the 5-model ensemble
    (mean per-class confidence; mask = majority vote of the models' masks).

Two GPUs: folds run two at a time, one per GPU, each in its own process. A time guard does not start a fold
that could push the Kaggle session past --time-budget-h; skipped folds are listed in metrics.json and can be
run later with --folds.

Outputs
  results/<RUN>/   config.json, metrics.json, summary.md, per_case_oof.csv, per_case_test.csv,
                   folds/fold<k>/ (per-fold CSVs, training curve, Ultralytics settings, log tail)
  work/<RUN>/      YOLO data folders, weights (last.pt) and predicted masks: kept in the Kaggle version
                   output, not committed to git.

Examples
  python scripts/train_yolo_seg.py --model yolov8s-seg.pt --imgsz 1024 --batch 8 --out /kaggle/working/results/S01
  python scripts/train_yolo_seg.py ... --folds 3,4          # only some folds
  python scripts/train_yolo_seg.py ... --export-only        # build the YOLO folders and stop (no training)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from femurseg import metrics as M  # noqa: E402
from femurseg import reference as REF  # noqa: E402
from femurseg.classes import CLASSES  # noqa: E402
from femurseg.data import DEFAULT_FOLDS_CSV, ImageStore, fold_split, load_records, read_png  # noqa: E402

YOLO_NAMES = {c.yolo_id: c.code for c in CLASSES if c.yolo_id >= 0}  # {0: '2AB', 1: '2CD', 2: '3AB', 3: '3CDE', 4: '4'}
DEFAULT_TAU = 0.25  # Ultralytics' default confidence threshold
CASE_COLS = ["file", "split", "fold", "model", "cls6_true", "n_det", "score", "conf_1", "conf_2", "conf_3", "conf_4",
             "conf_5", "top_cls6", "top_conf", "area_gt", "area_top", "dice_top", "iou_top", "hd95_top", "nsd2_top",
             "dice_union025", "area_union025"]


# ----------------------------------------------------------------------------------------------- arguments


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Ultralytics weights or yaml, e.g. yolov8s-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--gpus", default="0,1", help="comma list of GPU ids, or 'cpu' (repeat to run folds in parallel)")
    ap.add_argument("--workers", type=int, default=0, help="data-loader workers per fold; 0 = CPU cores / parallel folds")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "none"])
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--folds-csv", default=str(DEFAULT_FOLDS_CSV))
    ap.add_argument("--out", required=True, help="results folder of this run (results/<RUN>)")
    ap.add_argument("--work", default="", help="folder for weights and masks; default <out>/../../work/<RUN>")
    ap.add_argument("--conf-min", type=float, default=0.001, help="lowest confidence kept at prediction")
    ap.add_argument("--max-det", type=int, default=20)
    ap.add_argument("--no-test", action="store_true", help="do not predict the test set")
    ap.add_argument("--time-budget-h", type=float, default=11.0, help="stop starting folds after this many hours")
    ap.add_argument("--est-fold-min", type=float, default=90, help="first guess of minutes per fold (time guard)")
    ap.add_argument("--extra-train", default="{}", help="JSON dict of extra Ultralytics train arguments")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many training femurs")
    ap.add_argument("--export-only", action="store_true", help="build the YOLO data folders and stop")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--fold", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--device", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.out = Path(args.out).resolve()
    args.work = Path(args.work).resolve() if args.work else args.out.parent.parent / "work" / args.out.name
    args.fold_list = [int(f) for f in str(args.folds).split(",") if f.strip() != ""]
    args.gpu_list = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    return args


# ------------------------------------------------------------------------------------------ data per fold


def fold_records(args, k: int):
    recs = load_records(args.folds_csv)
    train, held = fold_split(recs, k)
    test = [r for r in recs if r.split == "test"]
    if args.limit:
        train, held, test = train[:args.limit], held[:max(args.limit // 4, 4)], test[:max(args.limit // 4, 4)]
    return recs, train, held, test


def export_yolo_fold(store: ImageStore, train, held, out_dir: Path) -> Path:
    """YOLO folder for one fold: symlinked images, one label file per image (empty = no fracture), data.yaml."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for part, recs in (("train", train), ("val", held)):
        (out_dir / "images" / part).mkdir(parents=True)
        (out_dir / "labels" / part).mkdir(parents=True)
        for r in recs:
            src = store.path(r, "image").resolve()
            dst = out_dir / "images" / part / r.file
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            text = ""
            if r.has_fracture:
                poly = store.polygon(r)
                if poly is None or poly[0] != r.cls6 - 1:
                    raise ValueError(f"{r.file}: cls6 {r.cls6} but label {None if poly is None else poly[0]}")
                text = Path(store.path(r, "label")).read_text()
            (out_dir / "labels" / part / (Path(r.file).stem + ".txt")).write_text(text)
    names = "\n".join(f"  {i}: '{n}'" for i, n in YOLO_NAMES.items())
    yaml = out_dir / "data.yaml"
    yaml.write_text(f"path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return yaml


# ------------------------------------------------------------------------------------------- one fold


def run_fold(args) -> None:
    """Worker process: train the model of one fold, then predict its held-out fold and the test set."""
    k, device = args.fold, args.device
    t0 = time.time()
    fold_out = args.out / "folds" / f"fold{k}"
    fold_out.mkdir(parents=True, exist_ok=True)
    recs, train, held, test = fold_records(args, k)
    store = ImageStore(args.data_root, recs)
    yaml = export_yolo_fold(store, train, held, args.work / "yolo_data" / f"fold{k}")
    print(f"[fold {k}] {len(train)} training femurs, {len(held)} held out, {len(test)} test; data {yaml}", flush=True)

    import torch
    from ultralytics import YOLO, __version__ as ul_version, settings

    if settings.get("sync"):
        settings.update({"sync": False})  # no usage analytics (normally already done by the controller)
    extra = json.loads(args.extra_train)
    cache = {"ram": "ram", "disk": "disk", "none": False}[args.cache]
    model = YOLO(args.model)
    model.train(data=str(yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=device,
                workers=args.workers, seed=args.seed, deterministic=True, cache=cache, val=False, plots=False,
                project=str(args.work / "train"), name=f"fold{k}", exist_ok=True, **extra)
    save_dir = Path(model.trainer.save_dir)
    last = save_dir / "weights" / "last.pt"
    t_train = time.time() - t0
    for name, dst in (("results.csv", "train_curve.csv"), ("args.yaml", "ultralytics_args.yaml")):
        if (save_dir / name).exists():
            shutil.copy2(save_dir / name, fold_out / dst)

    model = YOLO(str(last))
    mask_dir = args.work / "pred_masks"
    parts = [("oof", held)] + ([] if args.no_test else [("test", test)])
    for part, rs in parts:
        rows = predict_cases(model, store, rs, args, device, k, mask_dir / part / f"fold{k}")
        write_csv(fold_out / f"{part}.csv", rows)
    meta = {"fold": k, "device": device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "n_train": len(train), "n_held_out": len(held), "n_test": 0 if args.no_test else len(test),
            "train_minutes": round(t_train / 60, 2), "total_minutes": round((time.time() - t0) / 60, 2),
            "weights": str(last), "ultralytics": ul_version, "torch": torch.__version__,
            "python": platform.python_version(),
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None}
    (fold_out / "fold_meta.json").write_text(json.dumps(meta, indent=2))
    # The YOLO data folder (image links, label copies) is rebuilt from the fold file whenever needed; removing it keeps
    # Kaggle's version output small (Kaggle may copy the linked images into the output).
    shutil.rmtree(args.work / "yolo_data" / f"fold{k}", ignore_errors=True)
    print(f"[fold {k}] done in {meta['total_minutes']} min", flush=True)


def predict_cases(model, store: ImageStore, recs, args, device, k: int, mask_dir: Path) -> list[dict]:
    """Run the model on femurs and turn each result into one row of per-case numbers (and a saved mask)."""
    import cv2

    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    precision = {"quantize": "fp16"} if device not in ("cpu", "") else {}  # fp16 inference on GPU
    for i in range(0, len(recs), 16):
        chunk = {r.file: r for r in recs[i:i + 16]}
        results = model.predict(source=[str(store.path(r, "image")) for r in chunk.values()], imgsz=args.imgsz,
                                conf=args.conf_min, iou=0.7, max_det=args.max_det, retina_masks=True,
                                device=device, verbose=False, stream=True, **precision)
        for res in results:
            rec = chunk[Path(res.path).name]
            gt = store.mask(rec) > 0
            n = 0 if res.boxes is None else len(res.boxes)
            conf = res.boxes.conf.cpu().numpy() if n else np.zeros(0)
            cls = res.boxes.cls.cpu().numpy().astype(int) if n else np.zeros(0, int)
            masks = (res.masks.data.cpu().numpy() > 0.5) if (n and res.masks is not None) else np.zeros((0,) + gt.shape, bool)
            if masks.shape[1:] != gt.shape:  # safety: masks must be at the original resolution
                masks = np.stack([cv2.resize(m.astype(np.uint8), gt.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
                                  for m in masks]) if len(masks) else np.zeros((0,) + gt.shape, bool)
            row = {"file": rec.file, "split": rec.split, "fold": rec.fold, "model": k, "cls6_true": rec.cls6, "n_det": n}
            per_class = [float(conf[cls == c].max()) if (cls == c).any() else 0.0 for c in range(5)]
            row.update({f"conf_{c + 1}": round(v, 6) for c, v in enumerate(per_class)})
            row["score"] = round(max(per_class), 6)
            top_mask = np.zeros_like(gt)
            if n:
                t = int(np.argmax(conf))
                row["top_cls6"], row["top_conf"] = int(cls[t]) + 1, round(float(conf[t]), 6)
                top_mask = masks[t]
                cv2.imwrite(str(mask_dir / rec.file), top_mask.astype(np.uint8) * 255)
            else:
                row["top_cls6"], row["top_conf"] = 0, 0.0
            union = masks[conf >= DEFAULT_TAU].any(axis=0) if n else np.zeros_like(gt)
            row["area_gt"], row["area_top"], row["area_union025"] = int(gt.sum()), int(top_mask.sum()), int(union.sum())
            if gt.any():
                s = M.seg_metrics(top_mask, gt)
                row.update(dice_top=round(s["dice"], 6), iou_top=round(s["iou"], 6), hd95_top=s["hd95"],
                           nsd2_top=round(s["nsd2"], 6), dice_union025=round(M.dice(union, gt), 6))
            rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------- evaluation


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CASE_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CASE_COLS})


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in CASE_COLS:
            if c in ("file", "split", "model"):
                continue
            v = r.get(c, "")
            r[c] = float(v) if v not in ("", "nan") else (float("nan") if v == "nan" else None)
        r["cls6_true"], r["top_cls6"], r["fold"] = int(r["cls6_true"]), int(r["top_cls6"]), int(r["fold"])
    return rows


def evaluate(rows: list[dict], tau: float) -> dict:
    """All headline metrics for a set of per-case rows at fracture threshold ``tau``."""
    y = np.array([r["cls6_true"] for r in rows])
    score = np.array([r["score"] for r in rows])
    pred = np.array([r["top_cls6"] if r["top_conf"] >= tau else 0 for r in rows])
    out: dict = {"n": len(rows), "threshold": round(float(tau), 6), "auc_fracture": M.roc_auc(y > 0, score)}
    out.update({f"fracture_{k}": v for k, v in M.binary_metrics(y > 0, pred > 0).items()})
    mc = M.multiclass_metrics(y, pred)
    out.update(accuracy_6class=mc["accuracy"], macro_f1=mc["macro_f1"], balanced_accuracy=mc["balanced_accuracy"],
               kappa=mc["kappa"], garden12_recall=mc["per_class"]["2AB"]["recall"],
               garden12_detected=float(np.mean(pred[y == 1] > 0)) if (y == 1).any() else float("nan"),
               errors=M.error_breakdown(y, pred), per_class=mc["per_class"], confusion_matrix=mc["confusion_matrix"])
    fx = [r for r in rows if r["cls6_true"] > 0]
    det = [r for r in fx if r["top_conf"] >= tau]
    dice_fx = [r["dice_top"] if r["top_conf"] >= tau else 0.0 for r in fx]
    out.update(
        dice_fractured_mean=float(np.mean(dice_fx)) if fx else float("nan"),
        dice_fractured_median=float(np.median(dice_fx)) if fx else float("nan"),
        dice_detected_mean=float(np.mean([r["dice_top"] for r in det])) if det else float("nan"),
        iou_fractured_mean=float(np.mean([r["iou_top"] if r["top_conf"] >= tau else 0.0 for r in fx])) if fx else float("nan"),
        nsd2_fractured_mean=float(np.mean([r["nsd2_top"] if r["top_conf"] >= tau else 0.0 for r in fx])) if fx else float("nan"),
        hd95_detected_median=float(np.nanmedian([r["hd95_top"] for r in det])) if det else float("nan"),
        n_fractured=len(fx), n_fractured_missed=len(fx) - len(det),
        dice_per_class={CLASSES[c].code: float(np.mean([d for d, r in zip(dice_fx, fx) if r["cls6_true"] == c]))
                        for c in range(1, 6) if any(r["cls6_true"] == c for r in fx)},
    )
    probs = np.array([[1 - r["score"]] + [r[f"conf_{c}"] for c in range(1, 6)] for r in rows])
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-9, None)
    out.update(ece_derived=M.expected_calibration_error(probs, y), brier_derived=M.brier_score(probs, y))
    return out


def scalar_summary(ms: list[dict]) -> tuple[dict, dict]:
    keys = [k for k, v in ms[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    mean = {k: float(np.nanmean([m[k] for m in ms])) for k in keys}
    sd = {k: float(np.nanstd([m[k] for m in ms], ddof=1)) if len(ms) > 1 else float("nan") for k in keys}
    return mean, sd


def ensemble_rows(test_by_model: dict[int, list[dict]], store: ImageStore, recs_by_file: dict, mask_root: Path,
                  tau: float) -> list[dict]:
    """5-model ensemble on the test set: mean per-class confidence; mask = pixel-wise majority vote of the masks of
    the models that call a fracture themselves (top confidence >= tau), used only when the ensemble calls one."""
    models = sorted(test_by_model)
    files = [r["file"] for r in test_by_model[models[0]]]
    by = {m: {r["file"]: r for r in test_by_model[m]} for m in models}
    rows = []
    for f in files:
        rs = [by[m][f] for m in models]
        conf = np.array([[r[f"conf_{c}"] for c in range(1, 6)] for r in rs]).mean(axis=0)
        top = int(np.argmax(conf))
        row = {"file": f, "split": "test", "fold": -1, "model": "ensemble", "cls6_true": rs[0]["cls6_true"],
               "n_det": float(np.mean([r["n_det"] for r in rs])), "score": float(conf.max()),
               "top_cls6": top + 1 if conf.max() > 0 else 0, "top_conf": float(conf.max())}
        row.update({f"conf_{c + 1}": float(conf[c]) for c in range(5)})
        rec = recs_by_file[f]
        gt = store.mask(rec) > 0
        votes, n_vote = np.zeros(gt.shape, np.float32), 0
        if row["top_conf"] >= tau:  # the ensemble calls a fracture: vote among the models that also call one
            for m, r in zip(models, rs):
                p = mask_root / f"fold{m}" / f
                if r["top_conf"] >= tau and p.exists():
                    votes += read_png(p) > 0
                    n_vote += 1
        ens_mask = votes >= max(n_vote, 1) / 2
        row["area_gt"], row["area_top"] = int(gt.sum()), int(ens_mask.sum())
        if gt.any():
            s = M.seg_metrics(ens_mask, gt)
            row.update(dice_top=s["dice"], iou_top=s["iou"], hd95_top=s["hd95"], nsd2_top=s["nsd2"])
        rows.append(row)
    return rows


def aggregate(args, done: list[int], skipped: list[int], failed: list[int], t0: float) -> dict:
    recs = load_records(args.folds_csv)
    store = ImageStore(args.data_root, recs)
    recs_by_file = {r.file: r for r in recs if r.split == "test"}
    oof, test_by_model = [], {}
    for k in sorted(done):
        fd = args.out / "folds" / f"fold{k}"
        oof += read_csv(fd / "oof.csv")
        if (fd / "test.csv").exists():
            test_by_model[k] = read_csv(fd / "test.csv")
    gpus = sorted({json.loads((args.out / "folds" / f"fold{k}" / "fold_meta.json").read_text())["gpu"] for k in done})
    n_dev = len({g for g in args.gpu_list if g != "cpu"})
    res: dict = {"run": args.out.name, "gpu": ", ".join(gpus) + (f" x{n_dev}" if n_dev > 1 else ""),
                 "folds_requested": args.fold_list, "folds_completed": sorted(done),
                 "folds_skipped_time_guard": sorted(skipped), "folds_failed": sorted(failed),
                 "minutes_total": round((time.time() - t0) / 60, 1)}
    if not oof:
        return res
    tau_oof = M.youden_threshold([r["cls6_true"] > 0 for r in oof], [r["score"] for r in oof])
    res["thresholds"] = {"default_ultralytics": DEFAULT_TAU, "oof_youden": tau_oof}
    for name, tau in (("", tau_oof), ("at_default_threshold_0.25", DEFAULT_TAU)):
        block: dict = {"oof": evaluate(oof, tau),
                       "oof_per_fold": {str(k): {kk: v for kk, v in evaluate([r for r in oof if r["fold"] == k], tau).items()
                                                 if isinstance(v, (int, float))} for k in sorted(done)}}
        if test_by_model:
            per_model = [evaluate(test_by_model[k], tau) for k in sorted(test_by_model)]
            block["test_models_mean"], block["test_models_sd"] = scalar_summary(per_model)
            if len(test_by_model) >= 2:
                ens = ensemble_rows(test_by_model, store, recs_by_file, args.work / "pred_masks" / "test", tau)
                block["test_ensemble"] = evaluate(ens, tau)
                if name == "":
                    write_csv(args.out / "per_case_test_ensemble.csv", ens)
        if name == "":
            res.update(block)  # headline numbers (threshold fitted on OOF) at the top level
        else:
            res[name] = block
    write_csv(args.out / "per_case_oof.csv", oof)
    write_csv(args.out / "per_case_test.csv", [r for k in sorted(test_by_model) for r in test_by_model[k]])
    return res


def write_summary(args, res: dict) -> None:
    """summary.md: headline numbers at both thresholds, next to the paper's grid means for the same setting."""
    def f(src, block, key, scale=1.0, nd=3):
        v = (src.get(block) or {}).get(key)
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * scale:.{nd}f}"

    g = REF.GRID.get(("small", args.imgsz))  # the article's YOLOv8-small at the same image size (Table 2)
    paper = (f"{g['auc'][0]:.3f}", f"{100 * g['accuracy_6class'][0]:.1f}", f"{g['dice'][0]:.3f}") if g else ("", "", "")
    rows = (("Fracture AUC", "auc_fracture", 1, 3, paper[0]),
            ("Fracture sensitivity", "fracture_sensitivity", 1, 3, ""),
            ("Fracture specificity", "fracture_specificity", 1, 3, ""),
            ("6-class accuracy (%)", "accuracy_6class", 100, 1, paper[1]),
            ("6-class macro-F1", "macro_f1", 1, 3, ""),
            ("Dice, fractured femurs (missed = 0)", "dice_fractured_mean", 1, 3, ""),
            ("Dice, detected fractures only", "dice_detected_mean", 1, 3, paper[2]),
            ("Garden I-II recall", "garden12_recall", 1, 3, ""))
    tau = (res.get("thresholds") or {}).get("oof_youden", "n/a")
    n_oof = (res.get("oof") or {}).get("n", 0)
    lines = [f"# {res['run']}: summary", "",
             f"Model {args.model}, imgsz {args.imgsz}, {args.epochs} epochs, batch {args.batch}, final weights (last.pt).",
             f"Folds completed {res['folds_completed']}, skipped {res['folds_skipped_time_guard']}, "
             f"failed {res['folds_failed']}. Runtime {res['minutes_total']} min on {res.get('gpu') or 'n/a'}."]
    for title, src in ((f"Fracture threshold fitted on the OOF predictions (Youden): {tau}", res),
                       ("Ultralytics' default threshold 0.25 (probably the paper's setting; not reported there)",
                        res.get("at_default_threshold_0.25") or {})):
        lines += ["", f"## {title}", "",
                  f"| Metric | OOF (n={n_oof:,}) | Test, fold models mean ± SD | Test, ensemble "
                  "| Paper, YOLOv8-small at this image size (test, 6 runs) |", "|---|---|---|---|---|"]
        for label, key, scale, nd, p in rows:
            lines.append(f"| {label} | {f(src, 'oof', key, scale, nd)} | {f(src, 'test_models_mean', key, scale, nd)} ± "
                         f"{f(src, 'test_models_sd', key, scale, nd)} | {f(src, 'test_ensemble', key, scale, nd)} | {p} |")
    lines += ["", "Paper column: González et al. 2025, YOLOv8-small grid, mean of 6 seeds on the test set (their Dice "
              "definition is not stated). Their final selected model (small, 1024) reports AUC 0.981, sensitivity 0.952, "
              "specificity 0.985, 6-class accuracy 86.19% and Dice 0.77.",
              "Differences from the paper: one model per fold of our 5 folds instead of 6 seeds on one split; final "
              "weights instead of the best-validation-loss checkpoint; Ultralytics defaults of the pinned version."]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------- controller


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker:
        run_fold(args)
        return 0
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    n_par = max(1, len(args.gpu_list))
    if args.workers <= 0:
        args.workers = max(1, min(8, (os.cpu_count() or 2) // n_par))
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    weights = args.model if args.export_only else prepare_ultralytics(args.model, args.work / "start_weights")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(git_commit=git, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  start_weights=weights, start_weights_md5=file_md5(weights),
                  protocol="final weights (last.pt), val=False during training, threshold fitted on OOF (Youden)")
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    if args.export_only:
        for k in args.fold_list:
            recs, train, held, _ = fold_records(args, k)
            y = export_yolo_fold(ImageStore(args.data_root, recs), train, held, args.work / "yolo_data" / f"fold{k}")
            n_lab = sum(1 for p in (y.parent / "labels" / "train").glob("*.txt") if p.stat().st_size > 0)
            print(f"fold {k}: {len(train)} train ({n_lab} with a fracture polygon), {len(held)} val -> {y}")
        return 0

    todo: queue.Queue = queue.Queue()
    for k in args.fold_list:
        todo.put(k)
    done, skipped, failed, durations = [], [], [], []
    lock = threading.Lock()
    budget = args.time_budget_h * 3600

    def gpu_worker(gpu: str) -> None:
        while True:
            try:
                k = todo.get_nowait()
            except queue.Empty:
                return
            with lock:
                est = statistics.mean(durations) if durations else args.est_fold_min * 60
                if time.time() - t0 + 1.15 * est > budget:
                    skipped.append(k)
                    print(f"[guard] fold {k} not started: {(time.time() - t0) / 3600:.1f} h used, "
                          f"about {est / 3600:.1f} h needed, budget {args.time_budget_h} h", flush=True)
                    continue
            env = dict(os.environ)
            device = "cpu"
            if gpu != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = gpu
                device = "0"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--fold", str(k), "--device", device,
                   *sys_argv_without_control(argv), "--model", weights, "--workers", str(args.workers)]
            log = args.work / "logs" / f"fold{k}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            ts = time.time()
            print(f"[fold {k}] started on {'GPU ' + gpu if gpu != 'cpu' else 'CPU'}; log {log}", flush=True)
            with open(log, "w") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
            tail = log.read_text(errors="replace").splitlines()[-60:]
            (args.out / "folds" / f"fold{k}").mkdir(parents=True, exist_ok=True)
            (args.out / "folds" / f"fold{k}" / "log_tail.txt").write_text("\n".join(tail) + "\n")
            with lock:
                if rc == 0:
                    done.append(k)
                    durations.append(time.time() - ts)
                else:
                    failed.append(k)
            print(f"[fold {k}] {'finished' if rc == 0 else f'FAILED (exit {rc})'} after "
                  f"{(time.time() - ts) / 60:.1f} min", flush=True)

    threads = [threading.Thread(target=gpu_worker, args=(g,)) for g in args.gpu_list]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    res = aggregate(args, done, skipped, failed, t0)
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    if res.get("oof"):
        write_summary(args, res)
        print((args.out / "summary.md").read_text())
    print(f"folds done {sorted(done)}, skipped {sorted(skipped)}, failed {sorted(failed)}; "
          f"total {(time.time() - t0) / 60:.1f} min")
    return 0 if not failed and not skipped else (1 if failed else 3)


def prepare_ultralytics(model: str, dest: Path) -> str:
    """Switch off Ultralytics' usage analytics and download the starting weights once into ``dest``, before the
    fold processes start (two processes downloading the same file at the same time can corrupt it).
    Returns the local path of the weights (a .yaml name is returned unchanged: untrained model, smoke tests)."""
    from ultralytics import settings
    from ultralytics.utils.downloads import attempt_download_asset

    if settings.get("sync"):
        settings.update({"sync": False})
    if not model.endswith(".pt"):
        return model
    p = Path(model)
    if not p.exists():
        dest.mkdir(parents=True, exist_ok=True)
        p = Path(attempt_download_asset(str(dest / p.name)))  # official Ultralytics release asset
    return str(p.resolve())


def file_md5(path: str) -> str:
    import hashlib

    p = Path(path)
    return hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else ""


def sys_argv_without_control(argv) -> list[str]:
    """The controller's own arguments, minus --folds and --gpus, passed on to each worker."""
    src = list(sys.argv[1:] if argv is None else argv)
    out, skip = [], False
    for a in src:
        if skip:
            skip = False
            continue
        if a in ("--folds", "--gpus"):
            skip = True
            continue
        if a.startswith("--folds=") or a.startswith("--gpus="):
            continue
        out.append(a)
    return out


if __name__ == "__main__":
    sys.exit(main())
