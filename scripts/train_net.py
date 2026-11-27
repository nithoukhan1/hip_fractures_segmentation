#!/usr/bin/env python3
"""Baselines S04-S09 and the core model M0 (S10): classifiers and segmenters at 768 x 512 px on our 5 folds.

Networks (--net, see src/femurseg/nets.py):
  cls:<timm model>           S04 densenet121.tv_in1k, S05 efficientnet_b4.ra2_in1k, S06 convnext_tiny.fb_in22k_ft_in1k (= A0)
  seg:<smp arch>:<encoder>   S07 Unet:resnet34, S08 DeepLabV3Plus:resnet34, S09 Segformer:mit_b2
  m0:<timm model>            S10 = M0: U-Net with a ConvNeXt-T encoder, binary mask + 6-class head (= A1 = R0);
                             --extra-tests deletion adds the region-deletion test (run A1_m0_deletion, Analysis/06 §13)
  slg:<timm model>:<A2..A6>  Paper A, SLG-Net and its ablation chain (src/femurseg/slg.py; protocol Analysis/06):
                             M0 + region pooling (A2), + geometry (A3), + location -> grade hierarchy (A4), + shared
                             severity trunk (A5), + logit adjustment (A6). Teacher forcing over the first --tf-epochs
                             epochs; --extra-tests oracle,deletion adds the interpretability tests of the A6 models.
  ref:<timm model>:<R1..R7>  Paper B, RefFrac and its comparisons (src/femurseg/refnet.py; protocol Analysis/08): M0 plus
                             healthy reference femurs from the fixed table of step B-1 (--ref-table, --pairs-table),
                             encoded by the same encoder, aligned to the target and fused at 1/16 and 1/32.
                             --extra-tests self,random,fractured,retrieved,blind predicts the held-out fold and the test
                             set again with other references (Analysis/08 §2.6).

Protocol (the rules of S01/S02, Analysis/04 §3.4):
  * one model per fold, fixed training length (--epochs) and FINAL weights; the held-out fold is never used
    during training;
  * one recipe for every network: ImageNet weights; input 768 x 512 (the 1024 x 1536 crop at half size, grey copied
    to 3 channels); augmentations of src/femurseg/torchdata.py; AdamW (lr 2e-4, weight decay 0.05, none on biases
    and norms); 2 warm-up epochs, then cosine decay to 0; batch 16; mixed precision (fp16); gradient clipping 1.0;
  * losses: cross-entropy (classifiers); pixel cross-entropy + Dice over the 5 fracture classes (segmenters);
    cross-entropy + binary cross-entropy + Dice of the fracture mask (M0, mask loss weight --seg-weight);
  * outputs and decision rule of src/femurseg/nets.py; the fracture threshold is fitted on the out-of-fold (OOF)
    predictions (Youden) and applied unchanged to the test set; results at the plain threshold 0.5 are reported too;
  * test set: every fold model predicts it; we report mean +/- SD over the 5 models and the 5-model ensemble (mean
    class probabilities; mask = mean fracture-probability map >= 0.5).

Before the folds start, the controller (1) reads every femur once into a resized cache on local disk, and (2) runs a
pre-flight on the first GPU: it downloads the ImageNet weights, finds the largest micro-batch that fits in memory
(halving from --batch; gradients are then accumulated, so the effective batch stays --batch) and times a training
step. The estimate feeds the time guard. With --check-nets, only the pre-flight runs, for several networks
(run S00_net_check). Two GPUs: folds run two at a time, as in train_yolo_seg.py (src/femurseg/runner.py).

Outputs
  results/<RUN>/  config.json, preflight.json, metrics.json, summary.md, per_case_oof.csv, per_case_test.csv,
                  per_case_test_ensemble.csv, folds/fold<k>/ (per-fold CSVs, train_curve.csv, fold_meta.json, log tail);
                  SLG-Net and M0 with --extra-tests: the same per-case files with the suffix _oracle (SLG-Net only),
                  _deletion and _deletion_control, and their metrics under "tests" in metrics.json;
                  RefFrac: the same per-case files for every test mode (_self, _random, ...), and
                  per_case_<oof|test>_references.csv: the references each femur got in each mode and the pointing
                  game of the difference map (base mode)
  work/<RUN>/     weights/fold<k>.pt; fracture-probability maps prob_maps/<oof|test>/fold<k>/<file> (768 x 512 PNG,
                  value = 255 x probability; RefFrac also prob_maps/<oof|test>_<mode>/); logs. Kept in the Kaggle
                  version output, not committed to git.

Examples
  python scripts/train_net.py --net cls:convnext_tiny.fb_in22k_ft_in1k --out /kaggle/working/results/S06_convnext_tiny
  python scripts/train_net.py --net seg:Unet:resnet34 --folds 0 --gpus 0 --out ...      # one fold only
  python scripts/train_net.py --check-nets cls:densenet121.tv_in1k,seg:Unet:resnet34 --out .../S00_net_check
  python scripts/train_net.py --net slg:convnext_tiny.fb_in22k_ft_in1k:A6 --folds 0 --no-test --extra-tests oracle \
      --out .../A6_dev_base                                                          # fold-0 development (Analysis/06)
  python scripts/train_net.py --net ref:convnext_tiny.fb_in22k_ft_in1k:R4 --folds 0 --no-test --extra-tests self,random \
      --out .../R4_dev_base                                                          # fold-0 development (Analysis/08)
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from femurseg import metrics as M  # noqa: E402
from femurseg import reference as REF  # noqa: E402
from femurseg import refnet as RN  # noqa: E402
from femurseg.data import DEFAULT_FOLDS_CSV, fold_split, load_records, read_png  # noqa: E402
from femurseg.evaluation import evaluate, read_csv, scalar_summary, seg_case, write_csv  # noqa: E402
from femurseg.nets import TOPK, build_net, make_loss, param_groups, parse_net, postprocess  # noqa: E402
from femurseg.runner import argv_without, file_md5, git_commit, run_folds  # noqa: E402
from femurseg.slg import priors_from_counts  # noqa: E402
from femurseg.torchdata import Cache, FemurDataset, RefFemurDataset, build_cache, normalize, worker_init_fn  # noqa: E402

PLAIN_TAU = 0.5
EXTRA_TESTS = {"slg": ("oracle", "deletion"),       # SLG-Net; "deletion" also writes "deletion_control"
               "m0": ("deletion",),                 # M0: deletion only (its head never sees the mask, so no oracle)
               "ref": tuple(RN.TEST_POLICY)}         # RefFrac: self, random, fractured, retrieved, blind
TEST_MODES = {"oracle": ("oracle",), "deletion": ("deletion", "deletion_control"), **{t: (t,) for t in RN.TEST_POLICY}}
REF_TABLES = REPO / "results" / "R00_reference_bank_r2"  # the fixed reference table of step B-1 (Analysis/07 §9)
REF_COLS = ["file", "split", "fold", "model", "cls6_true", "mode", "source", "refs", "diff_hit", "diff_chance",
            "diff_peak_y", "diff_peak_x"]
DELETE_DILATE_PX, DELETE_SIGMA_PX = 8, 16           # at 768 x 512; scaled with the input height otherwise
PAPER_TABLE3 = {"densenet121": "DenseNet121", "efficientnet_b4": "EfficientNetB4"}  # same architecture in the article


# ----------------------------------------------------------------------------------------------- arguments


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default="", help="cls:<timm model> | seg:<smp arch>:<encoder> | m0:<timm model>")
    ap.add_argument("--check-nets", default="", help="comma list of --net strings: only run the pre-flight of each")
    ap.add_argument("--size", default="768x512", help="input height x width (multiples of 32)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16, help="effective batch size")
    ap.add_argument("--micro-batch", type=int, default=0, help="images per forward pass; 0 = largest that fits")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=2.0)
    ap.add_argument("--drop-rate", type=float, default=0.2, help="dropout before the classifier layer")
    ap.add_argument("--seg-weight", type=float, default=1.0, help="M0 and SLG-Net: weight of the mask loss")
    ap.add_argument("--tf-epochs", type=float, default=30.0,
                    help="SLG-Net teacher forcing: the chance that the heads get the released mask instead of the "
                         "predicted one falls linearly from 1 to 0 over this many epochs (0 = never)")
    ap.add_argument("--slg-scales", default="16,32", help="SLG-Net: encoder scales of region pooling (16 and/or 32)")
    ap.add_argument("--slg-detach", action="store_true", help="SLG-Net: the decision losses do not reach the mask")
    ap.add_argument("--la-tau", type=float, default=1.0, help="SLG-Net A6: strength of logit adjustment")
    ap.add_argument("--joint-weight", type=float, default=0.5,
                    help="SLG-Net A4-A6: weight of the chain-rule 6-class likelihood in the loss")
    ap.add_argument("--extra-tests", default="", help="SLG-Net: comma list of oracle, deletion (interpretability "
                                                      "tests on the held-out fold and the test set); M0: deletion; "
                                                      "RefFrac: of self, random, fractured, retrieved, blind (other "
                                                      "references)")
    ap.add_argument("--ref-table", default=str(REF_TABLES / "references.csv"),
                    help="RefFrac: the reference table of step B-1 (references.csv)")
    ap.add_argument("--pairs-table", default=str(REF_TABLES / "pairs.csv"), help="RefFrac: the pair index (pairs.csv)")
    ap.add_argument("--ref-k", type=int, default=0, help="RefFrac: number of references; 0 = the variant's")
    ap.add_argument("--ref-window", type=float, default=RN.DEFAULT_WINDOW,
                    help="RefFrac: half-width of the attention window as a fraction of the image height (0 = global)")
    ap.add_argument("--ref-margin", type=float, default=None,
                    help="RefFrac: weight of AASN's margin loss; default = the variant's (0.5 in R1, 0 otherwise)")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping")
    ap.add_argument("--topk", type=int, default=TOPK, help="segmenters: pixels averaged for the fracture score")
    ap.add_argument("--no-pretrained", action="store_true", help="random initialisation (tests only)")
    ap.add_argument("--no-amp", action="store_true", help="full precision on the GPU")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--gpus", default="0,1", help="comma list of GPU ids, or 'cpu' (repeat to run folds in parallel)")
    ap.add_argument("--workers", type=int, default=0, help="data-loader workers per fold; 0 = CPU cores / parallel folds")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--folds-csv", default=str(DEFAULT_FOLDS_CSV))
    ap.add_argument("--cache-dir", default="/tmp/femurseg_cache", help="resized image cache (local disk)")
    ap.add_argument("--out", required=True, help="results folder of this run (results/<RUN>)")
    ap.add_argument("--work", default="", help="folder for weights and maps; default <out>/../../work/<RUN>")
    ap.add_argument("--no-test", action="store_true", help="do not predict the test set")
    ap.add_argument("--no-maps", action="store_true", help="do not save probability maps (then no ensemble masks)")
    ap.add_argument("--time-budget-h", type=float, default=11.0, help="stop starting folds after this many hours")
    ap.add_argument("--est-fold-min", type=float, default=0, help="first guess of minutes per fold; 0 = pre-flight")
    ap.add_argument("--preflight-steps", type=int, default=8, help="timed training steps in the pre-flight")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many training femurs")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--preflight", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--preflight-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--cache", default="", help=argparse.SUPPRESS)
    ap.add_argument("--fold", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--device", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not args.net and not args.check_nets:
        ap.error("give --net (a training run) or --check-nets (pre-flight only)")
    args.out = Path(args.out).resolve()
    args.work = Path(args.work).resolve() if args.work else args.out.parent.parent / "work" / args.out.name
    args.fold_list = [int(f) for f in str(args.folds).split(",") if f.strip() != ""]
    args.gpu_list = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    h, w = (int(v) for v in args.size.lower().split("x"))
    if h % 32 or w % 32:
        ap.error("--size: height and width must be multiples of 32")
    args.size_hw = (h, w)
    args.slg_scale_list = tuple(int(v) for v in str(args.slg_scales).split(",") if v.strip())
    args.test_list = [t.strip() for t in str(args.extra_tests).split(",") if t.strip()]
    kind = args.net.split(":")[0] if args.net else ""
    allowed = EXTRA_TESTS.get(kind, ())
    bad = [t for t in args.test_list if t not in allowed]
    if bad:
        ap.error(f"--extra-tests {bad}: " + (f"choose from {', '.join(allowed)}" if allowed else
                                             "only SLG-Net (--net slg:...), M0 (--net m0:..., deletion) and "
                                             "RefFrac (--net ref:...) have tests"))
    args.modes = ["base"] + [m for t in args.test_list for m in TEST_MODES[t]]
    if kind == "ref":
        try:
            ref_cfg(args, parse_net(args.net))
        except ValueError as e:
            ap.error(str(e))
    return args


def net_options(args) -> dict:
    """Keyword arguments of nets.build_net for this run."""
    return {"slg_scales": args.slg_scale_list, "slg_detach": args.slg_detach, "ref_k": args.ref_k,
            "ref_margin": args.ref_margin, "ref_window": args.ref_window}


def ref_cfg(args, spec) -> dict:
    """RefFrac: the variant's settings with this run's overrides (refnet.variant_config)."""
    return RN.variant_config(spec.variant, args.ref_k, args.ref_margin, args.ref_window)


def model_inputs(spec, gt=None, tf_prob: float = 0.0, refs=None) -> dict:
    """Extra forward arguments: SLG-Net gets the released mask for teacher forcing or the oracle test; RefFrac gets
    the reference images (B k, 3, H, W)."""
    if spec.kind == "slg" and gt is not None:
        return {"gt": gt, "tf_prob": tf_prob}
    if spec.kind == "ref" and refs is not None:
        return {"refs": refs}
    return {}


def reference_batch(cache: Cache, srcs: list[list[tuple[tuple[str, str], bool]]], dev):
    """RefFrac at prediction time: the references of several femurs, as cached (mirrored where the table says),
    stacked femur by femur and normalised: (B k, 3, H, W)."""
    import torch

    imgs = np.stack([np.stack([RN.reference_image(cache.images, cache.row[key], mir) for key, mir in s]) for s in srcs])
    return normalize(torch.from_numpy(imgs).flatten(0, 1)[:, None], dev)


def fold_records(args, k: int):
    recs = load_records(args.folds_csv)
    train, held = fold_split(recs, k)
    test = [r for r in recs if r.split == "test"]
    if args.limit:
        train, held, test = train[:args.limit], held[:max(args.limit // 4, 4)], test[:max(args.limit // 4, 4)]
    return recs, train, held, test


def torch_device(text: str):
    import torch

    return torch.device("cpu") if text in ("", "cpu") else torch.device(f"cuda:{text}")


def versions() -> dict:
    import albumentations
    import segmentation_models_pytorch
    import timm
    import torch

    return {"torch": torch.__version__, "timm": timm.__version__, "smp": segmentation_models_pytorch.__version__,
            "albumentations": albumentations.__version__, "numpy": np.__version__, "python": platform.python_version()}


# --------------------------------------------------------------------------------------------- one fold


def run_fold(args) -> None:
    """Worker process: train the model of one fold, then predict its held-out fold and the test set."""
    import torch
    from torch.utils.data import DataLoader

    k, dev = args.fold, torch_device(args.device)
    t0 = time.time()
    fold_out = args.out / "folds" / f"fold{k}"
    fold_out.mkdir(parents=True, exist_ok=True)
    recs, train, held, test = fold_records(args, k)
    cache = Cache(args.cache)
    spec = parse_net(args.net)
    torch.manual_seed(args.seed + k)
    np.random.seed(args.seed + k)
    torch.set_num_threads(max(1, args.workers))
    torch.backends.cudnn.benchmark = True
    print(f"[fold {k}] {spec.text}: {len(train)} training femurs, {len(held)} held out, {len(test)} test; "
          f"device {dev}", flush=True)

    model = build_net(spec, pretrained=not args.no_pretrained, drop_rate=args.drop_rate, **net_options(args))
    model = model.to(dev).to(memory_format=torch.channels_last)
    counts = Counter(r.cls6 for r in train)
    priors = priors_from_counts(counts) if spec.kind == "slg" else None  # logit adjustment: training fold only
    micro = args.micro_batch or args.batch
    accum = max(1, args.batch // micro)
    gen = torch.Generator()
    gen.manual_seed(args.seed + k)
    table, rcfg, train_sources = None, {}, Counter()
    if spec.kind == "ref":  # RefFrac: each training femur's references, from the fixed table (Analysis/08 §2.1)
        rcfg = ref_cfg(args, spec)
        table = RN.RefTable(args.ref_table, args.pairs_table, recs)
        if k not in table.fold_models:
            raise ValueError(f"the reference table has no rows for fold model {k}")
        srcs = [table.sources(k, r, rcfg["policy"], rcfg["k"]) for r in train]
        train_sources = Counter(table.source_name(r, rcfg["policy"], s) for r, s in zip(train, srcs))
        ds = RefFemurDataset(cache, cache.rows(train), [[(cache.row[key], mir) for key, mir in s] for s in srcs],
                             augment=True, seed=args.seed + k)
        print(f"[fold {k}] RefFrac {spec.variant}: {rcfg}; training references {dict(train_sources)}", flush=True)
    else:
        ds = FemurDataset(cache, cache.rows(train), augment=True, seed=args.seed + k)
    dl = DataLoader(ds, batch_size=micro, shuffle=True, drop_last=True, num_workers=args.workers,
                    pin_memory=dev.type == "cuda", persistent_workers=args.workers > 0, worker_init_fn=worker_init_fn,
                    generator=gen)
    steps_per_epoch = max(1, len(dl) // accum)
    total = args.epochs * steps_per_epoch
    warm = max(1, int(round(args.warmup_epochs * steps_per_epoch)))
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    amp = dev.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = make_loss(spec, args.seg_weight, priors=priors, la_tau=args.la_tau, joint_weight=args.joint_weight,
                        ref_margin=rcfg.get("margin", 0.0))
    tf_steps = args.tf_epochs * len(dl)  # teacher forcing (SLG-Net): micro-steps until the heads see only own masks
    curve, micro_step, tf_prob = [], 0, 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        te, sums, n = time.time(), defaultdict(float), 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            x, m, y = batch[0], batch[1], batch[2]
            x = normalize(x, dev)
            m, y = m.to(dev, non_blocking=True).long(), y.to(dev, non_blocking=True).long()
            refs = normalize(batch[3].flatten(0, 1), dev) if len(batch) > 3 else None  # RefFrac: (B k, 3, H, W)
            tf_prob = max(0.0, 1.0 - micro_step / tf_steps) if tf_steps > 0 else 0.0
            extra = model_inputs(spec, (m > 0).unsqueeze(1).float(), tf_prob, refs)
            micro_step += 1
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                out = model(x, **extra)
            loss, parts = loss_fn(out, m, y)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            sums["loss"] += float(loss.detach())
            for name, v in parts.items():
                sums[name] += v
            n += 1
        row = {"epoch": epoch, "lr": round(opt.param_groups[0]["lr"], 8),
               **{name: round(v / max(n, 1), 5) for name, v in sums.items()}, "seconds": round(time.time() - te, 1)}
        if spec.kind == "slg":
            row["tf_prob"] = round(tf_prob, 4)  # chance of the released mask in the epoch's last step
        curve.append(row)
        print(f"[fold {k}] epoch {epoch}/{args.epochs} " + " ".join(f"{a} {b}" for a, b in row.items() if a != "epoch"),
              flush=True)
        if not math.isfinite(row["loss"]):
            raise RuntimeError(f"fold {k}: the training loss is {row['loss']} in epoch {epoch}")
    t_train = time.time() - t0
    with open(fold_out / "train_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    wdir = args.work / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), wdir / f"fold{k}.pt")

    parts = [("oof", held)] + ([] if args.no_test else [("test", test)])
    for part, rs in parts:
        map_dir = None if (args.no_maps or not spec.has_mask) else args.work / "prob_maps" / part / f"fold{k}"
        infos: list[dict] = []
        ref_infos: list[dict] = []
        rows = predict_cases(model, spec, cache, rs, args, dev, k, map_dir, amp, info_out=infos, table=table,
                             ref_out=ref_infos)
        for mode, mode_rows in rows.items():
            write_csv(fold_out / (f"{part}.csv" if mode == "base" else f"{part}_{mode}.csv"), mode_rows)
        if infos:  # where the deletion test's control region went, per femur
            with open(fold_out / f"{part}_deletion_info.csv", "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(infos[0]))
                wr.writeheader()
                wr.writerows(infos)
        if ref_infos:  # RefFrac: the references of every femur in every mode, and the pointing game
            write_ref_csv(fold_out / f"{part}_references.csv", ref_infos)
    slg_meta = {} if spec.kind != "slg" else {
        "variant": spec.variant, "tf_epochs": args.tf_epochs, "slg_scales": list(args.slg_scale_list),
        "slg_detach": args.slg_detach, "la_tau": args.la_tau, "joint_weight": args.joint_weight,
        "train_counts": {str(c): counts[c] for c in range(6)}, "priors": priors, "extra_tests": args.test_list}
    if spec.kind == "ref":
        h, w = args.size_hw
        slg_meta = {"variant": spec.variant, **rcfg,
                    "window_cells": {f"1/{s}": RN.window_cells(rcfg["window"], h // s) for s in (16, 32)}
                    if rcfg["align"] == "attention" else None,
                    "reference_table": str(args.ref_table), "reference_table_md5": file_md5(args.ref_table),
                    "pairs_table": str(args.pairs_table), "pairs_table_md5": file_md5(args.pairs_table),
                    "train_sources": dict(train_sources), "train_counts": {str(c): counts[c] for c in range(6)},
                    "extra_tests": args.test_list}
    if spec.kind == "m0" and args.test_list:
        slg_meta = {"extra_tests": args.test_list}
    meta = {"fold": k, "net": spec.text, "device": str(dev), **slg_meta,
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "n_train": len(train), "n_held_out": len(held), "n_test": 0 if args.no_test else len(test),
            "micro_batch": micro, "accumulation": accum, "optimizer_steps": total, "warmup_steps": warm,
            "train_minutes": round(t_train / 60, 2), "total_minutes": round((time.time() - t0) / 60, 2),
            "final_loss": curve[-1]["loss"], "weights": str(wdir / f"fold{k}.pt"),
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2), **versions(),
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2) if dev.type == "cuda" else None}
    (fold_out / "fold_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[fold {k}] done in {meta['total_minutes']} min", flush=True)


def predict_cases(model, spec, cache: Cache, recs, args, dev, k: int, map_dir: Path | None,
                  amp: bool, info_out: list | None = None, table=None, ref_out: list | None = None
                  ) -> dict[str, list[dict]]:
    """One row of per-case numbers per femur (src/femurseg/evaluation.py) for every mode in args.modes, and the
    saved probability maps (base mode only; RefFrac: every mode, in prob_maps/<part>_<mode>/).

    Modes: "base" = the model as it is used; SLG-Net and M0 tests (Analysis/06 §5 and §13; M0 has no oracle):
    "oracle" = the heads get the released mask (the network's own mask and its Dice are unchanged, so they are
    copied from base); "deletion" = the predicted region (fracture probability >= 0.5, dilated) is replaced by a
    blurred copy of the image; "deletion_control" =
    the same region shape blurred at another place of the image (see deletion_images; where the control went is
    appended to ``info_out``, one dict per femur). RefFrac tests (Analysis/08 §2.6): the same femurs with other
    references from ``table`` (refnet.TEST_POLICY: self, random, fractured, retrieved = nearest, blind); the
    references every femur got in every mode, and the pointing game of the base mode's difference map, are appended
    to ``ref_out``."""
    import cv2
    import torch

    model.eval()
    if map_dir is not None:
        map_dir.mkdir(parents=True, exist_ok=True)
    full_h, full_w = cache.full_hw
    bs = max(1, args.micro_batch or args.batch)
    rows: dict[str, list[dict]] = {m: [] for m in args.modes}
    rcfg = ref_cfg(args, spec) if spec.kind == "ref" else {}

    def run(x, **kw):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            return postprocess(spec, model(x, **kw), args.topk)

    def mode_dir(mode: str) -> Path | None:
        """prob_maps/<part>/fold<k> for the base mode, prob_maps/<part>_<mode>/fold<k> for a RefFrac test mode."""
        if map_dir is None or mode == "base":
            return map_dir
        d = map_dir.parent.parent / f"{map_dir.parent.name}_{mode}" / map_dir.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add(mode: str, pp: dict, chunk, idx, save_maps: bool, copy_seg_from: list[dict] | None = None) -> list[dict]:
        conf, score = pp["conf"].cpu().numpy(), pp["score"].cpu().numpy()
        top = pp["top_cls6"].cpu().numpy()
        fmap = pp["fmap"].cpu().numpy() if pp["fmap"] is not None else None
        new = []
        for j, rec in enumerate(chunk):
            sc = round(float(score[j]), 8)
            row = {"file": rec.file, "split": rec.split, "fold": rec.fold, "model": k, "cls6_true": rec.cls6,
                   "score": sc, "top_cls6": int(top[j]), "top_conf": sc}
            row.update({f"conf_{c + 1}": round(float(conf[j, c]), 8) for c in range(5)})
            gt = cache.gt_full(idx[j])
            row["area_gt"] = int(gt.sum())
            if copy_seg_from is not None:
                row.update({c: copy_seg_from[j][c] for c in ("area_top", "dice_top", "iou_top", "hd95_top", "nsd2_top",
                                                              "nsd10_top", "nsd20_top") if c in copy_seg_from[j]})
            elif fmap is not None:
                f = fmap[j].astype(np.float32)
                if save_maps and map_dir is not None:
                    cv2.imwrite(str(mode_dir(mode) / rec.file), np.round(f * 255).astype(np.uint8))
                mask = cv2.resize(f, (full_w, full_h), interpolation=cv2.INTER_LINEAR) >= 0.5
                row["area_top"] = int(mask.sum())
                if gt.any():
                    row.update(seg_case(mask, gt))
            new.append(row)
        rows[mode] += new
        return new

    with torch.no_grad():
        for i in range(0, len(recs), bs):
            chunk = recs[i:i + bs]
            idx = cache.rows(chunk)
            imgs = np.stack([cache.images[j] for j in idx])
            x = normalize(torch.from_numpy(imgs)[:, None], dev)
            if spec.kind == "ref":  # every mode = the same femurs with other references
                for mode in args.modes:
                    policy = rcfg["policy"] if mode == "base" else RN.TEST_POLICY[mode]
                    srcs = [table.sources(k, rec, policy, rcfg["k"]) for rec in chunk]
                    pp = run(x, refs=reference_batch(cache, srcs, dev))
                    add(mode, pp, chunk, idx, save_maps=True)
                    diff = pp["diff16"].float().cpu().numpy() if mode == "base" else None
                    for j, rec in enumerate(chunk):
                        info = {"file": rec.file, "split": rec.split, "fold": rec.fold, "model": k,
                                "cls6_true": rec.cls6, "mode": mode, "source": table.source_name(rec, policy, srcs[j]),
                                "refs": ";".join(f"{sp}/{f}" + (" mirrored" if mir else "") for (sp, f), mir in srcs[j])}
                        if diff is not None and rec.cls6 > 0:
                            info.update(RN.pointing_game(diff[j], np.asarray(cache.masks[idx[j]])))
                        if ref_out is not None:
                            ref_out.append(info)
                continue
            pp = run(x)
            base = add("base", pp, chunk, idx, save_maps=True)
            if "oracle" in rows:
                gt = torch.from_numpy((np.stack([cache.masks[j] for j in idx]) > 0).astype(np.float32))[:, None]
                add("oracle", run(x, **model_inputs(spec, gt.to(dev), 1.0)), chunk, idx, False, copy_seg_from=base)
            if "deletion" in rows:
                fmap = pp["fmap"].float().cpu().numpy()
                pairs = [deletion_images(imgs[j], fmap[j] >= 0.5, rec.file) for j, rec in enumerate(chunk)]
                if info_out is not None:
                    info_out += [{"file": rec.file, "fold": rec.fold, "model": k, "cls6_true": rec.cls6, **pr[2]}
                                 for rec, pr in zip(chunk, pairs)]
                for mode, which in (("deletion", 0), ("deletion_control", 1)):
                    xm = normalize(torch.from_numpy(np.stack([pr[which] for pr in pairs]))[:, None], dev)
                    add(mode, run(xm), chunk, idx, save_maps=False)
    return rows


def deletion_images(img: np.ndarray, region: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Region-deletion test inputs for one image at the network's input size (uint8).

    deletion: the predicted region, dilated by DELETE_DILATE_PX, is replaced by the image blurred with a Gaussian of
    DELETE_SIGMA_PX (which removes the fracture line but keeps the local brightness, so no artificial edge appears).
    control: the same region shape is blurred at another place: the first of up to 200 random shifts (seeded by the
    file name) that keeps the shape inside the image, does not overlap the region and has a mean brightness of at
    least 0.8 x the region's (on bone rather than background); otherwise the brightest shift without overlap.
    A femur without a predicted region is left unchanged in both. The third value says what happened:
    region_px (deleted pixels) and control = matched | dimmer | none (no place without overlap) | no_region."""
    import cv2

    if not region.any():
        return img, img, {"region_px": 0, "control": "no_region"}
    h, w = img.shape
    scale = h / 768
    rad = max(1, round(DELETE_DILATE_PX * scale))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    reg = cv2.dilate(region.astype(np.uint8), ker) > 0
    blur = cv2.GaussianBlur(img, (0, 0), DELETE_SIGMA_PX * scale)
    deleted = np.where(reg, blur, img)
    ys, xs = np.nonzero(reg)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    target = float(img[reg].mean())
    rng = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
    best, best_mu, placed = None, -1.0, "none"
    if y1 - y0 + 1 < h and x1 - x0 + 1 < w:
        for _ in range(200):
            dy, dx = int(rng.integers(-y0, h - y1)), int(rng.integers(-x0, w - x1))
            shifted = np.zeros_like(reg)
            shifted[ys + dy, xs + dx] = True
            if (shifted & reg).any():
                continue
            mu = float(img[shifted].mean())
            if mu >= 0.8 * target:
                best, placed = shifted, "matched"
                break
            if mu > best_mu:
                best, best_mu, placed = shifted, mu, "dimmer"
    control = np.where(best, blur, img) if best is not None else img
    return deleted, control, {"region_px": int(reg.sum()), "control": placed}


def write_ref_csv(path: Path, rows: list[dict]) -> None:
    """RefFrac: one row per femur, model and mode with the references it got (REF_COLS)."""
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=REF_COLS, restval="", extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


# ---------------------------------------------------------------------------------------------- pre-flight


def run_preflight(args) -> dict:
    """Worker process: build the network (downloading its weights), find the largest micro-batch that fits, time a
    training step and an inference pass, and estimate the minutes per fold. Writes JSON to --preflight-out."""
    import torch

    dev = torch_device(args.device)
    spec = parse_net(args.net)
    res: dict = {"net": spec.text, "ok": False, "device": str(dev),
                 "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu", "pretrained": not args.no_pretrained}
    try:
        t = time.time()
        model = build_net(spec, pretrained=not args.no_pretrained, drop_rate=args.drop_rate, **net_options(args))
        res["build_seconds"] = round(time.time() - t, 1)  # includes the weight download on first use
        res["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
        model = model.to(dev).to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        amp = dev.type == "cuda" and not args.no_amp
        rk = ref_cfg(args, spec)["k"] if spec.kind == "ref" else 0
        loss_fn = make_loss(spec, args.seg_weight, joint_weight=args.joint_weight,
                            ref_margin=ref_cfg(args, spec)["margin"] if spec.kind == "ref" else 0.0)
        opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=1e-7)
        if rk:
            res["references_per_femur"] = rk
        h, w = args.size_hw
        total_gb = torch.cuda.get_device_properties(dev).total_memory / 1e9 if dev.type == "cuda" else None
        cands = [args.micro_batch] if args.micro_batch else [b for b in (16, 8, 4, 2, 1) if b <= args.batch]
        res["tried"] = []
        for mb in cands:
            try:
                entry = _timed_steps(model, loss_fn, opt, amp, dev, mb, args)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom(e):
                    raise
                entry = {"micro_batch": mb, "oom": True}
            opt.zero_grad(set_to_none=True)
            gc.collect()  # the failed step's tensors are gone with its frame; return their memory to the GPU
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            res["tried"].append(entry)
            if entry.get("oom"):
                continue
            if total_gb and entry["peak_gb"] > 0.85 * total_gb:
                entry["note"] = "over 85% of the GPU memory: next size down"
                continue
            res.update(micro_batch=mb, accumulation=max(1, args.batch // mb), s_per_micro_step=entry["s_per_step"],
                       peak_gb=entry["peak_gb"], gpu_total_gb=round(total_gb, 1) if total_gb else None)
            break
        if "micro_batch" not in res:
            raise RuntimeError("no micro-batch fits in the GPU memory")
        mb = res["micro_batch"]
        model.eval()
        with torch.no_grad():
            x = normalize(torch.randint(0, 256, (mb, 1, h, w), dtype=torch.uint8), dev)
            refs = normalize(torch.randint(0, 256, (mb * rk, 1, h, w), dtype=torch.uint8), dev) if rk else None
            times = []
            for _ in range(4):
                ts = time.time()
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    postprocess(spec, model(x, **model_inputs(spec, refs=refs)), args.topk)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                times.append(time.time() - ts)
        res["s_per_image_inference"] = round(statistics.median(times[1:]) / mb, 4)
        _, train, held, test = fold_records(args, 0)
        micro_steps = args.epochs * (len(train) // mb)
        post_s = 0.05 if spec.has_mask else 0.0  # mask resize, metrics and PNG per femur (CPU)
        n_pass = len(getattr(args, "modes", ["base"]))  # the SLG-Net tests predict every femur again
        n_test = 0 if args.no_test else len(test)
        pred_s = (len(held) + n_test) * n_pass * (res["s_per_image_inference"] + post_s)
        # 1.10: data loading and logging overhead; +1 min to build the model and start the data loader
        res["est_fold_min"] = round((micro_steps * res["s_per_micro_step"] * 1.10 + pred_s) / 60 + 1, 1)
        res["ok"] = True
    except Exception as e:  # recorded, so that --check-nets can report every network
        res["error"] = f"{type(e).__name__}: {e}"
    if args.preflight_out:
        Path(args.preflight_out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


def _is_oom(e: BaseException) -> bool:
    text = str(e)
    return ("out of memory" in text.lower() or "ALLOC_FAILED" in text or "NOT_ENOUGH" in text
            or type(e).__name__ == "OutOfMemoryError")


def _timed_steps(model, loss_fn, opt, amp: bool, dev, mb: int, args) -> dict:
    """Pre-flight: a few full training steps on random images of micro-batch ``mb``; median time (after 2 warm-up
    steps) and peak GPU memory. In a function of its own, so that all its tensors are freed when it ends (also after
    an out-of-memory error)."""
    import torch

    h, w = args.size_hw
    spec = parse_net(args.net)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)  # fresh per try: a failed step can leave it half-way
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (mb, 1, h, w), dtype=torch.uint8, generator=gen)
    rk = ref_cfg(args, spec)["k"] if spec.kind == "ref" else 0
    refs = torch.randint(0, 256, (mb * rk, 1, h, w), dtype=torch.uint8, generator=gen) if rk else None
    y = torch.randint(0, 6, (mb,), generator=gen)
    m = torch.zeros(mb, h, w, dtype=torch.long)
    for b in range(mb):  # a fracture-sized blob for the fractured examples
        if y[b] > 0:
            m[b, h // 3: h // 3 + h // 8, w // 3: w // 3 + w // 6] = int(y[b])
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    model.train()
    times = []
    for _ in range(max(3, args.preflight_steps)):
        ts = time.time()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            out = model(normalize(x, dev), **model_inputs(spec, (m > 0).unsqueeze(1).float().to(dev), 0.5,
                                                          normalize(refs, dev) if refs is not None else None))
        loss, _ = loss_fn(out, m.to(dev), y.to(dev))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        times.append(time.time() - ts)
    peak = torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None
    return {"micro_batch": mb, "s_per_step": round(statistics.median(times[2:]), 4),
            "peak_gb": round(peak, 2) if peak is not None else None}


def preflight_subprocess(args, argv, net: str, gpu: str, cache: Path, tag: str) -> dict:
    out = args.work / "preflight" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    device = "cpu"
    if gpu != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = gpu
        device = "0"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--preflight", "--device", device,
           *argv_without(argv, ("--folds", "--gpus", "--check-nets")), "--net", net, "--cache", str(cache),
           "--preflight-out", str(out), "--folds", "0"]
    log = args.work / "logs" / f"preflight_{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    if out.exists():
        res = json.loads(out.read_text())
    else:
        res = {"net": net, "ok": False, "error": f"pre-flight process exited with code {rc}; see {log}"}
    res["log_tail"] = log.read_text(errors="replace").splitlines()[-15:] if not res.get("ok") else []
    return res


def check_nets(args, argv, t0: float, cache: Path, cache_s: float) -> int:
    """Run S00_net_check: the pre-flight of every network in --check-nets, two at a time on two GPUs."""
    nets = [n.strip() for n in args.check_nets.split(",") if n.strip()]
    results: dict[str, dict] = {}
    lock = threading.Lock()
    todo = list(nets)

    def worker(gpu: str) -> None:
        while True:
            with lock:
                if not todo:
                    return
                net = todo.pop(0)
            tag = net.replace(":", "_").replace(".", "_")
            print(f"[check] {net} on {'GPU ' + gpu if gpu != 'cpu' else 'CPU'}", flush=True)
            r = preflight_subprocess(args, argv, net, gpu, cache, tag)
            print(f"[check] {net}: {'ok' if r.get('ok') else 'FAILED ' + str(r.get('error'))}", flush=True)
            with lock:
                results[net] = r

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpu_list]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rounds = math.ceil(5 / max(1, len(args.gpu_list)))
    rep = {"run": args.out.name, "nets": [results[n] for n in nets], "cache_build_seconds": round(cache_s, 1),
           "size_hw": list(args.size_hw), "epochs": args.epochs, "batch": args.batch, "folds_per_gpu_round": rounds,
           "minutes_total": round((time.time() - t0) / 60, 1), **versions()}
    (args.out / "net_check.json").write_text(json.dumps(rep, indent=2))
    lines = [f"# {args.out.name}: network pre-flight", "",
             f"Input {args.size_hw[0]} x {args.size_hw[1]}, effective batch {args.batch}, {args.epochs} epochs. "
             f"Cache built in {cache_s:.0f} s. Estimated run time = {rounds} rounds of folds (5 folds, "
             f"{len(args.gpu_list)} at a time) x minutes per fold.", "",
             "| Network | OK | Params (M) | Build + download (s) | Micro-batch x accumulation | Peak GPU memory (GB) "
             "| s per micro-step | Est. min per fold | Est. min per run |", "|---|---|---|---|---|---|---|---|---|"]
    for n in nets:
        r = results[n]
        if r.get("ok"):
            lines.append(f"| `{n}` | yes | {r['params_m']} | {r['build_seconds']} | {r['micro_batch']} x {r['accumulation']} "
                         f"| {r.get('peak_gb')} | {r['s_per_micro_step']} | {r['est_fold_min']} | "
                         f"{round(rounds * r['est_fold_min'])} |")
        else:
            lines.append(f"| `{n}` | NO: {r.get('error', '')[:120]} | | | | | | | |")
    (args.out / "net_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(r.get("ok") for r in results.values()) else 1


# ----------------------------------------------------------------------------------------------- evaluation


def ensemble_rows(test_by_model: dict[int, list[dict]], cache: Cache, map_root: Path | None) -> list[dict]:
    """5-model ensemble on the test set: mean class probabilities and fracture score; mask = mean of the models'
    fracture-probability maps >= 0.5 (segmenters and M0)."""
    import cv2

    models = sorted(test_by_model)
    by = {m: {r["file"]: r for r in test_by_model[m]} for m in models}
    full_h, full_w = cache.full_hw
    rows = []
    for f in [r["file"] for r in test_by_model[models[0]]]:
        rs = [by[m][f] for m in models]
        conf = np.array([[r[f"conf_{c}"] for c in range(1, 6)] for r in rs]).mean(axis=0)
        score = float(np.mean([r["score"] for r in rs]))
        row = {"file": f, "split": "test", "fold": -1, "model": "ensemble", "cls6_true": rs[0]["cls6_true"],
               "score": score, "top_cls6": int(np.argmax(conf)) + 1, "top_conf": score}
        row.update({f"conf_{c + 1}": float(conf[c]) for c in range(5)})
        idx = cache.row[("test", f)]
        gt = cache.gt_full(idx)
        row["area_gt"] = int(gt.sum())
        if map_root is not None:
            maps = [read_png(map_root / f"fold{m}" / f).astype(np.float32) / 255 for m in models
                    if (map_root / f"fold{m}" / f).exists()]
            if maps:
                mean = np.mean(maps, axis=0)
                mask = cv2.resize(mean, (full_w, full_h), interpolation=cv2.INTER_LINEAR) >= 0.5
                row["area_top"] = int(mask.sum())
                if gt.any():
                    row.update(seg_case(mask, gt, nd=None))
        rows.append(row)
    return rows


def test_effects(base: list[dict], other: list[dict], tau: float, fractured: bool = True) -> dict:
    """How a test mode changes the decisions on the fractured femurs (or, with fractured=False, the healthy ones;
    rows matched by file and model): mean change of the fracture probability and of the probability of the true
    class, the share of correctly classified femurs that become wrong, and the share of wrong ones that become
    right. (The keys keep the name n_fractured for either group.)"""
    def key(r):
        return r["file"], str(r["model"])

    def correct(r):
        return (r["top_cls6"] if r["top_conf"] >= tau else 0) == r["cls6_true"]

    ob = {key(r): r for r in other}
    fx = [r for r in base if (r["cls6_true"] > 0) == fractured and key(r) in ob]
    ok, bad = [r for r in fx if correct(r)], [r for r in fx if not correct(r)]

    def mean(v):
        return float(np.mean(v)) if len(v) else float("nan")

    def p_true(r):  # probability of the true class; for a healthy femur 1 - P(fracture)
        return r[f"conf_{r['cls6_true']}"] if r["cls6_true"] > 0 else 1.0 - r["score"]

    return {"n_fractured": len(fx), "n_correct_base": len(ok),
            "mean_change_fracture_prob": mean([ob[key(r)]["score"] - r["score"] for r in fx]),
            "mean_change_true_class_prob": mean([p_true(ob[key(r)]) - p_true(r) for r in fx]),
            "share_correct_lost": mean([not correct(ob[key(r)]) for r in ok]),
            "share_wrong_fixed": mean([correct(ob[key(r)]) for r in bad])}


def aggregate(args, cache: Cache, done: list[int], skipped: list[int], failed: list[int], t0: float, pf: dict) -> dict:
    spec = parse_net(args.net)
    oof, test_by_model = [], {}
    modes = list(getattr(args, "modes", ["base"]))[1:]
    oof_m: dict[str, list[dict]] = {m: [] for m in modes}
    test_m: dict[str, dict[int, list[dict]]] = {m: {} for m in modes}
    for k in sorted(done):
        fd = args.out / "folds" / f"fold{k}"
        oof += read_csv(fd / "oof.csv")
        if (fd / "test.csv").exists():
            test_by_model[k] = read_csv(fd / "test.csv")
        for m in modes:
            oof_m[m] += read_csv(fd / f"oof_{m}.csv")
            if (fd / f"test_{m}.csv").exists():
                test_m[m][k] = read_csv(fd / f"test_{m}.csv")
    gpus = sorted({json.loads((args.out / "folds" / f"fold{k}" / "fold_meta.json").read_text())["gpu"] for k in done})
    n_dev = len({g for g in args.gpu_list if g != "cpu"})
    res: dict = {"run": args.out.name, "net": spec.text, "gpu": ", ".join(gpus) + (f" x{n_dev}" if n_dev > 1 else ""),
                 "folds_requested": args.fold_list, "folds_completed": sorted(done),
                 "folds_skipped_time_guard": sorted(skipped), "folds_failed": sorted(failed),
                 "minutes_total": round((time.time() - t0) / 60, 1),
                 "micro_batch": pf.get("micro_batch"), "accumulation": pf.get("accumulation")}
    if not oof:
        return res
    tau_oof = M.youden_threshold([r["cls6_true"] > 0 for r in oof], [r["score"] for r in oof])
    res["thresholds"] = {"oof_youden": tau_oof, "plain": PLAIN_TAU}
    maps = None if (args.no_maps or not spec.has_mask) else args.work / "prob_maps" / "test"
    ens = ensemble_rows(test_by_model, cache, maps) if len(test_by_model) >= 2 else []
    for name, tau in (("", tau_oof), (f"at_threshold_{PLAIN_TAU}", PLAIN_TAU)):
        block: dict = {"oof": evaluate(oof, tau),
                       "oof_per_fold": {str(k): {kk: v for kk, v in evaluate([r for r in oof if r["fold"] == k], tau).items()
                                                 if isinstance(v, (int, float))} for k in sorted(done)}}
        if test_by_model:
            per_model = [evaluate(test_by_model[k], tau) for k in sorted(test_by_model)]
            block["test_models_mean"], block["test_models_sd"] = scalar_summary(per_model)
            if ens:
                block["test_ensemble"] = evaluate(ens, tau)
        if name == "":
            res.update(block)  # headline numbers (threshold fitted on OOF) at the top level
        else:
            res[name] = block
    write_csv(args.out / "per_case_oof.csv", oof)
    write_csv(args.out / "per_case_test.csv", [r for k in sorted(test_by_model) for r in test_by_model[k]])
    if ens:
        write_csv(args.out / "per_case_test_ensemble.csv", ens)
    if modes:  # tests, at the run's own OOF threshold (SLG-Net: Analysis/06 §5; RefFrac: Analysis/08 §2.6)
        res["tests"] = {}
        for m in modes:
            blk: dict = {"oof": evaluate(oof_m[m], tau_oof), "effects_oof": test_effects(oof, oof_m[m], tau_oof)}
            if spec.kind == "ref":  # a reference can also make a healthy femur look fractured
                blk["effects_healthy_oof"] = test_effects(oof, oof_m[m], tau_oof, fractured=False)
            write_csv(args.out / f"per_case_oof_{m}.csv", oof_m[m])
            if test_m[m]:
                write_csv(args.out / f"per_case_test_{m}.csv", [r for k in sorted(test_m[m]) for r in test_m[m][k]])
                if len(test_m[m]) >= 2 and ens:
                    # SLG-Net: classification only (no maps in its test modes); RefFrac saves maps in every mode
                    mode_maps = maps.parent / f"test_{m}" if (spec.kind == "ref" and maps is not None) else None
                    ens_m = ensemble_rows(test_m[m], cache, mode_maps)
                    blk["test_ensemble"] = evaluate(ens_m, tau_oof)
                    blk["effects_test_ensemble"] = test_effects(ens, ens_m, tau_oof)
                    if spec.kind == "ref":
                        blk["effects_healthy_test_ensemble"] = test_effects(ens, ens_m, tau_oof, fractured=False)
                    write_csv(args.out / f"per_case_test_ensemble_{m}.csv", ens_m)
            res["tests"][m] = blk
        if "deletion_control" in res["tests"]:
            placed = Counter()
            for k in sorted(done):
                f = args.out / "folds" / f"fold{k}" / "oof_deletion_info.csv"
                if f.exists():
                    with open(f, newline="") as fh:
                        placed.update(r["control"] for r in csv.DictReader(fh))
            res["tests"]["deletion_control"]["control_placement_oof"] = dict(placed)
    if spec.kind == "ref":
        res["reference"] = reference_summary(args, done)
    return res


def reference_summary(args, done: list[int]) -> dict:
    """RefFrac: per_case_<oof|test>_references.csv (all folds), the pointing game of the difference map (base mode,
    fractured femurs: does the 1/16 cell with the largest |T - R~| touch the released mask?) and which references the
    femurs got in each mode."""
    out: dict = {}
    for part in ("oof", "test"):
        rows: list[dict] = []
        for k in sorted(done):
            f = args.out / "folds" / f"fold{k}" / f"{part}_references.csv"
            if f.exists():
                with open(f, newline="") as fh:
                    rows += list(csv.DictReader(fh))
        if not rows:
            continue
        write_ref_csv(args.out / f"per_case_{part}_references.csv", rows)
        pg = [r for r in rows if r["mode"] == "base" and r["diff_hit"] != ""]
        out[part] = {"pointing_game": {"n": len(pg),
                                       "hit_rate": float(np.mean([int(r["diff_hit"]) for r in pg])) if pg else float("nan"),
                                       "chance": float(np.mean([float(r["diff_chance"]) for r in pg])) if pg else float("nan")},
                     "sources": dict(sorted(Counter(f"{r['mode']}: {r['source']}" for r in rows).items()))}
    return out


def write_summary(args, res: dict) -> None:
    """summary.md: headline numbers at both thresholds, next to the article's numbers."""
    spec = parse_net(args.net)

    def f(src, block, key, scale=1.0, nd=3):
        v = (src.get(block) or {}).get(key)
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * scale:.{nd}f}"

    base = spec.arch.split(".")[0]
    t3 = REF.TABLE3.get(PAPER_TABLE3.get(base, ""), None) if spec.kind == "cls" else None
    if t3:
        ref_title = f"Paper, {PAPER_TABLE3[base]} (test, Table 3)"
        paper = (f"{t3['auc']:.3f}", f"{100 * t3['accuracy_6class']:.1f}", "")
    else:
        ref_title = "Paper, final model YOLOv8s-seg @1024 (test)"
        paper = (f"{REF.FINAL['auc']:.3f}", f"{100 * REF.FINAL['accuracy_6class']:.1f}", f"{REF.FINAL['dice']:.2f}")
    rows = [("Fracture AUC", "auc_fracture", 1, 3, paper[0]),
            ("Fracture sensitivity", "fracture_sensitivity", 1, 3, ""),
            ("Fracture specificity", "fracture_specificity", 1, 3, ""),
            ("6-class accuracy (%)", "accuracy_6class", 100, 1, paper[1]),
            ("6-class macro-F1", "macro_f1", 1, 3, ""),
            ("Garden I-II recall", "garden12_recall", 1, 3, "")]
    if spec.has_mask:
        rows += [("Dice, fractured femurs (missed = 0)", "dice_fractured_mean", 1, 3, ""),
                 ("Dice, detected fractures only", "dice_detected_mean", 1, 3, paper[2]),
                 ("NSD at 10 px, fractured femurs", "nsd10_fractured_mean", 1, 3, "")]
    tau = (res.get("thresholds") or {}).get("oof_youden", float("nan"))
    n_oof = (res.get("oof") or {}).get("n", 0)
    has_test = "test_ensemble" in res  # --no-test (development runs): out-of-fold columns only
    slg_text = "" if spec.kind != "slg" else (
        f" SLG-Net {spec.variant}: "
        + (f"teacher forcing over {args.tf_epochs:g} epochs" if args.tf_epochs > 0 else "no teacher forcing")
        + ", region pooling at 1/"
        + " and 1/".join(str(v) for v in args.slg_scale_list)
        + (", mask detached from the decision losses" if args.slg_detach else "")
        + (f", logit adjustment tau {args.la_tau:g}" if spec.variant == "A6" else "") + ".")
    if spec.kind == "ref":
        rc = ref_cfg(args, spec)
        slg_text = (f" RefFrac {spec.variant}: {rc['k']} reference(s) per femur ({rc['policy']}), alignment "
                    f"{rc['align']}" + (f" (window +-{rc['window']:g} of the image height)" if rc["align"] == "attention"
                                        and rc["window"] > 0 else (" (global)" if rc["align"] == "attention" else ""))
                    + f", fusion {rc['fusion']}" + (f", margin loss {rc['margin']:g}" if rc["margin"] > 0 else "")
                    + "; references encoded by the shared encoder without gradient.")
    lines = [f"# {res['run']}: summary", "",
             f"Network `{spec.text}`, input {args.size_hw[0]} x {args.size_hw[1]}, {args.epochs} epochs, batch "
             f"{args.batch} (micro-batch {res.get('micro_batch')} x {res.get('accumulation')}), lr {args.lr}, "
             f"final weights.{slg_text}",
             f"Folds completed {res['folds_completed']}, skipped {res['folds_skipped_time_guard']}, "
             f"failed {res['folds_failed']}. Runtime {res['minutes_total']} min on {res.get('gpu') or 'n/a'}."]
    for title, src in ((f"Fracture threshold fitted on the OOF predictions (Youden): {tau:.4f}", res),
                       (f"Plain threshold {PLAIN_TAU} on the fracture probability", res.get(f"at_threshold_{PLAIN_TAU}") or {})):
        if not has_test:
            lines += ["", f"## {title}", "", f"| Metric | OOF (n={n_oof:,}) |", "|---|---|"]
            lines += [f"| {label} | {f(src, 'oof', key, scale, nd)} |" for label, key, scale, nd, _ in rows]
            continue
        lines += ["", f"## {title}", "",
                  f"| Metric | OOF (n={n_oof:,}) | Test, fold models mean ± SD | Test, ensemble | {ref_title} |",
                  "|---|---|---|---|---|"]
        for label, key, scale, nd, p in rows:
            lines.append(f"| {label} | {f(src, 'oof', key, scale, nd)} | {f(src, 'test_models_mean', key, scale, nd)} ± "
                         f"{f(src, 'test_models_sd', key, scale, nd)} | {f(src, 'test_ensemble', key, scale, nd)} | {p} |")
    if res.get("tests") and spec.kind == "ref":
        test_col = (lambda blk: f" {f(blk, 'test_ensemble', 'dice_fractured_mean')} |") if has_test else (lambda blk: "")
        lines += ["", "## Reference tests (RefFrac, Analysis/08 §2.6), at the OOF threshold", "",
                  "| Mode | OOF Dice, fractured | OOF macro-F1 | OOF Garden I-II recall | Change of P(fracture), "
                  "fractured | Change of P(fracture), healthy | Healthy femurs that become fractured |"
                  + (" Test ensemble Dice, fractured |" if has_test else ""),
                  "|---|---|---|---|---|---|---|" + ("---|" if has_test else ""),
                  f"| as used | {f(res, 'oof', 'dice_fractured_mean')} | {f(res, 'oof', 'macro_f1')} | "
                  f"{f(res, 'oof', 'garden12_recall')} | | | |" + test_col(res)]
        for m, blk in res["tests"].items():
            e = blk.get("effects_healthy_oof") or {}
            lines.append(f"| {m} | {f(blk, 'oof', 'dice_fractured_mean')} | {f(blk, 'oof', 'macro_f1')} "
                         f"| {f(blk, 'oof', 'garden12_recall')} | {f(blk, 'effects_oof', 'mean_change_fracture_prob')} "
                         f"| {f(blk, 'effects_healthy_oof', 'mean_change_fracture_prob')} "
                         f"| {f(blk, 'effects_healthy_oof', 'share_correct_lost', 100, 1)}% of "
                         f"{e.get('n_correct_base', 'n/a')} |" + test_col(blk))
    elif res.get("tests"):
        test_col = (lambda blk: f" {f(blk, 'test_ensemble', 'accuracy_6class', 100, 1)} |") if has_test else (lambda blk: "")
        lines += ["", "## Interpretability tests (" + ("SLG-Net, Analysis/06 §5" if spec.kind == "slg" else
                                                    "M0, Analysis/06 §13") + "), at the OOF threshold", "",
                  "| Mode | OOF 6-class accuracy (%) | OOF macro-F1 | OOF Garden I-II recall | Change of P(true class), "
                  "fractured | Correct fractured femurs that become wrong |" + (" Test ensemble accuracy (%) |" if has_test else ""),
                  "|---|---|---|---|---|---|" + ("---|" if has_test else ""),
                  f"| as used | {f(res, 'oof', 'accuracy_6class', 100, 1)} | {f(res, 'oof', 'macro_f1')} | "
                  f"{f(res, 'oof', 'garden12_recall')} | | |" + test_col(res)]
        for m, blk in res["tests"].items():
            e = blk.get("effects_oof") or {}
            lines.append(f"| {m.replace('_', ' ')} | {f(blk, 'oof', 'accuracy_6class', 100, 1)} | {f(blk, 'oof', 'macro_f1')} "
                         f"| {f(blk, 'oof', 'garden12_recall')} | {f(blk, 'effects_oof', 'mean_change_true_class_prob')} "
                         f"| {f(blk, 'effects_oof', 'share_correct_lost', 100, 1)}% of {e.get('n_correct_base', 'n/a')} |"
                         + test_col(blk))
    pgs = [(part, (blk or {}).get("pointing_game") or {}) for part, blk in (res.get("reference") or {}).items()]
    if pgs:
        lines += ["", "Pointing game of the difference map |T - R~| at 1/16 (base mode, fractured femurs: the cell "
                  "with the largest difference touches the released mask): "
                  + "; ".join(f"{'OOF' if part == 'oof' else 'test (fold models)'} {pg.get('hit_rate', float('nan')):.3f} "
                              f"(chance {pg.get('chance', float('nan')):.3f}, n = {pg.get('n')})" for part, pg in pgs)
                  + "."]
    if has_test:
        lines += ["", "Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected "
                  "fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours "
                  "on our 5 folds (one model per fold, final weights)."]
    else:
        lines += ["", "Development run (`--no-test`): the held-out folds only; the test set was not predicted."]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------- controller


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker:
        run_fold(args)
        return 0
    if args.preflight:
        return 0 if run_preflight(args).get("ok") else 1
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    n_par = max(1, len(args.gpu_list))
    if args.workers <= 0:
        args.workers = max(1, min(8, (os.cpu_count() or 2) // n_par))
    recs = load_records(args.folds_csv)
    tc = time.time()
    cache = build_cache(args.data_root, recs, args.size_hw, args.cache_dir)
    cache_s = time.time() - tc
    print(f"image cache {cache} ready in {cache_s:.0f} s", flush=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(git_commit=git_commit(REPO), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  cache=str(cache), cache_seconds=round(cache_s, 1), **versions(),
                  protocol="final weights, fixed epochs, held-out fold unused in training, threshold fitted on OOF (Youden)")
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    if args.check_nets:
        return check_nets(args, argv, t0, cache, cache_s)

    pf = preflight_subprocess(args, argv, args.net, args.gpu_list[0], cache, "preflight")
    (args.out / "preflight.json").write_text(json.dumps(pf, indent=2))
    if not pf.get("ok"):
        print(f"pre-flight failed: {pf.get('error')}\n" + "\n".join(pf.get("log_tail", [])), flush=True)
        (args.out / "metrics.json").write_text(json.dumps({"run": args.out.name, "net": args.net,
                                                          "preflight_error": pf.get("error")}, indent=2))
        return 1
    est = args.est_fold_min or pf["est_fold_min"]
    print(f"pre-flight: micro-batch {pf['micro_batch']} x {pf['accumulation']}, {pf['s_per_micro_step']} s per "
          f"micro-step, peak {pf.get('peak_gb')} GB; about {est} min per fold", flush=True)
    worker_args = [*argv_without(argv), "--cache", str(cache), "--micro-batch", str(pf["micro_batch"]),
                   "--workers", str(args.workers)]
    done, skipped, failed = run_folds(Path(__file__).resolve(), worker_args, args.fold_list, args.gpu_list, args.out,
                                      args.work, t0, args.time_budget_h, est)
    res = aggregate(args, Cache(cache), done, skipped, failed, t0, pf)
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    if res.get("oof"):
        write_summary(args, res)
        print((args.out / "summary.md").read_text())
    print(f"folds done {sorted(done)}, skipped {sorted(skipped)}, failed {sorted(failed)}; "
          f"total {(time.time() - t0) / 60:.1f} min")
    return 0 if not failed and not skipped else (1 if failed else 3)


if __name__ == "__main__":
    sys.exit(main())
