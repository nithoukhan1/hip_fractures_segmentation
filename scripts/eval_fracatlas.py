#!/usr/bin/env python3
"""Zero-shot test of trained models on the FracAtlas external test set (Analysis/11 section 8).

Every model is used exactly as it was trained on HUSJA-UA: its 5 fold models, their ensemble (as on the HUSJA-UA
test set), and its own HUSJA-UA out-of-fold fracture threshold. Nothing is fitted on FracAtlas.

Input
  * the FracAtlas crops made by scripts/make_fracatlas_crops.py: <crops>/cases.csv, images/<hip>.png and
    masks/<hip>.png (1024 x 1536, as a HUSJA-UA image; mask 0 or 255);
  * the weights of each run: <weights>/<RUN>/fold0.pt ... fold4.pt (for a YOLO run, the fold's final weights last.pt
    saved as fold<k>.pt);
  * each run's filed folder in this repository (results/<RUN>/config.json and metrics.json): the network, its input
    size and its HUSJA-UA threshold.
  On Kaggle, --crops and --weights default to the folders under /kaggle/input that hold cases.csv and <RUN>/fold0.pt.

Per hip (one row of per_hip.csv per run)
  * score = the ensemble's fracture score (mean of the 5 fold models), called fractured if score >= the threshold;
  * the ensemble mask (mean of the fold models' fracture-probability maps >= 0.5; YOLO: majority vote of the masks
    of the fold models that call a fracture, as on the HUSJA-UA test set), and against the released FracAtlas
    outline: Dice, IoU and HD95 (pixels of the 1024 x 1536 crop);
  * pointing game: whether the peak of the mean probability map lies inside the released outline (YOLO: the centre
    of the top detection's mask). Classifiers have no map.
FracAtlas has no fracture type, so only fracture yes/no and the mask are scored.

Metrics (metrics.json, and summary.md over all runs; the standard set of Analysis/11 section 2.2)
  accuracy, sensitivity, specificity, precision and F1 of "fractured"; fracture AUC (DeLong interval); Dice and IoU
  of the fractured hips where a missed fracture counts 0; HD95 median over the fractures called; pointing game.
  95% intervals: bootstrap over patient groups (2,000 resamples; images with file numbers within 2 of each other
  count as one patient, see patient_groups) for accuracy, sensitivity, specificity and Dice; DeLong for the AUC.

    python scripts/eval_fracatlas.py S10_m0_convnext_unet S01_yolov8s_seg_1024 --crops FracAtlas_crops \\
        --weights fracatlas_weights --out results/fracatlas/zero_shot_check
    python scripts/eval_fracatlas.py --summarise results/fracatlas/zero_shot_check   # tables again, from the CSVs
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from femurseg import metrics as M  # noqa: E402
from femurseg import stats as S  # noqa: E402
from femurseg.data import read_png  # noqa: E402
from femurseg.evaluation import seg_case  # noqa: E402

N_FOLDS = 5
KAGGLE_INPUT = Path("/kaggle/input")
NO_INPUT_EXIT = 2


# ------------------------------------------------------------------------------------------------- inputs


def find_crops(root: Path) -> Path | None:
    """The folder under ``root`` holding cases.csv with images/ and masks/ beside it."""
    for p in sorted(root.rglob("cases.csv")):
        if (p.parent / "images").is_dir() and (p.parent / "masks").is_dir():
            return p.parent
    return None


def find_weights(root: Path, run: str) -> Path | None:
    """The folder <...>/<run>/ under ``root`` that holds fold0.pt."""
    for p in sorted(root.rglob(f"{run}/fold0.pt")):
        return p.parent
    return None


def read_cases(crops: Path) -> list[dict]:
    with open(crops / "cases.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] in ("fractured", "healthy")]
    return [r for r in rows if (crops / "images" / f"{r['hip']}.png").exists()]


def run_info(results: Path, run: str) -> dict:
    d = results / run
    cfg = json.loads((d / "config.json").read_text())
    tau = (json.loads((d / "metrics.json").read_text()).get("thresholds") or {}).get("oof_youden")
    if tau is None:
        raise SystemExit(f"{run}: metrics.json has no out-of-fold threshold")
    kind = "yolo" if "net" not in cfg else "net"
    return {"cfg": cfg, "tau": float(tau), "kind": kind}


# ---------------------------------------------------------------------------------------------- prediction


def peak_inside(prob: np.ndarray, gt: np.ndarray) -> int:
    """1 if the most probable pixel of ``prob`` (resized to the mask) lies inside the released outline ``gt``."""
    y, x = np.unravel_index(int(np.argmax(prob)), prob.shape)
    gy, gx = int(y * gt.shape[0] / prob.shape[0]), int(x * gt.shape[1] / prob.shape[1])
    return int(gt[min(gy, gt.shape[0] - 1), min(gx, gt.shape[1] - 1)])


def predict_net(info: dict, wdir: Path, crops: Path, cases: list[dict], device: str, batch: int,
                mask_dir: Path | None) -> list[dict]:
    """Ensemble of the 5 fold models of a scripts/train_net.py run on every hip."""
    import cv2
    import torch

    from femurseg.nets import build_net, parse_net, postprocess
    from femurseg.torchdata import normalize

    cfg = info["cfg"]
    spec = parse_net(cfg["net"])
    if spec.kind == "ref":
        raise SystemExit(f"{cfg['net']}: RefFrac needs HUSJA-UA reference femurs; it is not part of this test")
    h, w = cfg["size_hw"]
    dev = torch.device(device)
    amp = dev.type == "cuda" and not cfg.get("no_amp", False)
    opts = {"slg_scales": tuple(cfg.get("slg_scale_list") or (16, 32)), "slg_detach": bool(cfg.get("slg_detach")),
            "zoom_window": int(cfg.get("zoom_window") or 640), "zoom_input": int(cfg.get("zoom_input") or 512)}
    imgs = {}
    for c in cases:
        img = read_png(crops / "images" / f"{c['hip']}.png")
        imgs[c["hip"]] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if img.shape != (h, w) else img
    scores = {c["hip"]: [] for c in cases}
    maps = {c["hip"]: [] for c in cases}
    for k in range(N_FOLDS):
        model = build_net(spec, pretrained=False, drop_rate=float(cfg.get("drop_rate", 0.2)), **opts)
        state = torch.load(wdir / f"fold{k}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model = model.to(dev).to(memory_format=torch.channels_last).eval()
        with torch.no_grad():
            for i in range(0, len(cases), batch):
                chunk = cases[i:i + batch]
                x = normalize(torch.from_numpy(np.stack([imgs[c["hip"]] for c in chunk]))[:, None], dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    pp = postprocess(spec, model(x), int(cfg.get("topk", 500)))
                sc = pp["score"].float().cpu().numpy()
                fm = pp["fmap"].float().cpu().numpy() if pp["fmap"] is not None else None
                for j, c in enumerate(chunk):
                    scores[c["hip"]].append(float(sc[j]))
                    if fm is not None:
                        maps[c["hip"]].append(fm[j])
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    rows = []
    for c in cases:
        gt = read_png(crops / "masks" / f"{c['hip']}.png") > 0
        row = base_row(c, info["tau"], float(np.mean(scores[c["hip"]])), scores[c["hip"]])
        if maps[c["hip"]]:
            mean = np.mean(maps[c["hip"]], axis=0).astype(np.float32)
            mask = cv2.resize(mean, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR) >= 0.5
            row["area_pred"] = int(mask.sum())
            if mask_dir is not None:
                cv2.imwrite(str(mask_dir / f"{c['hip']}.png"), mask.astype(np.uint8) * 255)
            if gt.any():
                row.update(seg_case(mask, gt, nd=None), pointing_hit=peak_inside(mean, gt))
        rows.append(row)
    return rows


def predict_yolo(info: dict, wdir: Path, crops: Path, cases: list[dict], device: str,
                 mask_dir: Path | None) -> list[dict]:
    """Ensemble of the 5 fold models of a scripts/train_yolo_seg.py run, as its test-set ensemble: mean per-class
    confidence, score = its maximum; mask = majority vote of the fold models that call a fracture themselves."""
    import cv2
    from ultralytics import YOLO

    cfg, tau = info["cfg"], info["tau"]
    per = {c["hip"]: [] for c in cases}   # per fold: (per-class confidences, top mask or None)
    paths = [str(crops / "images" / f"{c['hip']}.png") for c in cases]
    for k in range(N_FOLDS):
        model = YOLO(str(wdir / f"fold{k}.pt"))
        prec = {"quantize": "fp16"} if device != "cpu" else {}  # fp16 on GPU, as scripts/train_yolo_seg.py
        for i in range(0, len(paths), 16):
            for res in model.predict(source=paths[i:i + 16], imgsz=int(cfg["imgsz"]), conf=float(cfg["conf_min"]),
                                     iou=0.7, max_det=int(cfg["max_det"]), retina_masks=True, device=device,
                                     verbose=False, stream=True, **prec):
                hip = Path(res.path).stem
                n = 0 if res.boxes is None else len(res.boxes)
                conf = res.boxes.conf.cpu().numpy() if n else np.zeros(0)
                cls = res.boxes.cls.cpu().numpy().astype(int) if n else np.zeros(0, int)
                pc = np.array([conf[cls == c].max() if (cls == c).any() else 0.0 for c in range(5)])
                top = None
                if n and res.masks is not None:
                    top = res.masks.data.cpu().numpy()[int(np.argmax(conf))] > 0.5
                per[hip].append((pc, top))
    rows = []
    for c in cases:
        gt = read_png(crops / "masks" / f"{c['hip']}.png") > 0
        fold = per[c["hip"]]
        conf = np.mean([pc for pc, _ in fold], axis=0)
        row = base_row(c, tau, float(conf.max()), [float(pc.max()) for pc, _ in fold])
        votes, n_vote = np.zeros(gt.shape, np.float32), 0
        if conf.max() >= tau:
            for pc, top in fold:
                if pc.max() >= tau and top is not None:
                    if top.shape != gt.shape:
                        top = cv2.resize(top.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                         interpolation=cv2.INTER_NEAREST) > 0
                    votes += top
                    n_vote += 1
        mask = votes >= max(n_vote, 1) / 2 if n_vote else np.zeros(gt.shape, bool)
        row["area_pred"] = int(mask.sum())
        if mask_dir is not None:
            cv2.imwrite(str(mask_dir / f"{c['hip']}.png"), mask.astype(np.uint8) * 255)
        if gt.any():
            row.update(seg_case(mask, gt, nd=None))
            if mask.any():
                ys, xs = np.nonzero(mask)
                row["pointing_hit"] = int(gt[int(round(ys.mean())), int(round(xs.mean()))])
            else:
                row["pointing_hit"] = 0
        rows.append(row)
    return rows


def base_row(case: dict, tau: float, score: float, fold_scores: list[float]) -> dict:
    row = {"hip": case["hip"], "image_id": case["image_id"], "side": case["side"], "label": case["label"],
           "y": int(case["label"] == "fractured"), "score": round(score, 8), "called": int(score >= tau),
           "threshold": tau, "area_gt": int(case.get("mask_pixels") or 0)}
    row.update({f"score_fold{k}": round(v, 8) for k, v in enumerate(fold_scores)})
    return row


# ------------------------------------------------------------------------------------------------- metrics


def patient_groups(image_ids: list[str], gap: int = 2) -> dict[str, str]:
    """Image -> patient group for the bootstrap. FracAtlas has no patient numbers, and second views of one patient
    have nearby file numbers (e.g. IMG0003766 and IMG0003767). Images whose numbers lie within ``gap`` of each other
    are chained into one group: this can merge two patients, which only widens the intervals."""
    def num(i: str) -> int:
        digits = "".join(ch for ch in i if ch.isdigit())
        return int(digits) if digits else -1

    ids = sorted(set(image_ids), key=num)
    out, cur, last = {}, None, None
    for i in ids:
        n = num(i)
        if cur is None or n < 0 or last < 0 or n - last > gap:
            cur = i
        out[i] = cur
        last = n
    return out


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def summarise_rows(rows: list[dict], n_boot: int = 2000, seed: int = 2026) -> dict:
    """The standard metrics of one run from its per-hip rows (fractured = positive)."""
    y = np.array([int(r["y"]) for r in rows])
    score = np.array([_f(r["score"]) for r in rows])
    pred = np.array([int(r["called"]) for r in rows])
    pg = patient_groups([r["image_id"] for r in rows])
    groups = np.array([pg[r["image_id"]] for r in rows])
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    sens = tp / (tp + fn) if tp + fn else float("nan")
    def boot(stat, *arrays) -> list[float]:
        b = S.bootstrap_ci(stat, *arrays, n_boot=n_boot, groups=groups, seed=seed)
        return [b["lo"], b["hi"]]

    def rate(yy, pp, cls):
        sel = yy == cls
        return float((pp[sel] == cls).mean()) if sel.any() else float("nan")

    out = {"n": len(rows), "n_fractured": int(y.sum()), "n_images": len({r["image_id"] for r in rows}),
           "n_patient_groups": len(set(groups)), "threshold": _f(rows[0]["threshold"]),
           "tp": tp, "fn": fn, "tn": tn, "fp": fp,
           "accuracy": (tp + tn) / len(rows), "accuracy_ci": boot(lambda a, b: float((a == b).mean()), y, pred),
           "sensitivity": sens, "sensitivity_ci": boot(lambda a, b: rate(a, b, 1), y, pred) if tp + fn else None,
           "specificity": tn / (tn + fp) if tn + fp else float("nan"),
           "specificity_ci": boot(lambda a, b: rate(a, b, 0), y, pred) if tn + fp else None,
           "precision": prec, "f1": 2 * prec * sens / (prec + sens) if prec == prec and prec + sens > 0 else float("nan")}
    if 0 < y.sum() < len(y):
        out["auc"] = S.delong_ci(y, score)
    fr = [r for r in rows if int(r["y"]) == 1]
    if fr and "dice_top" in fr[0]:
        called = np.array([int(r["called"]) for r in fr])
        dice = np.array([_f(r["dice_top"]) for r in fr]) * called      # a missed fracture counts 0
        iou = np.array([_f(r["iou_top"]) for r in fr]) * called
        g = np.array([pg[r["image_id"]] for r in fr])
        b = S.bootstrap_ci(lambda d: float(np.mean(d)), dice, n_boot=n_boot, groups=g, seed=seed)
        out["dice"] = {"estimate": b["estimate"], "lo": b["lo"], "hi": b["hi"]}
        out["iou"] = float(iou.mean())
        hd = [_f(r["hd95_top"]) for r, c in zip(fr, called) if c and np.isfinite(_f(r["hd95_top"]))]
        out["hd95_median"] = float(np.median(hd)) if hd else float("nan")
        out["hd95_iqr"] = [float(np.percentile(hd, 25)), float(np.percentile(hd, 75))] if hd else None
        hits = [int(r["pointing_hit"]) for r in fr if str(r.get("pointing_hit", "")) != ""]
        out["pointing_game"] = {"hits": int(sum(hits)), "n": len(hits)}
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def write_summary(out: Path, report: dict) -> None:
    def f3(v):
        return "–" if v is None or v != v else f"{v:.3f}"

    lines = ["# FracAtlas external test, zero-shot", "",
             "Every model as trained on HUSJA-UA: the ensemble of its 5 fold models and its own HUSJA-UA out-of-fold "
             "threshold; nothing fitted on FracAtlas (`scripts/eval_fracatlas.py`, Analysis/11 section 8). "
             "Fractured hips are the positive class. Dice and IoU: fractured hips, a missed fracture counts 0. HD95 "
             "in pixels of the 1024 × 1536 crop, over the fractures called. Pointing game: the peak of the "
             "probability map (YOLO: the centre of its mask) inside the released outline. 95% intervals: bootstrap "
             "over patient groups (images with file numbers within 2 of each other count as one patient) for "
             "accuracy, sensitivity, specificity and Dice; DeLong for the AUC.", "",
             "| Model | Hips (fractured) | Accuracy | Sensitivity | Specificity | Precision | F1 | Fracture AUC | Dice "
             "| IoU | HD95, px | Pointing game |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, m in report["runs"].items():
        sc, spc, auc, dice = m.get("sensitivity_ci"), m.get("specificity_ci"), m.get("auc"), m.get("dice")
        pg = m.get("pointing_game")
        lines.append(
            f"| {run} | {m['n']} ({m['n_fractured']}) | {f3(m['accuracy'])}"
            + (f" ({m['accuracy_ci'][0]:.2f}–{m['accuracy_ci'][1]:.2f})" if m.get("accuracy_ci") else "")
            + f" | {m['tp']}/{m['tp'] + m['fn']} = {f3(m['sensitivity'])}" + (f" ({sc[0]:.2f}–{sc[1]:.2f})" if sc else "")
            + f" | {m['tn']}/{m['tn'] + m['fp']} = {f3(m['specificity'])}" + (f" ({spc[0]:.2f}–{spc[1]:.2f})" if spc else "")
            + f" | {f3(m['precision'])} | {f3(m['f1'])} | "
            + (f"{auc['auc']:.3f} ({auc['lo']:.3f}–{min(1.0, auc['hi']):.3f})" if auc else "–")
            + " | " + (f"{dice['estimate']:.3f} ({dice['lo']:.3f}–{dice['hi']:.3f})" if dice else "–")
            + f" | {f3(m.get('iou'))} | "
            + (f"{m['hd95_median']:.0f}" if m.get("hd95_median") == m.get("hd95_median") and m.get("hd95_median") is not None else "–")
            + " | " + (f"{pg['hits']}/{pg['n']}" if pg else "–") + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def summarise_folder(out: Path, n_boot: int = 2000) -> dict:
    report = {"runs": {}}
    for csv_path in sorted(out.glob("*/per_hip.csv")):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        report["runs"][csv_path.parent.name] = summarise_rows(rows, n_boot)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    write_summary(out, report)
    return report


# ---------------------------------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="filed runs to test (results/<RUN>/)")
    ap.add_argument("--crops", default="", help="FracAtlas crop folder (default: found under /kaggle/input)")
    ap.add_argument("--weights", default="", help="folder holding <RUN>/fold<k>.pt (default: found under /kaggle/input)")
    ap.add_argument("--results", default=str(REPO / "results"), help="the repository's filed runs")
    ap.add_argument("--out", default="", help="output folder")
    ap.add_argument("--gpus", default="0", help="the first one is used; 'cpu' for CPU")
    ap.add_argument("--workers", type=int, default=2, help=argparse.SUPPRESS)   # accepted for kaggle/run.py
    ap.add_argument("--work", default="", help=argparse.SUPPRESS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-masks", action="store_true", help="do not save the ensemble masks")
    ap.add_argument("--summarise", default="", help="only rebuild metrics.json and summary.md of this output folder")
    a = ap.parse_args(argv)
    if a.summarise:
        summarise_folder(Path(a.summarise), a.n_boot)
        print(f"written: {a.summarise}/metrics.json and summary.md")
        return 0
    if not a.runs or not a.out:
        ap.error("give the runs and --out")
    crops = Path(a.crops) if a.crops else find_crops(KAGGLE_INPUT)
    if crops is None or not (crops / "cases.csv").exists():
        print("NO FRACATLAS CROPS FOUND: add the dataset with FracAtlas_crops (cases.csv, images/, masks/) as an "
              "input of the notebook (right panel -> Input -> Add Input).", flush=True)
        return NO_INPUT_EXIT
    wroot = Path(a.weights) if a.weights else KAGGLE_INPUT
    device = "cpu" if a.gpus.split(",")[0] in ("cpu", "") else f"cuda:{a.gpus.split(',')[0]}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = read_cases(crops)
    print(f"{len(cases)} hips ({sum(c['label'] == 'fractured' for c in cases)} fractured) from {crops}", flush=True)
    t0, done, failed = time.time(), [], {}
    for run in a.runs:
        wdir = find_weights(wroot, run)
        if wdir is None:
            failed[run] = f"no weights {run}/fold0.pt under {wroot}"
            print(f"{run}: {failed[run]}", flush=True)
            continue
        info = run_info(Path(a.results), run)
        rd = out / run
        rd.mkdir(parents=True, exist_ok=True)
        mdir = None if a.no_masks else rd / "masks"
        if mdir is not None:
            mdir.mkdir(exist_ok=True)
        t1 = time.time()
        try:
            rows = (predict_yolo(info, wdir, crops, cases, device.replace("cuda:", ""), mdir) if info["kind"] == "yolo"
                    else predict_net(info, wdir, crops, cases, device, a.batch, mdir))
        except SystemExit as e:
            failed[run] = str(e)
            print(f"{run}: {e}", flush=True)
            continue
        write_csv(rd / "per_hip.csv", rows)
        m = summarise_rows(rows, a.n_boot)
        (rd / "metrics.json").write_text(json.dumps({"run": run, "weights": str(wdir), **m}, indent=2, default=float))
        done.append(run)
        auc = m.get("auc", {}).get("auc", float("nan"))
        print(f"{run}: AUC {auc:.3f}, sensitivity {m['sensitivity']:.3f}, specificity {m['specificity']:.3f}, "
              f"Dice {m.get('dice', {}).get('estimate', float('nan')):.3f} ({time.time() - t1:.0f} s)", flush=True)
    report = summarise_folder(out, a.n_boot)
    report.update(done=done, failed=failed, minutes=round((time.time() - t0) / 60, 2), crops=str(crops),
                  n_hips=len(cases), device=device)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"written: {out}/metrics.json and summary.md; {len(done)} done, {len(failed)} failed")
    return 0 if done and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
