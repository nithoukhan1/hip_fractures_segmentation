#!/usr/bin/env python3
"""FracAtlas external test set: one 1024 x 1536 crop per hip, made like a HUSJA-UA image (Analysis/11 section 8).

Input: the FracAtlas folder (read only) and results/fracatlas/landmarks.csv (one row per included hip: the pelvic
midline x of the image, and the centre of the femoral head, as fractions of the image width and height, read by eye
without the fracture outlines).

The crop (section 8.2), the same for every hip:
  * width W = |midline x - head x| / MEDIAL_SHARE: in the 20 HUSJA-UA calibration images the femoral head lies
    X_LATERAL = 0.57 of the width from the lateral edge, so the medial edge (0.43 of the width from the head) falls
    at the pelvic midline, as in a half of a pelvis radiograph;
  * height 1.5 W (portrait 2:3), the head at Y_HEAD = 0.33 of the height from the top;
  * the original orientation is kept (HUSJA-UA images are not mirrored); outside the X-ray is black;
  * resized to 1024 x 1536, then windowed between the 2nd and 98th intensity percentiles of the X-ray pixels.
The FracAtlas polygons go through the same transform (the released mask of each hip).

Label of a hip: fractured if the centroid of a FracAtlas polygon lies on its side of the midline and inside its crop;
excluded if a polygon on its side lies only below the crop (a femoral shaft fracture); healthy otherwise.

Writes to --out: images/<hip>.png, masks/<hip>.png (0 or 255), cases.csv, and check_sheet_<n>.jpg (every crop with
its outline in red, for a visual check before any model sees the crops).

    python scripts/make_fracatlas_crops.py --fracatlas ".../FracAtlas" --out ".../FracAtlas_crops"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
X_LATERAL, Y_HEAD = 0.57, 0.33          # where the femoral head lies in a HUSJA-UA image (calibration, section 8.2)
MEDIAL_SHARE = 1.0 - X_LATERAL          # head to medial edge = head to pelvic midline
OUT_W, OUT_H = 1024, 1536


def crop_box(w: int, h: int, xm: float, hx: float, hy: float, side: str) -> tuple[float, float, float, float]:
    """(x0, y0, W, H) of a hip's crop in pixels of the original image."""
    bw = abs(xm - hx) * w / MEDIAL_SHARE
    bh = 1.5 * bw
    cx, cy = hx * w, hy * h
    x0 = cx - X_LATERAL * bw if side == "R" else cx - MEDIAL_SHARE * bw   # R hip: lateral edge on the image's left
    return x0, cy - Y_HEAD * bh, bw, bh


def warp(img: np.ndarray, box, interp=cv2.INTER_AREA) -> np.ndarray:
    x0, y0, bw, bh = box
    s = OUT_W / bw
    m = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    return cv2.warpAffine(img, m, (OUT_W, OUT_H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def window(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v = img[valid]
    if v.size == 0:
        return img
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    out = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    out[~valid] = 0
    return out.round().astype(np.uint8)


def polygons(coco: dict) -> dict[str, list[np.ndarray]]:
    by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    out: dict[str, list[np.ndarray]] = {}
    for a in coco["annotations"]:
        seg = a["segmentation"]
        seg = json.loads(seg) if isinstance(seg, str) else seg
        for poly in seg:
            out.setdefault(by_id[a["image_id"]], []).append(np.asarray(poly, np.float32).reshape(-1, 2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fracatlas", required=True, help="the FracAtlas folder (dataset.csv, images/, Annotations/)")
    ap.add_argument("--landmarks", default=str(REPO / "results" / "fracatlas" / "landmarks.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fa, out = Path(a.fracatlas), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    paths = {f: Path(d) / f for d, _, fs in os.walk(fa / "images") for f in fs}
    polys = polygons(json.loads((fa / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json").read_text()))
    rows = list(csv.DictReader(open(a.landmarks, newline="")))
    cases, thumbs = [], []
    for r in rows:
        img = cv2.imread(str(paths[r["image_id"]]), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        xm, hx, hy, side = float(r["midline_x"]), float(r["head_x"]), float(r["head_y"]), r["side"]
        box = crop_box(w, h, xm, hx, hy, side)
        x0, y0, bw, bh = box
        hip = f"{Path(r['image_id']).stem}_{side}"
        mask_full = np.zeros((h, w), np.uint8)
        n_in, n_below = 0, 0
        for p in polys.get(r["image_id"], []):
            c = p.mean(axis=0)
            on_side = (c[0] < xm * w) if side == "R" else (c[0] > xm * w)
            if not on_side:
                continue
            if x0 <= c[0] <= x0 + bw and y0 <= c[1] <= y0 + bh:
                cv2.fillPoly(mask_full, [p.round().astype(np.int32)], 1)
                n_in += 1
            elif c[1] > y0 + bh:
                n_below += 1
        label = "fractured" if n_in else ("excluded_shaft" if n_below else "healthy")
        valid = warp(np.ones_like(img) * 255, box, cv2.INTER_NEAREST) > 0
        crop = window(warp(img, box), valid)
        mask = (warp(mask_full.astype(np.float32), box, cv2.INTER_LINEAR) >= 0.5).astype(np.uint8) * 255
        if label != "excluded_shaft":
            cv2.imwrite(str(out / "images" / f"{hip}.png"), crop)
            cv2.imwrite(str(out / "masks" / f"{hip}.png"), mask)
        cases.append({"hip": hip, "image_id": r["image_id"], "side": side, "label": label,
                      "group": r["order_group"], "polygons": n_in, "mask_pixels": int((mask > 0).sum()),
                      "image_w": w, "image_h": h, "crop_x0": round(x0, 1), "crop_y0": round(y0, 1),
                      "crop_w": round(bw, 1), "crop_h": round(bh, 1), "scale": round(OUT_W / bw, 3),
                      "share_outside_xray": round(float(1 - valid.mean()), 3)})
        t = cv2.cvtColor(cv2.resize(crop, (160, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(cv2.resize(mask, (160, 240), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(t, cnts, -1, (0, 0, 255), 1)
        cv2.putText(t, f"{hip[3:]} {label[:4]}", (2, 234), 0, 0.38, (0, 255, 255), 1)
        thumbs.append(t)
    with open(out / "cases.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(cases[0]))
        wr.writeheader()
        wr.writerows(cases)
    per = 50
    for k in range(0, len(thumbs), per):
        ts = thumbs[k:k + per] + [np.zeros_like(thumbs[0])] * (-len(thumbs[k:k + per]) % 10)
        sheet = np.vstack([np.hstack(ts[j:j + 10]) for j in range(0, len(ts), 10)])
        cv2.imwrite(str(out / f"check_sheet_{k // per + 1}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    n = {lab: sum(c["label"] == lab for c in cases) for lab in ("fractured", "healthy", "excluded_shaft")}
    print(f"{len(cases)} hips: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
