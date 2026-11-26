"""
EDA for the HUSJA-UA Femoral Fracture dataset (González et al., Insights Imaging 2025).

Usage:
    python eda_femoral.py --root "<path>/Femoral Fractures" --out "<path>/EDA"

Expected layout:  <root>/{train,val,test}/{images,labels,masks}
  images : 8-bit grayscale PNG femur ROIs
  masks  : 8-bit PNG, 0 = background, k = fracture class (k = YOLO class id + 1, verified below)
  labels : YOLO-seg txt (class x1 y1 x2 y2 ... normalised), only present for fractured femurs

Works locally or on Kaggle (e.g. --root /kaggle/input/datasets/utopianstar/dataset/femoral/data).
"""
import argparse, os, json, hashlib, collections
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = {  # YOLO class id -> name (mapping matches Table 1 counts of the paper exactly)
    0: "Neck Garden I-II",
    1: "Neck Garden III-IV",
    2: "Pertroch Evans I-II",
    3: "Pertroch Evans III-V",
    4: "Subtrochanteric",
}
NOFRAC = "No fracture"
ORDER = [NOFRAC] + [CLASSES[k] for k in range(5)]


def ahash(img, n=16):
    s = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
    return (s > s.mean()).flatten()


def parse_label(path):
    polys = []
    if not os.path.exists(path):
        return polys
    for line in open(path):
        p = line.split()
        if len(p) < 7:
            polys.append((int(p[0]) if p else -1, np.zeros((0, 2))))
            continue
        c = int(p[0]); xy = np.array(p[1:], dtype=float)
        if len(xy) % 2:
            xy = xy[:-1]
        polys.append((c, xy.reshape(-1, 2)))
    return polys


def analyse(root, splits):
    rows = []
    for s in splits:
        idir = os.path.join(root, s, "images")
        if not os.path.isdir(idir) or not os.listdir(idir):
            print(f"[warn] split '{s}' missing or empty")
            continue
        for f in sorted(os.listdir(idir)):
            stem = os.path.splitext(f)[0]
            img = cv2.imread(os.path.join(idir, f), cv2.IMREAD_UNCHANGED)
            mpath = os.path.join(root, s, "masks", f)
            mask = cv2.imread(mpath, cv2.IMREAD_UNCHANGED) if os.path.exists(mpath) else None
            polys = parse_label(os.path.join(root, s, "labels", stem + ".txt"))
            H, W = img.shape[:2]
            r = dict(split=s, file=f, femur_id=stem.split("_")[0],
                     xray_id=stem.split("_")[1] if "_" in stem else "",
                     H=H, W=W, dtype=str(img.dtype), channels=1 if img.ndim == 2 else img.shape[2])
            g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            r.update(mean=float(g.mean()), std=float(g.std()),
                     p02=float(np.percentile(g, 2)), p98=float(np.percentile(g, 98)),
                     frac_black=float((g == 0).mean()), frac_white=float((g == 255).mean()),
                     sharpness=float(cv2.Laplacian(g, cv2.CV_64F).var()),
                     md5=hashlib.md5(g.tobytes()).hexdigest())
            r["_ahash"] = ahash(g)
            # mask
            if mask is not None:
                vals = [int(v) for v in np.unique(mask) if v != 0]
                r["mask_values"] = ",".join(map(str, vals))
                fg = mask > 0
                r["mask_area_px"] = int(fg.sum())
                r["mask_area_frac"] = float(fg.mean())
                n, lab, st, cen = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
                r["mask_components"] = n - 1
                if fg.any():
                    ys, xs = np.nonzero(fg)
                    r.update(mask_cx=xs.mean() / W, mask_cy=ys.mean() / H,
                             mask_bw=(xs.max() - xs.min() + 1) / W, mask_bh=(ys.max() - ys.min() + 1) / H)
                    cnts, _ = cv2.findContours(fg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    per = sum(cv2.arcLength(c, True) for c in cnts)
                    r["mask_compactness"] = float(4 * np.pi * fg.sum() / (per ** 2 + 1e-9))
                    hull = cv2.convexHull(np.vstack(cnts))
                    r["mask_solidity"] = float(fg.sum() / (cv2.contourArea(hull) + 1e-9))
            # labels
            r["n_polys"] = len(polys)
            r["label_classes"] = ",".join(str(c) for c, _ in polys)
            r["poly_vertices"] = ",".join(str(len(p)) for _, p in polys)
            if polys:
                cls = collections.Counter(c for c, _ in polys).most_common(1)[0][0]
                r["class_id"] = cls
                r["class_name"] = CLASSES.get(cls, f"unk{cls}")
                ras = np.zeros((H, W), np.uint8)
                for _, p in polys:
                    if len(p) >= 3:
                        pts = np.round(p * [W, H]).astype(np.int32)
                        cv2.fillPoly(ras, [pts], 1)
                if mask is not None:
                    fg = mask > 0
                    inter = (fg & (ras > 0)).sum(); uni = (fg | (ras > 0)).sum()
                    r["poly_mask_iou"] = float(inter / uni) if uni else np.nan
                    r["mask_minus_cls"] = (int(np.median(mask[fg])) - cls) if fg.any() else np.nan
                allp = np.vstack([p for _, p in polys if len(p)])
                r["poly_out_of_bounds"] = bool(((allp < 0) | (allp > 1)).any())
            else:
                r["class_id"] = -1
                r["class_name"] = NOFRAC
            rows.append(r)
    return pd.DataFrame(rows)


def leakage(df):
    """exact (md5) and near (aHash Hamming<=8 on 16x16) duplicates across/within splits."""
    out = {}
    dup = df[df.duplicated("md5", keep=False)]
    out["exact_duplicate_groups"] = dup.groupby("md5")["file"].apply(list).tolist()
    H = np.stack(df["_ahash"].values).astype(np.uint8)
    d = (H[:, None, :] != H[None, :, :]).sum(-1)
    iu = np.triu_indices(len(df), 1)
    near = [(df.iloc[i]["split"] + "/" + df.iloc[i]["file"], df.iloc[j]["split"] + "/" + df.iloc[j]["file"], int(d[i, j]))
            for i, j in zip(*iu) if d[i, j] <= 8]
    out["near_duplicate_pairs_ahash_le8"] = near
    out["near_duplicate_cross_split"] = [p for p in near if p[0].split("/")[0] != p[1].split("/")[0]]
    xr = df.groupby("xray_id")["split"].nunique()
    out["xray_ids_in_multiple_splits"] = xr[xr > 1].index.tolist()
    return out


def figures(df, root, out):
    fd = os.path.join(out, "figures"); os.makedirs(fd, exist_ok=True)
    splits = df.split.unique().tolist()
    # 1 class distribution
    ct = pd.crosstab(df.class_name, df.split).reindex(ORDER).fillna(0)
    ax = ct.plot.barh(figsize=(8, 4.5), color=["#2a6f97", "#e07a5f", "#81b29a"][:len(splits)])
    ax.set_xlabel("femurs"); ax.set_ylabel(""); ax.invert_yaxis(); ax.set_title("Class distribution per split")
    for c in ax.containers: ax.bar_label(c, fontsize=8, padding=2)
    plt.tight_layout(); plt.savefig(f"{fd}/01_class_distribution.png", dpi=150); plt.close()
    fr = df[df.class_id >= 0]
    # 2 mask area by class
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [fr[fr.class_name == c].mask_area_frac * 100 for c in ORDER[1:]]
    ax.boxplot(data, tick_labels=[c.replace(" ", "\n", 1) for c in ORDER[1:]], showfliers=True)
    ax.set_ylabel("fracture mask area (% of ROI)"); ax.set_title("Fracture region size by class")
    plt.tight_layout(); plt.savefig(f"{fd}/02_mask_area_by_class.png", dpi=150); plt.close()
    # 3 centroid scatter
    fig, ax = plt.subplots(figsize=(4.5, 6.5))
    cols = plt.cm.tab10(np.arange(5))
    for k in range(5):
        s = fr[fr.class_id == k]
        ax.scatter(s.mask_cx, s.mask_cy, s=8, color=cols[k], label=CLASSES[k], alpha=.7)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect(df.H.median() / df.W.median())
    ax.set_title("Fracture centroid (normalised ROI coords)"); ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{fd}/03_centroids_by_class.png", dpi=150); plt.close()
    # 4 mask frequency heatmap per class (downsampled)
    h, w = 192, 128
    heat = {k: np.zeros((h, w)) for k in range(5)}
    for _, r in fr.iterrows():
        m = cv2.imread(os.path.join(root, r.split, "masks", r.file), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        heat[r.class_id] += cv2.resize((m > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    fig, axs = plt.subplots(1, 5, figsize=(13, 4))
    for k in range(5):
        n = max((fr.class_id == k).sum(), 1)
        axs[k].imshow(heat[k] / n, cmap="magma", vmin=0); axs[k].set_title(CLASSES[k], fontsize=8); axs[k].axis("off")
    plt.suptitle("Pixel-wise fracture frequency (train+test)"); plt.tight_layout()
    plt.savefig(f"{fd}/04_mask_frequency_heatmaps.png", dpi=150); plt.close()
    # 5 intensity stats
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    for s in splits:
        d = df[df.split == s]
        axs[0].hist(d["mean"], 40, alpha=.6, label=s); axs[1].hist(d["std"], 40, alpha=.6, label=s)
        axs[2].hist(np.log10(d.sharpness + 1), 40, alpha=.6, label=s)
    axs[0].set_title("mean intensity"); axs[1].set_title("intensity std (contrast)"); axs[2].set_title("log10 Laplacian var (sharpness)")
    axs[0].legend(); plt.tight_layout(); plt.savefig(f"{fd}/05_intensity_stats.png", dpi=150); plt.close()
    # 6 sample grid with overlays
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(3, 6, figsize=(14, 10))
    for j, c in enumerate(ORDER):
        sub = df[df.class_name == c]
        pick = sub.iloc[rng.choice(len(sub), min(3, len(sub)), replace=False)]
        for i, (_, r) in enumerate(pick.iterrows()):
            im = cv2.imread(os.path.join(root, r.split, "images", r.file), 0)
            rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            m = cv2.imread(os.path.join(root, r.split, "masks", r.file), 0)
            cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cnts, -1, (255, 40, 40), 6)
            axs[i, j].imshow(rgb); axs[i, j].axis("off")
            if i == 0: axs[i, j].set_title(c, fontsize=9)
    plt.tight_layout(); plt.savefig(f"{fd}/06_samples_per_class.png", dpi=110); plt.close()
    # 7 mask area vs compactness
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k in range(5):
        s = fr[fr.class_id == k]
        ax.scatter(s.mask_area_frac * 100, s.mask_solidity, s=8, color=cols[k], label=CLASSES[k], alpha=.7)
    ax.set_xscale("log"); ax.set_xlabel("mask area (% ROI, log)"); ax.set_ylabel("solidity")
    ax.legend(fontsize=7); ax.set_title("Mask shape vs size"); plt.tight_layout()
    plt.savefig(f"{fd}/07_area_vs_solidity.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = analyse(a.root, a.splits.split(","))
    lk = leakage(df)
    df.drop(columns=["_ahash"]).to_csv(os.path.join(a.out, "per_image_stats.csv"), index=False)
    fr = df[df.class_id >= 0]
    summ = {
        "n_images_per_split": df.split.value_counts().to_dict(),
        "class_counts": pd.crosstab(df.class_name, df.split).reindex(ORDER).fillna(0).astype(int).to_dict(),
        "image_shapes": df.groupby(["H", "W"]).size().rename("n").reset_index().to_dict("records"),
        "dtypes": df.dtype.value_counts().to_dict(),
        "mask_values_vs_yolo_class(mask-cls)": fr.mask_minus_cls.value_counts().to_dict(),
        "polys_per_labelled_image": fr.n_polys.value_counts().to_dict(),
        "multi_class_label_files": int(fr.label_classes.str.contains(",").sum() and
                                       fr.label_classes.apply(lambda s: len(set(s.split(","))) > 1).sum()),
        "mask_components_fractured": fr.mask_components.value_counts().to_dict(),
        "nofracture_with_nonempty_mask": int((df[df.class_id < 0].mask_area_px > 0).sum()),
        "fractured_with_empty_mask": int((fr.mask_area_px == 0).sum()),
        "poly_mask_iou": fr.poly_mask_iou.describe().to_dict(),
        "poly_mask_iou_below_0.9": fr[fr.poly_mask_iou < 0.9][["split", "file", "poly_mask_iou"]].to_dict("records"),
        "poly_out_of_bounds": int(fr.poly_out_of_bounds.sum()),
        "mask_area_frac_by_class": fr.groupby("class_name").mask_area_frac.describe().to_dict("index"),
        "mask_bbox_by_class": fr.groupby("class_name")[["mask_bw", "mask_bh", "mask_cx", "mask_cy"]].mean().to_dict("index"),
        "intensity": df.groupby("split")[["mean", "std", "p02", "p98", "frac_black", "frac_white"]].mean().to_dict("index"),
        "low_contrast_std_lt_30": df[df["std"] < 30][["split", "file", "std"]].to_dict("records"),
        "high_black_fraction_gt_0.2": df[df.frac_black > 0.2][["split", "file", "frac_black"]].to_dict("records"),
        "femurs_per_xray": df.groupby(["split", "xray_id"]).size().groupby("split").value_counts().to_dict(),
        "unique_xray_ids_per_split": df.groupby("split").xray_id.nunique().to_dict(),
        "leakage": lk,
    }
    def conv(o):
        if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [conv(v) for v in o]
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    json.dump(conv(summ), open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    figures(df, a.root, a.out)
    print(json.dumps(conv({k: v for k, v in summ.items() if k not in ("leakage", "poly_mask_iou_below_0.9")}), indent=1)[:6000])
    print("leakage:", {k: (len(v) if isinstance(v, list) else v) for k, v in lk.items()})


if __name__ == "__main__":
    main()
