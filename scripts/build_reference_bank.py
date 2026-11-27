#!/usr/bin/env python3
"""Paper B (RefFrac), step B-1: the pair index and the table of healthy reference femurs (run R00_reference_bank).

Protocol: Analysis/07_Step4_PaperB_Reference_Bank_Protocol.md. Trains nothing. Step by step:
  1. reads every femur (1,043 training + 210 test) into the same 768 x 512 cache as the training runs;
  2. encodes each femur as released and mirrored left-right with the frozen ImageNet weights of M0's backbone
     (global average of the last stage, after the network's final normalisation: 768 numbers per image), and reads
     its pelvis side from the image (the brighter half of the upper image; protocol §8);
  3. picks each femur's references from the bank of each of the 5 fold models (src/femurseg/refbank.py): the 5 most
     similar healthy femurs, one random healthy femur and one random fractured femur, each mirrored when its pelvis
     side differs from the femur's;
  4. runs the sanity checks of the protocol (§4) and draws the nearest-neighbour figure of fold 0.

Writes to --out (results/<RUN>/, small, committed): pairs.csv, references.csv, sanity.json, summary.md,
nn_fold0.png, config.json; and to --work (not committed): features.npz (features, files, splits, pelvis sides).

Examples
  python scripts/build_reference_bank.py --out /kaggle/working/results/R00_reference_bank      (Kaggle, one GPU)
  python scripts/build_reference_bank.py --data-root "<folder with train/ and test/>" --device cpu --out <folder>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from femurseg import refbank as RB  # noqa: E402
from femurseg.classes import BY_CLS6  # noqa: E402
from femurseg.data import DEFAULT_FOLDS_CSV, load_records  # noqa: E402
from femurseg.torchdata import Cache, build_cache, normalize  # noqa: E402

NET = "convnext_tiny.fb_in22k_ft_in1k"  # M0's backbone (S06, S10), ImageNet weights, frozen


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--folds-csv", default=str(DEFAULT_FOLDS_CSV))
    ap.add_argument("--size", default="768x512", help="input height x width, as the training runs")
    ap.add_argument("--cache-dir", default="/tmp/femurseg_cache")
    ap.add_argument("--net", default=NET, help="timm name of the frozen feature network")
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (CPU smoke test only)")
    ap.add_argument("--device", default="", help="cuda, cuda:1 or cpu (default: cuda if available)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True, help="results folder of this run (results/<RUN>)")
    ap.add_argument("--work", default="", help="folder for features.npz; default <out>/../../work/<RUN>")
    a = ap.parse_args(argv)
    a.out = Path(a.out).resolve()
    a.work = Path(a.work).resolve() if a.work else a.out.parent.parent / "work" / a.out.name
    a.size_hw = tuple(int(v) for v in a.size.lower().split("x"))
    return a


def encode(cache: Cache, rows: list[int], net: str, pretrained: bool, device: str, batch: int) -> np.ndarray:
    """(N, 2, D) float32 features of the cached images: [as released, mirrored left-right]."""
    import timm
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(2026)  # only matters for --no-pretrained (random weights)
    model = timm.create_model(net, pretrained=pretrained, num_classes=0).eval().to(dev)
    model = model.to(memory_format=torch.channels_last)
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            x = torch.from_numpy(np.stack([cache.images[r] for r in rows[i:i + batch]]))[:, None]
            x = normalize(x, dev)
            f = torch.stack([model(x), model(torch.flip(x, dims=[3]))], dim=1)  # W is the last axis
            out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def figure(path: Path, cache: Cache, records, table: list[dict]) -> list[str]:
    """Fold 0: one held-out femur per class (seeded) with its three nearest healthy references."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_of = {r.file: i for i, r in zip(cache.rows(records), records)}
    held = [t for t in table if t["fold_model"] == 0 and t["role"] == "held_out"]
    picks = []
    for c in range(6):
        cand = sorted((t for t in held if t["cls6"] == c), key=lambda t: t["file"])
        if cand:
            picks.append(cand[int(hashlib.md5(f"figure|{c}".encode()).hexdigest()[:8], 16) % len(cand)])
    fig, axes = plt.subplots(len(picks), 4, figsize=(8, 3.1 * len(picks)), squeeze=False)
    for ax_row, t in zip(axes, picks):
        tiles = [(t["file"], False, f"{t['file']}\n{BY_CLS6[t['cls6']].name}")]
        tiles += [(t[f"ref{j}"], bool(t[f"ref{j}_mirrored"]),
                   f"ref {j}: cos {t[f'ref{j}_cos']:.3f}" + ("\nmirrored" if t[f"ref{j}_mirrored"] else "\nas released"))
                  for j in (1, 2, 3)]
        for ax, (file, mirrored, title) in zip(ax_row, tiles):
            img = np.asarray(cache.images[row_of[file]])[::2, ::2]
            ax.imshow(img[:, ::-1] if mirrored else img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle("Fold model 0: held-out femurs (left) and their 3 nearest healthy references", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))  # keep the title clear of the first row
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return [t["file"] for t in picks]


def summary_md(cfg: dict, counts: dict, checks: dict, picks: list[str], sides: dict) -> str:
    m1, m2, m5 = checks["1_mirror_check"], checks["2_retrieval_vs_chance"], checks["5_partner_rank"]
    close = checks["3_duplicates"][f"references_with_cos_ge_{RB.DUPLICATE_COS}"]
    lines = [f"# {cfg['run']}: pair index and reference bank (Paper B, step B-1)", "",
             f"Features: `{cfg['net']}` ({'ImageNet weights' if cfg['pretrained'] else 'RANDOM weights: smoke test'}"
             f", frozen), input {cfg['size_hw'][0]} x {cfg['size_hw'][1]}, {cfg['feature_dim']} numbers per image; "
             f"{cfg['n_femurs']} femurs, each as released and mirrored. Orientation: {cfg['orientation_rule']}. "
             f"Took {cfg['seconds']:.0f} s on {cfg['device']}. Protocol: "
             "`Analysis/07_Step4_PaperB_Reference_Bank_Protocol.md`.", "",
             "## Data (pair types)", "", "| Split | Femurs | Healthy | Fractured | Pelvis on the left | Pair types |",
             "|---|---|---|---|---|---|"]
    for split, c in counts.items():
        types = "; ".join(f"{k}: {v}" for k, v in sorted(c["by_pair_type"].items()))
        left = sum(v == "left" for k, v in sides.items() if k[0] == split)
        lines.append(f"| {split} | {c['femurs']} | {c['healthy']} | {c['fractured']} | {left} | {types} |")
    lines += ["", "## Checks (protocol §4)", "",
              f"1. **Mirror check:** the orientation rule mirrors the partner (the other hip) for "
              f"{m1['partner_mirrored_by_rule']} of {m1['femurs_with_partner']} femurs with a partner "
              f"({m1['share']:.1%}). Rule: {m1['rule']}. With the ImageNet features alone (the version-1 rule, "
              f"reported only), the mirrored partner is closer for {m1['features_mirrored_partner_closer']} "
              f"({m1['features_share']:.1%}).",
              f"2. **Retrieval vs chance** (held-out femurs): mean cosine of the 3 nearest minus that of the random "
              f"healthy femur = {m2['top3_minus_random_mean']:.3f} (mean over the 5 fold models).", "",
              "| Fold model | Held-out: 3 nearest | Held-out: random healthy | Held-out: nearest, median (IQR) | "
              "Test: nearest, median (IQR) |", "|---|---|---|---|---|"]
    for p, s in zip(m2["per_fold_model"], checks["4_held_out_vs_test_top1"]):
        h, t = s["held_out"], s["test"]
        lines.append(f"| {p['fold_model']} | {p['held_out_top3_mean_cos']:.3f} | {p['held_out_random_cos']:.3f} | "
                     f"{h['median']:.3f} ({h['q1']:.3f}-{h['q3']:.3f}) | {t['median']:.3f} ({t['q1']:.3f}-{t['q3']:.3f}) |")
    lines += ["",
              f"3. **Duplicates:** no reference has the same pixels as its femur (the script stops otherwise); "
              f"{len(close)} references have cosine >= {RB.DUPLICATE_COS}" + (":" if close else "."),
              *[f"   - fold model {c['fold_model']}: {c['file']} -> {c['reference']} (cos {c['cos']:.4f})"
                for c in close[:20]],
              "4. **Held-out vs test:** see the last two columns of the table above.",
              f"5. **Partner rank** ({m5['femurs']} held-out and test femurs with a healthy partner, counted once per "
              f"fold model for test femurs): the own partner, mirrored, would rank {m5['median_rank']:.0f} (median) "
              f"among the bank; rank 1 for {m5['share_rank_1']:.1%}, top 5 for {m5['share_top5']:.1%}.",
              "6. **Bank:** " + "; ".join(f"fold model {b['fold_model']}: {b['healthy_femurs']} healthy "
                                          f"({', '.join(f'{k}: {v}' for k, v in sorted(b['healthy_by_pair_type'].items()))})"
                                          for b in checks["6_bank"]) + ".",
              f"7. **Figure** `nn_fold0.png`: {', '.join(picks)} (one held-out femur per class, seeded).", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    a = parse_args(argv)
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    a.work.mkdir(parents=True, exist_ok=True)
    records = load_records(a.folds_csv)
    cache = Cache(build_cache(a.data_root, records, a.size_hw, a.cache_dir))
    rows = cache.rows(records)
    f = encode(cache, rows, a.net, not a.no_pretrained, a.device, a.batch)
    feats = {r.key: f[i] for i, r in enumerate(records)}
    sides = {r.key: RB.pelvis_side(cache.images[row]) for r, row in zip(records, rows)}
    np.savez_compressed(a.work / "features.npz", files=np.array([r.file for r in records]),
                        splits=np.array([r.split for r in records]), features=f.astype(np.float16),
                        pelvis_sides=np.array([sides[r.key] for r in records]))
    pairs = RB.pair_rows(records, feats, sides)
    table = RB.reference_table(records, feats, sides)
    checks = RB.sanity(records, feats, table, pairs)
    counts = RB.counts(records)
    write_csv(a.out / "pairs.csv", pairs)
    write_csv(a.out / "references.csv", table)
    picks = figure(a.out / "nn_fold0.png", cache, records, table)
    import torch

    cfg = {"run": a.out.name, "net": a.net, "pretrained": not a.no_pretrained, "size_hw": list(a.size_hw),
           "feature_dim": int(f.shape[-1]), "n_femurs": len(records), "top_k": RB.TOP_K, "folds_csv": str(a.folds_csv),
           "orientation_rule": "version 2 (protocol §8): a reference is mirrored when its pelvis side (the brighter "
                               "half of the upper image) differs from the femur's",
           "device": a.device or ("cuda" if torch.cuda.is_available() else "cpu"),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", "seconds": time.time() - t0,
           "torch": torch.__version__, "timm": __import__("timm").__version__}
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2))
    (a.out / "sanity.json").write_text(json.dumps({"counts": counts, "checks": checks}, indent=2))
    (a.out / "summary.md").write_text(summary_md(cfg, counts, checks, picks, sides))
    m1 = checks["1_mirror_check"]
    print(f"done in {cfg['seconds']:.0f} s: {len(table)} reference rows, mirror check {m1['share']:.1%} (features "
          f"alone {m1['features_share']:.1%}); see {a.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
