#!/usr/bin/env python3
"""Paper B step B-1: review of the orientation rule (Analysis/07 §8). Can a femur's side be read from its pixels?

Check 1 of R00_reference_bank found that the frozen ImageNet features hardly see left and right (the mirrored partner
was closer for only 58.4% of 810 femurs, below the protocol's 60% line). This script tests orientation signals that
do not use those features. It trains nothing, uses no label and needs only numpy and OpenCV (about a minute on a
laptop):
  1. a 96 x 64 thumbnail of every released femur image (area interpolation);
  2. pixel mirror check: for every femur with a partner (the other hip of its x-ray), is the partner closer mirrored
     than as released? Correlation of the thumbnails after a Gaussian blur of sigma 2: the protocol's check 1 with
     pixels instead of ImageNet features;
  3. pelvis side of every femur by the rule of version 2 (femurseg.refbank.pelvis_side: the brighter half of the
     upper image) and how often the two femurs of an x-ray get opposite sides, as they must; two variants (whole
     image, upper third) are reported only;
  4. 20,000 random pairs of femurs from different x-ray groups (seed 2026): how often the side rule ("mirror when
     the sides differ") agrees with the orientation the thumbnails prefer.

  python scripts/check_orientation.py --data-root "<folder with train/ and test/>" --out results/analysis/C10_paperB_orientation_review
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from femurseg.data import DEFAULT_FOLDS_CSV, load_records, read_png  # noqa: E402
from femurseg.paths import build_index, find_data_root  # noqa: E402
from femurseg.refbank import pelvis_side  # noqa: E402

THUMB_HW = (96, 64)
BLUR_SIGMA = 2.0
N_RANDOM_PAIRS = 20000
AMBIGUOUS = 0.05  # |left - right| / (left + right) of the upper half below this: the side is called ambiguous


def thumbnails(data_root: str, records) -> np.ndarray:
    import cv2

    index = build_index(find_data_root(data_root))
    out = np.zeros((len(records),) + THUMB_HW, np.float32)
    for i, r in enumerate(records):
        out[i] = cv2.resize(read_png(index[r.key]["image"]), THUMB_HW[::-1], interpolation=cv2.INTER_AREA)
    return out


def unit_rows(a: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1).astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)


def opposite_pairs(records, side: np.ndarray) -> tuple[int, int, list[list[str]]]:
    """(pairs with opposite sides, pairs, the pairs with the same side)."""
    idx = {r.file: i for i, r in enumerate(records)}
    pairs = [(r.file, r.pair_file) for r in records if r.pair_file and r.file < r.pair_file]
    same = [[a, b] for a, b in pairs if side[idx[a]] == side[idx[b]]]
    return len(pairs) - len(same), len(pairs), same


def review(records, th: np.ndarray) -> dict:
    import cv2

    idx = {r.file: i for i, r in enumerate(records)}
    blurred = np.stack([cv2.GaussianBlur(x, (0, 0), BLUR_SIGMA) for x in th])
    zr, zm = unit_rows(blurred), unit_rows(blurred[:, :, ::-1])
    partners = [(idx[r.file], idx[r.pair_file]) for r in records if r.pair_file]
    pixel_wins = sum(float(zr[a] @ zm[b]) > float(zr[a] @ zr[b]) for a, b in partners)

    h, w = THUMB_HW
    side = np.array([pelvis_side(x) for x in th])
    top = th[:, : h // 2]
    left, right = top[:, :, : w // 2].mean(axis=(1, 2)), top[:, :, w // 2:].mean(axis=(1, 2))
    rel = np.abs(left - right) / np.maximum(left + right, 1e-9)
    ok, n_pairs, same = opposite_pairs(records, side)
    variants = {}
    for name, rows in (("whole image", slice(0, h)), ("upper third", slice(0, h // 3))):
        part = th[:, rows]
        s = np.where(part[:, :, : w // 2].mean(axis=(1, 2)) > part[:, :, w // 2:].mean(axis=(1, 2)), "left", "right")
        v_ok, _, _ = opposite_pairs(records, s)
        variants[name] = {"opposite": v_ok, "pairs": n_pairs, "share": v_ok / n_pairs}

    rng = np.random.default_rng(2026)
    groups = np.array([r.group for r in records])
    n = agree = 0
    while n < N_RANDOM_PAIRS:
        a, b = rng.integers(0, len(records), 2)
        if groups[a] == groups[b]:
            continue
        n += 1
        agree += (float(zr[a] @ zm[b]) > float(zr[a] @ zr[b])) == (side[a] != side[b])

    split = np.array([r.split for r in records])
    return {
        "femurs": len(records), "thumbnail_hw": list(THUMB_HW), "blur_sigma": BLUR_SIGMA,
        "pixel_mirror_check": {"femurs_with_partner": len(partners), "mirrored_partner_closer": int(pixel_wins),
                               "share": pixel_wins / len(partners)},
        "pelvis_side": {"left_share": {"all": float(np.mean(side == "left")),
                                       "train": float(np.mean(side[split == "train"] == "left")),
                                       "test": float(np.mean(side[split == "test"] == "left"))},
                        "pairs_opposite": ok, "pairs": n_pairs, "share": ok / n_pairs,
                        "pairs_with_the_same_side": same,
                        "relative_difference": {"median": float(np.median(rel)), "p05": float(np.percentile(rel, 5)),
                                                "min": float(rel.min()),
                                                f"femurs_below_{AMBIGUOUS}": int((rel < AMBIGUOUS).sum()),
                                                f"files_below_{AMBIGUOUS}": [records[i].file for i in
                                                                             np.where(rel < AMBIGUOUS)[0]]}},
        "variants_reported_only": variants,
        "random_pairs_other_groups": {"pairs": n, "side_rule_agrees_with_pixels": int(agree), "share": agree / n,
                                      "seed": 2026},
    }


def report_md(res: dict, seconds: float) -> str:
    pm, ps, rp = res["pixel_mirror_check"], res["pelvis_side"], res["random_pairs_other_groups"]
    rd = ps["relative_difference"]
    lines = ["# C10: review of Paper B's orientation rule (Analysis/07 §8)", "",
             f"Made by `scripts/check_orientation.py` from the released images of all {res['femurs']} femurs "
             f"({res['thumbnail_hw'][0]} x {res['thumbnail_hw'][1]} thumbnails; no label used) in {seconds:.0f} s.", "",
             "| Test | Result |", "|---|---|",
             f"| Pixel mirror check: the partner (other hip) correlates better mirrored | {pm['mirrored_partner_closer']}"
             f" of {pm['femurs_with_partner']} femurs ({pm['share']:.1%}) |",
             f"| Same check with the frozen ImageNet features (R00_reference_bank, version 1) | 473 of 810 (58.4%) |",
             f"| Pelvis side rule: the two femurs of an x-ray get opposite sides | {ps['pairs_opposite']} of "
             f"{ps['pairs']} x-rays ({ps['share']:.1%}) |",
             *[f"| Variant, reported only ({k}) | {v['opposite']} of {v['pairs']} ({v['share']:.1%}) |"
               for k, v in res["variants_reported_only"].items()],
             f"| Random pairs from different x-ray groups: the side rule agrees with the thumbnails | "
             f"{rp['side_rule_agrees_with_pixels']} of {rp['pairs']} ({rp['share']:.1%}) |",
             f"| Pelvis on the left (all / train / test) | {ps['left_share']['all']:.1%} / "
             f"{ps['left_share']['train']:.1%} / {ps['left_share']['test']:.1%} |", "",
             f"- X-rays whose two femurs got the same side: {', '.join(' + '.join(p) for p in ps['pairs_with_the_same_side']) or 'none'}.",
             f"- Relative brightness difference of the two upper halves: median {rd['median']:.3f}, 5th percentile "
             f"{rd['p05']:.3f}; {rd[f'femurs_below_{AMBIGUOUS}']} femurs below {AMBIGUOUS} (side ambiguous).", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--folds-csv", default=str(DEFAULT_FOLDS_CSV))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    records = load_records(a.folds_csv)
    res = review(records, thumbnails(a.data_root, records))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "orientation_review.json").write_text(json.dumps(res, indent=2))
    (out / "orientation_review.md").write_text(report_md(res, time.time() - t0))
    print(f"pixel mirror check {res['pixel_mirror_check']['share']:.1%}; pelvis side opposite in "
          f"{res['pelvis_side']['share']:.1%} of x-rays; written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
