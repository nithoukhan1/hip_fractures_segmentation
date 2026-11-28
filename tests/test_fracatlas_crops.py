"""Checks of the FracAtlas crop geometry (scripts/make_fracatlas_crops.py; needs cv2)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import make_fracatlas_crops as FC  # noqa: E402


def test_crop_box_puts_the_head_and_the_midline_in_place():
    w, h = 400, 300
    for side, hx, xm in (("R", 0.25, 0.5), ("L", 0.75, 0.5)):
        x0, y0, bw, bh = FC.crop_box(w, h, xm, hx, 0.5, side)
        assert abs(bh - 1.5 * bw) < 1e-9 and abs(bw - 100 / FC.MEDIAL_SHARE) < 1e-9
        head_from_left = (hx * w - x0) / bw
        assert abs(head_from_left - (FC.X_LATERAL if side == "R" else FC.MEDIAL_SHARE)) < 1e-9
        medial_edge = x0 + bw if side == "R" else x0
        assert abs(medial_edge - xm * w) < 1e-6          # the medial edge is the midline
        assert abs((0.5 * h - y0) / bh - FC.Y_HEAD) < 1e-9


def test_warp_maps_the_box_to_the_output_and_pads_black():
    img = np.full((300, 400), 200, np.uint8)
    out = FC.warp(img, (-50.0, 0.0, 200.0, 300.0))
    assert out.shape == (FC.OUT_H, FC.OUT_W)
    assert out[:, : int(0.2 * FC.OUT_W)].max() == 0 and out[10, -10] == 200


if __name__ == "__main__":
    test_crop_box_puts_the_head_and_the_midline_in_place()
    test_warp_maps_the_box_to_the_output_and_pads_black()
    print("ok")
