"""Checks of the segment-and-zoom network (src/femurseg/zoomnet.py; needs torch, timm, smp).

Run with ``python tests/test_zoomnet.py`` or pytest. No downloads: every network is built without ImageNet weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from femurseg import nets as N  # noqa: E402
from femurseg import zoomnet as Z  # noqa: E402

CONVNEXT = "convnext_tiny.fb_in22k_ft_in1k"


def _net():
    import torch

    torch.manual_seed(0)
    return N.build_net(N.parse_net(f"zoom:{CONVNEXT}"), pretrained=False)


def test_parse():
    spec = N.parse_net(f"zoom:{CONVNEXT}")
    assert spec.kind == "zoom" and spec.has_mask and spec.text == f"zoom:{CONVNEXT}"
    try:
        N.parse_net(f"zoom:{CONVNEXT}:x")
    except ValueError:
        return
    raise AssertionError("zoom with a variant accepted")


def test_mask_centre_and_fallback():
    import torch

    p = torch.zeros(2, 1, 40, 30)
    p[0, 0, 10:20, 5:15] = 0.9                       # a region: centroid (15, 10) in pixel centres
    p[1, 0, 30, 20] = 0.3                            # nothing >= 0.5: the most probable pixel wins
    c = Z.mask_centre(p, topk=1)
    assert torch.allclose(c[0], torch.tensor([15.0, 10.0]), atol=1e-4), c[0]
    assert torch.allclose(c[1], torch.tensor([30.5, 20.5]), atol=1e-4), c[1]
    cc = Z.clamp_centre(torch.tensor([[1.0, 1.0], [39.0, 29.0]]), (40, 30), 20)
    assert torch.allclose(cc, torch.tensor([[10.0, 10.0], [30.0, 20.0]]))


def test_crop_takes_the_window():
    import torch

    x = torch.arange(64 * 48, dtype=torch.float32).view(1, 1, 64, 48)
    # a 16-pixel window around the centre (24, 20) at its own size = the pixels 16..31 x 12..27, exactly
    out = Z.crop(x, torch.tensor([[24.0, 20.0]]), 16, 16)
    assert torch.allclose(out[0, 0], x[0, 0, 16:32, 12:28], atol=1e-3)


def test_window_hit():
    gt = np.zeros((100, 80), bool)
    gt[40:60, 30:50] = True
    assert Z.window_hit((50.0, 40.0), 40, gt) == {"zoom_hit": 1, "zoom_mask_inside": 1.0}
    far = Z.window_hit((15.0, 15.0), 20, gt)
    assert far["zoom_hit"] == 0 and far["zoom_mask_inside"] == 0.0
    assert Z.window_hit((50.0, 40.0), 40, np.zeros((100, 80), bool)) == {}


def test_forward_loss_and_teacher_forcing():
    import torch

    net = _net()
    net.window, net.zoom_input = 64, 64
    torch.manual_seed(1)
    x = torch.randn(2, 3, 256, 192)
    mask = torch.zeros(2, 256, 192, dtype=torch.long)
    mask[0, 200:230, 20:50] = 2                     # far from the centre of the image
    y = torch.tensor([2, 0])
    net.eval()
    gt = (mask > 0).float()[:, None]
    with torch.no_grad():
        out = net(x, gt=gt, tf_prob=1.0)            # the oracle: the crop goes to the released mask
    assert out["seg"].shape == (2, 1, 256, 192) and out["cls"].shape == (2, 6) and out["cls_global"].shape == (2, 6)
    assert torch.allclose(out["centre"][0], torch.tensor([215.0, 35.0]), atol=1e-3), out["centre"][0]
    # a healthy femur keeps its own centre, inside the image
    c1 = out["centre"][1]
    assert 32 <= float(c1[0]) <= 224 and 32 <= float(c1[1]) <= 160
    pp = N.postprocess(N.parse_net(f"zoom:{CONVNEXT}"), out)
    assert pp["fmap"].shape == (2, 256, 192) and pp["centre"].shape == (2, 2)
    assert torch.allclose(pp["conf"].sum(1), pp["score"])
    net.train()
    loss_fn = N.make_loss(N.parse_net(f"zoom:{CONVNEXT}"))
    total, parts = loss_fn(net(x, gt=gt, tf_prob=0.5), mask, y)
    assert set(parts) == {"ce", "ce_global", "seg_bce", "seg_dice"} and torch.isfinite(total)
    total.backward()
    assert net.zoom.stem[0].weight.grad is not None and net.unet.decoder is not None
    assert any(p.grad is not None for p in net.unet.encoder.parameters())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
