"""Checks of the PyTorch side of S04-S10: networks, decision rule, losses, image cache (needs torch, timm, smp, cv2).

Run with ``python tests/test_nets.py`` or pytest. No downloads: every network is built without ImageNet weights.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from femurseg import nets as N  # noqa: E402
from femurseg.torchdata import Cache, FemurDataset, build_cache, normalize, resize_pair  # noqa: E402

CONVNEXT = "convnext_tiny.fb_in22k_ft_in1k"


def test_parse_net():
    assert N.parse_net("cls:densenet121.tv_in1k") == N.NetSpec("cls", "densenet121.tv_in1k", "")
    assert N.parse_net("seg:Unet:resnet34") == N.NetSpec("seg", "Unet", "resnet34")
    assert N.parse_net(f"m0:{CONVNEXT}").has_mask and not N.parse_net("cls:x").has_mask
    for bad in ("unet", "seg:Unet", "cls:a:b", "m1:x"):
        try:
            N.parse_net(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} accepted")


def test_m0_classification_path_is_the_classifier():
    """M0 = the ConvNeXt-T classifier (A0) + a decoder: with the same weights, the class logits are identical."""
    import torch

    torch.manual_seed(0)
    m0 = N.build_net(N.parse_net(f"m0:{CONVNEXT}"), pretrained=False).eval()
    a0 = N.build_net(N.parse_net(f"cls:{CONVNEXT}"), pretrained=False).eval()
    sd = a0.body.state_dict()
    enc = {re.sub(r"^(stem|stages)\.(\d+)\.", r"\1_\2.", k): v for k, v in sd.items() if not k.startswith("head.")}
    m0.body.encoder.model.load_state_dict(enc, strict=True)
    m0.body.classification_head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("head.")}, strict=True)
    x = torch.randn(2, 3, 128, 96)
    with torch.no_grad():
        ya, ym = a0(x), m0(x)
    assert torch.equal(ya["cls"], ym["cls"])
    assert ym["seg"].shape == (2, 1, 128, 96)


def test_postprocess_classifier():
    import torch

    logits = torch.tensor([[2.0, 0.0, 1.0, -1.0, 0.0, 0.5], [-3.0, 0.0, 0.0, 0.0, 4.0, 0.0]])
    pp = N.postprocess(N.parse_net("cls:x"), {"cls": logits})
    p = torch.softmax(logits, 1)
    assert torch.allclose(pp["score"], 1 - p[:, 0]) and torch.allclose(pp["conf"].sum(1), pp["score"])
    assert pp["top_cls6"].tolist() == [2, 4] and pp["fmap"] is None


def test_postprocess_segmenter():
    import torch

    h, w = 64, 48
    seg = torch.full((2, 6, h, w), -5.0)
    seg[:, 0] = 5.0                       # background everywhere ...
    seg[0, 0, 10:30, 10:20] = -5.0        # ... except a region of class 3 (image 0)
    seg[0, 3, 10:30, 10:20] = 5.0
    seg[0, 0, 10:14, 10:20] = -5.0        # where a few pixels lean to class 4
    seg[0, 4, 10:14, 10:20] = 6.0
    pp = N.postprocess(N.parse_net("seg:Unet:resnet34"), {"seg": seg}, topk=50)
    assert pp["top_cls6"][0] == 3 and pp["score"][0] > 0.99          # most of the region says class 3
    assert pp["score"][1] < 0.01                                       # image 1: nothing
    assert torch.allclose(pp["conf"].sum(1), pp["score"], atol=1e-6)
    assert pp["fmap"].shape == (2, h, w) and pp["fmap"][0, 20, 15] > 0.99 and pp["fmap"][0, 50, 40] < 0.01
    # an empty predicted region: the class comes from the TOPK most fracture-like pixels
    seg2 = torch.full((1, 6, h, w), -5.0)
    seg2[:, 0] = 2.0
    seg2[0, 5, :5, :5] = 1.5              # f < 0.5 everywhere, highest where class 5 is strongest
    pp2 = N.postprocess(N.parse_net("seg:Unet:resnet34"), {"seg": seg2}, topk=25)
    assert pp2["top_cls6"][0] == 5 and pp2["score"][0] < 0.5


def test_losses_are_finite_and_complete():
    import torch

    torch.manual_seed(0)
    mask = torch.zeros(2, 64, 64, dtype=torch.long)
    mask[0, 20:40, 20:40] = 3
    y = torch.tensor([3, 0])
    outs = {"cls": {"cls": torch.randn(2, 6)}, "seg": {"seg": torch.randn(2, 6, 64, 64)},
            "m0": {"cls": torch.randn(2, 6), "seg": torch.randn(2, 1, 64, 64)}}
    parts_expected = {"cls": {"ce"}, "seg": {"seg_ce", "seg_dice"}, "m0": {"ce", "seg_bce", "seg_dice"}}
    for kind, out in outs.items():
        spec = N.NetSpec(kind, "x", "y" if kind == "seg" else "")
        total, parts = N.make_loss(spec)(out, mask, y)
        assert torch.isfinite(total) and set(parts) == parts_expected[kind]
        assert abs(float(total) - sum(parts.values())) < 1e-5


def test_resize_pair_keeps_mask_centred():
    img = np.zeros((8, 8), np.uint8)
    msk = np.zeros((8, 8), np.uint8)
    msk[2:6, 4:8] = 4                     # 4 x 4 block -> exactly the 2 x 2 block [1:3, 2:4] at half size
    img[msk > 0] = 200
    small_img, small_msk = resize_pair(img, msk, (4, 4))
    want = np.zeros((4, 4), np.uint8)
    want[1:3, 2:4] = 4
    assert np.array_equal(small_msk, want) and small_img[1, 2] == 200 and small_img[0, 0] == 0
    _, empty = resize_pair(img, np.zeros_like(msk), (4, 4))
    assert not empty.any()


def test_cache_round_trip():
    import torch
    from synthetic_data import make_dataset

    from femurseg.data import load_records, read_png
    from femurseg.paths import build_index, find_data_root

    with tempfile.TemporaryDirectory() as tmp:
        data_root, folds_csv = make_dataset(Path(tmp), n_per_fold=3, n_test=3)
        recs = load_records(folds_csv)
        folder = build_cache(data_root, recs, (192, 128), Path(tmp) / "cache", threads=2)
        assert build_cache(data_root, recs, (192, 128), Path(tmp) / "cache") == folder  # reused, not rebuilt
        cache = Cache(folder)
        index = build_index(find_data_root(data_root))
        for r in recs:
            row = cache.row[r.key]
            msk = read_png(index[r.key]["mask"])
            assert np.array_equal(cache.gt_full(row), msk > 0)
            assert cache.images[row].shape == (192, 128) and set(np.unique(cache.masks[row])) <= {0, r.cls6}
            assert (cache.masks[row] > 0).any() == (r.cls6 > 0)
        ds = FemurDataset(cache, cache.rows(recs), augment=True, seed=1)
        x, m, c = ds[1]
        assert x.shape == (1, 192, 128) and x.dtype == torch.uint8 and m.shape == (192, 128) and c == recs[1].cls6
        z = normalize(x[None], torch.device("cpu"))
        assert z.shape == (1, 3, 192, 128) and z.dtype == torch.float32


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} network tests passed")
