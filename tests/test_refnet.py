"""Checks of RefFrac (Paper B, src/femurseg/refnet.py), its reference table and its dataset (needs torch, timm, smp,
albumentations, cv2).

Run with ``python tests/test_refnet.py`` or pytest. No downloads: every network is built without ImageNet weights.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from femurseg import nets as N  # noqa: E402
from femurseg import refbank as RB  # noqa: E402
from femurseg import refnet as RN  # noqa: E402
from test_refbank import _features, _records  # noqa: E402

CONVNEXT = "convnext_tiny.fb_in22k_ft_in1k"


def _net(variant: str, **kw):
    import torch

    torch.manual_seed(0)
    return N.build_net(N.parse_net(f"ref:{CONVNEXT}:{variant}"), pretrained=False, **kw).eval()


def _tables(tmp: Path, recs=None):
    """references.csv and pairs.csv of the synthetic fold file of test_refbank, built by refbank as in step B-1."""
    recs = recs or _records()
    feats = _features(recs)
    sides = {r.key: ("left" if int(r.femur_id) % 2 else "right") for r in recs}
    table = RB.reference_table(recs, feats, sides)
    pairs = RB.pair_rows(recs, feats, sides)
    for name, rows in (("references.csv", table), ("pairs.csv", pairs)):
        with open(tmp / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return recs, tmp / "references.csv", tmp / "pairs.csv"


def test_parse_variants_and_overrides():
    for v in RN.VARIANTS:
        spec = N.parse_net(f"ref:{CONVNEXT}:{v}")
        assert spec.kind == "ref" and spec.variant == v and spec.has_mask and spec.text == f"ref:{CONVNEXT}:{v}"
    for bad in (f"ref:{CONVNEXT}", f"ref:{CONVNEXT}:R0", f"ref:{CONVNEXT}:R6", f"ref:{CONVNEXT}:R8"):
        try:
            N.parse_net(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} accepted")
    assert RN.VARIANTS["R4"] == {"policy": "nearest", "k": 3, "align": "attention", "fusion": "difference", "margin": 0.0}
    assert RN.variant_config("R1")["margin"] == 0.5 and RN.variant_config("R4", margin=0.5)["margin"] == 0.5
    assert RN.variant_config("R1", margin=0.0)["margin"] == 0.0 and RN.variant_config("R4", k=5)["k"] == 5
    for kw in ({"k": 6}, {"window": -0.1}, {"window": 1.5}):
        try:
            RN.variant_config("R4", **kw)
        except ValueError:
            continue
        raise AssertionError(f"{kw} accepted")
    assert RN.window_cells(0.125, 48) == 6 and RN.window_cells(0.125, 24) == 3 and RN.window_cells(0, 48) is None


def test_untrained_network_computes_m0():
    """The fusion starts as [I, 0, 0]: with any references, the untrained network gives exactly what it gives without
    them, which is M0's function (same trunk, same heads)."""
    import torch

    torch.manual_seed(1)
    x = torch.randn(2, 3, 128, 96)
    for v in ("R1", "R4", "R5", "R6a", "R6b"):
        net = _net(v)
        refs = torch.randn(2 * net.cfg["k"], 3, 128, 96)
        with torch.no_grad():
            a, b = net(x, refs=refs), net(x)
        assert torch.allclose(a["seg"], b["seg"], atol=1e-5) and torch.allclose(a["cls"], b["cls"], atol=1e-5), v
        assert a["diff16"].shape == (2, 8, 6) and "diff16" not in b, v
        assert ("margin_d2" in a) == (v == "R1"), v


def test_attention_window_and_values():
    import torch

    torch.manual_seed(0)
    h, w, c, k = 6, 4, 8, 2
    t, r = torch.randn(1, c, h, w), torch.randn(1, k, c, h, w)
    norm, wq, wk = torch.nn.LayerNorm(c), torch.nn.Linear(c, 4), torch.nn.Linear(c, 4)
    out, a = RN.cross_attend(t, r, norm, wq, wk, radius=1)
    assert torch.allclose(a.sum(-1), torch.ones(1, h * w), atol=1e-5)
    keys = (a[0, 0] > 0).nonzero().flatten().tolist()  # position (0, 0): (0,0) (0,1) (1,0) (1,1) of both references
    assert keys == [0, 1, 4, 5, 24, 25, 28, 29], keys
    assert int((a[0, 2 * w + 1] > 0).sum()) == 9 * k  # an inner position: a 3 x 3 window in each reference
    values = r[0].flatten(2).transpose(1, 2).reshape(k * h * w, c)  # the values are the references' own features
    assert torch.allclose((a[0] @ values).T.reshape(c, h, w), out[0], atol=1e-5)
    _, ag = RN.cross_attend(t, r, norm, wq, wk, radius=None)
    assert int((ag[0, 0] > 0).sum()) == k * h * w  # global: the whole of every reference


def test_alignment_none_and_identity_affine_give_the_mean():
    import torch

    torch.manual_seed(2)
    for v in ("R6a", "R6b"):
        net = _net(v)
        ts = [torch.randn(2, 384, 8, 6), torch.randn(2, 768, 4, 3)]
        rs = [torch.randn(2, 3, 384, 8, 6), torch.randn(2, 3, 768, 4, 3)]
        with torch.no_grad():
            al, extra = net.align(ts, rs)
        for o, r in zip(al, rs):
            assert torch.allclose(o, r.mean(dim=1), atol=1e-5), v
        if v == "R6b":  # the spatial transformer starts at the identity
            want = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]).expand(2, 3, 2, 3)
            assert torch.allclose(extra["theta"], want)


def test_references_get_no_gradient_and_training_step_runs():
    import torch

    net = _net("R4").train()
    x, refs = torch.randn(2, 3, 128, 96), torch.randn(6, 3, 128, 96)
    f16, f32 = net.encode_references(refs)
    assert not f16.requires_grad and not f32.requires_grad and f16.shape == (6, 384, 8, 6)
    mask = torch.zeros(2, 128, 96, dtype=torch.long)
    mask[0, 40:70, 30:60] = 2
    loss, parts = N.make_loss(N.parse_net(f"ref:{CONVNEXT}:R4"))(net(x, refs=refs), mask, torch.tensor([2, 0]))
    loss.backward()
    assert set(parts) == {"ce", "seg_bce", "seg_dice"}
    fuse = net.fuse[0].conv.weight.grad
    assert fuse is not None and fuse[:, 384:].abs().sum() > 0  # the difference channels learn from the first step
    assert net.attn[0].k.weight.grad is not None and net.unet.encoder.model.stem_0.weight.grad is not None


def test_losses():
    import torch

    torch.manual_seed(3)
    out = {"cls": torch.randn(2, 6), "seg": torch.randn(2, 1, 32, 32)}
    mask = torch.zeros(2, 32, 32, dtype=torch.long)
    mask[0, :16, :] = 3  # the upper half of femur 0 is fractured
    y = torch.tensor([3, 0])
    ref0, _ = RN.make_ref_loss(0.0)(out, mask, y)
    m0, _ = N.make_loss(N.parse_net(f"m0:{CONVNEXT}"))(out, mask, y)
    assert torch.allclose(ref0, m0)  # without the margin term: exactly M0's loss
    d2 = torch.full((2, 2, 2), 0.2)  # 1/16 grid of 2 x 2 cells; cells [0, 0, :] are inside the fracture
    total, parts = RN.make_ref_loss(0.5)({**out, "margin_d2": d2}, mask, y)
    want = (2 * (0.5 - 0.2) + 6 * 0.2) / 8  # inside: max(0, 0.5 - d2); outside: d2
    assert abs(parts["margin"] - want) < 1e-6 and torch.allclose(total, m0 + 0.5 * want)


def test_reference_table_policies():
    with tempfile.TemporaryDirectory() as d:
        recs, ref_csv, pairs_csv = _tables(Path(d))
        table = RN.RefTable(ref_csv, pairs_csv, recs)
        assert table.fold_models == [0, 1, 2, 3, 4]
        by = {r.key: r for r in recs}
        fx_pair = next(r for r in recs if r.split == "train" and r.fold == 0 and r.cls6 > 0 and r.pair_file)
        healthy_pair = by[(fx_pair.split, fx_pair.pair_file)]
        single = next(r for r in recs if r.split == "test" and not r.pair_file)
        for rec in (fx_pair, healthy_pair, single):
            row = table.row[(0, rec.split, rec.file)]
            near = table.sources(0, rec, "nearest", 3)
            assert near == [(("train", row[f"ref{j}"]), row[f"ref{j}_mirrored"] == "1") for j in (1, 2, 3)]
            for key, _ in near:
                b = by[key]
                assert b.cls6 == 0 and b.fold != 0 and b.group != rec.group
            assert table.sources(0, rec, "random", 3) == [(("train", row["random_healthy"]),
                                                            row["random_healthy_mirrored"] == "1")]
            assert by[table.sources(0, rec, "fractured", 1)[0][0]].cls6 > 0
            assert table.sources(0, rec, "self", 3) == [(rec.key, False)]
        # partner: only a healthy partner (the training policy of R7); blind: any partner; always mirrored
        partner = ((fx_pair.split, healthy_pair.file), True)
        assert table.sources(0, fx_pair, "partner", 1) == [partner] == table.sources(0, fx_pair, "blind", 1)
        assert table.sources(0, fx_pair, "partner", 3) == [partner] + table.sources(0, fx_pair, "nearest", 2)
        assert table.sources(0, healthy_pair, "partner", 1) == table.sources(0, healthy_pair, "nearest", 1)
        assert table.sources(0, healthy_pair, "blind", 1) == [((fx_pair.split, fx_pair.file), True)]
        assert table.sources(0, single, "blind", 1) == table.sources(0, single, "nearest", 1)
        assert table.source_name(fx_pair, "partner", table.sources(0, fx_pair, "partner", 1)) == "partner"
        assert table.source_name(healthy_pair, "partner", table.sources(0, healthy_pair, "partner", 1)) == "nearest"


def test_reference_table_refuses_a_table_that_breaks_the_rules():
    with tempfile.TemporaryDirectory() as d:
        recs, ref_csv, pairs_csv = _tables(Path(d))
        rows = list(csv.DictReader(open(ref_csv, newline="")))
        heads = list(rows[0])
        by = {r.file: r for r in recs}
        i = next(j for j, r in enumerate(rows) if r["role"] == "held_out")
        own_fold = next(r.file for r in recs if r.split == "train" and r.fold == int(rows[i]["fold_model"])
                        and r.cls6 == 0)
        fractured = next(r.file for r in recs if r.split == "train" and r.fold != int(rows[i]["fold_model"])
                         and r.cls6 > 0)
        cases = {"a reference from the held-out fold": ("ref1", own_fold),
                 "a fractured reference": ("ref2", fractured),
                 "a wrong class": ("cls6", str((by[rows[i]["file"]].cls6 + 1) % 6))}
        for what, (col, value) in cases.items():
            bad = [dict(r) for r in rows]
            bad[i][col] = value
            path = Path(d) / "bad.csv"
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=heads)
                w.writeheader()
                w.writerows(bad)
            try:
                RN.RefTable(path, pairs_csv, recs)
            except ValueError:
                continue
            raise AssertionError(f"{what}: accepted")
        try:  # a table of another fold file (one femur missing)
            RN.RefTable(ref_csv, pairs_csv, recs + [recs[0].__class__(**{**recs[0].__dict__, "file": "new.png"})])
        except ValueError:
            pass
        else:
            raise AssertionError("a femur without references accepted")


def test_dataset_flips_references_with_the_target():
    import torch

    from femurseg.torchdata import RefFemurDataset

    class FakeCache:
        meta = {"cls6": [2, 0, 0]}
        images = np.arange(3 * 4 * 6, dtype=np.uint8).reshape(3, 4, 6)
        masks = np.zeros((3, 4, 6), np.uint8)

    cache = FakeCache()
    cache.masks[0, :, :2] = 2
    refs = [[(1, False), (2, True)]]
    ds = RefFemurDataset(cache, [0], refs, augment=False)
    img, msk, y, r = ds[0]
    assert y == 2 and r.shape == (2, 1, 4, 6) and r.dtype == torch.uint8
    assert np.array_equal(r[0, 0].numpy(), cache.images[1]) and np.array_equal(r[1, 0].numpy(), cache.images[2][:, ::-1])
    ds = RefFemurDataset(cache, [0], refs, augment=True, seed=0)
    ds.aug = lambda image, mask: {"image": image, "mask": mask}  # only the joint flip is left
    ds.ref_aug = lambda image: {"image": image}
    seen = set()
    for s in range(20):
        np.random.seed(s)
        img, msk, _, r = ds[0]
        flipped = not np.array_equal(img[0].numpy(), cache.images[0])
        seen.add(flipped)
        f = (lambda a: a[:, ::-1]) if flipped else (lambda a: a)
        assert np.array_equal(img[0].numpy(), f(cache.images[0])) and np.array_equal(msk.numpy(), f(cache.masks[0]))
        assert np.array_equal(r[0, 0].numpy(), f(cache.images[1]))
        assert np.array_equal(r[1, 0].numpy(), f(cache.images[2][:, ::-1]))
    assert seen == {True, False}


def test_default_augmentation_is_unchanged():
    import albumentations as A

    from femurseg.torchdata import train_transforms

    full, no_flip = train_transforms(0), train_transforms(0, flip=False)
    assert isinstance(full.transforms[0], A.HorizontalFlip) and len(full.transforms) == 5
    assert [type(t) for t in full.transforms[1:]] == [type(t) for t in no_flip.transforms]
    img = np.random.default_rng(0).integers(0, 255, (64, 48), dtype=np.uint8)
    a = [train_transforms(7)(image=img)["image"] for _ in range(2)]
    assert np.array_equal(a[0], a[1])  # same seed, same draw


def test_pointing_game():
    diff = np.zeros((4, 3))
    diff[1, 2] = 5.0
    mask = np.zeros((8, 6), np.uint8)
    mask[2:4, 4:6] = 3  # exactly cell (1, 2) of the 4 x 3 grid
    pg = RN.pointing_game(diff, mask)
    assert pg["diff_hit"] == 1 and abs(pg["diff_chance"] - 1 / 12) < 1e-6 and pg["diff_peak_x"] == round(2.5 / 3, 4)
    diff[1, 2], diff[3, 0] = 0.0, 1.0
    assert RN.pointing_game(diff, mask)["diff_hit"] == 0


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
