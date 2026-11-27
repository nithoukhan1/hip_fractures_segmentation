"""Checks of SLG-Net (Paper A, src/femurseg/slg.py) and its region-deletion inputs (needs torch, timm, smp, cv2).

Run with ``python tests/test_slg.py`` or pytest. No downloads: every network is built without ImageNet weights.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from femurseg import nets as N  # noqa: E402
from femurseg import slg as G  # noqa: E402

CONVNEXT = "convnext_tiny.fb_in22k_ft_in1k"
COUNTS = {0: 363, 1: 57, 2: 254, 3: 134, 4: 199, 5: 36}  # the release's training set (Table 1)


def _net(variant: str, **kw):
    import torch

    torch.manual_seed(0)
    return N.build_net(N.parse_net(f"slg:{CONVNEXT}:{variant}"), pretrained=False, **kw).eval()


def _batch():
    import torch

    torch.manual_seed(1)
    x = torch.randn(3, 3, 128, 96)
    mask = torch.zeros(3, 128, 96, dtype=torch.long)
    mask[0, 30:60, 20:50] = 2   # neck, Garden III-IV
    mask[1, 70:90, 30:60] = 3   # pertrochanteric, Evans I-II
    return x, mask, torch.tensor([2, 3, 0])


def test_parse_variants():
    for v in G.VARIANTS:
        spec = N.parse_net(f"slg:{CONVNEXT}:{v}")
        assert spec.kind == "slg" and spec.variant == v and spec.has_mask and spec.text == f"slg:{CONVNEXT}:{v}"
    for bad in (f"slg:{CONVNEXT}", f"slg:{CONVNEXT}:A1", f"slg:{CONVNEXT}:A7"):
        try:
            N.parse_net(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} accepted")


def test_chain_rule_probabilities():
    import torch

    torch.manual_seed(0)
    z_f, z_loc, z_sev = torch.randn(5), torch.randn(5, 3), torch.randn(5, 2)
    p = G.chain_logp(z_f, z_loc, z_sev).exp()
    pf, pl, ps = torch.sigmoid(z_f), torch.softmax(z_loc, 1), torch.sigmoid(z_sev)
    want = torch.stack([1 - pf, pf * pl[:, 0] * (1 - ps[:, 0]), pf * pl[:, 0] * ps[:, 0],
                        pf * pl[:, 1] * (1 - ps[:, 1]), pf * pl[:, 1] * ps[:, 1], pf * pl[:, 2]], dim=1)
    assert torch.allclose(p, want, atol=1e-6) and torch.allclose(p.sum(1), torch.ones(5), atol=1e-6)
    assert torch.allclose(torch.softmax(p.log(), 1), p, atol=1e-6)  # postprocess's softmax returns them unchanged


def test_geometry_of_an_ellipse_and_an_empty_mask():
    import cv2
    import torch

    h, w = 384, 256
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (100, 150), (40, 20), 0, 0, 360, 1, -1)  # centre (x 100, y 150), half-axes x 40, y 20
    mt = torch.from_numpy(m.astype(np.float32))[None, None]
    q = G.mask_confidence(mt, 50)
    g = G.geometry(mt, q, empty_mass=0.0)[0]
    name = dict(zip(G.GEOM_NAMES, g.tolist()))
    assert abs(name["q"] - 1) < 1e-6 and abs(name["pbar"] - 1) < 1e-6
    assert abs(name["cy"] - 150.5 / h) < 2e-3 and abs(name["cx"] - 100.5 / w) < 2e-3
    assert abs(name["h"] - 40 / h) < 0.01 and abs(name["w"] - 80 / w) < 0.01  # 4 SD = full axes of the ellipse
    assert abs(name["rho"]) < 0.02
    assert abs(name["log_area"] - math.log10(m.sum() / (h * w) + 1e-4) / 4) < 1e-5
    e = G.geometry(torch.zeros(1, 1, h, w), torch.zeros(1), empty_mass=100.0)[0]
    en = dict(zip(G.GEOM_NAMES, e.tolist()))
    assert en["cy"] == 0.5 and en["cx"] == 0.5 and en["h"] < 1e-3 and en["rho"] == 0 and en["log_area"] == -1.0


def test_region_pool_is_the_mean_inside_the_mask():
    import torch

    f = torch.randn(2, 4, 6, 4)
    m = torch.zeros(2, 1, 12, 8)
    m[0, 0, 0:4, 0:4] = 1  # exactly the 2 x 2 cells [0:2, 0:2] of the feature grid
    r = G.region_pool(f, m)
    assert torch.allclose(r[0], f[0, :, 0:2, 0:2].mean(dim=(1, 2)), atol=1e-6)
    assert torch.equal(r[1], torch.zeros(4))  # empty mask: zero (and the network multiplies by q = 0 anyway)


def test_variants_have_their_components():
    nets = {v: _net(v) for v in G.VARIANTS}
    assert hasattr(nets["A2"], "fc") and not hasattr(nets["A2"], "geom_embed")
    assert hasattr(nets["A3"], "fc") and hasattr(nets["A3"], "geom_embed")
    assert hasattr(nets["A4"], "sev_trunks") and not hasattr(nets["A4"], "sev_trunk")
    for v in ("A5", "A6"):
        assert hasattr(nets[v], "sev_trunk") and hasattr(nets[v], "loc_embed") and not hasattr(nets[v], "fc")
    n = {v: sum(p.numel() for p in net.parameters()) for v, net in nets.items()}
    assert n["A5"] == n["A6"] and n["A4"] > n["A5"] > n["A3"] > n["A2"]


def test_outputs_losses_and_postprocess():
    import torch

    x, mask, y = _batch()
    gt = (mask > 0).float()[:, None]
    priors = G.priors_from_counts(COUNTS)
    assert abs(priors["log_odds_high"][0] - math.log(254 / 57)) < 1e-9
    assert abs(math.exp(priors["log_prior_loc"][2]) - 36 / (311 + 333 + 36)) < 1e-9
    for v in G.VARIANTS:
        net = _net(v)
        spec = N.parse_net(f"slg:{CONVNEXT}:{v}")
        out = net(x, gt=gt, tf_prob=0.5)
        assert out["seg"].shape == (3, 1, 128, 96) and out["cls"].shape == (3, 6)
        pp = N.postprocess(spec, out)
        assert torch.allclose(pp["conf"].sum(1), pp["score"], atol=1e-6) and pp["fmap"].shape == (3, 128, 96)
        total, parts = N.make_loss(spec, priors=priors)(out, mask, y)
        assert torch.isfinite(total)
        if G.VARIANTS[v]["hier"]:
            assert set(parts) == {"frac_bce", "loc_ce", "grade_bce", "joint_nll", "seg_bce", "seg_dice"}
            assert torch.allclose(pp["score"], torch.sigmoid(out["z_f"]), atol=1e-6)
            want = sum(parts[k] for k in ("frac_bce", "loc_ce", "grade_bce", "seg_bce", "seg_dice")) + 0.5 * parts["joint_nll"]
        else:
            assert set(parts) == {"ce", "seg_bce", "seg_dice"}
            want = sum(parts.values())
        assert abs(float(total.detach()) - want) < 1e-4


def test_logit_adjustment_changes_the_loss_not_the_prediction():
    import torch

    x, mask, y = _batch()
    priors = G.priors_from_counts(COUNTS)
    with torch.no_grad():
        out = _net("A6")(x)
    plain = N.make_loss(N.parse_net(f"slg:{CONVNEXT}:A5"), priors=priors)(out, mask, y)[1]   # A5: no adjustment
    adjusted = N.make_loss(N.parse_net(f"slg:{CONVNEXT}:A6"), priors=priors)(out, mask, y)[1]
    assert plain["frac_bce"] == adjusted["frac_bce"]                      # no adjustment at the fracture node
    assert plain["loc_ce"] != adjusted["loc_ce"] and plain["grade_bce"] != adjusted["grade_bce"]
    z_sev = out["z_sev"][0, 0] + priors["log_odds_high"][0]              # femur 0: neck, Garden III-IV
    assert abs(adjusted["grade_bce"] - float(torch.nn.functional.softplus(-z_sev)) / 2
               - float(torch.nn.functional.softplus(out["z_sev"][1, 1] + priors["log_odds_high"][1])) / 2) < 1e-5


def test_teacher_forcing_and_oracle_use_the_released_mask():
    import torch

    x, mask, _ = _batch()
    gt = (mask > 0).float()[:, None]
    net = _net("A6")
    with torch.no_grad():
        own = net(x)["cls"]
        oracle = net(x, gt=gt, tf_prob=1.0)["cls"]
        other = net(x, gt=torch.roll(gt, 1, dims=0), tf_prob=1.0)["cls"]
        never = net(x, gt=gt, tf_prob=0.0)["cls"]
    assert not torch.allclose(own, oracle) and not torch.allclose(oracle, other)
    assert torch.equal(own, never)  # tf_prob 0: the network's own mask
    assert torch.allclose(oracle[2].exp()[0], torch.sigmoid(-net.head_frac(torch.cat(  # femur 2: empty mask, r = 0
        [net.ctx_norm(net.ctx_pool(net.unet.encoder(x)[-1])).flatten(1)[2:],
         net.geom_embed(G.geometry(gt[2:], torch.zeros(1), 100.0 * 128 * 96 / G.REF_PIXELS))], 1)))[0, 0], atol=1e-5)


def test_detached_mask_gets_no_gradient_from_the_decision():
    x, mask, y = _batch()
    for detach, want_grad in ((False, True), (True, False)):
        net = _net("A6", slg_detach=detach).train()
        out = net(x)
        out["cls"].gather(1, y[:, None]).sum().backward()  # a decision loss only: no mask loss
        g = net.unet.segmentation_head[0].weight.grad
        has = g is not None and float(g.abs().sum()) > 0
        assert has == want_grad, (detach, has)


def test_deletion_images():
    spec = importlib.util.spec_from_file_location("train_net", REPO / "scripts" / "train_net.py")
    tn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tn)
    rng = np.random.default_rng(0)
    img = rng.integers(60, 200, (768, 512)).astype(np.uint8)
    region = np.zeros((768, 512), bool)
    region[300:360, 200:260] = True
    deleted, control, info = tn.deletion_images(img, region, "x.png")
    assert info["control"] == "matched" and info["region_px"] > region.sum()                 # dilated
    changed_d, changed_c = deleted != img, control != img
    assert changed_d[region].mean() > 0.9 and not changed_d[:250].any()          # the region (dilated) only
    assert changed_c.any() and not (changed_c & changed_d).any()                  # elsewhere, no overlap
    assert abs(int(changed_c.sum()) - int(changed_d.sum())) < 0.1 * changed_d.sum()  # same shape
    d2, c2, _ = tn.deletion_images(img, region, "x.png")
    assert np.array_equal(d2, deleted) and np.array_equal(c2, control)            # seeded by the file name
    same_d, same_c, info0 = tn.deletion_images(img, np.zeros_like(region), "x.png")
    assert np.array_equal(same_d, img) and np.array_equal(same_c, img)            # no region: unchanged
    assert info0["control"] == "no_region"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} SLG-Net tests passed")
