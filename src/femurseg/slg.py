"""SLG-Net ("Segment, Locate, Grade"), Paper A: M0 plus heads that make the decision from the predicted fracture region.

The network is M0 (src/femurseg/nets.py: U-Net with a ConvNeXt-T encoder and a binary fracture mask) with its flat
6-class head replaced. Each variant adds one component to the one before (Report 03 §3 and §7.2, Report 04 §4,
protocol in Analysis/06):

  A1 = M0      flat 6-class head on the global context c (average of the 1/32 features, the classifier's LayerNorm)
  A2  + region pooling (C2): flat head on [c, r]; r = the 1/16 and 1/32 encoder features averaged inside the mask,
        each LayerNorm-ed and multiplied by the mask confidence q (so a femur without a predicted region gets r = 0)
  A3  + region geometry (C3): flat head on [c, r, g]; g = 8 differentiable numbers of the mask (below), embedded
  A4  + hierarchy (C4): P(fracture) <- [c, g]; P(location | fracture) <- [c, r, g]; P(high grade | location) <- [r, g]
        with one grade trunk per location (neck, pertrochanteric); 6-class probabilities by the chain rule
  A5  + shared severity trunk (C5): one grade trunk for both locations, which gets a location embedding
  A6  + logit adjustment (C6, Menon et al. 2021) at the location and grade nodes during training = SLG-Net

Chain rule (always valid, consistent 6-class probabilities; the release does not grade subtrochanteric fractures):
  P(no fracture) = 1 - p_f
  P(neck, Garden I-II)      = p_f p_loc(neck) (1 - p_sev(neck))     P(neck, Garden III-IV)   = p_f p_loc(neck) p_sev(neck)
  P(pertroch, Evans I-II)   = p_f p_loc(per) (1 - p_sev(per))       P(pertroch, Evans III-V) = p_f p_loc(per) p_sev(per)
  P(subtrochanteric)        = p_f p_loc(sub)
The network returns these as log-probabilities in out["cls"], so softmax(out["cls"]) gives them back and every
decision rule and metric of M0 applies unchanged (score = 1 - P(no fracture) = p_f).

Mask fed to the heads (teacher forcing, C7): during training each femur uses the released mask with probability
tf_prob (scheduled by the training script, falling from 1 to 0) and its own predicted mask otherwise; at prediction
time always the predicted mask, except in the oracle test (the released mask). With detach_mask the decision losses
do not reach the mask (the mask is then learned from the mask loss only).

Geometry g (from the soft mask m at the input size, coordinates scaled to [0, 1]):
  q     mask confidence: mean of the top-K pixel probabilities (K = 500 at 768 x 512, a robust maximum)
  pbar  mean probability inside the soft region (sum m^2 / sum m)
  la    log10(area fraction + 1e-4) / 4
  cy cx centroid
  h w   4 x the standard deviations along y and x (the full axes of the equivalent ellipse)
  rho   correlation of y and x inside the region (orientation)
An empty mask is pulled to the image centre with zero spread (a pseudo-mass of 100 pixels at 768 x 512).
"""
from __future__ import annotations

VARIANTS: dict[str, dict[str, bool]] = {
    "A2": {"region": True, "geometry": False, "hier": False, "shared": False, "la": False},
    "A3": {"region": True, "geometry": True, "hier": False, "shared": False, "la": False},
    "A4": {"region": True, "geometry": True, "hier": True, "shared": False, "la": False},
    "A5": {"region": True, "geometry": True, "hier": True, "shared": True, "la": False},
    "A6": {"region": True, "geometry": True, "hier": True, "shared": True, "la": True},
}
LOC_OF_CLS6 = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}   # 0 = neck, 1 = pertrochanteric, 2 = subtrochanteric
HIGH_CLS6 = (2, 4)                              # Garden III-IV, Evans III-V
REF_PIXELS = 768 * 512                          # K and the pseudo-mass are set for this input size, scaled otherwise
TOPK_REF, EMPTY_MASS_REF = 500, 100.0
N_GEOM = 8
GEOM_NAMES = ("q", "pbar", "log_area", "cy", "cx", "h", "w", "rho")


def priors_from_counts(counts: dict[int, int]) -> dict[str, list[float]]:
    """Training-fold class frequencies -> the log-priors of logit adjustment: log P(location | fracture) (3) and the
    log-odds of a high grade at the neck and pertrochanteric nodes (2). ``counts`` maps cls6 -> number of femurs."""
    import math

    n = {c: max(int(counts.get(c, 0)), 1) for c in range(1, 6)}  # never log(0)
    loc = [n[1] + n[2], n[3] + n[4], n[5]]
    tot = sum(loc)
    return {"log_prior_loc": [math.log(v / tot) for v in loc],
            "log_odds_high": [math.log(n[2] / n[1]), math.log(n[4] / n[3])]}


def mask_confidence(m, k: int):
    """Mean of the k highest pixel probabilities of m (B, 1, H, W) -> (B,)."""
    flat = m.flatten(1)
    return flat.topk(min(k, flat.shape[1]), dim=1).values.mean(dim=1)


def geometry(m, q, empty_mass: float):
    """The 8 geometry numbers (B, 8) of the soft mask m (B, 1, H, W) in float32 (see the module docstring)."""
    import torch

    mm = m[:, 0].float()
    _, h, w = mm.shape
    ys = (torch.arange(h, device=mm.device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=mm.device, dtype=torch.float32) + 0.5) / w
    mass = mm.sum(dim=(1, 2))
    den = mass + empty_mass
    cy = ((mm.sum(dim=2) * ys).sum(dim=1) + 0.5 * empty_mass) / den
    cx = ((mm.sum(dim=1) * xs).sum(dim=1) + 0.5 * empty_mass) / den
    dy = ys[None, :, None] - cy[:, None, None]
    dx = xs[None, None, :] - cx[:, None, None]
    vy = (mm * dy * dy).sum(dim=(1, 2)) / den
    vx = (mm * dx * dx).sum(dim=(1, 2)) / den
    cov = (mm * dy * dx).sum(dim=(1, 2)) / den
    sy, sx = (vy + 1e-8).sqrt(), (vx + 1e-8).sqrt()
    rho = (cov / (sy * sx + 1e-4)).clamp(-1.0, 1.0)
    pbar = (mm * mm).sum(dim=(1, 2)) / den
    log_area = torch.log10(mass / (h * w) + 1e-4) / 4.0
    return torch.stack([q.float(), pbar, log_area, cy, cx, 4 * sy, 4 * sx, rho], dim=1)


def region_pool(feat, m):
    """Average of feat (B, C, h, w) inside the soft mask m (B, 1, H, W), in float32 -> (B, C). The mask is first
    averaged to the feature grid (the fraction of each cell inside the region)."""
    import torch.nn.functional as F

    ms = F.adaptive_avg_pool2d(m.float(), feat.shape[-2:])
    mass = ms.sum(dim=(2, 3))
    return (feat.float() * ms).sum(dim=(2, 3)) / mass.clamp_min(1e-3)


def chain_logp(z_f, z_loc, z_sev):
    """6-class log-probabilities (B, 6) from the node logits: fracture (B,), location (B, 3), high grade at the neck
    and pertrochanteric nodes (B, 2)."""
    import torch
    import torch.nn.functional as F

    lf, lnf = F.logsigmoid(z_f), F.logsigmoid(-z_f)
    lloc = F.log_softmax(z_loc, dim=1)
    lhi, llo = F.logsigmoid(z_sev), F.logsigmoid(-z_sev)
    return torch.stack([lnf,
                        lf + lloc[:, 0] + llo[:, 0], lf + lloc[:, 0] + lhi[:, 0],
                        lf + lloc[:, 1] + llo[:, 1], lf + lloc[:, 1] + lhi[:, 1],
                        lf + lloc[:, 2]], dim=1)


def build_slg(arch: str, variant: str, pretrained: bool = True, drop_rate: float = 0.2, scales=(16, 32),
              detach_mask: bool = False, hidden: int = 256):
    """Create SLG-Net variant ``variant`` (A2..A6) on the timm model ``arch`` (downloads weights when ``pretrained``)."""
    import segmentation_models_pytorch as smp
    import timm
    import torch
    import torch.nn as nn

    if variant not in VARIANTS:
        raise ValueError(f"unknown SLG-Net variant '{variant}'; expected one of {', '.join(VARIANTS)}")
    cfg = VARIANTS[variant]
    if cfg["hier"] and not cfg["geometry"]:
        raise ValueError(f"{variant}: the hierarchical heads use the geometry embedding")
    scales = tuple(sorted({int(s) for s in scales}))
    if not scales or any(s not in (16, 32) for s in scales):
        raise ValueError(f"region-pooling scales {scales}: use 16 and/or 32")

    class SLGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.variant, self.cfg, self.scales, self.detach_mask = variant, dict(cfg), scales, detach_mask
            # the M0 trunk: same encoder, decoder and mask head as smp.Unet in nets.build_net (m0)
            self.unet = smp.Unet(encoder_name=f"tu-{arch}", encoder_weights="imagenet" if pretrained else None,
                                 in_channels=3, classes=1)
            head = getattr(timm.create_model(arch, pretrained=pretrained, num_classes=6, drop_rate=drop_rate),
                           "head", None)
            if head is None or not hasattr(head, "norm") or not hasattr(head, "global_pool"):
                raise ValueError(f"slg:{arch}: the timm model has no pooled, normalised classifier head")
            self.ctx_pool, self.ctx_norm = head.global_pool, head.norm  # M0's global context path (pretrained norm)
            chans = self.unet.encoder.out_channels
            self.level = {16: len(chans) - 2, 32: len(chans) - 1}  # index of the 1/16 and 1/32 features
            d_c = chans[-1]
            d_r = sum(chans[self.level[s]] for s in scales)
            self.region_norm = nn.ModuleDict({str(s): nn.LayerNorm(chans[self.level[s]]) for s in scales})
            d_g = 64 if cfg["geometry"] else 0
            if cfg["geometry"]:
                self.geom_embed = nn.Sequential(nn.Linear(N_GEOM, d_g), nn.GELU())
            self.drop = nn.Dropout(drop_rate)
            if not cfg["hier"]:
                self.fc = nn.Linear(d_c + d_r + d_g, 6)
            else:
                self.head_frac = nn.Linear(d_c + d_g, 1)
                self.head_loc = nn.Linear(d_c + d_r + d_g, 3)
                if cfg["shared"]:
                    self.loc_embed = nn.Embedding(2, 16)
                    self.sev_trunk = nn.Linear(d_r + d_g + 16, hidden)
                    self.sev_out = nn.Linear(hidden, 1)
                else:
                    self.sev_trunks = nn.ModuleList([nn.Linear(d_r + d_g, hidden) for _ in range(2)])
                    self.sev_outs = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(2)])
            self.act = nn.GELU()

        def forward(self, x, gt=None, tf_prob: float = 0.0):
            """x (B, 3, H, W); gt (B, 1, H, W) released mask in {0, 1} (training: teacher forcing with probability
            tf_prob per femur; tf_prob >= 1: always the released mask, i.e. the oracle test)."""
            feats = self.unet.encoder(x)
            seg = self.unet.segmentation_head(self.unet.decoder(feats))
            out = {"seg": seg}
            with torch.autocast(device_type=x.device.type, enabled=False):
                m = torch.sigmoid(seg.float())
                if self.detach_mask:
                    m = m.detach()
                if gt is not None and tf_prob > 0:
                    gt = gt.float()
                    if tf_prob >= 1:
                        m = gt
                    else:
                        use = (torch.rand(m.shape[0], 1, 1, 1, device=m.device) < tf_prob).float()
                        m = use * gt + (1 - use) * m
                npix = m.shape[-2] * m.shape[-1]
                q = mask_confidence(m, max(1, round(TOPK_REF * npix / REF_PIXELS)))
                c = self.ctx_norm(self.ctx_pool(feats[-1].float())).flatten(1)
                r = torch.cat([q[:, None] * self.region_norm[str(s)](region_pool(feats[self.level[s]], m))
                               for s in self.scales], dim=1)
                parts = [c, r]
                if self.cfg["geometry"]:
                    g = geometry(m, q, EMPTY_MASS_REF * npix / REF_PIXELS)
                    e_g = self.geom_embed(g)
                    parts.append(e_g)
                    out["geom"] = g
                if not self.cfg["hier"]:
                    out["cls"] = self.fc(self.drop(torch.cat(parts, dim=1)))
                    return out
                z_f = self.head_frac(self.drop(torch.cat([c, e_g], dim=1)))[:, 0]
                z_loc = self.head_loc(self.drop(torch.cat([c, r, e_g], dim=1)))
                rg = self.drop(torch.cat([r, e_g], dim=1))
                if self.cfg["shared"]:
                    z_sev = torch.cat([self.sev_out(self.act(self.sev_trunk(torch.cat(
                        [rg, self.loc_embed.weight[i].expand(rg.shape[0], -1)], dim=1)))) for i in range(2)], dim=1)
                else:
                    z_sev = torch.cat([self.sev_outs[i](self.act(self.sev_trunks[i](rg))) for i in range(2)], dim=1)
                out.update(z_f=z_f, z_loc=z_loc, z_sev=z_sev, cls=chain_logp(z_f, z_loc, z_sev))
            return out

    return SLGNet()


def make_slg_loss(variant: str, seg_weight: float = 1.0, joint_weight: float = 0.5, priors: dict | None = None,
                  la_tau: float = 1.0):
    """Loss (outputs, mask (B, H, W) long, cls6 (B,) long) -> (total, {part: value}).

    Mask: binary cross-entropy + Dice on every femur (as M0). Flat variants (A2, A3): cross-entropy of the 6 classes.
    Hierarchical variants (A4-A6): fracture BCE (all femurs) + location cross-entropy (fractured femurs) + grade BCE
    (neck and pertrochanteric fractures, at their own location) + joint_weight x the negative log-likelihood of the
    chain-rule 6-class probability (all femurs). A6 adds la_tau x the training-fold log-priors to the location and
    grade logits in these losses only (logit adjustment); predictions use the plain logits."""
    import torch
    import torch.nn.functional as F
    from segmentation_models_pytorch.losses import DiceLoss

    cfg = VARIANTS[variant]
    dice_bin = DiceLoss(mode="binary", from_logits=True)
    adj_loc = adj_sev = None
    if cfg["la"] and priors is not None:
        adj_loc = torch.tensor(priors["log_prior_loc"], dtype=torch.float32) * la_tau
        adj_sev = torch.tensor(priors["log_odds_high"], dtype=torch.float32) * la_tau
    loc_map = torch.tensor([-1, 0, 0, 1, 1, 2])
    high_map = torch.tensor([0, 0, 1, 0, 1, 0])

    def loss(out: dict, mask, y):
        parts = {}
        seg = out["seg"].float()
        target = (mask > 0).float().unsqueeze(1)
        parts["seg_bce"] = F.binary_cross_entropy_with_logits(seg, target)
        parts["seg_dice"] = dice_bin(seg, target)
        if not cfg["hier"]:
            parts["ce"] = F.cross_entropy(out["cls"].float(), y)
            total = parts["ce"] + seg_weight * (parts["seg_bce"] + parts["seg_dice"])
            return total, {k: float(v.detach()) for k, v in parts.items()}
        z_f, z_loc, z_sev = out["z_f"].float(), out["z_loc"].float(), out["z_sev"].float()
        if adj_loc is not None:
            z_loc = z_loc + adj_loc.to(z_loc.device)
            z_sev = z_sev + adj_sev.to(z_sev.device)
        fx = y > 0
        loc = loc_map.to(y.device)[y]
        parts["frac_bce"] = F.binary_cross_entropy_with_logits(z_f, fx.float())
        parts["loc_ce"] = F.cross_entropy(z_loc[fx], loc[fx]) if fx.any() else z_loc.sum() * 0.0
        graded = fx & (loc < 2)
        if graded.any():
            sel = z_sev[graded].gather(1, loc[graded][:, None])[:, 0]
            parts["grade_bce"] = F.binary_cross_entropy_with_logits(sel, high_map.to(y.device)[y][graded].float())
        else:
            parts["grade_bce"] = z_sev.sum() * 0.0
        parts["joint_nll"] = F.nll_loss(chain_logp(z_f, z_loc, z_sev), y)
        total = (parts["frac_bce"] + parts["loc_ce"] + parts["grade_bce"] + joint_weight * parts["joint_nll"]
                 + seg_weight * (parts["seg_bce"] + parts["seg_dice"]))
        return total, {k: float(v.detach()) for k, v in parts.items()}

    return loss
