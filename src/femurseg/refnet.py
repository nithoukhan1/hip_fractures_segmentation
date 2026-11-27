"""RefFrac ("compare with a healthy reference"), Paper B: M0 plus healthy reference femurs, encoded by the same
encoder, aligned to the target femur and fused into M0's decoder and head.

Protocol: Analysis/08_Step5_PaperB_RefFrac_Protocol.md (design: Report 03 §4 and Report 04 §5; the references: Report
07 and its fixed tables in results/R00_reference_bank_r2/).

The network is M0 (src/femurseg/nets.py: U-Net with a ConvNeXt-T encoder, a binary fracture mask, and the timm
model's own classifier head on the 1/32 features). The target femur T and its references R_1..R_k go through the same
encoder (Siamese, shared weights); the references without gradient (stop-gradient: they cost forward passes only).
At 1/16 and 1/32 the target's features are replaced by

    F = W [T, T - R~, |T - R~|]     (a 1 x 1 convolution, 3C -> C)

where R~ is the reference feature aligned to each target position. W starts as [I, 0, 0], so the untrained network
computes exactly M0's function, and what the reference adds is learned. The decoder (with the target's own 1/4 and 1/8
skip features) and the classifier head then work on F. The variants (Analysis/08 §1):

  R1   AASN (Chen et al., ECCV 2020) adapted to femur crops: the nearest reference (k = 1), global affine alignment,
       concatenation, and AASN's margin loss
  R2   one random healthy reference (k = 1), local cross-attention, difference
  R3   the nearest reference (k = 1), local cross-attention, difference
  R4   the 3 nearest references, local cross-attention, difference = RefFrac
  R5   R4 with concatenation, W [T, R~], instead of the difference
  R6a  R4 without alignment: R~ = mean of the references at the same position
  R6b  R4 with global affine alignment instead of local cross-attention
  R7   R3 with the patient's own other femur (always mirrored) as the reference when it is healthy

Alignment:
  attention  one head; query W_q LN(T(p)), key W_k LN(R_j(p')) of dimension 128, value R_j(p') itself, so that R~
             stays in the encoder's feature space and T - R~ is meaningful. Each target position p attends to the
             positions p' of all k references within +- window x the image height of p (default 0.125, i.e.
             +-96 px at 768 x 512: 13 x 13 cells at 1/16 and 7 x 7 at 1/32), one softmax over all of them, so R~(p)
             is the best-matching healthy anatomy near p among the k references. window 0 = the whole reference.
  none       R~(p) = mean_j R_j(p).
  affine     per reference, a localisation network on [T_32, R_j,32] (1 x 1 convolution to 256 channels, GELU,
             3 x 3 convolution with stride 2 to 128, GELU, global average, linear to 6) gives an affine transform (a
             spatial transformer, initialised to the identity); R_j,16 and R_j,32 are resampled with it (bilinear,
             zeros outside); R~ = mean_j.
Margin loss (AASN's contrastive term): projection g = 1 x 1 convolution (C -> 128), BatchNorm, ReLU, then scaled to
length 1 at each position; d2 = |g(T_16) - g(R~_16)|^2, in [0, 2]; loss = mean over the 1/16 cells of
(1 - a) d2 + a max(0, 0.5 - d2), a = the fraction of the cell inside the released fracture mask. Weight 0.5 in R1 and
0 otherwise (--ref-margin).

Where the references come from (RefTable): the fixed table of step B-1. Policies:
  nearest    ref1..refk of the femur's row for the fold model (healthy femurs of the 4 training folds)
  random     random_healthy                      fractured  random_fractured (swap test b)
  partner    the femur's own other femur, mirrored, when it is healthy; the nearest otherwise (R7's training policy;
             it uses the partner's label, so at prediction time it is reported only to show that leak)
  blind      the partner whenever there is one, healthy or not; the nearest otherwise (no label used)
  self       the target itself (no information: how much does the decision depend on the comparison?)
"""
from __future__ import annotations

import csv
import functools
from pathlib import Path

import numpy as np

from .data import FemurRecord

VARIANTS: dict[str, dict] = {
    "R1": {"policy": "nearest", "k": 1, "align": "affine", "fusion": "concat", "margin": 0.5},
    "R2": {"policy": "random", "k": 1, "align": "attention", "fusion": "difference", "margin": 0.0},
    "R3": {"policy": "nearest", "k": 1, "align": "attention", "fusion": "difference", "margin": 0.0},
    "R4": {"policy": "nearest", "k": 3, "align": "attention", "fusion": "difference", "margin": 0.0},
    "R5": {"policy": "nearest", "k": 3, "align": "attention", "fusion": "concat", "margin": 0.0},
    "R6a": {"policy": "nearest", "k": 3, "align": "none", "fusion": "difference", "margin": 0.0},
    "R6b": {"policy": "nearest", "k": 3, "align": "affine", "fusion": "difference", "margin": 0.0},
    "R7": {"policy": "partner", "k": 1, "align": "attention", "fusion": "difference", "margin": 0.0},
}
POLICIES = ("nearest", "random", "fractured", "partner", "blind", "self")
TEST_POLICY = {"self": "self", "random": "random", "fractured": "fractured", "retrieved": "nearest", "blind": "blind"}
DEFAULT_WINDOW = 0.125    # +- 1/8 of the image height (96 px at 768 x 512)
ATTN_DIM = 128            # query/key dimension of the cross-attention
PROJ_DIM = 128            # margin loss: dimension of the projection g
MARGIN = 0.5              # AASN's margin on the squared distance
MAX_K = 5                 # the table keeps 5 nearest references per femur
REF_CHUNK = 16            # references are encoded in chunks of this many images (bounds the transient memory)


def variant_config(variant: str, k: int = 0, margin: float | None = None, window: float = DEFAULT_WINDOW) -> dict:
    """The settings of a variant, with the command-line overrides (k 0 and margin None keep the variant's)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown RefFrac variant '{variant}'; expected one of {', '.join(VARIANTS)}")
    cfg = dict(VARIANTS[variant])
    if k:
        if not 1 <= int(k) <= MAX_K:
            raise ValueError(f"--ref-k {k}: use 1..{MAX_K} (the table keeps {MAX_K} nearest references)")
        cfg["k"] = int(k)
    if margin is not None and margin >= 0:
        cfg["margin"] = float(margin)
    if window < 0 or window > 1:
        raise ValueError(f"--ref-window {window}: a fraction of the image height in [0, 1] (0 = the whole reference)")
    cfg["window"] = float(window)
    return cfg


# ------------------------------------------------------------------------------------------- the reference table


class RefTable:
    """The fixed tables of step B-1 (references.csv and pairs.csv of results/R00_reference_bank_r2): who serves as
    whose reference, for each fold model, and every femur's partner. No run retrieves anything itself.

    sources() returns, for one femur, fold model and policy, the references as ((split, file), mirrored) pairs: the
    key of the image cache and whether the image is mirrored left-right before use. Bank femurs are training femurs;
    a partner belongs to the femur's own x-ray (same split)."""

    def __init__(self, references_csv: str | Path, pairs_csv: str | Path, records: list[FemurRecord]):
        self.paths = (Path(references_csv), Path(pairs_csv))
        self.rec = {r.key: r for r in records}
        self.row: dict[tuple[int, str, str], dict] = {}
        with open(references_csv, newline="") as f:
            for r in csv.DictReader(f):
                self.row[(int(r["fold_model"]), r["split"], r["file"])] = r
        self.partner: dict[tuple[str, str], tuple[str, int]] = {}
        with open(pairs_csv, newline="") as f:
            for r in csv.DictReader(f):
                if r["partner"]:
                    self.partner[(r["split"], r["file"])] = (r["partner"], int(r["partner_cls6"]))
        self.fold_models = sorted({m for m, _, _ in self.row})
        self._validate(records)

    def _validate(self, records: list[FemurRecord]) -> None:
        """The tables must describe exactly the femurs of the fold file (same folds, classes and groups), and every
        reference must be a healthy training femur of the fold model's 4 training folds, outside the femur's group."""
        if not self.fold_models:
            raise ValueError(f"{self.paths[0]}: no rows")
        bad = []
        for m in self.fold_models:
            for rec in records:
                row = self.row.get((m, rec.split, rec.file))
                if row is None:
                    bad.append(f"fold model {m}: no row for {rec.split}/{rec.file}")
                elif (int(row["fold"]), int(row["cls6"]), row["group"]) != (rec.fold, rec.cls6, rec.group):
                    bad.append(f"fold model {m}: {rec.file} has fold/class/group {row['fold']}/{row['cls6']}/"
                               f"{row['group']} in the table but {rec.fold}/{rec.cls6}/{rec.group} in the fold file")
                else:
                    cols = [f"ref{j}" for j in range(1, MAX_K + 1)] + ["random_healthy", "random_fractured"]
                    for c in cols:
                        ref = self.rec.get(("train", row[c]))
                        want_healthy = c != "random_fractured"
                        if (ref is None or ref.fold == m or (ref.cls6 == 0) != want_healthy or ref.group == rec.group):
                            bad.append(f"fold model {m}: {c} of {rec.file} ({row[c]}) breaks the bank rules")
                if len(bad) > 5:
                    break
            if len(bad) > 5:
                break
        for (split, file), (pf, pc) in self.partner.items():
            p, me = self.rec.get((split, pf)), self.rec.get((split, file))
            if p is None or me is None or p.cls6 != pc or me.group != p.group:
                bad.append(f"pairs: partner {pf} of {file} does not match the fold file")
                break
        if bad:
            raise ValueError("the reference tables do not match the fold file:\n  " + "\n  ".join(bad[:6]))

    def sources(self, fold_model: int, rec: FemurRecord, policy: str, k: int) -> list[tuple[tuple[str, str], bool]]:
        """The references of ``rec`` for ``fold_model`` under ``policy`` (see the module docstring). nearest gives k
        references; partner and blind give the partner (if used) followed by the k - 1 nearest; random, fractured and
        self give one reference."""
        row = self.row[(fold_model, rec.split, rec.file)]

        def bank(col: str) -> tuple[tuple[str, str], bool]:
            return ("train", row[col]), row[f"{col}_mirrored"] == "1"

        nearest = [bank(f"ref{j}") for j in range(1, k + 1)]
        if policy == "nearest":
            return nearest
        if policy == "random":
            return [bank("random_healthy")]
        if policy == "fractured":
            return [bank("random_fractured")]
        if policy == "self":
            return [(rec.key, False)]
        if policy in ("partner", "blind"):
            p = self.partner.get(rec.key)
            if p is not None and (policy == "blind" or p[1] == 0):
                return [((rec.split, p[0]), True)] + nearest[:k - 1]  # the other hip: always mirrored
            return nearest
        raise ValueError(f"unknown reference policy '{policy}'; expected one of {', '.join(POLICIES)}")

    def source_name(self, rec: FemurRecord, policy: str, srcs: list) -> str:
        """What the first reference is: partner, nearest, random, fractured or self."""
        if policy in ("partner", "blind"):
            p = self.partner.get(rec.key)
            return "partner" if p is not None and srcs[0][0] == (rec.split, p[0]) else "nearest"
        return policy


def reference_image(images, row: int, mirrored: bool) -> np.ndarray:
    """One cached reference image (H, W) uint8, mirrored left-right when the table says so."""
    img = np.array(images[row])
    return np.ascontiguousarray(img[:, ::-1]) if mirrored else img


def window_cells(window: float, h: int) -> int | None:
    """Half-width of the attention window in cells of a feature grid of height h; None = the whole reference."""
    return None if window <= 0 else max(1, int(round(window * h)))


def pointing_game(diff: np.ndarray, mask: np.ndarray) -> dict:
    """Pointing game of a difference map (h, w) against the released mask at any size: does the cell with the
    largest difference touch the fracture? chance = the share of cells that touch it."""
    import cv2

    h, w = diff.shape
    grid = cv2.resize((mask > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0
    i, j = np.unravel_index(int(np.argmax(diff)), diff.shape)
    return {"diff_hit": int(grid[i, j]), "diff_chance": round(float(grid.mean()), 6),
            "diff_peak_y": round((i + 0.5) / h, 4), "diff_peak_x": round((j + 0.5) / w, 4)}


# ---------------------------------------------------------------------------------------------- the network


@functools.lru_cache(maxsize=32)
def _window_mask(h: int, w: int, radius: int, k: int, device: str):
    import torch

    ys = torch.arange(h).repeat_interleave(w)
    xs = torch.arange(w).repeat(h)
    m = ((ys[:, None] - ys[None, :]).abs() <= radius) & ((xs[:, None] - xs[None, :]).abs() <= radius)
    return m.repeat(1, k).to(device)  # (L, k L): reference j's positions are the columns j L .. (j + 1) L - 1


def window_mask(h: int, w: int, radius: int, k: int, device):
    """(h w, k h w) bool: target position p may attend to position p' of every reference iff |dy|, |dx| <= radius.
    Built once per grid, window, k and device."""
    return _window_mask(h, w, radius, k, str(device))


def cross_attend(t, r, norm, wq, wk, radius: int | None):
    """Local cross-attention of the target features t (B, C, h, w) over the references r (B, k, C, h, w).
    Returns the aligned reference features (B, C, h, w) and the attention weights (B, h w, k h w)."""
    import torch

    b, c, h, w = t.shape
    k = r.shape[1]
    tq = t.flatten(2).transpose(1, 2)                         # (B, L, C)
    rv = r.flatten(3).transpose(2, 3).reshape(b, k * h * w, c)  # (B, k L, C)
    q, key = wq(norm(tq)), wk(norm(rv))
    s = torch.matmul(q, key.transpose(1, 2)) * (q.shape[-1] ** -0.5)
    if radius is not None:
        s = s.masked_fill(~window_mask(h, w, radius, k, s.device), float("-inf"))
    a = torch.softmax(s.float(), dim=-1)
    out = torch.matmul(a.to(rv.dtype), rv)                    # (B, L, C)
    return out.transpose(1, 2).reshape(b, c, h, w), a


def build_refnet(arch: str, variant: str, pretrained: bool = True, drop_rate: float = 0.2, k: int = 0,
                 margin: float | None = None, window: float = DEFAULT_WINDOW):
    """Create RefFrac variant ``variant`` (R1..R7) on the timm model ``arch`` (downloads weights when
    ``pretrained``). k, margin and window override the variant's settings (variant_config)."""
    import segmentation_models_pytorch as smp
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = variant_config(variant, k, margin, window)

    class Fusion(nn.Module):
        """F = W [T, T - R, |T - R|] (difference) or W [T, R] (concatenation); W starts as [I, 0, ...]."""

        def __init__(self, c: int, kind: str):
            super().__init__()
            self.kind = kind
            self.conv = nn.Conv2d((3 if kind == "difference" else 2) * c, c, 1)
            with torch.no_grad():
                self.conv.weight.zero_()
                self.conv.bias.zero_()
                self.conv.weight[:, :c, 0, 0] = torch.eye(c)

        def forward(self, t, r):
            if self.kind == "difference":
                d = t - r
                return self.conv(torch.cat([t, d, d.abs()], dim=1))
            return self.conv(torch.cat([t, r], dim=1))

    class Attention(nn.Module):
        def __init__(self, c: int):
            super().__init__()
            self.norm = nn.LayerNorm(c)
            self.q = nn.Linear(c, ATTN_DIM)
            self.k = nn.Linear(c, ATTN_DIM)

    class AffineAlign(nn.Module):
        """Global affine alignment of each reference (a spatial transformer, Jaderberg et al. 2015)."""

        def __init__(self, c32: int):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(2 * c32, 256, 1), nn.GELU(),
                                     nn.Conv2d(256, 128, 3, stride=2, padding=1), nn.GELU(),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 6))
            with torch.no_grad():
                self.net[-1].weight.zero_()
                self.net[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

        def forward(self, t32, rs):
            """t32 (B, C, h, w); rs = [r16, r32], each (B, k, C, h, w) -> aligned means [(B, C, h, w)] x 2, theta."""
            b, k = rs[1].shape[:2]
            tt = t32.float().unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(b * k, *t32.shape[1:])
            theta = self.net(torch.cat([tt, rs[1].float().reshape(b * k, *rs[1].shape[2:])], dim=1)).view(b * k, 2, 3)
            out = []
            for r in rs:
                rf = r.float().reshape(b * k, *r.shape[2:])
                grid = F.affine_grid(theta, list(rf.shape), align_corners=False)
                warped = F.grid_sample(rf, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                out.append(warped.view(b, k, *rf.shape[1:]).mean(dim=1))
            return out, theta.view(b, k, 2, 3)

    class RefNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.variant, self.cfg = variant, dict(cfg)
            # the M0 trunk: same encoder, decoder, mask head and classifier head as nets.build_net (m0)
            self.unet = smp.Unet(encoder_name=f"tu-{arch}", encoder_weights="imagenet" if pretrained else None,
                                 in_channels=3, classes=1)
            head = getattr(timm.create_model(arch, pretrained=pretrained, num_classes=6, drop_rate=drop_rate),
                           "head", None)
            if head is None or not hasattr(head, "global_pool"):
                raise ValueError(f"ref:{arch}: the timm model has no pooled classifier head ('head' with global_pool)")
            self.unet.classification_head = head
            chans = self.unet.encoder.out_channels
            self.levels = (len(chans) - 2, len(chans) - 1)  # the 1/16 and 1/32 features
            cs = [chans[i] for i in self.levels]
            self.fuse = nn.ModuleList([Fusion(c, cfg["fusion"]) for c in cs])
            if cfg["align"] == "attention":
                self.attn = nn.ModuleList([Attention(c) for c in cs])
            elif cfg["align"] == "affine":
                self.affine = AffineAlign(cs[1])
            if cfg["margin"] > 0:
                self.proj = nn.Sequential(nn.Conv2d(cs[0], PROJ_DIM, 1), nn.BatchNorm2d(PROJ_DIM), nn.ReLU())

        def encode_references(self, refs):
            """refs (B k, 3, H, W) -> their 1/16 and 1/32 features, with the shared encoder and no gradient."""
            outs: list[list] = [[], []]
            with torch.no_grad():
                for chunk in refs.split(REF_CHUNK):
                    f = self.unet.encoder(chunk)
                    for i, lv in enumerate(self.levels):
                        outs[i].append(f[lv])
            return [torch.cat(o) for o in outs]

        def align(self, ts, rs):
            """ts = [T_16, T_32] (B, C, h, w); rs = [R_16, R_32] (B, k, C, h, w) -> [R~_16, R~_32], extra outputs."""
            if self.cfg["align"] == "none":
                return [r.mean(dim=1) for r in rs], {}
            if self.cfg["align"] == "affine":
                with torch.autocast(device_type=ts[0].device.type, enabled=False):
                    out, theta = self.affine(ts[1], rs)
                return [o.to(t.dtype) for o, t in zip(out, ts)], {"theta": theta}
            out, extra = [], {}
            for i, (t, r) in enumerate(zip(ts, rs)):
                at = self.attn[i]
                o, a = cross_attend(t, r, at.norm, at.q, at.k, window_cells(self.cfg["window"], t.shape[-2]))
                out.append(o)
                extra[f"attn{i}"] = a
            return out, extra

        def forward(self, x, refs=None, keep_attention: bool = False):
            """x (B, 3, H, W) the target; refs (B k, 3, H, W) its k references, femur by femur (k is read from the
            shapes, so it may differ from the k of training). Without refs the network is exactly M0."""
            feats = list(self.unet.encoder(x))
            out = {}
            if refs is not None:
                b = x.shape[0]
                if refs.shape[0] % b:
                    raise ValueError(f"{refs.shape[0]} reference images for {b} femurs")
                kk = refs.shape[0] // b
                rs = [r.reshape(b, kk, *r.shape[1:]) for r in self.encode_references(refs)]
                ts = [feats[lv] for lv in self.levels]
                aligned, extra = self.align(ts, rs)
                for i, lv in enumerate(self.levels):
                    feats[lv] = self.fuse[i](ts[i], aligned[i])
                with torch.autocast(device_type=x.device.type, enabled=False):
                    out["diff16"] = (ts[0].float() - aligned[0].float()).norm(dim=1)  # pointing-game map
                    if self.cfg["margin"] > 0:
                        g = F.normalize(self.proj(torch.cat([ts[0], aligned[0]]).float()), dim=1)
                        out["margin_d2"] = (g[:b] - g[b:]).pow(2).sum(dim=1)
                if keep_attention:
                    out.update(extra)
            out["seg"] = self.unet.segmentation_head(self.unet.decoder(feats))
            out["cls"] = self.unet.classification_head(feats[-1])
            return out

    return RefNet()


def make_ref_loss(margin_weight: float = 0.0, seg_weight: float = 1.0):
    """Loss (outputs, mask (B, H, W) long, cls6 (B,) long) -> (total, {part: value}): M0's loss (cross-entropy of the
    6 classes + seg_weight x (binary cross-entropy + Dice of the fracture mask)) + margin_weight x the margin loss."""
    import torch.nn.functional as F
    from segmentation_models_pytorch.losses import DiceLoss

    dice_bin = DiceLoss(mode="binary", from_logits=True)

    def loss(out: dict, mask, y):
        parts = {"ce": F.cross_entropy(out["cls"].float(), y)}
        seg = out["seg"].float()
        target = (mask > 0).float().unsqueeze(1)
        parts["seg_bce"] = F.binary_cross_entropy_with_logits(seg, target)
        parts["seg_dice"] = dice_bin(seg, target)
        total = parts["ce"] + seg_weight * (parts["seg_bce"] + parts["seg_dice"])
        if margin_weight > 0 and "margin_d2" in out:
            d2 = out["margin_d2"].float()
            a = F.adaptive_avg_pool2d(target, d2.shape[-2:])[:, 0]  # share of each cell inside the fracture
            parts["margin"] = ((1 - a) * d2 + a * F.relu(MARGIN - d2)).mean()
            total = total + margin_weight * parts["margin"]
        return total, {k: float(v.detach()) for k, v in parts.items()}

    return loss
