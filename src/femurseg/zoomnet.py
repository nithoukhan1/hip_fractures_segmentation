"""Segment-and-zoom network: the ConvNeXt U-Net finds the fracture, and a second encoder grades it from a
full-resolution crop around the found region (Analysis/11, method candidate 1; screening rules in its dated section
"Fold-0 screening rules").

Why: the ConvNeXt U-Net sees the femur at half resolution (768 x 512) and most of its out-of-fold errors are grade
errors (65 of 121): undisplaced and displaced neck fractures, and stable and unstable pertrochanteric fractures,
differ in details that are half a millimetre wide. The mask already tells where to look, so this network looks there
again at the full resolution of the release.

Input: the whole femur at full resolution, (B, 3, H, W) = 1536 x 1024 (``--size 1536x1024``).

  1. Global branch = the ConvNeXt U-Net unchanged, on the image averaged down by 2 (2 x 2 average = the area
     resize of the ConvNeXt U-Net's own cache): encoder, U-Net decoder, fracture mask, and its 6-class head. The mask
     logits are resized back to the input size, so the mask loss and Dice are at full resolution.
  2. Crop centre = the probability-weighted centroid of the predicted mask (pixels with p >= 0.5). Without such
     pixels (most healthy femurs): the centroid of the K most probable pixels, so every femur gets one crop at its
     most suspicious place. The centre does not carry gradients.
     Training: with probability tf_prob (teacher forcing, falling from 1 to 0 over --tf-epochs as in SLG-Net) the
     centroid of the released mask is used instead (femurs with a fracture only), and the centre is moved by up to
     +-JITTER pixels.
  3. Zoom branch: a WINDOW x WINDOW square of the full-resolution image around the centre (kept inside the image),
     resampled to ZOOM_INPUT x ZOOM_INPUT, through a second encoder of the same timm model (own ImageNet weights,
     global average pool and LayerNorm).
  4. Decision: LayerNorm of [global features, zoom features] (768 + 768 for ConvNeXt-T), dropout, linear, 6 classes.

Outputs: "seg" (B, 1, H, W) mask logits, "cls" (B, 6) the decision, "cls_global" (B, 6) the global head alone (an
auxiliary loss keeps it a full ConvNeXt U-Net), "centre" (B, 2) crop centre (y, x) in input pixels. "cls" is used as
the ConvNeXt U-Net's head is, so nets.postprocess treats this network like the ConvNeXt U-Net.

Loss: cross-entropy of "cls" + AUX_WEIGHT x cross-entropy of "cls_global" + the ConvNeXt U-Net's mask loss
(binary cross-entropy + Dice).
"""
from __future__ import annotations

WINDOW = 640        # side of the crop in full-resolution pixels (5/12 of the 1536-pixel height)
ZOOM_INPUT = 512    # the crop is resampled to this size for the zoom encoder (0.8 x full resolution)
JITTER = 32         # training: the crop centre moves by up to this many pixels
AUX_WEIGHT = 0.5    # weight of the global head's cross-entropy
TOPK_HALF = 500     # fallback centre: this many pixels at 768 x 512 (the ConvNeXt U-Net's TOPK), scaled


def mask_centre(p, topk: int, thr: float = 0.5):
    """p (B, 1, H, W) probabilities -> (B, 2) centre (y, x) in pixels: the p-weighted centroid of p >= thr, or of the
    ``topk`` most probable pixels where no pixel reaches thr."""
    import torch

    b, _, h, w = p.shape
    flat = p.float().flatten(1)                                         # (B, HW)
    k = min(topk, flat.shape[1])
    top_idx = flat.topk(k, dim=1).indices
    top_w = torch.zeros_like(flat).scatter_(1, top_idx, 1.0) * flat.clamp_min(1e-6)
    above = flat * (flat >= thr)
    has = above.sum(dim=1, keepdim=True) > 0
    wts = torch.where(has, above, top_w)
    wts = wts / wts.sum(dim=1, keepdim=True)
    ys = torch.arange(h, device=p.device, dtype=flat.dtype).repeat_interleave(w)
    xs = torch.arange(w, device=p.device, dtype=flat.dtype).repeat(h)
    return torch.stack([(wts * ys).sum(1), (wts * xs).sum(1)], dim=1) + 0.5  # pixel centres


def clamp_centre(c, hw: tuple[int, int], window: int):
    """Keep the window inside the image (an image smaller than the window: centre of the image)."""
    import torch

    h, w = hw
    half = window / 2
    cy = c[:, 0].clamp(min(half, h / 2), max(h - half, h / 2))
    cx = c[:, 1].clamp(min(half, w / 2), max(w - half, w / 2))
    return torch.stack([cy, cx], dim=1)


def crop(x, centre, window: int, out: int):
    """Square crops (B, C, out, out) of x (B, C, H, W): side ``window`` pixels around ``centre`` (B, 2) (y, x),
    bilinear resampling (align_corners=False pixel geometry)."""
    import torch
    import torch.nn.functional as F

    b, _, h, w = x.shape
    theta = torch.zeros(b, 2, 3, device=x.device, dtype=torch.float32)
    theta[:, 0, 0] = window / w
    theta[:, 1, 1] = window / h
    theta[:, 0, 2] = centre[:, 1].float() * 2 / w - 1
    theta[:, 1, 2] = centre[:, 0].float() * 2 / h - 1
    grid = F.affine_grid(theta, (b, x.shape[1], out, out), align_corners=False)
    return F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border", align_corners=False).to(x.dtype)


def window_hit(centre_yx, window: int, gt) -> dict:
    """Explainability of one femur's crop against its released mask gt (H, W) bool at the input size: whether the
    window holds the mask's centroid (pointing game), and the share of the mask inside the window."""
    import numpy as np

    ys, xs = np.nonzero(gt)
    if len(ys) == 0:
        return {}
    cy, cx = centre_yx
    y0, x0 = cy - window / 2, cx - window / 2
    inside = (ys + 0.5 >= y0) & (ys + 0.5 < y0 + window) & (xs + 0.5 >= x0) & (xs + 0.5 < x0 + window)
    gy, gx = ys.mean() + 0.5, xs.mean() + 0.5
    hit = bool(y0 <= gy < y0 + window and x0 <= gx < x0 + window)
    return {"zoom_hit": int(hit), "zoom_mask_inside": round(float(inside.mean()), 4)}


def build_zoomnet(arch: str, pretrained: bool = True, drop_rate: float = 0.2, window: int = WINDOW,
                  zoom_input: int = ZOOM_INPUT, jitter: int = JITTER):
    """Create the segment-and-zoom network on the timm model ``arch`` (downloads weights when ``pretrained``)."""
    import segmentation_models_pytorch as smp
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ZoomNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.window, self.zoom_input, self.jitter = window, zoom_input, jitter
            # the ConvNeXt U-Net, built exactly as nets.build_net (m0)
            self.unet = smp.Unet(encoder_name=f"tu-{arch}", encoder_weights="imagenet" if pretrained else None,
                                 in_channels=3, classes=1)
            head = getattr(timm.create_model(arch, pretrained=pretrained, num_classes=6, drop_rate=drop_rate),
                           "head", None)
            if head is None or not hasattr(head, "norm") or not hasattr(head, "global_pool"):
                raise ValueError(f"zoom:{arch}: the timm model has no pooled, normalised classifier head")
            self.head = head
            self.zoom = timm.create_model(arch, pretrained=pretrained, num_classes=0)  # pooled + LayerNorm features
            d = self.unet.encoder.out_channels[-1] + self.zoom.num_features
            self.fuse_norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(drop_rate)
            self.fc = nn.Linear(d, 6)

        def forward(self, x, gt=None, tf_prob: float = 0.0):
            """x (B, 3, H, W) at full resolution; gt (B, 1, H, W) released mask in {0, 1} (training: teacher forcing
            with probability tf_prob per femur; tf_prob >= 1: always the released mask, the oracle test)."""
            h, w = x.shape[-2:]
            half = F.avg_pool2d(x, 2)
            feats = self.unet.encoder(half)
            seg_half = self.unet.segmentation_head(self.unet.decoder(feats))
            seg = F.interpolate(seg_half, size=(h, w), mode="bilinear", align_corners=False)
            cls_global = self.head(feats[-1])
            with torch.autocast(device_type=x.device.type, enabled=False):
                p = torch.sigmoid(seg_half.detach().float())
                centre = mask_centre(p, TOPK_HALF) * 2                    # half-size pixels -> input pixels
                if gt is not None and tf_prob > 0:
                    g = gt.float()
                    has = g.flatten(1).sum(1) > 0
                    if has.any():
                        c_gt = mask_centre(g, 1)                          # weights = 1 inside the released mask
                        use = has & ((torch.rand(has.shape[0], device=x.device) < tf_prob) if tf_prob < 1 else has)
                        centre = torch.where(use[:, None], c_gt, centre)
                if self.training and self.jitter > 0:
                    centre = centre + (torch.rand_like(centre) * 2 - 1) * self.jitter
                centre = clamp_centre(centre, (h, w), self.window)
            z = self.zoom(crop(x, centre, self.window, self.zoom_input))
            c = self.head.norm(self.head.global_pool(feats[-1])).flatten(1)
            fused = self.fuse_norm(torch.cat([c.float(), z.float()], dim=1))
            return {"seg": seg, "cls": self.fc(self.drop(fused)), "cls_global": cls_global, "centre": centre}

    return ZoomNet()


def make_zoom_loss(seg_weight: float = 1.0, aux_weight: float = AUX_WEIGHT):
    """(outputs, mask (B, H, W) long, cls6 (B,) long) -> (total, {part: value})."""
    import torch.nn.functional as F
    from segmentation_models_pytorch.losses import DiceLoss

    dice_bin = DiceLoss(mode="binary", from_logits=True)

    def loss(out: dict, mask, y):
        seg = out["seg"].float()
        target = (mask > 0).float().unsqueeze(1)
        parts = {"ce": F.cross_entropy(out["cls"].float(), y),
                 "ce_global": F.cross_entropy(out["cls_global"].float(), y),
                 "seg_bce": F.binary_cross_entropy_with_logits(seg, target),
                 "seg_dice": dice_bin(seg, target)}
        total = parts["ce"] + aux_weight * parts["ce_global"] + seg_weight * (parts["seg_bce"] + parts["seg_dice"])
        return total, {k: float(v.detach()) for k, v in parts.items()}

    return loss
