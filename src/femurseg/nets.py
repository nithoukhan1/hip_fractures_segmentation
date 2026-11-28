"""The PyTorch networks of S04-S10 (and the starting point of Papers A and B), their losses and their outputs.

A network is named by a ``--net`` string:

  cls:<timm model>            image classifier, 6 outputs                      S04 DenseNet121, S05 EfficientNet-B4,
                                                                               S06 ConvNeXt-T (= A0)
  seg:<smp architecture>:<encoder>                                            S07 U-Net, S08 DeepLabV3+ (ResNet34),
                              semantic segmentation, 6 classes per pixel      S09 SegFormer (MiT-B2)
                              (0 = background, 1..5 = the femur's class)
  m0:<timm model>             M0 (= A1 of Paper A = R0 of Paper B): U-Net with the timm model as encoder, a binary
                              fracture mask, and the timm model's own classifier head (global pooling, norm, dropout,
                              6 outputs) on the encoder's last feature map. With the same timm model as an S06-type
                              classifier, the classification path is identical; M0 only adds the decoder and its loss.
  slg:<timm model>:<variant>  SLG-Net of Paper A (src/femurseg/slg.py): M0's encoder, decoder and mask, with heads that
                              decide from the predicted region; variant A2..A6 (the ablation chain, Analysis/06). Its
                              "cls" output holds 6-class log-probabilities (chain rule in A4-A6), so the M0 rule below
                              applies unchanged.
  ref:<timm model>:<variant>  RefFrac of Paper B (src/femurseg/refnet.py): M0 plus healthy reference femurs encoded by
                              the same encoder, aligned and fused into M0's decoder and head; variant R1..R7
                              (Analysis/08). Its outputs are M0's, so the M0 rule below applies unchanged.
  zoom:<timm model>           segment-and-zoom network (src/femurseg/zoomnet.py, Analysis/11): M0 on the image at half
                              resolution finds the fracture; a second encoder of the same timm model grades it from a
                              full-resolution crop around the predicted mask. Input at full resolution (--size
                              1536x1024). Its "cls" is a 6-class decision, so the M0 rule below applies unchanged.

Every network returns a dict with "cls" (B, 6) logits and/or "seg" (B, C, H, W) logits at the input size.

From outputs to one row per femur (postprocess), the same rule for every model:
  * classifier, M0 and SLG-Net: p = softmax(cls); conf_c = p_c (c = 1..5); score = top_conf = 1 - p_0;
    top_cls6 = 1 + argmax(p_1..p_5); mask (M0, SLG-Net) = sigmoid(seg) >= 0.5.
  * semantic segmentation: fracture probability map f = 1 - p_0 per pixel; score = top_conf = mean of the TOPK
    highest f (a robust maximum: a few hundred pixels, far fewer than the smallest fracture); the class shares come
    from the predicted region (f >= 0.5, or the TOPK pixels if it is empty): share_c = sum of p_c over the region /
    sum of f over the region; conf_c = score * share_c; top_cls6 = 1 + argmax share; mask = f >= 0.5.
Masks are compared with the released masks at full resolution (f is resized with bilinear interpolation first).
"""
from __future__ import annotations

from dataclasses import dataclass

TOPK = 500  # pixels at 768 x 512; the smallest released fracture mask has 1,722 there (median 23,045)


@dataclass(frozen=True)
class NetSpec:
    kind: str          # cls | seg | m0 | slg | ref | zoom
    arch: str          # timm model name (cls, m0, slg, ref) or smp architecture (seg)
    encoder: str       # smp encoder (seg); "" otherwise
    variant: str = ""  # SLG-Net variant A2..A6 (slg) or RefFrac variant R1..R7 (ref); "" otherwise

    @property
    def has_mask(self) -> bool:
        return self.kind in ("seg", "m0", "slg", "ref", "zoom")

    @property
    def text(self) -> str:
        tail = self.encoder or self.variant
        return f"{self.kind}:{self.arch}" + (f":{tail}" if tail else "")


def parse_net(text: str) -> NetSpec:
    parts = text.split(":")
    if parts[0] in ("cls", "m0", "zoom") and len(parts) == 2:
        return NetSpec(parts[0], parts[1], "")
    if parts[0] == "seg" and len(parts) == 3:
        return NetSpec("seg", parts[1], parts[2])
    if parts[0] == "slg" and len(parts) == 3:
        from .slg import VARIANTS

        if parts[2] not in VARIANTS:
            raise ValueError(f"--net '{text}': SLG-Net variant must be one of {', '.join(VARIANTS)}")
        return NetSpec("slg", parts[1], "", parts[2])
    if parts[0] == "ref" and len(parts) == 3:
        from .refnet import VARIANTS as REF_VARIANTS

        if parts[2] not in REF_VARIANTS:
            raise ValueError(f"--net '{text}': RefFrac variant must be one of {', '.join(REF_VARIANTS)}")
        return NetSpec("ref", parts[1], "", parts[2])
    raise ValueError(f"--net '{text}': expected cls:<timm model>, m0:<timm model>, seg:<smp arch>:<encoder>, "
                     "slg:<timm model>:<A2..A6>, ref:<timm model>:<R1..R7> or zoom:<timm model>")


def build_net(spec: NetSpec, pretrained: bool = True, drop_rate: float = 0.2, slg_scales=(16, 32),
              slg_detach: bool = False, ref_k: int = 0, ref_margin: float | None = None, ref_window: float = 0.125,
              zoom_window: int = 640, zoom_input: int = 512):
    """Create the network (downloads ImageNet weights when ``pretrained``). SLG-Net options: the encoder scales
    of region pooling and whether the decision losses may reach the mask (see slg.build_slg). RefFrac options: the
    number of references, the margin-loss weight and the attention window (None / 0 keep the variant's; see
    refnet.build_refnet). Segment-and-zoom options: the crop side in input pixels and the zoom encoder's input size
    (zoomnet.build_zoomnet)."""
    import torch.nn as nn

    if spec.kind == "zoom":
        from .zoomnet import build_zoomnet

        return build_zoomnet(spec.arch, pretrained=pretrained, drop_rate=drop_rate, window=zoom_window,
                             zoom_input=zoom_input)
    if spec.kind == "ref":
        from .refnet import build_refnet

        return build_refnet(spec.arch, spec.variant, pretrained=pretrained, drop_rate=drop_rate, k=ref_k,
                            margin=ref_margin, window=ref_window)
    if spec.kind == "slg":
        from .slg import build_slg

        return build_slg(spec.arch, spec.variant, pretrained=pretrained, drop_rate=drop_rate, scales=slg_scales,
                         detach_mask=slg_detach)

    class Net(nn.Module):
        def __init__(self, body, kind):
            super().__init__()
            self.body, self.kind = body, kind

        def forward(self, x):
            out = self.body(x)
            if self.kind == "cls":
                return {"cls": out}
            if self.kind == "seg":
                return {"seg": out}
            seg, cls = out  # smp returns (masks, labels) when it has a classification head
            return {"seg": seg, "cls": cls}

    if spec.kind == "cls":
        import timm

        return Net(timm.create_model(spec.arch, pretrained=pretrained, num_classes=6, drop_rate=drop_rate), "cls")
    import segmentation_models_pytorch as smp

    if spec.kind == "seg":
        model = getattr(smp, spec.arch)(encoder_name=spec.encoder, encoder_weights="imagenet" if pretrained else None,
                                        in_channels=3, classes=6)
        return Net(model, "seg")
    import timm

    model = smp.Unet(encoder_name=f"tu-{spec.arch}", encoder_weights="imagenet" if pretrained else None,
                     in_channels=3, classes=1)
    # the classifier head of the same timm model (for ConvNeXt: global average pool, LayerNorm, dropout, linear),
    # with its pretrained norm weights; smp's forward applies it to the encoder's last feature map
    head = getattr(timm.create_model(spec.arch, pretrained=pretrained, num_classes=6, drop_rate=drop_rate), "head", None)
    if head is None or not hasattr(head, "global_pool"):
        raise ValueError(f"m0:{spec.arch}: the timm model has no pooled classifier head ('head' with global_pool)")
    model.classification_head = head
    return Net(model, "m0")


def make_loss(spec: NetSpec, seg_weight: float = 1.0, priors: dict | None = None, la_tau: float = 1.0,
              joint_weight: float = 0.5, ref_margin: float = 0.0):
    """Loss function (outputs, mask (B, H, W) long, cls6 (B,) long) -> (total, {part: value}). SLG-Net: see
    slg.make_slg_loss (``priors`` = training-fold log-priors for logit adjustment, A6 only). RefFrac: M0's loss plus
    ref_margin x the margin loss (refnet.make_ref_loss)."""
    import torch.nn.functional as F
    from segmentation_models_pytorch.losses import DiceLoss

    if spec.kind == "zoom":
        from .zoomnet import make_zoom_loss

        return make_zoom_loss(seg_weight=seg_weight)
    if spec.kind == "ref":
        from .refnet import make_ref_loss

        return make_ref_loss(margin_weight=ref_margin, seg_weight=seg_weight)
    if spec.kind == "slg":
        from .slg import make_slg_loss

        return make_slg_loss(spec.variant, seg_weight=seg_weight, joint_weight=joint_weight, priors=priors,
                             la_tau=la_tau)

    dice_multi = DiceLoss(mode="multiclass", classes=[1, 2, 3, 4, 5], from_logits=True)
    dice_bin = DiceLoss(mode="binary", from_logits=True)

    def loss(out: dict, mask, y):
        parts = {}
        if "cls" in out:
            parts["ce"] = F.cross_entropy(out["cls"].float(), y)
        if spec.kind == "seg":
            seg = out["seg"].float()
            parts["seg_ce"] = F.cross_entropy(seg, mask)
            parts["seg_dice"] = dice_multi(seg, mask)
        elif spec.kind == "m0":
            seg = out["seg"].float()
            target = (mask > 0).float().unsqueeze(1)
            parts["seg_bce"] = F.binary_cross_entropy_with_logits(seg, target)
            parts["seg_dice"] = dice_bin(seg, target)
        seg_total = sum(v for k, v in parts.items() if k.startswith("seg"))
        total = parts.get("ce", 0.0) + (seg_weight if spec.kind == "m0" else 1.0) * seg_total
        return total, {k: float(v.detach()) for k, v in parts.items()}

    return loss


def postprocess(spec: NetSpec, out: dict, topk: int = TOPK) -> dict:
    """Per-image decision values (torch tensors on the model's device): conf (B, 5), score (B,), top_cls6 (B,),
    and fmap (B, H, W) = fracture probability per pixel (None for classifiers)."""
    import torch

    if spec.kind in ("cls", "m0", "slg", "ref", "zoom"):
        p = torch.softmax(out["cls"].float(), dim=1)  # SLG-Net A4-A6: "cls" = log-probabilities, softmax returns them
        conf = p[:, 1:]
        res = {"conf": conf, "score": conf.sum(dim=1), "top_cls6": conf.argmax(dim=1) + 1, "fmap": None}
        if spec.kind in ("m0", "slg", "ref", "zoom"):
            res["fmap"] = torch.sigmoid(out["seg"].float()[:, 0])
        if spec.kind == "ref":
            res["diff16"] = out.get("diff16")  # RefFrac: |T - R~| at 1/16, for the pointing game
        if spec.kind == "zoom":
            res["centre"] = out["centre"]  # segment-and-zoom: crop centre (y, x) in input pixels
        return res
    p = torch.softmax(out["seg"].float(), dim=1)                     # (B, 6, H, W)
    f = 1.0 - p[:, 0]                                                # fracture probability
    flat = f.flatten(1)
    k = min(topk, flat.shape[1])
    top_vals, top_idx = flat.topk(k, dim=1)
    score = top_vals.mean(dim=1)
    region = flat >= 0.5
    region_top = torch.zeros_like(region).scatter_(1, top_idx, True)
    region = torch.where(region.any(dim=1, keepdim=True), region, region_top)
    mass = (p[:, 1:].flatten(2) * region[:, None]).sum(dim=2)        # (B, 5)
    share = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return {"conf": score[:, None] * share, "score": score, "top_cls6": share.argmax(dim=1) + 1, "fmap": f}


def param_groups(model, weight_decay: float):
    """AdamW groups: no weight decay for biases and normalisation weights (1-D parameters)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
