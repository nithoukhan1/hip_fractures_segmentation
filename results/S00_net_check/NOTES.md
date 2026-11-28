# Run notes: S00_net_check

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** The Step 2 pre-flight (`Analysis/05_Step2_Baselines_Protocol.md` section 5). Before the seven other baselines are trained, check on Kaggle's T4s that the ImageNet weights of the 7 networks download, find the largest micro-batch (at most 16) that fits one T4 under 85% of its memory, time a training step, and estimate the minutes per fold and per run. Trains nothing. Expected: all 7 build; the largest may need micro-batch 8 with gradient accumulation.

**2. What changed compared with the previous run** (config, code commit, data, folds): First Kaggle run of `scripts/train_net.py`, code `a15364b` (main after PR #5). Same launcher, T4 x2, Internet on. Kaggle version number not recorded.

**3. Result** (from `net_check.json` and `net_check.md`):
- All 7 networks downloaded their ImageNet weights (2–5 s each) and trained on a T4.
- Micro-batch 16 × 1: DenseNet121, ConvNeXt-T, U-Net, DeepLabV3+ and ConvNeXt U-Net. Micro-batch 8 × 2: EfficientNet-B4 (16 needed 14.6 GB, over 85% of the 15.6 GB T4) and SegFormer (16 ran out of memory).
- Peak memory 6.3–11.2 GB at the chosen sizes. None is close to the 85% limit (13.3 GB), so each run's own pre-flight will pick the same micro-batch.
- Estimated training per run (3 rounds of folds): DenseNet-121 86, EfficientNet-B4 119, ConvNeXt-T classifier 90, ResNet-34 U-Net 94, ResNet-34 DeepLabV3+ 81, SegFormer-B2 190, ConvNeXt U-Net 161 min. With about 5 min of set-up per run, the 7 runs take about 14.6 h. Image cache built in 46 s; whole run 3.5 min.

**4. What we learnt:** The Step 2 code runs unchanged on Kaggle's GPU stack (torch 2.11 with CUDA 13, timm 1.0.30, smp 0.6.0). The times come from steps on random images. The data loader should not slow training: augmenting one 768 × 512 image takes about 1.5 ms on one CPU core (measured in our container), while a fold needs at most 36 images per second. One ConvNeXt U-Net step costs 1.8 times an ConvNeXt-T classifier step, because of the decoder. In EfficientNet-B4, batch-norm statistics come from 8 images instead of 16; SegFormer has a single batch-norm layer, in its decoder.

**5. Decision / next step:** The measured times are in `kaggle/runs.json` and in the protocol's addendum (section 7). Launch ConvNeXt U-Net and ConvNeXt-T classifier (ConvNeXt-T, about 1.7 h) first, then ResNet-34 U-Net, DenseNet-121, ResNet-34 DeepLabV3+, SegFormer-B2 and EfficientNet-B4.
