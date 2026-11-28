# Run notes: S09_segformer_b2

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Baseline 9: SegFormer (MiT-B2) 6-class semantic segmenter with the shared Step 2 recipe: the transformer segmenter among the baselines. It is the last run of the segmenter analysis (segmenters against YOLO, the segmentation question).

**2. What changed compared with the previous run** (config, code commit, data, folds):
- A transformer encoder (MiT-B2) and SegFormer's light decoder instead of ResNet34 with U-Net or DeepLabV3+ (ResNet-34 U-Net, ResNet-34 DeepLabV3+). Same loss, recipe and folds.
- Micro-batch 8 × 2 (16 did not fit a T4; `S00_net_check`). SegFormer has a single batch-norm layer, in its decoder.
- Code `f342b8a`, the commit of the Paper A development runs (ResNet-34 U-Net and ResNet-34 DeepLabV3+ used `4820f6b`). The newer commit added SLG-Net and restructured `train_net.py`; the segmenter recipe (`Analysis/05` sections 2–3) is unchanged.
- T4 x2, 168.8 min (estimate 3.3 h). Kaggle version number not recorded.
- The first launch (2026-11-27, commit `4820f6b`) had no dataset attached and stopped in 0.0 min. Its zip is kept as `results/_inbox/filed/results_S09_segformer_b2_failed_no_data.zip` and was not filed.

**3. Result** (`metrics.json`, `summary.md`):
- OOF: AUC .994, 6-class accuracy 87.9%, macro-F1 .817, Garden I-II recall .579, Dice fractured (missed = 0) .766, NSD@10 .238. Threshold .873.
- Test, 5 fold models: accuracy 86.8 ± 0.8%, macro-F1 .797 ± .020, Dice fractured .725 ± .008.
- Test ensemble: AUC .999, accuracy 88.1%, macro-F1 .804, Dice fractured .737.

**4. What we learnt** (segmenter analysis, `results/analysis/C04_segmenters_vs_yolo/report.md`):
- It is the best standard segmenter: its Dice is above YOLO's, U-Net's and DeepLabV3+'s out of fold (Holm p ≤ .014), and below only ConvNeXt U-Net's (−.011, p = .0003).
- It classifies as well as ConvNeXt U-Net (no significant difference) and has the highest test accuracy per fold model of all baselines (86.8%).
- Of the five models in the segmenter analysis, it misses the fewest fractures out of fold (16 of 680; YOLO 51).

**5. Decision / next step:** It enters Paper A's main table (SLG-Net main-table analysis) as the strongest segmentation baseline. With this run, all Step 2 baselines are filed: the baseline-table milestone (`results/analysis/M1_baseline_table/`).
