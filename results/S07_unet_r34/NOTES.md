# Run notes: S07_unet_r34

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Baseline 7: U-Net (ResNet34) 6-class semantic segmenter with the shared Step 2 recipe. It is the standard segmentation baseline for the Dice comparison with YOLO and ConvNeXt U-Net (segmenter analysis, the segmentation question).

**2. What changed compared with the previous run** (config, code commit, data, folds): A segmenter instead of a classifier. It is trained with pixel cross-entropy + Dice over the 5 fracture classes; the class and score come from the predicted region (Report 05 section 3). Code `4820f6b`, micro-batch 16 × 1, T4 x2, 87.2 min (estimate 1.7 h). Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`):
- OOF: AUC .990, 6-class accuracy 86.3%, macro-F1 .800, Garden I-II recall .614, Dice fractured (missed = 0) .748, NSD@10 .221. Threshold .700.
- Test, 5 fold models: accuracy 84.6 ± 1.6%, macro-F1 .770 ± .027, Dice fractured .698 ± .009.
- Test ensemble: AUC .999, accuracy 86.7%, macro-F1 .799, Dice fractured .711.

**4. What we learnt:**
- It classifies as well as the best image classifiers out of fold, and has the best Garden I-II recall of all baselines so far (.614).
- Its Dice is close to YOLO's (YOLOv8s-seg: .740 OOF, .693 test models) and below ConvNeXt U-Net's (.777 OOF, .734 test models). The formal tests come in the segmenter analysis.

**5. Decision / next step:** the segmenter analysis (ResNet-34 U-Net, ResNet-34 DeepLabV3+, SegFormer-B2 and ConvNeXt U-Net against YOLOv8s-seg) once SegFormer-B2 is filed.
