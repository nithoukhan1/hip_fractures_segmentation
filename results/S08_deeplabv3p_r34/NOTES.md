# Run notes: S08_deeplabv3p_r34

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Baseline 8: DeepLabV3+ (ResNet34) 6-class semantic segmenter with the shared Step 2 recipe: the second standard segmentation family (atrous context). It enters the segmenter analysis.

**2. What changed compared with the previous run** (config, code commit, data, folds): The decoder only, compared with ResNet-34 U-Net (same ResNet34 encoder, loss, recipe and folds). Code `4820f6b`, micro-batch 16 × 1, T4 x2, 73.0 min (estimate 1.5 h). Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`):
- OOF: AUC .990, 6-class accuracy 86.2%, macro-F1 .800, Garden I-II recall .596, Dice fractured .749, NSD@10 .222. Threshold .954.
- Test, 5 fold models: accuracy 85.5 ± 1.9%, macro-F1 .784 ± .022, Dice fractured .704 ± .005.
- Test ensemble: AUC .999, accuracy 86.2%, macro-F1 .778, Dice fractured .698.

**4. What we learnt:**
- It is practically the same as U-Net (ResNet-34 U-Net) on every endpoint, and below ConvNeXt U-Net's Dice.
- Its fracture threshold (.954) is high because the segmenter's score (the mean of the 500 highest pixel probabilities) is close to 1 for most fractured femurs.

**5. Decision / next step:** the segmenter analysis once SegFormer-B2 is filed.
