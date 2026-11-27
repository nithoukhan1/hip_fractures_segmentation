# Run notes: S07_unet_r34

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Baseline 7: U-Net (ResNet34) 6-class semantic segmenter with the shared Step 2 recipe. It is the standard segmentation baseline for the Dice comparison with YOLO and M0 (analysis C04, RQ1).

**2. What changed compared with the previous run** (config, code commit, data, folds): A segmenter instead of a classifier. It is trained with pixel cross-entropy + Dice over the 5 fracture classes; the class and score come from the predicted region (Report 05 §3). Code `4820f6b`, micro-batch 16 × 1, T4 x2, 87.2 min (estimate 1.7 h). Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`):
- OOF: AUC .990, 6-class accuracy 86.3%, macro-F1 .800, Garden I-II recall .614, Dice fractured (missed = 0) .748, NSD@10 .221. Threshold .700.
- Test, 5 fold models: accuracy 84.6 ± 1.6%, macro-F1 .770 ± .027, Dice fractured .698 ± .009.
- Test ensemble: AUC .999, accuracy 86.7%, macro-F1 .799, Dice fractured .711.

**4. What we learnt:**
- It classifies as well as the best image classifiers out of fold, and has the best Garden I-II recall of all baselines so far (.614).
- Its Dice is close to YOLO's (S01: .740 OOF, .693 test models) and below M0's (.777 OOF, .734 test models). The formal tests come in C04.

**5. Decision / next step:** Analysis C04 (S07, S08, S09 and S10 against S01) once S09 is filed.
