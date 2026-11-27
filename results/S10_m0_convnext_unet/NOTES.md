# Run notes: S10_m0_convnext_unet

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** The core model M0 (= A1 of Paper A = R0 of Paper B): S06 plus a U-Net decoder and its mask loss. Analysis C05 asks whether learning the mask helps the classification. Expected: classification at least as good as S06's, and a mask close to YOLO's.

**2. What changed compared with the previous run** (config, code commit, data, folds): Compared with S06, only the decoder and its loss (binary cross-entropy + Dice, weight 1). Same encoder, 6-class head, recipe, folds and seeds. Code `18e4774`, micro-batch 16 × 1, T4 x2, 140.1 min (estimate 170). Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`; analysis `results/analysis/C05_mask_task_s06_vs_s10/`):
- OOF: AUC .995, 6-class accuracy 88.4%, macro-F1 .825, Garden I-II recall .544, Dice fractured .777.
- Test, 5 fold models: accuracy 86.0 ± 0.5%, macro-F1 .776 ± .017, Dice fractured .734 ± .005.
- Test ensemble: AUC .999, accuracy 86.7%, macro-F1 .779, Dice fractured .744, 1 detection error. OOF threshold .018.

**4. What we learnt:**
- The mask task helps. Compared with S06, OOF accuracy is +2.0 points (Holm p = .026) and macro-F1 +.042 (Holm p = .013), and M0 is better on all 5 held-out folds. The test set shows the same direction, not significant.
- The gain comes from detection and location. Grade errors are unchanged and are 65 of M0's 121 OOF errors.
- Its mask is already better than YOLO's: Dice fractured .777 against .740 OOF, and .734 against .693 for the test fold models (formal test in C04).
- The probabilities are over-confident: OOF threshold .018, expected calibration error .108.

**5. Decision / next step:** M0 is confirmed as the shared starting point. Paper A's A2–A6 and Paper B's R runs build on it. Their targets are grade errors (Garden I-II above all) and calibration.
