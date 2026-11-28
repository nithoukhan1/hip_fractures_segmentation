# Run notes: N_convnext_s_fold0_seed2

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** fold-0 screening of the new method candidates (`Analysis/11_New_Direction_Metrics_FracAtlas_Method_Plan.md` section 7): candidate 3, the ConvNeXt U-Net with the ConvNeXt-S encoder, seed 2027. Hypothesis: it beats the ConvNeXt U-Net by at least .02 in macro-F1 (mean of two seeds) without losing more than .01 Dice.

**2. What changed compared with the previous run** (config, code commit, data, folds): compared with the ConvNeXt U-Net, as `N_convnext_s_fold0`, seed 2027. Everything else as the ConvNeXt U-Net (60 epochs, batch 16, final weights). Fold 0 only, no test set. Code `eb22c1f`, one T4 (GPU 1 of the group `N_screen_round3`).

**3. Result** (fold 0, 208 femurs; `results/analysis/N_screening_fold0/`, threshold fitted again on fold 0):
- Accuracy 180/208 = 0.865, macro-F1 0.788 (0.686–0.863), sensitivity 0.993, specificity 0.986, fracture AUC 0.998.
- Dice of fractured femurs 0.761, IoU 0.641, 1 of 136 fractures missed.
- The base (the two ConvNeXt U-Net runs on fold 0): macro-F1 .817, Dice .755.

**4. What we learnt:** mean of this candidate's two seeds: macro-F1 0.801, Dice 0.760. The pass mark is macro-F1 ≥ .837 and Dice ≥ .745, so the candidate **does not pass** the screening (Analysis/11 section 7.5).

**5. Decision / next step:** no candidate passed. By the rule fixed before the runs, we stop and the person decides the next step (Analysis/11 section 7.5).
