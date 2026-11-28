# Run notes: N_segzoom_fold0_seed2

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** fold-0 screening of the new method candidates (`Analysis/11_New_Direction_Metrics_FracAtlas_Method_Plan.md` section 7): candidate 1, the segment-and-zoom network, seed 2027. Hypothesis: it beats the ConvNeXt U-Net by at least .02 in macro-F1 (mean of two seeds) without losing more than .01 Dice.

**2. What changed compared with the previous run** (config, code commit, data, folds): compared with the ConvNeXt U-Net, the same network as `N_segzoom_fold0`, seed 2027. Everything else as the ConvNeXt U-Net (60 epochs, batch 16, final weights). Fold 0 only, no test set. Code `50e2039`, one T4 (GPU 1 of the group `N_screen_round2`).

**3. Result** (fold 0, 208 femurs; `results/analysis/N_screening_fold0/`, threshold fitted again on fold 0):
- Accuracy 186/208 = 0.894, macro-F1 0.843 (0.766–0.906), sensitivity 0.985, specificity 0.986, fracture AUC 0.997.
- Dice of fractured femurs 0.763, IoU 0.645, 2 of 136 fractures missed.
- Pointing game of the zoom crop: the crop window holds the centre of the released mask in 135 of 136 fractured femurs (0.993); on average 0.990 of the mask lies inside the window.
- The base (the two ConvNeXt U-Net runs on fold 0): macro-F1 .817, Dice .755.

**4. What we learnt:** mean of this candidate's two seeds: macro-F1 0.829, Dice 0.763. The pass mark is macro-F1 ≥ .837 and Dice ≥ .745, so the candidate **does not pass** the screening (Analysis/11 section 7.5).

**5. Decision / next step:** no candidate passed. By the rule fixed before the runs, we stop and the person decides the next step (Analysis/11 section 7.5).
