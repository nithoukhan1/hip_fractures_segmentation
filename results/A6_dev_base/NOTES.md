# Run notes: A6_dev_base

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper A fold-0 development, round 1 (`Analysis/06_Step3_PaperA_Protocol.md` section 4). SLG-Net with the base settings (teacher forcing over 30 epochs, end-to-end mask, region pooling at 1/16 + 1/32), trained on folds 1–4 and scored on fold 0. It is the go/no-go check: fold-0 macro-F1 must reach .79 (ConvNeXt U-Net − .03).

**2. What changed compared with the previous run** (config, code commit, data, folds): ConvNeXt U-Net plus SLG-Net's heads (region pooling, geometry, hierarchy, shared trunk, logit adjustment). Fold 0 only, no test set, oracle test. Code `f342b8a`, one T4 (GPU 0 of the group `A_dev_round1`), 49.3 min. Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`; analysis `results/analysis/C06_paperA_dev_fold0/`):
- Fold 0 (208 femurs): AUC .998, accuracy 184/208 (88.5%), macro-F1 .776 (.675–.853), Dice fractured .754, Garden I-II recall .545. Threshold .022.
- Oracle test (released mask given to the heads): accuracy 190/208, macro-F1 .839. 6 of the 23 wrong fractured femurs are fixed.

**4. What we learnt:**
- **It fails the go/no-go** (.776 < .79), although its accuracy equals ConvNeXt U-Net's (184/208).
- The gap is in the smallest classes: 3 of 7 subtrochanteric femurs right against ConvNeXt U-Net's 5. It is not significant (p = .25).
- The heads do use the region: a better mask would help.

**5. Decision / next step:** Design review (SLG-Net development analysis, section 3). The cause is the 1/16 region features, which the pooling-scale question removes. The proposed freeze uses pooling at 1/32 only.
