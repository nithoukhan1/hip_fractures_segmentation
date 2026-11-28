# Run notes: A6_dev_scale32_s2

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper A's second-seed check (`Analysis/06` section 10): `A6_dev_scale32` (region pooling at 1/32 only, the frozen setting) repeated with seed 2027.

**2. What changed compared with the previous run** (config, code commit, data, folds): Only `--seed 2027`, compared with `A6_dev_scale32`. Code `f96f84c`, one T4 (GPU 1 of the group `A_dev_seed2`), 48.9 min. Kaggle version number not recorded.

**3. Result** (`metrics.json`; analysis `results/analysis/C11_paperA_seed2/`):
- Fold 0: AUC .999, accuracy 185/208, macro-F1 .827 (.728–.896), Dice fractured .770, Garden I-II recall .727. Threshold .0488.
- Oracle test: accuracy 189/208, macro-F1 .859; 5 of 23 wrong fractured femurs fixed.

**4. What we learnt:** Same macro-F1 as the first seed (.827), Dice +.010. Unlike seed 2026, the released mask helps these heads, so the SLG-Net development analysis's "1/32 heads do not use the region" was seed noise.

**5. Decision / next step:** the SLG-Net second-seed check: all three conditions of section 10 hold (mean +.054 macro-F1, Dice +.016, .827 ≥ .79). Pooling at 1/32 only stays; the five final runs are valid.
