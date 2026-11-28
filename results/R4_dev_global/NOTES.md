# Run notes: R4_dev_global

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B development the whole-reference question (`Analysis/08` section 4): a local attention window (±1/8 of the image height), or the whole reference? Same as `R4_dev_base` with `--ref-window 0`.

**2. What changed compared with the previous run** (config, code commit, data, folds): The attention window only. Code `2619b39` (main after PR #13), one T4, micro-batch 16 x 1 (12.6 GB), fold 0 only, no test set. Kaggle version number not recorded. GPU 0 of the group `R_dev_round2`, 87.0 min.

**3. Result** (`metrics.json`; the RefFrac development analysis):
- Fold 0: Dice fractured .778 (.753–.800), macro-F1 .796, accuracy 181/208, Garden I-II recall .545. Threshold .0128.
- `self` test: Dice −.035, P(fracture) −.063. Pointing game 77.9%.

**4. What we learnt:** Dice +.009 over the base mean, below .01 and within the seed noise; macro-F1 −.022.

**5. Decision / next step:** the RefFrac development analysis: the local window stays.
