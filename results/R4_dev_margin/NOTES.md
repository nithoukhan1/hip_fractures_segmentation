# Run notes: R4_dev_margin

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B development question D2 (`Analysis/08` §4): does AASN's margin loss help RefFrac use its references? Same as `R4_dev_base` with `--ref-margin 0.5`.

**2. What changed compared with the previous run** (config, code commit, data, folds): The margin loss only. Code `2619b39` (main after PR #13), one T4, micro-batch 16 x 1 (12.6 GB), fold 0 only, no test set. Kaggle version number not recorded. GPU 1 of the group `R_dev_round2`, 82.0 min.

**3. Result** (`metrics.json`; analysis C12):
- Fold 0: Dice fractured .755 (.721–.786), macro-F1 .829, accuracy 185/208, Garden I-II recall .727. Threshold .9745.
- `self` test: Dice .318 (−.438), 56% of the correct decisions lost. Pointing game 54.4%.

**4. What we learnt:** The margin loss makes the model depend on the difference far more, without better Dice (−.014 vs the base mean).

**5. Decision / next step:** C12: no margin loss in R2–R7 (R1 keeps AASN's own loss).
