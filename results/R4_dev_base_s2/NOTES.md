# Run notes: R4_dev_base_s2

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B fold-0 development (`Analysis/08` section 4): `R4_dev_base` repeated with seed 2027, so that the base is a mean of two seeds and the seed noise N is measured.

**2. What changed compared with the previous run** (config, code commit, data, folds): Only `--seed 2027`. Code `2619b39` (main after PR #13), one T4, micro-batch 16 x 1 (12.6 GB), fold 0 only, no test set. Kaggle version number not recorded. GPU 1 of the group `R_dev_round1`, 80.8 min.

**3. Result** (`metrics.json`; the RefFrac development analysis):
- Fold 0: Dice fractured .774 (.747–.798), macro-F1 .819, accuracy 186/208, Garden I-II recall .636. Threshold .0007.
- `self` test: Dice .658 (−.116), P(fracture) −.531, 17 more fractures missed. `random` test: Dice .769 (−.005).
- Pointing game: 91.9% (chance 8.0%).

**4. What we learnt:** Seed noise on fold 0: Dice .009, macro-F1 .002. The pointing game varies far more between seeds (64% vs 92%).

**5. Decision / next step:** Base = mean of the two seeds (Dice .770, macro-F1 .818): passes the go/no-go of the RefFrac development analysis.
