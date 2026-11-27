# Run notes: A6_dev_detach

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Development question D2 (`Analysis/06_Step3_PaperA_Protocol.md` §4): should the decision losses reach the mask? Same as A6_dev_base, but the mask given to the heads is detached (`--slg-detach`).

**2. What changed compared with the previous run** (config, code commit, data, folds): The gradient path from the heads to the mask only, compared with A6_dev_base. Code `f342b8a`, one T4 (GPU 0 of the group `A_dev_round2`), 49.9 min. Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`; analysis `results/analysis/C06_paperA_dev_fold0/`):
- Fold 0: AUC .998, accuracy 184/208, macro-F1 .794 (+.019 vs the base), Dice fractured .754, Garden I-II recall .727. Threshold .0016.
- Oracle test: macro-F1 .838; 2 of 24 wrong fractured femurs fixed.

**4. What we learnt:**
- The gain (+.0190) is short of the rule's .02 by .001. Within noise either way.
- One fold-0 femur has its true class written as probability 0 (below the 8 decimals of the per-case file). Its calibration temperature (3.38) is fitted since the `run_stats.py` change of 2026-11-27.

**5. Decision / next step:** By the rule, the end-to-end mask stays.
