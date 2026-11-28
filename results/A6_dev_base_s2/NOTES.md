# Run notes: A6_dev_base_s2

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper A's second-seed check (`Analysis/06_Step3_PaperA_Protocol.md` section 10), asked for by the person together with the final runs: `A6_dev_base` repeated with seed 2027 (round 1 used 2026).

**2. What changed compared with the previous run** (config, code commit, data, folds): Only `--seed 2027`, compared with `A6_dev_base`. Code `f96f84c` (training code identical to `f342b8a`), one T4 (GPU 0 of the group `A_dev_seed2`), 49.4 min. Kaggle version number not recorded.

**3. Result** (`metrics.json`; analysis `results/analysis/C11_paperA_seed2/`):
- Fold 0: AUC .998, accuracy 180/208, macro-F1 .771 (.668–.847), Dice fractured .744, Garden I-II recall .545. Threshold .0006.
- Oracle test: accuracy 186/208, macro-F1 .839; 6 of 27 wrong fractured femurs fixed.

**4. What we learnt:** The base reproduces its first seed within .004 macro-F1 and .010 Dice: it stays below the go/no-go line (.79) with either seed.

**5. Decision / next step:** With `A6_dev_scale32_s2`, the SLG-Net second-seed check keeps pooling at 1/32 only; the freeze `A-v1-frozen` stands.
