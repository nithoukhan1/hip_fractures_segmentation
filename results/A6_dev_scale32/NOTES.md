# Run notes: A6_dev_scale32

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Development the pooling-scale question (`Analysis/06_Step3_PaperA_Protocol.md` section 4): region pooling at 1/16 + 1/32, or at 1/32 only? Same as A6_dev_base with `--slg-scales 32`.

**2. What changed compared with the previous run** (config, code commit, data, folds): The scales of region pooling only, compared with A6_dev_base. Code `f342b8a`, one T4 (GPU 1 of the group `A_dev_round2`), 50.6 min. Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`; analysis `results/analysis/C06_paperA_dev_fold0/`):
- Fold 0: AUC .999, accuracy 185/208 (88.9%), macro-F1 .827 (.727–.897; +.052 vs the base, +.005 vs ConvNeXt U-Net), Dice fractured .760 (+.006 vs the base), Garden I-II recall .818. Threshold .0017.
- Oracle test: accuracy 183/208, macro-F1 .810. No wrong fractured femur is fixed, and 2 correct ones are lost.

**4. What we learnt:**
- The only alternative that passes the rule, and the best of the four runs.
- With pooling at 1/32 only, the heads match ConvNeXt U-Net, so the base's shortfall came from the 1/16 features.
- On fold 0 these heads do not gain from the released mask. Watch the deletion test of the final SLG-Net (interpretability analysis).

**5. Decision / next step:** Region pooling at 1/32 only enters the proposed freeze (`A-v1-frozen`); final the five SLG-Net chain runs are in `kaggle/runs.json`, pending the person's approval.
