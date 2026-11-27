# Run notes: A2_region

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper A final run A2 (`Analysis/06_Step3_PaperA_Protocol.md` §1, §7, §9–§11): M0 + region pooling (C2) at 1/32: a flat 6-class head on [c, r]. Does looking inside the predicted region help? Frozen settings: teacher forcing over 30 epochs, end-to-end mask, region pooling at 1/32 only.

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `f96f84c` = tag `A-v1-frozen` (the launcher log shows this commit), T4 x2, two folds at a time, all 5 folds trained, exit 0. Kaggle version number not recorded. 151.4 min.

**3. Result** (`metrics.json`, `summary.md`; threshold fitted on the OOF femurs only):
- Out of fold (1,043 femurs): AUC .996, accuracy 921/1043 (88.3%), macro-F1 .820, Dice fractured .772, Garden I-II recall .544. Threshold .0581.
- Test ensemble (210 femurs): AUC .999, accuracy 181/210, macro-F1 .785, Dice fractured .738, Garden I-II recall .250 (16 femurs).
- Oracle test (released mask for the heads): OOF accuracy 930/1043, macro-F1 .832; 11.1% of the wrong fractured femurs fixed, 1.6% of the correct ones lost.

**4. What we learnt:** Analysed with the other final runs in C07 (ablation chain), C08 (interpretability) and C09 (main table).

**5. Decision / next step:** Final run of Paper A; analyses C07–C09 (Step 19).
