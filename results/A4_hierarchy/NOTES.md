# Run notes: A4_hierarchy

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper A final run the hierarchy model (`Analysis/06_Step3_PaperA_Protocol.md` sections 1, 7, 9–11): the geometry model + fracture → location → grade heads, one grade trunk per location. Does deciding location first, then grade, help? Frozen settings: teacher forcing over 30 epochs, end-to-end mask, region pooling at 1/32 only.

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `f96f84c` = tag `A-v1-frozen` (the launcher log shows this commit), T4 x2, two folds at a time, all 5 folds trained, exit 0. Kaggle version number not recorded. 142.6 min.

**3. Result** (`metrics.json`, `summary.md`; threshold fitted on the OOF femurs only):
- Out of fold (1,043 femurs): AUC .994, accuracy 916/1043 (87.8%), macro-F1 .823, Dice fractured .770, Garden I-II recall .649. Threshold .0149.
- Test ensemble (210 femurs): AUC .999, accuracy 183/210, macro-F1 .800, Dice fractured .732, Garden I-II recall .375.
- Oracle test: OOF accuracy 921/1043, macro-F1 .825; 11.5% of the wrong fractured femurs fixed, 1.6% of the correct ones lost.

**4. What we learnt:** Analysed with the other final runs in the SLG-Net ablation analysis (ablation chain), the SLG-Net interpretability analysis and the SLG-Net main-table analysis (main table).

**5. Decision / next step:** Final run of Paper A; SLG-Net analyses (Step 19).
