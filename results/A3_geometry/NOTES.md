# Run notes: A3_geometry

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper A final run the geometry model (`Analysis/06_Step3_PaperA_Protocol.md` sections 1, 7, 9–11): the region-pooling model + region geometry: 8 numbers describing the predicted region. Does the region's shape and position help? Frozen settings: teacher forcing over 30 epochs, end-to-end mask, region pooling at 1/32 only.

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `f96f84c` = tag `A-v1-frozen` (the launcher log shows this commit), T4 x2, two folds at a time, all 5 folds trained, exit 0. Kaggle version number not recorded. 142.5 min.

**3. Result** (`metrics.json`, `summary.md`; threshold fitted on the OOF femurs only):
- Out of fold (1,043 femurs): AUC .996, accuracy 927/1043 (88.9%), macro-F1 .838, Dice fractured .777, Garden I-II recall .684. Threshold .0577.
- Test ensemble (210 femurs): AUC .999, accuracy 180/210, macro-F1 .786, Dice fractured .734, Garden I-II recall .312.
- Oracle test: OOF accuracy 937/1043, macro-F1 .845; 8.5% of the wrong fractured femurs fixed, 1.6% of the correct ones lost.

**4. What we learnt:** Analysed with the other final runs in the SLG-Net ablation analysis (ablation chain), the SLG-Net interpretability analysis and the SLG-Net main-table analysis (main table).

**5. Decision / next step:** Final run of Paper A; SLG-Net analyses (Step 19).
