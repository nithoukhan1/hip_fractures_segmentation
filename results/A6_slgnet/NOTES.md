# Run notes: A6_slgnet

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper A final run A6 = SLG-Net (`Analysis/06_Step3_PaperA_Protocol.md` §1, §7, §9–§11): A5 + logit adjustment (C6) with the training-fold priors. Does it help the rare classes? Paper A's main model, with both interpretability tests. Frozen settings: teacher forcing over 30 epochs, end-to-end mask, region pooling at 1/32 only.

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `f96f84c` = tag `A-v1-frozen` (the launcher log shows this commit), T4 x2, two folds at a time, all 5 folds trained, exit 0. Kaggle version number not recorded. 174.3 min.

**3. Result** (`metrics.json`, `summary.md`; threshold fitted on the OOF femurs only):
- Out of fold (1,043 femurs): AUC .996, accuracy 907/1043 (87.0%), macro-F1 .806, Dice fractured .774, Garden I-II recall .614. Threshold .0156.
- Test ensemble (210 femurs): AUC .999, accuracy 186/210, macro-F1 .820, Dice fractured .737, Garden I-II recall .312.
- Oracle test: OOF accuracy 912/1043, macro-F1 .810. **Region deletion:** OOF accuracy 907 → 576, macro-F1 .806 → .324, 64.7% of the correct fractured decisions lost, P(true class) −.479 on fractured femurs; **control** (same shape elsewhere): 902, macro-F1 .805, 1.3% lost.

**4. What we learnt:** Analysed with the other final runs in C07 (ablation chain), C08 (interpretability) and C09 (main table). The deletion test already shows that its decisions depend on the predicted region (C08 tests it formally).

**5. Decision / next step:** Final run of Paper A; analyses C07–C09 (Step 19).
