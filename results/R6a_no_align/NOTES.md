# Run notes: R6a_no_align

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B final run R6a: as R4 without alignment; settings frozen by C12 (Analysis/08 sections 1, 2, 5 and 8), tag B-v1-frozen (1c88912); 5 folds and the test set; analyses C13-C16 (Step 20).

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `1c88912` = tag `B-v1-frozen` (the launcher log shows this commit), T4 x2, all 5 folds trained, exit 0. 229.8 min. Question: as R4 without alignment: is alignment needed?.

**3. Result** (`metrics.json`, `summary.md`; threshold fitted on the OOF femurs only):
- Out of fold (1,043 femurs): AUC 0.995, accuracy 908/1043, macro-F1 0.807, Dice fractured 0.777, Garden I-II recall 0.561. Threshold 0.0013.
- Test ensemble (210 femurs): AUC 0.999, accuracy 180/210, macro-F1 0.788, Dice fractured 0.741, Garden I-II recall 0.312.
- Reference tests, OOF (`../analysis/C13_paperB_reference/reftests.md`): self (vs base): Dice 0.777 → 0.222, Garden I-II 0.561 → 0.000, fractures lost 477/662; random (vs base): Dice 0.777 → 0.773, Garden I-II 0.561 → 0.526, fractures lost 5/662. Pointing game 0.48 (chance 0.079).

**4. What we learnt:** Without alignment, accuracy is unchanged but Garden I-II recall is lower than R4's out of fold (8 vs 0 femurs, Holm .031; test equal). It collapses without a reference (self test) and points less well (.48) (C15).

**5. Decision / next step:** Final run of Paper B; analysed in C13–C16 (Step 20, `Analysis/08` §10). How Paper B is presented goes to the person.
