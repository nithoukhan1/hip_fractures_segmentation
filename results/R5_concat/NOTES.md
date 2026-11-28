# Run notes: R5_concat

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B final run R5: as R4 with concatenation [T, R] instead of the difference; settings frozen by C12 (Analysis/08 sections 1, 2, 5 and 8), tag B-v1-frozen (1c88912); 5 folds and the test set; analyses C13-C16 (Step 20).

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `1c88912` = tag `B-v1-frozen` (the launcher log shows this commit), T4 x2, all 5 folds trained, exit 0. 241.3 min. Question: as R4 with concatenation: difference or concatenation?.

**3. Result** (`metrics.json`, `summary.md`; threshold fitted on the OOF femurs only):
- Out of fold (1,043 femurs): AUC 0.996, accuracy 925/1043, macro-F1 0.827, Dice fractured 0.780, Garden I-II recall 0.561. Threshold 0.0608.
- Test ensemble (210 femurs): AUC 0.999, accuracy 178/210, macro-F1 0.741, Dice fractured 0.740, Garden I-II recall 0.188.
- Reference tests, OOF (`../analysis/C13_paperB_reference/reftests.md`): self (vs base): Dice 0.780 → 0.733, Garden I-II 0.561 → 0.421, fractures lost 34/662; random (vs base): Dice 0.780 → 0.779, Garden I-II 0.561 → 0.544, fractures lost 1/662. Pointing game 0.35 (chance 0.079).

**4. What we learnt:** Concatenation instead of the difference changes no accuracy endpoint (C15), but the model relies on the comparison much less (self test loses 34 vs 201 fractures) and points less well (.35 vs .53).

**5. Decision / next step:** Final run of Paper B; analysed in C13–C16 (Step 20, `Analysis/08` §10). How Paper B is presented goes to the person.
