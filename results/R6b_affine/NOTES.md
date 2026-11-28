# Run notes: R6b_affine

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B final run R6b: as R4 with one global affine alignment per reference; settings frozen by C12 (Analysis/08 sections 1, 2, 5 and 8), tag B-v1-frozen (1c88912); 5 folds and the test set; analyses C13-C16 (Step 20).

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `1c88912` = tag `B-v1-frozen` (the launcher log shows this commit), T4 x2, all 5 folds trained, exit 0. 265.4 min. Question: as R4 with a global affine alignment: local or global alignment?.

**3. Result** (`metrics.json`, `summary.md`; threshold fitted on the OOF femurs only):
- Out of fold (1,043 femurs): AUC 0.995, accuracy 915/1043, macro-F1 0.814, Dice fractured 0.776, Garden I-II recall 0.561. Threshold 0.0167.
- Test ensemble (210 femurs): AUC 0.998, accuracy 181/210, macro-F1 0.792, Dice fractured 0.742, Garden I-II recall 0.312.
- Reference tests, OOF (`../analysis/C13_paperB_reference/reftests.md`): self (vs base): Dice 0.776 → 0.133, Garden I-II 0.561 → 0.000, fractures lost 553/660; random (vs base): Dice 0.776 → 0.772, Garden I-II 0.561 → 0.561, fractures lost 4/660. Pointing game 0.39 (chance 0.079).

**4. What we learnt:** A global affine alignment instead of local cross-attention changes no accuracy endpoint (C15); it collapses without a reference (self test) and points less well (.39).

**5. Decision / next step:** Final run of Paper B; analysed in C13–C16 (Step 20, `Analysis/08` §10). How Paper B is presented goes to the person.
