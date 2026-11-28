# Run notes: R3_nearest_ref

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B final run the nearest-reference model: as RefFrac with only the nearest healthy reference (k = 1); settings frozen by the RefFrac development analysis (Analysis/08 sections 1, 2, 5 and 8), tag B-v1-frozen (1c88912); 5 folds and the test set; the RefFrac analyses (Step 20).

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `1c88912` = tag `B-v1-frozen` (the launcher log shows this commit), T4 x2, all 5 folds trained, exit 0. 170.9 min. Question: as RefFrac with one nearest reference: does retrieval help (vs the random-reference model), do 3 references beat 1 (RefFrac vs the nearest-reference model)?.

**3. Result** (`metrics.json`, `summary.md`; threshold fitted on the OOF femurs only):
- Out of fold (1,043 femurs): AUC 0.993, accuracy 918/1043, macro-F1 0.826, Dice fractured 0.775, Garden I-II recall 0.684. Threshold 0.0008.
- Test ensemble (210 femurs): AUC 0.999, accuracy 177/210, macro-F1 0.765, Dice fractured 0.742, Garden I-II recall 0.375.
- Reference tests, OOF (`../analysis/C13_paperB_reference/reftests.md`): self (vs base): Dice 0.775 → 0.756, Garden I-II 0.684 → 0.614, fractures lost 17/669; random (vs base): Dice 0.775 → 0.775, Garden I-II 0.684 → 0.684, fractures lost 2/669. Pointing game 0.55 (chance 0.079).

**4. What we learnt:** One nearest reference is no better than a random one (nearest-reference model vs the random-reference model, the reference analysis) and no worse than three (RefFrac vs the nearest-reference model). It is the own-hip model's comparison for the own hip (own-hip analysis).

**5. Decision / next step:** Final run of Paper B; analysed in the RefFrac analyses (Step 20, `Analysis/08` section 10). How Paper B is presented goes to the person.
