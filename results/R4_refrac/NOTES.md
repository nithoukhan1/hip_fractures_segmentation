# Run notes: R4_refrac

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B final run RefFrac: the 3 nearest healthy references, local cross-attention, difference fusion; settings frozen by the RefFrac development analysis (Analysis/08 sections 1, 2, 5 and 8), tag B-v1-frozen (1c88912); 5 folds and the test set; the RefFrac analyses (Step 20).

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `1c88912` = tag `B-v1-frozen` (the launcher log shows this commit), T4 x2, all 5 folds trained, exit 0. 241.5 min. Question: the main method of Paper B (3 nearest references, local cross-attention, difference).

**3. Result** (`metrics.json`, `summary.md`; threshold fitted on the OOF femurs only):
- Out of fold (1,043 femurs): AUC 0.994, accuracy 920/1043, macro-F1 0.826, Dice fractured 0.777, Garden I-II recall 0.702. Threshold 0.0071.
- Test ensemble (210 femurs): AUC 1.000, accuracy 180/210, macro-F1 0.779, Dice fractured 0.739, Garden I-II recall 0.312.
- Reference tests, OOF (`../analysis/C13_paperB_reference/reftests.md`): self (vs base): Dice 0.777 → 0.536, Garden I-II 0.702 → 0.158, fractures lost 201/663; random (vs base): Dice 0.777 → 0.775, Garden I-II 0.702 → 0.667, fractures lost 2/663. Pointing game 0.53 (chance 0.079).

**4. What we learnt:** RefFrac ties ConvNeXt U-Net in Dice and macro-F1 and has the highest out-of-fold Garden I-II recall of all models (.702 vs ConvNeXt U-Net .544; p .035, Holm .14 in the reference analysis), not seen on the test set (.312 = ConvNeXt U-Net). It depends strongly on its references (self test) but not on their match (random). The reference analysis, the RefFrac component analysis, the RefFrac main-table analysis.

**5. Decision / next step:** Final run of Paper B; analysed in the RefFrac analyses (Step 20, `Analysis/08` section 10). How Paper B is presented goes to the person.
