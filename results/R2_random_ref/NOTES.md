# Run notes: R2_random_ref

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B final run the random-reference model: as the nearest-reference model with a random healthy reference of the fold's bank; settings frozen by the RefFrac development analysis (Analysis/08 sections 1, 2, 5 and 8), tag B-v1-frozen (1c88912); 5 folds and the test set; the RefFrac analyses (Step 20).

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `1c88912` = tag `B-v1-frozen` (the launcher log shows this commit), T4 x2, all 5 folds trained, exit 0. 192.5 min. Question: as the nearest-reference model with a random healthy reference: does any healthy second image help (vs ConvNeXt U-Net)?.

**3. Result** (`metrics.json`, `summary.md`; threshold fitted on the OOF femurs only):
- Out of fold (1,043 femurs): AUC 0.994, accuracy 916/1043, macro-F1 0.814, Dice fractured 0.781, Garden I-II recall 0.614. Threshold 0.0020.
- Test ensemble (210 femurs): AUC 0.999, accuracy 182/210, macro-F1 0.804, Dice fractured 0.741, Garden I-II recall 0.438.
- Reference tests, OOF (`../analysis/C13_paperB_reference/reftests.md`): self (vs base): Dice 0.781 → 0.705, Garden I-II 0.614 → 0.526, fractures lost 64/665; random (vs base): Dice 0.781 → 0.781, Garden I-II 0.614 → 0.614, fractures lost 0/665. Pointing game 0.68 (chance 0.079).

**4. What we learnt:** A random healthy reference gives the same results as a retrieved one (reference analysis): it has the best Dice of the reference analysis (.781, n.s.) and the best pointing game (.68).

**5. Decision / next step:** Final run of Paper B; analysed in the RefFrac analyses (Step 20, `Analysis/08` section 10). How Paper B is presented goes to the person.
