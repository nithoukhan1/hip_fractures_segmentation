# Run notes: R1_aasn

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B final run the AASN-style model: AASN adapted (nearest reference, global affine alignment, concatenation, margin loss 0.5); settings frozen by the RefFrac development analysis (Analysis/08 sections 1, 2, 5 and 8), tag B-v1-frozen (1c88912); 5 folds and the test set; the RefFrac analyses (Step 20).

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `1c88912` = tag `B-v1-frozen` (the launcher log shows this commit), T4 x2, all 5 folds trained, exit 0. 177.6 min. Question: the closest published method (AASN, Chen et al. 2020), adapted.

**3. Result** (`metrics.json`, `summary.md`; threshold fitted on the OOF femurs only):
- Out of fold (1,043 femurs): AUC 0.994, accuracy 915/1043, macro-F1 0.820, Dice fractured 0.781, Garden I-II recall 0.579. Threshold 0.0011.
- Test ensemble (210 femurs): AUC 0.999, accuracy 181/210, macro-F1 0.798, Dice fractured 0.738, Garden I-II recall 0.438.
- Reference tests, OOF (`../analysis/C13_paperB_reference/reftests.md`): self (vs base): Dice 0.781 → 0.768, Garden I-II 0.579 → 0.561, fractures lost 11/668; random (vs base): Dice 0.781 → 0.781, Garden I-II 0.579 → 0.561, fractures lost 1/668. Pointing game 0.42 (chance 0.079).

**4. What we learnt:** The adapted published method (AASN) matches RefFrac in Dice and macro-F1 but hardly uses its reference (self test loses 11 fractures) and finds fewer Garden I-II fractures out of fold (.579 vs .702, Holm .077); on the test set it finds 7 of 16 vs RefFrac's 5 (RefFrac component analysis).

**5. Decision / next step:** Final run of Paper B; analysed in the RefFrac analyses (Step 20, `Analysis/08` section 10). How Paper B is presented goes to the person.
