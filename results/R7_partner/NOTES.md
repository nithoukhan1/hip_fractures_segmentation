# Run notes: R7_partner

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B final run the own-hip model: trained with the femur's own healthy partner (mirrored), else the nearest; test modes blind, retrieved, random, fractured; settings frozen by the RefFrac development analysis (Analysis/08 sections 1, 2, 5 and 8), tag B-v1-frozen (1c88912); 5 folds and the test set; the RefFrac analyses (Step 20).

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `1c88912` = tag `B-v1-frozen` (the launcher log shows this commit), T4 x2, all 5 folds trained, exit 0. 193.1 min. Question: the patient's own other hip as the reference (own-hip question) and the pairing shortcut (pairing-shortcut question).

**3. Result** (`metrics.json`, `summary.md`; threshold fitted on the OOF femurs only):
- Base mode (training policy; uses the partner's label, so only the size of the leak). Out of fold (1,043 femurs): AUC 0.995, accuracy 916/1043, macro-F1 0.819, Dice fractured 0.780, Garden I-II recall 0.632. Threshold 0.0601.
- Test ensemble (210 femurs): AUC 0.999, accuracy 182/210, macro-F1 0.796, Dice fractured 0.741, Garden I-II recall 0.375.
- Reference tests, OOF (`../analysis/C13_paperB_reference/reftests.md`): blind (vs base): Dice 0.780 → 0.780, Garden I-II 0.632 → 0.632, fractures lost 0/663; retrieved (vs blind): Dice 0.780 → 0.781, Garden I-II 0.632 → 0.632, fractures lost 1/664; random (vs blind): Dice 0.780 → 0.779, Garden I-II 0.632 → 0.632, fractures lost 1/664; fractured (vs blind): Dice 0.780 → 0.743, Garden I-II 0.632 → 0.596, fractures lost 11/664. Pointing game 0.66 (chance 0.079).

**4. What we learnt:** The own other hip (blind mode, label-free) is no better than a retrieved femur (Dice −.001); the own-hip model did not learn the pairing shortcut (base and blind give the same decisions); a fractured reference lowers P(fracture) of fractured femurs by .116 (own-hip analysis).

**5. Decision / next step:** Final run of Paper B; analysed in the RefFrac analyses (Step 20, `Analysis/08` section 10). How Paper B is presented goes to the person.
