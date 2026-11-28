# Run notes: R4_dev_base

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B fold-0 development (`Analysis/08_Step5_PaperB_RefFrac_Protocol.md` section 4): RefFrac with the base settings (3 nearest healthy references, local cross-attention within ±1/8 of the image height, difference fusion, no margin loss), seed 2026. Go/no-go and the base of the whole-reference question and the margin-loss question.

**2. What changed compared with the previous run** (config, code commit, data, folds): First RefFrac run; ConvNeXt U-Net plus the reference path. Code `2619b39` (main after PR #13), one T4, micro-batch 16 x 1 (12.6 GB), fold 0 only, no test set. Kaggle version number not recorded. GPU 0 of the group `R_dev_round1`, 83.2 min.

**3. Result** (`metrics.json`; analysis `results/analysis/C12_paperB_dev_fold0/`):
- Fold 0: Dice fractured .765 (.735–.792; ConvNeXt U-Net .749), macro-F1 .817, accuracy 183/208, Garden I-II recall .636. Threshold .0001.
- `self` test: Dice .701 (−.065), P(fracture) on fractured femurs −.457, 11 more fractures missed. `random` test: Dice .761 (−.004).
- Pointing game: 64.0% (chance 8.0%).

**4. What we learnt:** The model uses its references strongly; which healthy femur it gets matters little at prediction time.

**5. Decision / next step:** With `R4_dev_base_s2` it forms the RefFrac development analysis's base: go/no-go passes; the base settings are frozen (`B-v1-frozen`).
