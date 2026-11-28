# Run notes: A6_dev_no_tf

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Development the teacher-forcing question (`Analysis/06_Step3_PaperA_Protocol.md` section 4): is teacher forcing needed? Same as A6_dev_base, but the heads always get the predicted mask (`--tf-epochs 0`).

**2. What changed compared with the previous run** (config, code commit, data, folds): Teacher forcing only, compared with A6_dev_base. Code `f342b8a`, one T4 (GPU 1 of the group `A_dev_round1`), 51.9 min. Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`; analysis `results/analysis/C06_paperA_dev_fold0/`):
- Fold 0: AUC .998, accuracy 184/208, macro-F1 .783 (+.008 vs the base), Dice fractured .754, Garden I-II recall .727. Threshold .0004.
- Oracle test: accuracy 185/208, macro-F1 .791. Only 1 of 23 wrong fractured femurs is fixed.

**4. What we learnt:** Without teacher forcing, the heads hardly use the region: the released mask changes one femur, against six for the base.

**5. Decision / next step:** By the rule (needs ≥ +.02), teacher forcing over 30 epochs stays.
