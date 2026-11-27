# Run notes: S05_efficientnet_b4

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Baseline 5: EfficientNet-B4 image classifier, an article baseline (Table 3), with the shared Step 2 recipe. It enters analysis C03.

**2. What changed compared with the previous run** (config, code commit, data, folds): The network only, as S04. It used micro-batch 8 × 2: 16 needed 14.6 GB, over the 85% rule of S00_net_check. Code `4820f6b`, T4 x2, 115.2 min (estimate 2.1 h). Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`; analysis `results/analysis/C03_classifiers_vs_yolo/`):
- OOF: AUC .990, 6-class accuracy 84.8%, macro-F1 .755, Garden I-II recall .386. Threshold .449.
- Test, 5 fold models: AUC .990 ± .002, accuracy 80.6 ± 1.5%, macro-F1 .736 ± .021.
- Test ensemble: AUC .994, accuracy 81.0%, macro-F1 .744.

**4. What we learnt:**
- It reproduces the article's EfficientNet-B4 (AUC .990, accuracy 79.0%).
- It is the weakest classifier here: out of fold, its macro-F1 is no better than YOLO's (+.011, p = .59), although its AUC is (+.015, p = .001). It has the weakest Garden I-II recall (.386).
- Temperature 3.27 (OOF ECE .122 → .025). One OOF femur has its true class written as probability 0, below the per-case file's 8 decimals; `run_stats.py` now handles this (2026-11-27).

**5. Decision / next step:** Kept as a baseline in the tables. No further EfficientNet runs (the EfficientNetV2-S sensitivity run of Report 04 §4 stays planned for Paper A).
