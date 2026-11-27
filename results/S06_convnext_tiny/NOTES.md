# Run notes: S06_convnext_tiny

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Baseline 6 = A0 of Paper A: the ConvNeXt-T image classifier with the shared Step 2 recipe (`Analysis/05_Step2_Baselines_Protocol.md`). It is the classification reference for M0 (analysis C05) and has the backbone of both papers. Expected: close to the article's classifiers (its DenseNet121: 83.8%).

**2. What changed compared with the previous run** (config, code commit, data, folds): First training run of `scripts/train_net.py`. 768 × 512 input, 60 epochs, batch 16 (micro-batch 16 × 1, as S00_net_check measured), AdamW lr 2e-4, final weights. Code `18e4774` (main after PR #6), T4 x2, 87.5 min (estimate 100). Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`; analysis `results/analysis/C05_mask_task_s06_vs_s10/`):
- OOF: AUC .990, 6-class accuracy 86.4%, macro-F1 .784, Garden I-II recall .474.
- Test, 5 fold models: accuracy 83.5 ± 1.6%, macro-F1 .732 ± .024.
- Test ensemble: AUC 1.000, accuracy 84.8%, macro-F1 .742. OOF threshold .123.

**4. What we learnt:** An image classifier at 768 × 512 classifies about as well as YOLO on the test set (fold models 83.5% against S01's 84.6%) and better on the OOF femurs (86.4% against 83.2%). The training loss reaches about 0 by epoch 60, and the probabilities are over-confident (expected calibration error .127 OOF, against .079 for S01). Garden I-II recall on the test set is low (.25–.28, 16 femurs).

**5. Decision / next step:** Reference for C05, which is done: M0 is better. Enters C03 (classifiers S04–S06 vs S01) when S04 and S05 are filed.
