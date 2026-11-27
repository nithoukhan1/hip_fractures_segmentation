# Run notes: S04_densenet121

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Baseline 4: DenseNet121 image classifier, the article's best classifier (Table 3), with the shared Step 2 recipe. Analysis C03 asks whether image classifiers grade better than YOLO and whether our retraining reproduces the article's numbers.

**2. What changed compared with the previous run** (config, code commit, data, folds): The network only. The recipe, folds, seeds and decision rule are those of S06 (`Analysis/05_Step2_Baselines_Protocol.md`). Code `4820f6b`, micro-batch 16 × 1, T4 x2, 82.5 min (S00_net_check estimate 1.6 h). Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`; analysis `results/analysis/C03_classifiers_vs_yolo/`):
- OOF: AUC .992, 6-class accuracy 86.8%, macro-F1 .802, Garden I-II recall .579. Threshold .118.
- Test, 5 fold models: AUC .997 ± .001, accuracy 84.8 ± 2.0%, macro-F1 .749 ± .049.
- Test ensemble: AUC .999, accuracy 85.7%, macro-F1 .760.

**4. What we learnt:**
- It reproduces the article's DenseNet121 (AUC .993, accuracy 83.8%, Table 3).
- It beats YOLO (S01) out of fold: accuracy +3.5 points and macro-F1 +.058, both Holm p = .002. This comes mainly from half as many detection errors (32 against 62).
- On the test set, YOLO's ensemble is ahead (88.1% against 85.7%), not significantly.
- It is the best plain classifier: equal to ConvNeXt-T (S06) and better than EfficientNet-B4 (macro-F1 +.047, p = .022).
- Temperature 2.44 (OOF ECE .116 → .026).

**5. Decision / next step:** DenseNet121 joins Paper A's main table (C09) as the strongest plain classifier. ConvNeXt-T stays the backbone of M0, because it is as good and the mask task was shown on it (C05).
