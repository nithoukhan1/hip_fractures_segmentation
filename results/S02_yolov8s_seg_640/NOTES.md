# Run notes: S02_yolov8s_seg_640

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Baseline 2: the paper's best-Dice setting, YOLOv8s-seg @640, same protocol. Hypothesis: 640 px gives better masks than 1024 px (article: Dice .780 ± .011 vs .761 ± .008) with slightly lower accuracy (82.0 ± 3.0%).

**2. What changed compared with the previous run** (config, code commit, data, folds): Only the image size (640 instead of 1024) and Ultralytics' default batch 16 (instead of 8). Same code `29e1852`, folds, epochs and settings as S01. Run in a parallel Kaggle session (notebook `S02_yolov8s_seg_640.ipynb`, started 10:54 UTC). Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`; analysis in `results/analysis/C01_yolo_1024_vs_640/`):
- All 5 folds done, exit 0, 81 min (23–27 min of training per fold, 4.3 GB GPU memory). OOF threshold (Youden) 0.045.
- Test, 5 fold models: AUC .979 ± .007, accuracy 82.2 ± 2.1%, macro-F1 .758 ± .036, Dice detected .741 ± .002, Dice fractured (missed = 0) .696 ± .009, Garden I–II recall .412 ± .201.
- Test, 5-model ensemble: AUC .994, accuracy 83.8% (176/210, exact CI 78.1–88.5), macro-F1 .787, 34 errors (detection 8, location 5, grade 21).
- OOF (n = 1,043): AUC .975, accuracy 82.6%, macro-F1 .751, Dice fractured .746, Garden I–II recall .544 (.411–.673).

**4. What we learnt:** Accuracy matches the article (82.2 vs 82.0%). The hypothesis about Dice did not hold: Dice is the same at 640 and 1024 px (.741 vs .741 detected; paired Wilcoxon p = .50 OOF, .56 test) and lower than the article's .780. S01 vs S02 shows no significant difference on any endpoint after the Holm correction (test-ensemble accuracy 88.1 vs 83.8%, raw p = .049, Holm p = .20; OOF accuracy 83.2 vs 82.6%, p = .64). The same over-segmentation of the smaller test neck masks is seen as in S01.

**5. Decision / next step:** Keep S02 as the second YOLO baseline in the tables (the article's best-Dice setting). Resolution is not a lever for YOLO on this data, so S03 (YOLO11s-seg) runs at 1024 px to compare like with like with S01.
