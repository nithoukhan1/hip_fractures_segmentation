# Run notes: S01_yolov8s_seg_1024

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Baseline 1: the paper's final model, YOLOv8s-seg @1024, retrained on our 5 folds (paper protocol, 100 epochs, final weights). Hypothesis: with the article's setting we get its test numbers (AUC .986 ± .008, 6-class accuracy 84.4 ± 2.5%, Dice .761 ± .008; Table 2, six runs).

**2. What changed compared with the previous run** (config, code commit, data, folds): First training run. Code `29e1852` (main after PR #3), `scripts/train_yolo_seg.py`, Ultralytics 8.4.219, COCO weights, 100 epochs, batch 8, defaults (AdamW lr 0.001111), `val=False`, final weights. Folds `splits/folds_v1.csv` (Kaggle's val/ pooled into train). Two folds at a time on 2 × T4, Save Version (notebook `S01_yolov8s_seg_1024.ipynb`, started 10:38 UTC). Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`; analysis in `results/analysis/C01_yolo_1024_vs_640/`):
- All 5 folds done, exit 0, 176 min (54–59 min of training per fold, 5.3 GB GPU memory). OOF threshold (Youden) 0.127.
- Test, 5 fold models: AUC .985 ± .005, accuracy 84.6 ± 2.1%, macro-F1 .791 ± .018, Dice detected .741 ± .009, Dice fractured (missed = 0) .693 ± .016, Garden I–II recall .438 ± .117.
- Test, 5-model ensemble: AUC .994 (.988–1.000), accuracy 88.1% (185/210, exact CI 82.9–92.1), macro-F1 .845, 25 errors (detection 6, location 2, grade 17) vs the article's 29 (8, 6, 15).
- OOF (n = 1,043): AUC .975, accuracy 83.2%, macro-F1 .744, Dice fractured .740, Garden I–II recall .509 (.379–.641).

**4. What we learnt:** The hypothesis held for classification (accuracy and AUC equal to the article's) but not fully for Dice (.741 vs .761). Grade errors dominate (location right in 96.6% of found fractures, grade right in 86.1% of those); Garden I–II and Evans III–V are the weak classes. Test neck masks are about 35–40% smaller than training neck masks, so the model over-segments them on the test set (predicted / true area about 2 for Garden I–II, 1.8 for III–IV): this explains most of the OOF-to-test Dice drop (.800 → .741). The article's selection rule (best held-out accuracy) picks fold 1, which scores 83.8% on test: one selected run is not a reliable estimate.

**5. Decision / next step:** YOLOv8s-seg is the reference YOLO baseline for both papers (fold models mean ± SD, ensemble, OOF). Report Dice per location and OOF next to test. Next: YOLO11s-seg (@1024, same protocol) and the Step 2 the six non-YOLO baselines and ConvNeXt U-Net.
