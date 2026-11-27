# Run notes: S03_yolo11s_seg_1024

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Baseline 3: YOLO11s-seg at 1024 px with exactly S01's protocol, so that it differs from S01 only in the architecture. Question of analysis C02: does a newer YOLO change anything? Expected: about the same as S01.

**2. What changed compared with the previous run** (config, code commit, data, folds): Model `yolo11s-seg.pt` instead of `yolov8s-seg.pt`. Same folds, data, Ultralytics 8.4.219, 100 epochs, batch 8, final weights, OOF threshold. Code `b1a2719` (main after PR #4), T4 x2, 177.6 min. Kaggle version number not recorded.

**3. Result** (`metrics.json`, `summary.md`; analysis `results/analysis/C02_yolov8_vs_yolo11/`):
- OOF: AUC .974, 6-class accuracy 82.2%, macro-F1 .719, Dice fractured .743.
- Test, 5 fold models: accuracy 81.1 ± 4.0%, macro-F1 .750 ± .062, Dice fractured .693 ± .007.
- Test ensemble: AUC .994, accuracy 85.2%, macro-F1 .801, Dice fractured .730. OOF threshold .133.

**4. What we learnt:** No difference from S01 on any endpoint (smallest Holm-corrected p = .72). Classification is slightly lower and varies more between fold models (78.1–88.1% against 82.4–88.1%). Segmentation is the same (Dice on detected fractures .800 OOF for both).

**5. Decision / next step:** S01 stays the YOLO baseline of both papers. S03 enters the baseline table as it is. No further YOLO variants.
