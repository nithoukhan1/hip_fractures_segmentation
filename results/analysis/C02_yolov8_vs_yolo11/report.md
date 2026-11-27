# C02: Does a newer YOLO (YOLO11s-seg, S03) change anything compared with YOLOv8s-seg (S01)?

*Written 2026-11-27 from the filed runs `results/S01_yolov8s_seg_1024/` and `results/S03_yolo11s_seg_1024/` (register lines 3 and 6). Every number below comes from the two files in this folder:*
- `stats.md` / `stats.json`: 95% intervals and paired S01 vs S03 tests (`python scripts/run_stats.py S01_yolov8s_seg_1024 S03_yolo11s_seg_1024 --compare --name C02_yolov8_vs_yolo11`).
- `breakdown.md` / `breakdown.json`: per class, mean ± SD over folds and fold models, errors, areas, comparison with the article (`python scripts/run_breakdown.py S01_yolov8s_seg_1024 S03_yolo11s_seg_1024 --name C02_yolov8_vs_yolo11`).

The two runs differ only in the architecture: same folds, data, Ultralytics version (8.4.219), image size (1024), 100 epochs, batch 8, final weights, and the fracture threshold fitted on the OOF predictions (Youden).

## 1. Verdict: no. YOLO11s-seg is not better than YOLOv8s-seg on any endpoint

| Endpoint | S01 YOLOv8s-seg | S03 YOLO11s-seg | S01 − S03 | p (Holm) |
|---|---|---|---|---|
| **OOF, 1,043 femurs:** fracture AUC | .975 (.967–.983) | .974 (.963–.983) | +.001 | 1.00 |
| 6-class accuracy | 83.2% (81.0–85.4) | 82.2% (79.8–84.4) | +1.1 points | 1.00 |
| Macro-F1 (primary) | .744 (.707–.779) | .719 (.678–.754) | +.025 | .95 |
| Dice, fractured femurs, missed = 0 (primary) | .740 (.722–.755) | .743 (.726–.759) | −.003 | 1.00 |
| **Test, 5-model ensemble, 210 femurs:** fracture AUC | .994 | .994 | .000 | 1.00 |
| 6-class accuracy | 88.1% | 85.2% | +2.9 points | .72 |
| Macro-F1 | .845 | .801 | +.044 | .72 |
| Dice, fractured femurs | .725 | .730 | −.004 | 1.00 |
| **Test, 5 fold models, mean ± SD:** 6-class accuracy | 84.6 ± 2.1% | 81.1 ± 4.0% | | |
| Macro-F1 | .791 ± .018 | .750 ± .062 | | |
| Dice, fractured femurs | .693 ± .016 | .693 ± .007 | | |

Intervals are 95% cluster-bootstrap intervals over x-rays. Tests: DeLong (AUC), McNemar (accuracy), randomisation with whole x-rays swapped (macro-F1), Wilcoxon signed-rank (Dice), Holm-corrected within each set.

1. **Nothing differs significantly** (smallest Holm-corrected p = .72). The point estimates for classification favour S01, and the Dice estimates favour S03 by less than .005.
2. **S03's fold models vary more:** test accuracy 78.1–88.1% against 82.4–88.1% for S01. Its best fold model (fold 3, 88.1%) equals S01's best, so a single selected run could make YOLO11 look as good as YOLOv8, or worse. This is the reason we report all 5 fold models and the ensemble.
3. **Segmentation is the same:** Dice on detected fractures .800 OOF for both; .741 ± .009 (S01) and .737 ± .005 (S03) on the test set. Both over-segment the test neck fractures (predicted / true area 1.27 and 1.33 on the test set, against about 1.0 OOF), because the test masks of neck fractures are 35–40% smaller than the training masks (C01, annotation shift).
4. **Errors:** S03 makes as many detection errors as S01 (OOF 65 vs 62) and slightly more location errors (45 vs 37). Grade errors are the same (76 vs 76).

## 2. Decision

- **S01 (YOLOv8s-seg @1024) stays the YOLO baseline of both papers.** It is the article's own model, and YOLO11 adds nothing.
- S03 enters the baseline table as it is, as evidence that a newer YOLO does not change the picture.
- No further YOLO variants are run, which saves about 3 GPU-hours per variant.
