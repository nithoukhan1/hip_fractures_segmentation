# S03_yolo11s_seg_1024: summary

Model yolo11s-seg.pt, imgsz 1024, 100 epochs, batch 8, final weights (last.pt).
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 176.9 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.132935

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, YOLOv8-small at this image size (test, 6 runs) |
|---|---|---|---|---|
| Fracture AUC | 0.974 | 0.988 ± 0.006 | 0.994 | 0.986 |
| Fracture sensitivity | 0.928 | 0.941 ± 0.008 | 0.972 |  |
| Fracture specificity | 0.956 | 0.948 ± 0.028 | 0.969 |  |
| 6-class accuracy (%) | 82.2 | 81.1 ± 4.0 | 85.2 | 84.4 |
| 6-class macro-F1 | 0.719 | 0.750 ± 0.062 | 0.801 |  |
| Dice, fractured femurs (missed = 0) | 0.743 | 0.693 ± 0.007 | 0.730 |  |
| Dice, detected fractures only | 0.800 | 0.737 ± 0.005 | 0.750 | 0.761 |
| Garden I-II recall | 0.386 | 0.463 ± 0.137 | 0.438 |  |

## Ultralytics' default threshold 0.25 (probably the paper's setting; not reported there)

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, YOLOv8-small at this image size (test, 6 runs) |
|---|---|---|---|---|
| Fracture AUC | 0.974 | 0.988 ± 0.006 | 0.994 | 0.986 |
| Fracture sensitivity | 0.912 | 0.913 ± 0.017 | 0.952 |  |
| Fracture specificity | 0.961 | 0.969 ± 0.027 | 0.985 |  |
| 6-class accuracy (%) | 81.8 | 80.9 ± 3.2 | 84.3 | 84.4 |
| 6-class macro-F1 | 0.712 | 0.738 ± 0.049 | 0.781 |  |
| Dice, fractured femurs (missed = 0) | 0.730 | 0.675 ± 0.010 | 0.715 |  |
| Dice, detected fractures only | 0.801 | 0.740 ± 0.007 | 0.752 | 0.761 |
| Garden I-II recall | 0.351 | 0.388 ± 0.143 | 0.375 |  |

Paper column: González et al. 2025, YOLOv8-small grid, mean of 6 seeds on the test set (their Dice definition is not stated). Their final selected model (small, 1024) reports AUC 0.981, sensitivity 0.952, specificity 0.985, 6-class accuracy 86.19% and Dice 0.77.
Differences from the paper: one model per fold of our 5 folds instead of 6 seeds on one split; final weights instead of the best-validation-loss checkpoint; Ultralytics defaults of the pinned version.
