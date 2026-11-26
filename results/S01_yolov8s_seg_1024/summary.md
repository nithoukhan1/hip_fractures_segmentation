# S01_yolov8s_seg_1024: summary

Model yolov8s-seg.pt, imgsz 1024, 100 epochs, batch 8, final weights (last.pt).
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 175.4 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.127197

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, same setting (test, 6 seeds) |
|---|---|---|---|---|
| Fracture AUC | 0.975 | 0.985 ± 0.005 | 0.994 | .986 |
| Fracture sensitivity | 0.925 | 0.935 ± 0.014 | 0.966 |  |
| Fracture specificity | 0.970 | 0.978 ± 0.008 | 0.985 |  |
| 6-class accuracy (%) | 83.2 | 84.6 ± 2.1 | 88.1 | 84.4 |
| 6-class macro-F1 | 0.744 | 0.791 ± 0.018 | 0.845 |  |
| Dice, fractured femurs (missed = 0) | 0.740 | 0.693 ± 0.016 | 0.725 |  |
| Dice, detected fractures only | 0.800 | 0.741 ± 0.009 | 0.751 | .761 |
| Garden I-II recall | 0.509 | 0.438 ± 0.117 | 0.500 |  |

## Ultralytics' default threshold 0.25 (probably the paper's setting; not reported there)

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, same setting (test, 6 seeds) |
|---|---|---|---|---|
| Fracture AUC | 0.975 | 0.985 ± 0.005 | 0.994 | .986 |
| Fracture sensitivity | 0.910 | 0.914 ± 0.017 | 0.931 |  |
| Fracture specificity | 0.978 | 0.991 ± 0.008 | 0.985 |  |
| 6-class accuracy (%) | 82.8 | 84.0 ± 2.1 | 87.1 | 84.4 |
| 6-class macro-F1 | 0.737 | 0.779 ± 0.020 | 0.839 |  |
| Dice, fractured femurs (missed = 0) | 0.729 | 0.681 ± 0.018 | 0.701 |  |
| Dice, detected fractures only | 0.801 | 0.744 ± 0.007 | 0.753 | .761 |
| Garden I-II recall | 0.456 | 0.400 ± 0.130 | 0.500 |  |

Paper column: González et al. 2025, YOLOv8-small grid, mean of 6 seeds on the test set (their Dice definition is not stated). Their final selected model (small, 1024) reports AUC 0.981, sensitivity 0.952, specificity 0.985, 6-class accuracy 86.19% and Dice 0.77.
Differences from the paper: one model per fold of our 5 folds instead of 6 seeds on one split; final weights instead of the best-validation-loss checkpoint; Ultralytics defaults of the pinned version.
