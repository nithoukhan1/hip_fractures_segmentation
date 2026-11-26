# S02_yolov8s_seg_640: summary

Model yolov8s-seg.pt, imgsz 640, 100 epochs, batch 16, final weights (last.pt).
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 80.5 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.045197

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, same setting (test, 6 seeds) |
|---|---|---|---|---|
| Fracture AUC | 0.975 | 0.979 ± 0.007 | 0.994 | .985 |
| Fracture sensitivity | 0.935 | 0.939 ± 0.013 | 0.972 |  |
| Fracture specificity | 0.970 | 0.972 ± 0.007 | 0.938 |  |
| 6-class accuracy (%) | 82.6 | 82.2 ± 2.1 | 83.8 | 82.0 |
| 6-class macro-F1 | 0.751 | 0.758 ± 0.036 | 0.787 |  |
| Dice, fractured femurs (missed = 0) | 0.746 | 0.696 ± 0.009 | 0.731 |  |
| Dice, detected fractures only | 0.798 | 0.741 ± 0.002 | 0.752 | .780 |
| Garden I-II recall | 0.544 | 0.412 ± 0.201 | 0.562 |  |

## Ultralytics' default threshold 0.25 (probably the paper's setting; not reported there)

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, same setting (test, 6 seeds) |
|---|---|---|---|---|
| Fracture AUC | 0.975 | 0.979 ± 0.007 | 0.994 | .985 |
| Fracture sensitivity | 0.899 | 0.906 ± 0.014 | 0.917 |  |
| Fracture specificity | 0.983 | 0.991 ± 0.008 | 1.000 |  |
| 6-class accuracy (%) | 81.8 | 81.2 ± 2.2 | 82.9 | 82.0 |
| 6-class macro-F1 | 0.735 | 0.737 ± 0.040 | 0.738 |  |
| Dice, fractured femurs (missed = 0) | 0.720 | 0.676 ± 0.010 | 0.699 |  |
| Dice, detected fractures only | 0.802 | 0.746 ± 0.003 | 0.762 | .780 |
| Garden I-II recall | 0.439 | 0.338 ± 0.151 | 0.312 |  |

Paper column: González et al. 2025, YOLOv8-small grid, mean of 6 seeds on the test set (their Dice definition is not stated). Their final selected model (small, 1024) reports AUC 0.981, sensitivity 0.952, specificity 0.985, 6-class accuracy 86.19% and Dice 0.77.
Differences from the paper: one model per fold of our 5 folds instead of 6 seeds on one split; final weights instead of the best-validation-loss checkpoint; Ultralytics defaults of the pinned version.
