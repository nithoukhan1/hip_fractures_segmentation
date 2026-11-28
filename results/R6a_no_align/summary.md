# R6a_no_align: summary

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R6a`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. RefFrac the no-alignment model: 3 reference(s) per femur (nearest), alignment none, fusion difference; references encoded by the shared encoder without gradient.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 229.8 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0013

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.995 | 0.997 ± 0.002 | 0.999 | 0.981 |
| Fracture sensitivity | 0.974 | 0.975 ± 0.006 | 0.986 |  |
| Fracture specificity | 0.983 | 0.985 ± 0.015 | 0.969 |  |
| 6-class accuracy (%) | 87.1 | 85.9 ± 1.5 | 85.7 | 86.2 |
| 6-class macro-F1 | 0.807 | 0.786 ± 0.039 | 0.788 |  |
| Garden I-II recall | 0.561 | 0.338 ± 0.114 | 0.312 |  |
| Dice, fractured femurs (missed = 0) | 0.777 | 0.727 ± 0.004 | 0.741 |  |
| Dice, detected fractures only | 0.798 | 0.746 ± 0.004 | 0.751 | 0.77 |
| NSD at 10 px, fractured femurs | 0.243 | 0.194 ± 0.003 | 0.198 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.995 | 0.997 ± 0.002 | 0.999 | 0.981 |
| Fracture sensitivity | 0.954 | 0.959 ± 0.010 | 0.966 |  |
| Fracture specificity | 0.994 | 1.000 ± 0.000 | 1.000 |  |
| 6-class accuracy (%) | 87.0 | 85.5 ± 1.0 | 85.7 | 86.2 |
| 6-class macro-F1 | 0.804 | 0.779 ± 0.029 | 0.789 |  |
| Garden I-II recall | 0.509 | 0.300 ± 0.081 | 0.312 |  |
| Dice, fractured femurs (missed = 0) | 0.767 | 0.720 ± 0.005 | 0.729 |  |
| Dice, detected fractures only | 0.803 | 0.751 ± 0.004 | 0.755 | 0.77 |
| NSD at 10 px, fractured femurs | 0.239 | 0.192 ± 0.004 | 0.195 |  |

## Reference tests (RefFrac, Analysis/08 section 2.6), at the OOF threshold

| Mode | OOF Dice, fractured | OOF macro-F1 | OOF Garden I-II recall | Change of P(fracture), fractured | Change of P(fracture), healthy | Healthy femurs that become fractured | Test ensemble Dice, fractured |
|---|---|---|---|---|---|---|---|
| as used | 0.777 | 0.807 | 0.561 | | | | 0.741 |
| self | 0.222 | 0.364 | 0.000 | -0.880 | -0.007 | 0.0% of 357 | 0.488 |
| random | 0.773 | 0.795 | 0.526 | -0.000 | 0.003 | 3.1% of 357 | 0.740 |

Pointing game of the difference map |T - R~| at 1/16 (base mode, fractured femurs: the cell with the largest difference touches the released mask): OOF 0.484 (chance 0.079, n = 680); test (fold models) 0.474 (chance 0.071, n = 725).

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
