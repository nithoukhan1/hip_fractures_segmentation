# R2_random_ref: summary

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R2`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. RefFrac the random-reference model: 1 reference(s) per femur (random), alignment attention (window +-0.125 of the image height), fusion difference; references encoded by the shared encoder without gradient.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 192.5 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0020

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.999 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.978 | 0.982 ± 0.006 | 0.993 |  |
| Fracture specificity | 0.978 | 0.988 ± 0.020 | 0.954 |  |
| 6-class accuracy (%) | 87.8 | 86.1 ± 2.7 | 86.7 | 86.2 |
| 6-class macro-F1 | 0.814 | 0.780 ± 0.036 | 0.804 |  |
| Garden I-II recall | 0.614 | 0.312 ± 0.099 | 0.438 |  |
| Dice, fractured femurs (missed = 0) | 0.781 | 0.731 ± 0.006 | 0.741 |  |
| Dice, detected fractures only | 0.798 | 0.744 ± 0.006 | 0.746 | 0.77 |
| NSD at 10 px, fractured femurs | 0.241 | 0.192 ± 0.005 | 0.198 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.999 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.957 | 0.970 ± 0.008 | 0.986 |  |
| Fracture specificity | 0.986 | 1.000 ± 0.000 | 1.000 |  |
| 6-class accuracy (%) | 87.3 | 86.0 ± 2.0 | 87.6 | 86.2 |
| 6-class macro-F1 | 0.811 | 0.777 ± 0.025 | 0.806 |  |
| Garden I-II recall | 0.579 | 0.287 ± 0.095 | 0.375 |  |
| Dice, fractured femurs (missed = 0) | 0.771 | 0.725 ± 0.007 | 0.741 |  |
| Dice, detected fractures only | 0.805 | 0.747 ± 0.004 | 0.751 | 0.77 |
| NSD at 10 px, fractured femurs | 0.239 | 0.190 ± 0.004 | 0.198 |  |

## Reference tests (RefFrac, Analysis/08 section 2.6), at the OOF threshold

| Mode | OOF Dice, fractured | OOF macro-F1 | OOF Garden I-II recall | Change of P(fracture), fractured | Change of P(fracture), healthy | Healthy femurs that become fractured | Test ensemble Dice, fractured |
|---|---|---|---|---|---|---|---|
| as used | 0.781 | 0.814 | 0.614 | | | | 0.741 |
| self | 0.705 | 0.687 | 0.526 | -0.442 | -0.014 | 0.0% of 355 | 0.727 |
| random | 0.781 | 0.814 | 0.614 | 0.000 | 0.000 | 0.0% of 355 | 0.741 |

Pointing game of the difference map |T - R~| at 1/16 (base mode, fractured femurs: the cell with the largest difference touches the released mask): OOF 0.681 (chance 0.079, n = 680); test (fold models) 0.663 (chance 0.071, n = 725).

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
