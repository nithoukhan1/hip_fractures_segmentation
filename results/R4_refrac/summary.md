# R4_refrac: summary

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R4`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. RefFrac: 3 reference(s) per femur (nearest), alignment attention (window +-0.125 of the image height), fusion difference; references encoded by the shared encoder without gradient.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 241.5 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0071

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.998 ± 0.002 | 1.000 | 0.981 |
| Fracture sensitivity | 0.975 | 0.981 ± 0.006 | 0.986 |  |
| Fracture specificity | 0.981 | 0.997 ± 0.007 | 0.985 |  |
| 6-class accuracy (%) | 88.2 | 86.0 ± 0.9 | 85.7 | 86.2 |
| 6-class macro-F1 | 0.826 | 0.777 ± 0.021 | 0.779 |  |
| Garden I-II recall | 0.702 | 0.325 ± 0.052 | 0.312 |  |
| Dice, fractured femurs (missed = 0) | 0.777 | 0.730 ± 0.004 | 0.739 |  |
| Dice, detected fractures only | 0.797 | 0.744 ± 0.001 | 0.749 | 0.77 |
| NSD at 10 px, fractured femurs | 0.241 | 0.192 ± 0.005 | 0.196 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.998 ± 0.002 | 1.000 | 0.981 |
| Fracture sensitivity | 0.965 | 0.961 ± 0.006 | 0.966 |  |
| Fracture specificity | 0.989 | 1.000 ± 0.000 | 1.000 |  |
| 6-class accuracy (%) | 88.0 | 85.1 ± 1.0 | 84.8 | 86.2 |
| 6-class macro-F1 | 0.821 | 0.762 ± 0.028 | 0.762 |  |
| Garden I-II recall | 0.649 | 0.287 ± 0.105 | 0.250 |  |
| Dice, fractured femurs (missed = 0) | 0.775 | 0.722 ± 0.003 | 0.729 |  |
| Dice, detected fractures only | 0.803 | 0.751 ± 0.003 | 0.755 | 0.77 |
| NSD at 10 px, fractured femurs | 0.240 | 0.190 ± 0.004 | 0.192 |  |

## Reference tests (RefFrac, Analysis/08 section 2.6), at the OOF threshold

| Mode | OOF Dice, fractured | OOF macro-F1 | OOF Garden I-II recall | Change of P(fracture), fractured | Change of P(fracture), healthy | Healthy femurs that become fractured | Test ensemble Dice, fractured |
|---|---|---|---|---|---|---|---|
| as used | 0.777 | 0.826 | 0.702 | | | | 0.739 |
| self | 0.536 | 0.627 | 0.158 | -0.586 | -0.011 | 0.0% of 356 | 0.680 |
| random | 0.775 | 0.817 | 0.667 | -0.002 | 0.002 | 0.8% of 356 | 0.737 |

Pointing game of the difference map |T - R~| at 1/16 (base mode, fractured femurs: the cell with the largest difference touches the released mask): OOF 0.525 (chance 0.079, n = 680); test (fold models) 0.499 (chance 0.071, n = 725).

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
