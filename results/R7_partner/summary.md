# R7_partner: summary

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R7`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. RefFrac the own-hip model: 1 reference(s) per femur (partner), alignment attention (window +-0.125 of the image height), fusion difference; references encoded by the shared encoder without gradient.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 193.1 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0601

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.995 | 0.998 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.975 | 0.978 ± 0.006 | 0.986 |  |
| Fracture specificity | 0.992 | 0.994 ± 0.008 | 0.985 |  |
| 6-class accuracy (%) | 87.8 | 85.5 ± 1.6 | 86.7 | 86.2 |
| 6-class macro-F1 | 0.819 | 0.772 ± 0.020 | 0.796 |  |
| Garden I-II recall | 0.632 | 0.375 ± 0.088 | 0.375 |  |
| Dice, fractured femurs (missed = 0) | 0.780 | 0.728 ± 0.003 | 0.741 |  |
| Dice, detected fractures only | 0.800 | 0.745 ± 0.003 | 0.752 | 0.77 |
| NSD at 10 px, fractured femurs | 0.243 | 0.190 ± 0.004 | 0.199 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.995 | 0.998 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.966 | 0.966 ± 0.010 | 0.979 |  |
| Fracture specificity | 0.992 | 0.997 ± 0.007 | 1.000 |  |
| 6-class accuracy (%) | 87.4 | 85.4 ± 1.6 | 86.7 | 86.2 |
| 6-class macro-F1 | 0.812 | 0.772 ± 0.017 | 0.789 |  |
| Garden I-II recall | 0.614 | 0.350 ± 0.056 | 0.312 |  |
| Dice, fractured femurs (missed = 0) | 0.775 | 0.723 ± 0.004 | 0.737 |  |
| Dice, detected fractures only | 0.802 | 0.749 ± 0.005 | 0.753 | 0.77 |
| NSD at 10 px, fractured femurs | 0.242 | 0.188 ± 0.005 | 0.197 |  |

## Reference tests (RefFrac, Analysis/08 section 2.6), at the OOF threshold

| Mode | OOF Dice, fractured | OOF macro-F1 | OOF Garden I-II recall | Change of P(fracture), fractured | Change of P(fracture), healthy | Healthy femurs that become fractured | Test ensemble Dice, fractured |
|---|---|---|---|---|---|---|---|
| as used | 0.780 | 0.819 | 0.632 | | | | 0.741 |
| blind | 0.780 | 0.819 | 0.632 | 0.000 | -0.002 | 0.0% of 360 | 0.741 |
| retrieved | 0.781 | 0.822 | 0.632 | 0.002 | 0.000 | 0.0% of 360 | 0.741 |
| random | 0.778 | 0.815 | 0.632 | 0.002 | 0.000 | 0.3% of 360 | 0.742 |
| fractured | 0.724 | 0.761 | 0.561 | -0.116 | -0.005 | 0.0% of 360 | 0.718 |

Pointing game of the difference map |T - R~| at 1/16 (base mode, fractured femurs: the cell with the largest difference touches the released mask): OOF 0.657 (chance 0.079, n = 680); test (fold models) 0.641 (chance 0.071, n = 725).

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
