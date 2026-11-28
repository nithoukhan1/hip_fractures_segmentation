# R5_concat: summary

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R5`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. RefFrac the concatenation model: 3 reference(s) per femur (nearest), alignment attention (window +-0.125 of the image height), fusion concat; references encoded by the shared encoder without gradient.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 241.3 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0608

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.996 | 0.998 ± 0.002 | 0.999 | 0.981 |
| Fracture sensitivity | 0.974 | 0.975 ± 0.009 | 0.986 |  |
| Fracture specificity | 0.992 | 0.994 ± 0.008 | 0.985 |  |
| 6-class accuracy (%) | 88.7 | 85.8 ± 0.9 | 84.8 | 86.2 |
| 6-class macro-F1 | 0.827 | 0.765 ± 0.021 | 0.741 |  |
| Garden I-II recall | 0.561 | 0.275 ± 0.084 | 0.188 |  |
| Dice, fractured femurs (missed = 0) | 0.780 | 0.729 ± 0.006 | 0.740 |  |
| Dice, detected fractures only | 0.802 | 0.747 ± 0.003 | 0.751 | 0.77 |
| NSD at 10 px, fractured femurs | 0.246 | 0.196 ± 0.004 | 0.201 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.996 | 0.998 ± 0.002 | 0.999 | 0.981 |
| Fracture sensitivity | 0.966 | 0.971 ± 0.012 | 0.979 |  |
| Fracture specificity | 0.992 | 1.000 ± 0.000 | 1.000 |  |
| 6-class accuracy (%) | 88.7 | 85.9 ± 1.1 | 85.2 | 86.2 |
| 6-class macro-F1 | 0.829 | 0.767 ± 0.023 | 0.746 |  |
| Garden I-II recall | 0.561 | 0.275 ± 0.084 | 0.188 |  |
| Dice, fractured femurs (missed = 0) | 0.778 | 0.727 ± 0.007 | 0.737 |  |
| Dice, detected fractures only | 0.805 | 0.748 ± 0.004 | 0.752 | 0.77 |
| NSD at 10 px, fractured femurs | 0.245 | 0.196 ± 0.004 | 0.201 |  |

## Reference tests (RefFrac, Analysis/08 section 2.6), at the OOF threshold

| Mode | OOF Dice, fractured | OOF macro-F1 | OOF Garden I-II recall | Change of P(fracture), fractured | Change of P(fracture), healthy | Healthy femurs that become fractured | Test ensemble Dice, fractured |
|---|---|---|---|---|---|---|---|
| as used | 0.780 | 0.827 | 0.561 | | | | 0.740 |
| self | 0.733 | 0.772 | 0.421 | -0.117 | -0.006 | 0.0% of 360 | 0.733 |
| random | 0.779 | 0.820 | 0.544 | 0.001 | 0.001 | 0.8% of 360 | 0.741 |

Pointing game of the difference map |T - R~| at 1/16 (base mode, fractured femurs: the cell with the largest difference touches the released mask): OOF 0.353 (chance 0.079, n = 680); test (fold models) 0.272 (chance 0.071, n = 725).

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
