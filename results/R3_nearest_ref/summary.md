# R3_nearest_ref: summary

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R3`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. RefFrac R3: 1 reference(s) per femur (nearest), alignment attention (window +-0.125 of the image height), fusion difference; references encoded by the shared encoder without gradient.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 170.9 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0008

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.993 | 0.996 ± 0.002 | 0.999 | 0.981 |
| Fracture sensitivity | 0.984 | 0.988 ± 0.006 | 1.000 |  |
| Fracture specificity | 0.972 | 0.972 ± 0.017 | 0.938 |  |
| 6-class accuracy (%) | 88.0 | 85.1 ± 1.1 | 84.3 | 86.2 |
| 6-class macro-F1 | 0.826 | 0.773 ± 0.023 | 0.765 |  |
| Garden I-II recall | 0.684 | 0.350 ± 0.056 | 0.375 |  |
| Dice, fractured femurs (missed = 0) | 0.775 | 0.731 ± 0.003 | 0.742 |  |
| Dice, detected fractures only | 0.788 | 0.740 ± 0.003 | 0.742 | 0.77 |
| NSD at 10 px, fractured femurs | 0.241 | 0.193 ± 0.003 | 0.199 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.993 | 0.996 ± 0.002 | 0.999 | 0.981 |
| Fracture sensitivity | 0.956 | 0.970 ± 0.012 | 0.979 |  |
| Fracture specificity | 0.983 | 0.991 ± 0.014 | 1.000 |  |
| 6-class accuracy (%) | 87.3 | 84.8 ± 0.8 | 85.2 | 86.2 |
| 6-class macro-F1 | 0.814 | 0.761 ± 0.022 | 0.767 |  |
| Garden I-II recall | 0.596 | 0.300 ± 0.052 | 0.312 |  |
| Dice, fractured femurs (missed = 0) | 0.770 | 0.725 ± 0.005 | 0.739 |  |
| Dice, detected fractures only | 0.805 | 0.748 ± 0.006 | 0.754 | 0.77 |
| NSD at 10 px, fractured femurs | 0.240 | 0.192 ± 0.003 | 0.199 |  |

## Reference tests (RefFrac, Analysis/08 §2.6), at the OOF threshold

| Mode | OOF Dice, fractured | OOF macro-F1 | OOF Garden I-II recall | Change of P(fracture), fractured | Change of P(fracture), healthy | Healthy femurs that become fractured | Test ensemble Dice, fractured |
|---|---|---|---|---|---|---|---|
| as used | 0.775 | 0.826 | 0.684 | | | | 0.742 |
| self | 0.756 | 0.738 | 0.614 | -0.227 | -0.011 | 0.0% of 353 | 0.725 |
| random | 0.775 | 0.819 | 0.684 | -0.001 | 0.002 | 0.8% of 353 | 0.741 |

Pointing game of the difference map |T - R~| at 1/16 (base mode, fractured femurs: the cell with the largest difference touches the released mask): OOF 0.547 (chance 0.079, n = 680); test (fold models) 0.570 (chance 0.071, n = 725).

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
