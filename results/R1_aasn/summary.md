# R1_aasn: summary

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R1`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. RefFrac the AASN-style model: 1 reference(s) per femur (nearest), alignment affine, fusion concat, margin loss 0.5; references encoded by the shared encoder without gradient.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 177.6 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0011

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.999 ± 0.000 | 0.999 | 0.981 |
| Fracture sensitivity | 0.982 | 0.986 ± 0.005 | 0.986 |  |
| Fracture specificity | 0.981 | 0.982 ± 0.017 | 0.954 |  |
| 6-class accuracy (%) | 87.7 | 85.4 ± 1.1 | 86.2 | 86.2 |
| 6-class macro-F1 | 0.820 | 0.777 ± 0.016 | 0.798 |  |
| Garden I-II recall | 0.579 | 0.375 ± 0.062 | 0.438 |  |
| Dice, fractured femurs (missed = 0) | 0.781 | 0.728 ± 0.003 | 0.738 |  |
| Dice, detected fractures only | 0.795 | 0.738 ± 0.003 | 0.748 | 0.77 |
| NSD at 10 px, fractured femurs | 0.243 | 0.188 ± 0.007 | 0.194 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.999 ± 0.000 | 0.999 | 0.981 |
| Fracture sensitivity | 0.974 | 0.967 ± 0.010 | 0.979 |  |
| Fracture specificity | 0.986 | 0.994 ± 0.008 | 1.000 |  |
| 6-class accuracy (%) | 87.6 | 84.9 ± 1.2 | 87.1 | 86.2 |
| 6-class macro-F1 | 0.817 | 0.770 ± 0.020 | 0.808 |  |
| Garden I-II recall | 0.561 | 0.350 ± 0.084 | 0.438 |  |
| Dice, fractured femurs (missed = 0) | 0.780 | 0.721 ± 0.006 | 0.734 |  |
| Dice, detected fractures only | 0.801 | 0.746 ± 0.004 | 0.749 | 0.77 |
| NSD at 10 px, fractured femurs | 0.243 | 0.186 ± 0.007 | 0.193 |  |

## Reference tests (RefFrac, Analysis/08 section 2.6), at the OOF threshold

| Mode | OOF Dice, fractured | OOF macro-F1 | OOF Garden I-II recall | Change of P(fracture), fractured | Change of P(fracture), healthy | Healthy femurs that become fractured | Test ensemble Dice, fractured |
|---|---|---|---|---|---|---|---|
| as used | 0.781 | 0.820 | 0.579 | | | | 0.738 |
| self | 0.768 | 0.820 | 0.561 | -0.077 | -0.009 | 0.0% of 356 | 0.735 |
| random | 0.781 | 0.813 | 0.561 | -0.000 | 0.000 | 0.0% of 356 | 0.737 |

Pointing game of the difference map |T - R~| at 1/16 (base mode, fractured femurs: the cell with the largest difference touches the released mask): OOF 0.419 (chance 0.079, n = 680); test (fold models) 0.415 (chance 0.071, n = 725).

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
