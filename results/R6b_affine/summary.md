# R6b_affine: summary

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R6b`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. RefFrac the affine-alignment model: 3 reference(s) per femur (nearest), alignment affine, fusion difference; references encoded by the shared encoder without gradient.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 265.4 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0167

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.995 | 0.998 ± 0.001 | 0.998 | 0.981 |
| Fracture sensitivity | 0.971 | 0.974 ± 0.003 | 0.986 |  |
| Fracture specificity | 0.989 | 0.988 ± 0.013 | 0.954 |  |
| 6-class accuracy (%) | 87.7 | 85.3 ± 0.7 | 86.2 | 86.2 |
| 6-class macro-F1 | 0.814 | 0.763 ± 0.024 | 0.792 |  |
| Garden I-II recall | 0.561 | 0.287 ± 0.105 | 0.312 |  |
| Dice, fractured femurs (missed = 0) | 0.776 | 0.729 ± 0.004 | 0.742 |  |
| Dice, detected fractures only | 0.799 | 0.749 ± 0.004 | 0.753 | 0.77 |
| NSD at 10 px, fractured femurs | 0.240 | 0.194 ± 0.007 | 0.205 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.995 | 0.998 ± 0.001 | 0.998 | 0.981 |
| Fracture sensitivity | 0.960 | 0.957 ± 0.006 | 0.966 |  |
| Fracture specificity | 0.992 | 0.994 ± 0.008 | 1.000 |  |
| 6-class accuracy (%) | 87.4 | 84.9 ± 0.9 | 86.7 | 86.2 |
| 6-class macro-F1 | 0.810 | 0.760 ± 0.021 | 0.796 |  |
| Garden I-II recall | 0.526 | 0.287 ± 0.105 | 0.312 |  |
| Dice, fractured femurs (missed = 0) | 0.771 | 0.719 ± 0.006 | 0.731 |  |
| Dice, detected fractures only | 0.803 | 0.751 ± 0.004 | 0.757 | 0.77 |
| NSD at 10 px, fractured femurs | 0.239 | 0.190 ± 0.007 | 0.200 |  |

## Reference tests (RefFrac, Analysis/08 section 2.6), at the OOF threshold

| Mode | OOF Dice, fractured | OOF macro-F1 | OOF Garden I-II recall | Change of P(fracture), fractured | Change of P(fracture), healthy | Healthy femurs that become fractured | Test ensemble Dice, fractured |
|---|---|---|---|---|---|---|---|
| as used | 0.776 | 0.814 | 0.561 | | | | 0.742 |
| self | 0.133 | 0.250 | 0.000 | -0.864 | -0.008 | 0.0% of 359 | 0.173 |
| random | 0.772 | 0.813 | 0.561 | -0.003 | 0.004 | 0.6% of 359 | 0.743 |

Pointing game of the difference map |T - R~| at 1/16 (base mode, fractured femurs: the cell with the largest difference touches the released mask): OOF 0.394 (chance 0.079, n = 680); test (fold models) 0.389 (chance 0.071, n = 725).

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
