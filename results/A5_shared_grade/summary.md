# A5_shared_grade: summary

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A5`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. SLG-Net A5: teacher forcing over 30 epochs, region pooling at 1/32.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 144.7 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0077

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.998 ± 0.003 | 1.000 | 0.981 |
| Fracture sensitivity | 0.982 | 0.978 ± 0.009 | 0.993 |  |
| Fracture specificity | 0.986 | 0.994 ± 0.008 | 0.985 |  |
| 6-class accuracy (%) | 88.0 | 85.5 ± 1.0 | 87.1 | 86.2 |
| 6-class macro-F1 | 0.820 | 0.774 ± 0.015 | 0.792 |  |
| Garden I-II recall | 0.614 | 0.300 ± 0.052 | 0.312 |  |
| Dice, fractured femurs (missed = 0) | 0.769 | 0.727 ± 0.005 | 0.734 |  |
| Dice, detected fractures only | 0.783 | 0.743 ± 0.002 | 0.739 | 0.77 |
| NSD at 10 px, fractured femurs | 0.238 | 0.200 ± 0.003 | 0.206 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.998 ± 0.003 | 1.000 | 0.981 |
| Fracture sensitivity | 0.953 | 0.960 ± 0.011 | 0.972 |  |
| Fracture specificity | 0.992 | 0.997 ± 0.007 | 1.000 |  |
| 6-class accuracy (%) | 87.1 | 85.0 ± 1.0 | 86.7 | 86.2 |
| 6-class macro-F1 | 0.805 | 0.768 ± 0.017 | 0.779 |  |
| Garden I-II recall | 0.561 | 0.300 ± 0.052 | 0.250 |  |
| Dice, fractured femurs (missed = 0) | 0.763 | 0.719 ± 0.007 | 0.734 |  |
| Dice, detected fractures only | 0.801 | 0.749 ± 0.004 | 0.754 | 0.77 |
| NSD at 10 px, fractured femurs | 0.237 | 0.197 ± 0.003 | 0.206 |  |

## Interpretability tests (SLG-Net, Analysis/06 §5), at the OOF threshold

| Mode | OOF 6-class accuracy (%) | OOF macro-F1 | OOF Garden I-II recall | Change of P(true class), fractured | Correct fractured femurs that become wrong | Test ensemble accuracy (%) |
|---|---|---|---|---|---|---|
| as used | 88.0 | 0.820 | 0.614 | | | 87.1 |
| oracle | 88.6 | 0.826 | 0.614 | 0.010 | 1.6% of 560 | 88.1 |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
