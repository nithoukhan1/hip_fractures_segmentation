# A4_hierarchy: summary

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A4`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. SLG-Net A4: teacher forcing over 30 epochs, region pooling at 1/32.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 142.6 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0149

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.998 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.979 | 0.975 ± 0.004 | 0.993 |  |
| Fracture specificity | 0.986 | 0.985 ± 0.015 | 0.969 |  |
| 6-class accuracy (%) | 87.8 | 86.3 ± 1.3 | 87.1 | 86.2 |
| 6-class macro-F1 | 0.823 | 0.775 ± 0.016 | 0.800 |  |
| Garden I-II recall | 0.649 | 0.287 ± 0.084 | 0.375 |  |
| Dice, fractured femurs (missed = 0) | 0.770 | 0.727 ± 0.005 | 0.732 |  |
| Dice, detected fractures only | 0.786 | 0.745 ± 0.004 | 0.737 | 0.77 |
| NSD at 10 px, fractured femurs | 0.236 | 0.193 ± 0.006 | 0.195 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.998 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.956 | 0.966 ± 0.007 | 0.972 |  |
| Fracture specificity | 0.992 | 0.994 ± 0.008 | 1.000 |  |
| 6-class accuracy (%) | 87.2 | 86.1 ± 1.6 | 86.7 | 86.2 |
| 6-class macro-F1 | 0.814 | 0.771 ± 0.018 | 0.779 |  |
| Garden I-II recall | 0.579 | 0.263 ± 0.052 | 0.250 |  |
| Dice, fractured femurs (missed = 0) | 0.763 | 0.724 ± 0.004 | 0.732 |  |
| Dice, detected fractures only | 0.798 | 0.750 ± 0.003 | 0.753 | 0.77 |
| NSD at 10 px, fractured femurs | 0.234 | 0.191 ± 0.005 | 0.195 |  |

## Interpretability tests (SLG-Net, Analysis/06 §5), at the OOF threshold

| Mode | OOF 6-class accuracy (%) | OOF macro-F1 | OOF Garden I-II recall | Change of P(true class), fractured | Correct fractured femurs that become wrong | Test ensemble accuracy (%) |
|---|---|---|---|---|---|---|
| as used | 87.8 | 0.823 | 0.649 | | | 87.1 |
| oracle | 88.3 | 0.825 | 0.649 | 0.006 | 1.6% of 558 | 88.1 |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
