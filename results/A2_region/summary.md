# A2_region: summary

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A2`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. SLG-Net the region-pooling model: teacher forcing over 30 epochs, region pooling at 1/32.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 151.4 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0581

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.996 | 0.999 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.978 | 0.985 ± 0.008 | 0.993 |  |
| Fracture specificity | 0.986 | 0.988 ± 0.013 | 0.954 |  |
| 6-class accuracy (%) | 88.3 | 86.3 ± 1.0 | 86.2 | 86.2 |
| 6-class macro-F1 | 0.820 | 0.779 ± 0.010 | 0.785 |  |
| Garden I-II recall | 0.544 | 0.263 ± 0.052 | 0.250 |  |
| Dice, fractured femurs (missed = 0) | 0.772 | 0.726 ± 0.007 | 0.738 |  |
| Dice, detected fractures only | 0.790 | 0.737 ± 0.009 | 0.743 | 0.77 |
| NSD at 10 px, fractured femurs | 0.238 | 0.198 ± 0.010 | 0.213 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.996 | 0.999 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.965 | 0.972 ± 0.015 | 0.979 |  |
| Fracture specificity | 0.989 | 0.994 ± 0.008 | 1.000 |  |
| 6-class accuracy (%) | 87.7 | 86.0 ± 1.1 | 87.1 | 86.2 |
| 6-class macro-F1 | 0.812 | 0.775 ± 0.011 | 0.791 |  |
| Garden I-II recall | 0.526 | 0.250 ± 0.044 | 0.250 |  |
| Dice, fractured femurs (missed = 0) | 0.768 | 0.721 ± 0.008 | 0.735 |  |
| Dice, detected fractures only | 0.796 | 0.741 ± 0.008 | 0.750 | 0.77 |
| NSD at 10 px, fractured femurs | 0.236 | 0.196 ± 0.008 | 0.212 |  |

## Interpretability tests (SLG-Net, Analysis/06 section 5), at the OOF threshold

| Mode | OOF 6-class accuracy (%) | OOF macro-F1 | OOF Garden I-II recall | Change of P(true class), fractured | Correct fractured femurs that become wrong | Test ensemble accuracy (%) |
|---|---|---|---|---|---|---|
| as used | 88.3 | 0.820 | 0.544 | | | 86.2 |
| oracle | 89.2 | 0.832 | 0.544 | 0.002 | 1.6% of 563 | 89.0 |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
