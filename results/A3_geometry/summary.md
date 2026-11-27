# A3_geometry: summary

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A3`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. SLG-Net A3: teacher forcing over 30 epochs, region pooling at 1/32.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 142.5 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0577

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.996 | 0.998 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.981 | 0.982 ± 0.010 | 0.993 |  |
| Fracture specificity | 0.972 | 0.969 ± 0.011 | 0.908 |  |
| 6-class accuracy (%) | 88.9 | 86.5 ± 1.1 | 85.7 | 86.2 |
| 6-class macro-F1 | 0.838 | 0.782 ± 0.012 | 0.786 |  |
| Garden I-II recall | 0.684 | 0.312 ± 0.077 | 0.312 |  |
| Dice, fractured femurs (missed = 0) | 0.777 | 0.729 ± 0.007 | 0.734 |  |
| Dice, detected fractures only | 0.792 | 0.742 ± 0.007 | 0.739 | 0.77 |
| NSD at 10 px, fractured femurs | 0.246 | 0.199 ± 0.004 | 0.206 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.996 | 0.998 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.968 | 0.972 ± 0.007 | 0.972 |  |
| Fracture specificity | 0.978 | 0.982 ± 0.007 | 1.000 |  |
| 6-class accuracy (%) | 88.6 | 86.5 ± 1.3 | 87.6 | 86.2 |
| 6-class macro-F1 | 0.833 | 0.781 ± 0.010 | 0.788 |  |
| Garden I-II recall | 0.632 | 0.300 ± 0.052 | 0.250 |  |
| Dice, fractured femurs (missed = 0) | 0.772 | 0.725 ± 0.005 | 0.731 |  |
| Dice, detected fractures only | 0.797 | 0.745 ± 0.004 | 0.752 | 0.77 |
| NSD at 10 px, fractured femurs | 0.244 | 0.198 ± 0.004 | 0.204 |  |

## Interpretability tests (SLG-Net, Analysis/06 §5), at the OOF threshold

| Mode | OOF 6-class accuracy (%) | OOF macro-F1 | OOF Garden I-II recall | Change of P(true class), fractured | Correct fractured femurs that become wrong | Test ensemble accuracy (%) |
|---|---|---|---|---|---|---|
| as used | 88.9 | 0.838 | 0.684 | | | 85.7 |
| oracle | 89.8 | 0.845 | 0.614 | -0.005 | 1.6% of 574 | 88.6 |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
