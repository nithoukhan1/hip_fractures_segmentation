# S10_m0_convnext_unet: summary

Network `m0:convnext_tiny.fb_in22k_ft_in1k`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 139.9 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0182

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.995 | 0.999 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.972 | 0.983 ± 0.006 | 0.993 |  |
| Fracture specificity | 0.992 | 1.000 ± 0.000 | 1.000 |  |
| 6-class accuracy (%) | 88.4 | 86.0 ± 0.5 | 86.7 | 86.2 |
| 6-class macro-F1 | 0.825 | 0.776 ± 0.017 | 0.779 |  |
| Garden I-II recall | 0.544 | 0.312 ± 0.077 | 0.312 |  |
| Dice, fractured femurs (missed = 0) | 0.777 | 0.734 ± 0.005 | 0.744 |  |
| Dice, detected fractures only | 0.799 | 0.746 ± 0.002 | 0.749 | 0.77 |
| NSD at 10 px, fractured femurs | 0.248 | 0.196 ± 0.004 | 0.202 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.995 | 0.999 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.957 | 0.974 ± 0.011 | 0.972 |  |
| Fracture specificity | 0.992 | 1.000 ± 0.000 | 1.000 |  |
| 6-class accuracy (%) | 88.0 | 85.5 ± 0.5 | 85.7 | 86.2 |
| 6-class macro-F1 | 0.821 | 0.773 ± 0.016 | 0.762 |  |
| Garden I-II recall | 0.544 | 0.312 ± 0.077 | 0.250 |  |
| Dice, fractured femurs (missed = 0) | 0.771 | 0.728 ± 0.009 | 0.734 |  |
| Dice, detected fractures only | 0.806 | 0.748 ± 0.003 | 0.755 | 0.77 |
| NSD at 10 px, fractured femurs | 0.246 | 0.195 ± 0.005 | 0.200 |  |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
