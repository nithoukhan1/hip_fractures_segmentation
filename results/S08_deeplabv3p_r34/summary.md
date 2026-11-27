# S08_deeplabv3p_r34: summary

Network `seg:DeepLabV3Plus:resnet34`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 72.8 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.9539

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.990 | 0.993 ± 0.004 | 0.999 | 0.981 |
| Fracture sensitivity | 0.959 | 0.961 ± 0.010 | 0.931 |  |
| Fracture specificity | 0.964 | 0.978 ± 0.014 | 1.000 |  |
| 6-class accuracy (%) | 86.2 | 85.5 ± 1.9 | 86.2 | 86.2 |
| 6-class macro-F1 | 0.800 | 0.784 ± 0.022 | 0.778 |  |
| Garden I-II recall | 0.596 | 0.375 ± 0.125 | 0.250 |  |
| Dice, fractured femurs (missed = 0) | 0.749 | 0.704 ± 0.005 | 0.698 |  |
| Dice, detected fractures only | 0.781 | 0.733 ± 0.006 | 0.750 | 0.77 |
| NSD at 10 px, fractured femurs | 0.222 | 0.188 ± 0.004 | 0.185 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.990 | 0.993 ± 0.004 | 0.999 | 0.981 |
| Fracture sensitivity | 0.972 | 0.977 ± 0.017 | 0.993 |  |
| Fracture specificity | 0.948 | 0.957 ± 0.030 | 0.985 |  |
| 6-class accuracy (%) | 86.0 | 85.4 ± 1.6 | 88.6 | 86.2 |
| 6-class macro-F1 | 0.801 | 0.786 ± 0.020 | 0.824 |  |
| Garden I-II recall | 0.614 | 0.388 ± 0.120 | 0.438 |  |
| Dice, fractured femurs (missed = 0) | 0.751 | 0.708 ± 0.005 | 0.721 |  |
| Dice, detected fractures only | 0.773 | 0.725 ± 0.009 | 0.726 | 0.77 |
| NSD at 10 px, fractured femurs | 0.222 | 0.189 ± 0.005 | 0.192 |  |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
