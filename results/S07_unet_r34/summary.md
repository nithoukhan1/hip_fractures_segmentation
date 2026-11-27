# S07_unet_r34: summary

Network `seg:Unet:resnet34`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 87.0 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.6997

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.990 | 0.993 ± 0.005 | 0.999 | 0.981 |
| Fracture sensitivity | 0.969 | 0.972 ± 0.005 | 0.966 |  |
| Fracture specificity | 0.970 | 0.972 ± 0.013 | 1.000 |  |
| 6-class accuracy (%) | 86.3 | 84.6 ± 1.6 | 86.7 | 86.2 |
| 6-class macro-F1 | 0.800 | 0.770 ± 0.027 | 0.799 |  |
| Garden I-II recall | 0.614 | 0.375 ± 0.088 | 0.375 |  |
| Dice, fractured femurs (missed = 0) | 0.748 | 0.698 ± 0.009 | 0.711 |  |
| Dice, detected fractures only | 0.772 | 0.718 ± 0.009 | 0.736 | 0.77 |
| NSD at 10 px, fractured femurs | 0.221 | 0.172 ± 0.014 | 0.179 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.990 | 0.993 ± 0.005 | 0.999 | 0.981 |
| Fracture sensitivity | 0.971 | 0.972 ± 0.005 | 0.979 |  |
| Fracture specificity | 0.964 | 0.966 ± 0.013 | 0.985 |  |
| 6-class accuracy (%) | 86.2 | 84.4 ± 1.3 | 87.1 | 86.2 |
| 6-class macro-F1 | 0.801 | 0.769 ± 0.025 | 0.809 |  |
| Garden I-II recall | 0.632 | 0.375 ± 0.088 | 0.438 |  |
| Dice, fractured femurs (missed = 0) | 0.748 | 0.698 ± 0.009 | 0.711 |  |
| Dice, detected fractures only | 0.771 | 0.718 ± 0.009 | 0.726 | 0.77 |
| NSD at 10 px, fractured femurs | 0.221 | 0.172 ± 0.014 | 0.180 |  |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
