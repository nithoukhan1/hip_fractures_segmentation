# S09_segformer_b2: summary

Network `seg:Segformer:mit_b2`, input 768 x 512, 60 epochs, batch 16 (micro-batch 8 x 2), lr 0.0002, final weights.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 168.6 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.8732

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.997 ± 0.002 | 0.999 | 0.981 |
| Fracture sensitivity | 0.976 | 0.986 ± 0.005 | 0.986 |  |
| Fracture specificity | 0.970 | 0.982 ± 0.020 | 1.000 |  |
| 6-class accuracy (%) | 87.9 | 86.8 ± 0.8 | 88.1 | 86.2 |
| 6-class macro-F1 | 0.817 | 0.797 ± 0.020 | 0.804 |  |
| Garden I-II recall | 0.579 | 0.412 ± 0.095 | 0.375 |  |
| Dice, fractured femurs (missed = 0) | 0.766 | 0.725 ± 0.008 | 0.737 |  |
| Dice, detected fractures only | 0.784 | 0.735 ± 0.005 | 0.747 | 0.77 |
| NSD at 10 px, fractured femurs | 0.238 | 0.194 ± 0.009 | 0.202 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.997 ± 0.002 | 0.999 | 0.981 |
| Fracture sensitivity | 0.981 | 0.989 ± 0.004 | 0.986 |  |
| Fracture specificity | 0.953 | 0.969 ± 0.019 | 0.985 |  |
| 6-class accuracy (%) | 87.5 | 86.4 ± 0.9 | 87.6 | 86.2 |
| 6-class macro-F1 | 0.810 | 0.794 ± 0.022 | 0.802 |  |
| Garden I-II recall | 0.579 | 0.412 ± 0.095 | 0.375 |  |
| Dice, fractured femurs (missed = 0) | 0.766 | 0.725 ± 0.008 | 0.737 |  |
| Dice, detected fractures only | 0.781 | 0.733 ± 0.008 | 0.747 | 0.77 |
| NSD at 10 px, fractured femurs | 0.238 | 0.194 ± 0.009 | 0.202 |  |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
