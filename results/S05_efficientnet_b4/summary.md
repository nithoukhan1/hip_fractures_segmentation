# S05_efficientnet_b4: summary

Network `cls:efficientnet_b4.ra2_in1k`, input 768 x 512, 60 epochs, batch 16 (micro-batch 8 x 2), lr 0.0002, final weights.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 115.1 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.4486

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, EfficientNetB4 (test, Table 3) |
|---|---|---|---|---|
| Fracture AUC | 0.990 | 0.990 ± 0.002 | 0.994 | 0.990 |
| Fracture sensitivity | 0.954 | 0.932 ± 0.013 | 0.924 |  |
| Fracture specificity | 0.975 | 0.972 ± 0.013 | 0.985 |  |
| 6-class accuracy (%) | 84.8 | 80.6 ± 1.5 | 81.0 | 79.0 |
| 6-class macro-F1 | 0.755 | 0.736 ± 0.021 | 0.744 |  |
| Garden I-II recall | 0.386 | 0.338 ± 0.114 | 0.312 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, EfficientNetB4 (test, Table 3) |
|---|---|---|---|---|
| Fracture AUC | 0.990 | 0.990 ± 0.002 | 0.994 | 0.990 |
| Fracture sensitivity | 0.953 | 0.931 ± 0.015 | 0.924 |  |
| Fracture specificity | 0.975 | 0.975 ± 0.014 | 0.985 |  |
| 6-class accuracy (%) | 84.7 | 80.6 ± 1.2 | 81.0 | 79.0 |
| 6-class macro-F1 | 0.754 | 0.735 ± 0.020 | 0.744 |  |
| Garden I-II recall | 0.386 | 0.338 ± 0.114 | 0.312 |  |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
