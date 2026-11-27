# S04_densenet121: summary

Network `cls:densenet121.tv_in1k`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 82.4 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.1175

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, DenseNet121 (test, Table 3) |
|---|---|---|---|---|
| Fracture AUC | 0.992 | 0.997 ± 0.001 | 0.999 | 0.993 |
| Fracture sensitivity | 0.962 | 0.967 ± 0.009 | 0.986 |  |
| Fracture specificity | 0.983 | 0.988 ± 0.013 | 0.969 |  |
| 6-class accuracy (%) | 86.8 | 84.8 ± 2.0 | 85.7 | 83.8 |
| 6-class macro-F1 | 0.802 | 0.749 ± 0.049 | 0.760 |  |
| Garden I-II recall | 0.579 | 0.312 ± 0.133 | 0.250 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, DenseNet121 (test, Table 3) |
|---|---|---|---|---|
| Fracture AUC | 0.992 | 0.997 ± 0.001 | 0.999 | 0.993 |
| Fracture sensitivity | 0.950 | 0.957 ± 0.012 | 0.966 |  |
| Fracture specificity | 0.983 | 0.997 ± 0.007 | 1.000 |  |
| 6-class accuracy (%) | 86.2 | 84.5 ± 2.2 | 86.2 | 83.8 |
| 6-class macro-F1 | 0.793 | 0.743 ± 0.047 | 0.763 |  |
| Garden I-II recall | 0.526 | 0.287 ± 0.130 | 0.250 |  |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
