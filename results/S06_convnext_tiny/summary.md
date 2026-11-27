# S06_convnext_tiny: summary

Network `cls:convnext_tiny.fb_in22k_ft_in1k`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 87.5 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.1228

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.990 | 0.998 ± 0.001 | 1.000 | 0.981 |
| Fracture sensitivity | 0.966 | 0.963 ± 0.017 | 0.993 |  |
| Fracture specificity | 0.978 | 0.994 ± 0.014 | 0.969 |  |
| 6-class accuracy (%) | 86.4 | 83.5 ± 1.6 | 84.8 | 86.2 |
| 6-class macro-F1 | 0.784 | 0.732 ± 0.024 | 0.742 |  |
| Garden I-II recall | 0.474 | 0.275 ± 0.034 | 0.250 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.990 | 0.998 ± 0.001 | 1.000 | 0.981 |
| Fracture sensitivity | 0.959 | 0.957 ± 0.016 | 0.952 |  |
| Fracture specificity | 0.983 | 0.994 ± 0.014 | 1.000 |  |
| 6-class accuracy (%) | 86.3 | 83.3 ± 1.8 | 83.8 | 86.2 |
| 6-class macro-F1 | 0.782 | 0.726 ± 0.029 | 0.713 |  |
| Garden I-II recall | 0.456 | 0.263 ± 0.052 | 0.188 |  |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
