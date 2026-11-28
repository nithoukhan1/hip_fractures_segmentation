# A1_m0_deletion: summary

Network `m0:convnext_tiny.fb_in22k_ft_in1k`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 147.6 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0084

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.999 ± 0.001 | 1.000 | 0.981 |
| Fracture sensitivity | 0.971 | 0.985 ± 0.006 | 1.000 |  |
| Fracture specificity | 0.989 | 0.985 ± 0.015 | 0.938 |  |
| 6-class accuracy (%) | 88.3 | 85.0 ± 1.2 | 85.2 | 86.2 |
| 6-class macro-F1 | 0.825 | 0.756 ± 0.024 | 0.783 |  |
| Garden I-II recall | 0.614 | 0.300 ± 0.149 | 0.438 |  |
| Dice, fractured femurs (missed = 0) | 0.778 | 0.732 ± 0.005 | 0.738 |  |
| Dice, detected fractures only | 0.802 | 0.743 ± 0.003 | 0.738 | 0.77 |
| NSD at 10 px, fractured femurs | 0.245 | 0.191 ± 0.002 | 0.196 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.999 ± 0.001 | 1.000 | 0.981 |
| Fracture sensitivity | 0.956 | 0.967 ± 0.019 | 0.972 |  |
| Fracture specificity | 0.992 | 0.997 ± 0.007 | 1.000 |  |
| 6-class accuracy (%) | 87.7 | 84.8 ± 1.7 | 85.7 | 86.2 |
| 6-class macro-F1 | 0.815 | 0.750 ± 0.031 | 0.772 |  |
| Garden I-II recall | 0.561 | 0.275 ± 0.122 | 0.312 |  |
| Dice, fractured femurs (missed = 0) | 0.770 | 0.724 ± 0.009 | 0.730 |  |
| Dice, detected fractures only | 0.806 | 0.749 ± 0.006 | 0.751 | 0.77 |
| NSD at 10 px, fractured femurs | 0.243 | 0.189 ± 0.003 | 0.195 |  |

## Interpretability tests (M0, Analysis/06 §13), at the OOF threshold

| Mode | OOF 6-class accuracy (%) | OOF macro-F1 | OOF Garden I-II recall | Change of P(true class), fractured | Correct fractured femurs that become wrong | Test ensemble accuracy (%) |
|---|---|---|---|---|---|---|
| as used | 88.3 | 0.825 | 0.614 | | | 85.2 |
| deletion | 63.8 | 0.462 | 0.298 | -0.442 | 48.8% of 562 | 55.7 |
| deletion control | 87.7 | 0.816 | 0.596 | 0.001 | 1.4% of 562 | 85.7 |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
