# A6_slgnet: summary

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A6`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. SLG-Net: teacher forcing over 30 epochs, region pooling at 1/32, logit adjustment tau 1.
Folds completed [0, 1, 2, 3, 4], skipped [], failed []. Runtime 174.3 min on Tesla T4 x2.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0156

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.996 | 0.999 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.972 | 0.972 ± 0.007 | 0.986 |  |
| Fracture specificity | 0.983 | 0.991 ± 0.008 | 0.985 |  |
| 6-class accuracy (%) | 87.0 | 86.3 ± 1.5 | 88.6 | 86.2 |
| 6-class macro-F1 | 0.806 | 0.789 ± 0.015 | 0.820 |  |
| Garden I-II recall | 0.614 | 0.287 ± 0.034 | 0.312 |  |
| Dice, fractured femurs (missed = 0) | 0.774 | 0.725 ± 0.007 | 0.737 |  |
| Dice, detected fractures only | 0.796 | 0.745 ± 0.003 | 0.747 | 0.77 |
| NSD at 10 px, fractured femurs | 0.243 | 0.190 ± 0.012 | 0.198 |  |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=1,043) | Test, fold models mean ± SD | Test, ensemble | Paper, final model YOLOv8s-seg @1024 (test) |
|---|---|---|---|---|
| Fracture AUC | 0.996 | 0.999 ± 0.001 | 0.999 | 0.981 |
| Fracture sensitivity | 0.965 | 0.956 ± 0.016 | 0.959 |  |
| Fracture specificity | 0.986 | 0.997 ± 0.007 | 1.000 |  |
| 6-class accuracy (%) | 86.8 | 85.7 ± 2.0 | 87.1 | 86.2 |
| 6-class macro-F1 | 0.802 | 0.783 ± 0.023 | 0.800 |  |
| Garden I-II recall | 0.596 | 0.275 ± 0.034 | 0.250 |  |
| Dice, fractured femurs (missed = 0) | 0.772 | 0.716 ± 0.011 | 0.723 |  |
| Dice, detected fractures only | 0.800 | 0.749 ± 0.003 | 0.754 | 0.77 |
| NSD at 10 px, fractured femurs | 0.242 | 0.188 ± 0.011 | 0.193 |  |

## Interpretability tests (SLG-Net, Analysis/06 section 5), at the OOF threshold

| Mode | OOF 6-class accuracy (%) | OOF macro-F1 | OOF Garden I-II recall | Change of P(true class), fractured | Correct fractured femurs that become wrong | Test ensemble accuracy (%) |
|---|---|---|---|---|---|---|
| as used | 87.0 | 0.806 | 0.614 | | | 88.6 |
| oracle | 87.4 | 0.810 | 0.632 | 0.009 | 0.5% of 550 | 89.5 |
| deletion | 55.2 | 0.324 | 0.088 | -0.479 | 64.7% of 550 | 50.0 |
| deletion control | 86.5 | 0.805 | 0.614 | -0.008 | 1.3% of 550 | 88.1 |

Paper column: González et al. 2025 (Table 3; their final model's Dice is the mean over detected fractures, definition not stated). Their classifiers and YOLO models were trained on one split, ours on our 5 folds (one model per fold, final weights).
