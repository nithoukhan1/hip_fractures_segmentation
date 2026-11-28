# N_convnext_s_fold0: summary

Network `m0:convnext_small.fb_in22k_ft_in1k`, input 768 x 512, 60 epochs, batch 16 (micro-batch 4 x 4), lr 0.0002, final weights.
Folds completed [0], skipped [], failed []. Runtime 72.3 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0018

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.996 |
| Fracture sensitivity | 0.978 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 89.4 |
| 6-class macro-F1 | 0.814 |
| Garden I-II recall | 0.455 |
| Dice, fractured femurs (missed = 0) | 0.759 |
| Dice, detected fractures only | 0.777 |
| NSD at 10 px, fractured femurs | 0.244 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.996 |
| Fracture sensitivity | 0.941 |
| Fracture specificity | 1.000 |
| 6-class accuracy (%) | 88.0 |
| 6-class macro-F1 | 0.772 |
| Garden I-II recall | 0.364 |
| Dice, fractured femurs (missed = 0) | 0.750 |
| Dice, detected fractures only | 0.797 |
| NSD at 10 px, fractured femurs | 0.240 |

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
