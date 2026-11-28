# N_res1152_fold0: summary

Network `m0:convnext_tiny.fb_in22k_ft_in1k`, input 1152 x 768, 60 epochs, batch 16 (micro-batch 8 x 2), lr 0.0002, final weights.
Folds completed [0], skipped [], failed []. Runtime 116.7 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0958

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.997 |
| Fracture sensitivity | 0.978 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 88.0 |
| 6-class macro-F1 | 0.783 |
| Garden I-II recall | 0.545 |
| Dice, fractured femurs (missed = 0) | 0.768 |
| Dice, detected fractures only | 0.785 |
| NSD at 10 px, fractured femurs | 0.242 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.997 |
| Fracture sensitivity | 0.963 |
| Fracture specificity | 1.000 |
| 6-class accuracy (%) | 88.0 |
| 6-class macro-F1 | 0.776 |
| Garden I-II recall | 0.455 |
| Dice, fractured femurs (missed = 0) | 0.763 |
| Dice, detected fractures only | 0.792 |
| NSD at 10 px, fractured femurs | 0.242 |

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
