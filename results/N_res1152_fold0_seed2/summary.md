# N_res1152_fold0_seed2: summary

Network `m0:convnext_tiny.fb_in22k_ft_in1k`, input 1152 x 768, 60 epochs, batch 16 (micro-batch 8 x 2), lr 0.0002, final weights.
Folds completed [0], skipped [], failed []. Runtime 102.4 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.3886

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.999 |
| Fracture sensitivity | 0.993 |
| Fracture specificity | 1.000 |
| 6-class accuracy (%) | 87.0 |
| 6-class macro-F1 | 0.790 |
| Garden I-II recall | 0.545 |
| Dice, fractured femurs (missed = 0) | 0.779 |
| Dice, detected fractures only | 0.785 |
| NSD at 10 px, fractured femurs | 0.234 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.999 |
| Fracture sensitivity | 0.985 |
| Fracture specificity | 1.000 |
| 6-class accuracy (%) | 86.5 |
| 6-class macro-F1 | 0.787 |
| Garden I-II recall | 0.545 |
| Dice, fractured femurs (missed = 0) | 0.775 |
| Dice, detected fractures only | 0.787 |
| NSD at 10 px, fractured femurs | 0.234 |

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
