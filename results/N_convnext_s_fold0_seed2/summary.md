# N_convnext_s_fold0_seed2: summary

Network `m0:convnext_small.fb_in22k_ft_in1k`, input 768 x 512, 60 epochs, batch 16 (micro-batch 4 x 4), lr 0.0002, final weights.
Folds completed [0], skipped [], failed []. Runtime 70.9 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0001

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.998 |
| Fracture sensitivity | 0.993 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 86.5 |
| 6-class macro-F1 | 0.788 |
| Garden I-II recall | 0.636 |
| Dice, fractured femurs (missed = 0) | 0.761 |
| Dice, detected fractures only | 0.766 |
| NSD at 10 px, fractured femurs | 0.224 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.998 |
| Fracture sensitivity | 0.956 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 85.6 |
| 6-class macro-F1 | 0.767 |
| Garden I-II recall | 0.636 |
| Dice, fractured femurs (missed = 0) | 0.754 |
| Dice, detected fractures only | 0.789 |
| NSD at 10 px, fractured femurs | 0.222 |

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
