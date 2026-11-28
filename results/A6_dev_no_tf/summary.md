# A6_dev_no_tf: summary

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A6`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. SLG-Net: no teacher forcing, region pooling at 1/16 and 1/32, logit adjustment tau 1.
Folds completed [0], skipped [], failed []. Runtime 51.8 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0004

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.998 |
| Fracture sensitivity | 0.985 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 88.5 |
| 6-class macro-F1 | 0.783 |
| Garden I-II recall | 0.727 |
| Dice, fractured femurs (missed = 0) | 0.754 |
| Dice, detected fractures only | 0.765 |
| NSD at 10 px, fractured femurs | 0.233 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.998 |
| Fracture sensitivity | 0.949 |
| Fracture specificity | 1.000 |
| 6-class accuracy (%) | 88.5 |
| 6-class macro-F1 | 0.799 |
| Garden I-II recall | 0.636 |
| Dice, fractured femurs (missed = 0) | 0.754 |
| Dice, detected fractures only | 0.795 |
| NSD at 10 px, fractured femurs | 0.233 |

## Interpretability tests (SLG-Net, Analysis/06 section 5), at the OOF threshold

| Mode | OOF 6-class accuracy (%) | OOF macro-F1 | OOF Garden I-II recall | Change of P(true class), fractured | Correct fractured femurs that become wrong |
|---|---|---|---|---|---|
| as used | 88.5 | 0.783 | 0.727 | | |
| oracle | 88.9 | 0.791 | 0.727 | 0.002 | 0.0% of 113 |

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
