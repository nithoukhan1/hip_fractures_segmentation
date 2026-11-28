# A6_dev_base: summary

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A6`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. SLG-Net: teacher forcing over 30 epochs, region pooling at 1/16 and 1/32, logit adjustment tau 1.
Folds completed [0], skipped [], failed []. Runtime 49.2 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0217

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.998 |
| Fracture sensitivity | 0.978 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 88.5 |
| 6-class macro-F1 | 0.775 |
| Garden I-II recall | 0.545 |
| Dice, fractured femurs (missed = 0) | 0.754 |
| Dice, detected fractures only | 0.771 |
| NSD at 10 px, fractured femurs | 0.233 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.998 |
| Fracture sensitivity | 0.941 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 88.5 |
| 6-class macro-F1 | 0.794 |
| Garden I-II recall | 0.545 |
| Dice, fractured femurs (missed = 0) | 0.748 |
| Dice, detected fractures only | 0.795 |
| NSD at 10 px, fractured femurs | 0.232 |

## Interpretability tests (SLG-Net, Analysis/06 section 5), at the OOF threshold

| Mode | OOF 6-class accuracy (%) | OOF macro-F1 | OOF Garden I-II recall | Change of P(true class), fractured | Correct fractured femurs that become wrong |
|---|---|---|---|---|---|
| as used | 88.5 | 0.775 | 0.545 | | |
| oracle | 91.3 | 0.839 | 0.545 | 0.014 | 0.0% of 113 |

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
