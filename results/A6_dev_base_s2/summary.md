# A6_dev_base_s2: summary

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A6`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. SLG-Net: teacher forcing over 30 epochs, region pooling at 1/16 and 1/32, logit adjustment tau 1.
Folds completed [0], skipped [], failed []. Runtime 49.4 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0006

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.998 |
| Fracture sensitivity | 0.985 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 86.5 |
| 6-class macro-F1 | 0.771 |
| Garden I-II recall | 0.545 |
| Dice, fractured femurs (missed = 0) | 0.744 |
| Dice, detected fractures only | 0.755 |
| NSD at 10 px, fractured femurs | 0.231 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.998 |
| Fracture sensitivity | 0.934 |
| Fracture specificity | 1.000 |
| 6-class accuracy (%) | 87.0 |
| 6-class macro-F1 | 0.802 |
| Garden I-II recall | 0.545 |
| Dice, fractured femurs (missed = 0) | 0.739 |
| Dice, detected fractures only | 0.791 |
| NSD at 10 px, fractured femurs | 0.229 |

## Interpretability tests (SLG-Net, Analysis/06 section 5), at the OOF threshold

| Mode | OOF 6-class accuracy (%) | OOF macro-F1 | OOF Garden I-II recall | Change of P(true class), fractured | Correct fractured femurs that become wrong |
|---|---|---|---|---|---|
| as used | 86.5 | 0.771 | 0.545 | | |
| oracle | 89.4 | 0.839 | 0.727 | 0.017 | 0.0% of 109 |

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
