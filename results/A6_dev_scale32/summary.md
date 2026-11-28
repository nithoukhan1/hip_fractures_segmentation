# A6_dev_scale32: summary

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A6`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. SLG-Net: teacher forcing over 30 epochs, region pooling at 1/32, logit adjustment tau 1.
Folds completed [0], skipped [], failed []. Runtime 50.5 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0017

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.999 |
| Fracture sensitivity | 0.985 |
| Fracture specificity | 1.000 |
| 6-class accuracy (%) | 88.9 |
| 6-class macro-F1 | 0.827 |
| Garden I-II recall | 0.818 |
| Dice, fractured femurs (missed = 0) | 0.760 |
| Dice, detected fractures only | 0.771 |
| NSD at 10 px, fractured femurs | 0.238 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.999 |
| Fracture sensitivity | 0.949 |
| Fracture specificity | 1.000 |
| 6-class accuracy (%) | 87.5 |
| 6-class macro-F1 | 0.810 |
| Garden I-II recall | 0.636 |
| Dice, fractured femurs (missed = 0) | 0.748 |
| Dice, detected fractures only | 0.788 |
| NSD at 10 px, fractured femurs | 0.235 |

## Interpretability tests (SLG-Net, Analysis/06 section 5), at the OOF threshold

| Mode | OOF 6-class accuracy (%) | OOF macro-F1 | OOF Garden I-II recall | Change of P(true class), fractured | Correct fractured femurs that become wrong |
|---|---|---|---|---|---|
| as used | 88.9 | 0.827 | 0.818 | | |
| oracle | 88.0 | 0.810 | 0.727 | -0.004 | 1.8% of 113 |

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
