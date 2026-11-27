# R4_dev_base: summary

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R4`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. RefFrac R4: 3 reference(s) per femur (nearest), alignment attention (window +-0.125 of the image height), fusion difference; references encoded by the shared encoder without gradient.
Folds completed [0], skipped [], failed []. Runtime 83.1 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0001

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.997 |
| Fracture sensitivity | 0.978 |
| Fracture specificity | 1.000 |
| 6-class accuracy (%) | 88.0 |
| 6-class macro-F1 | 0.817 |
| Garden I-II recall | 0.636 |
| Dice, fractured femurs (missed = 0) | 0.765 |
| Dice, detected fractures only | 0.783 |
| NSD at 10 px, fractured femurs | 0.241 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.997 |
| Fracture sensitivity | 0.941 |
| Fracture specificity | 1.000 |
| 6-class accuracy (%) | 86.1 |
| 6-class macro-F1 | 0.783 |
| Garden I-II recall | 0.636 |
| Dice, fractured femurs (missed = 0) | 0.749 |
| Dice, detected fractures only | 0.796 |
| NSD at 10 px, fractured femurs | 0.238 |

## Reference tests (RefFrac, Analysis/08 §2.6), at the OOF threshold

| Mode | OOF Dice, fractured | OOF macro-F1 | OOF Garden I-II recall | Change of P(fracture), fractured | Change of P(fracture), healthy | Healthy femurs that become fractured |
|---|---|---|---|---|---|---|
| as used | 0.765 | 0.817 | 0.636 | | | |
| self | 0.701 | 0.768 | 0.545 | -0.457 | -0.000 | 0.0% of 72 |
| random | 0.761 | 0.817 | 0.636 | 0.004 | 0.000 | 0.0% of 72 |

Pointing game of the difference map |T - R~| at 1/16 (base mode, fractured femurs: the cell with the largest difference touches the released mask): OOF 0.640 (chance 0.080, n = 136).

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
