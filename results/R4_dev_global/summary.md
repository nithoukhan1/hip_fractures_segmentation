# R4_dev_global: summary

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R4`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. RefFrac: 3 reference(s) per femur (nearest), alignment attention (global), fusion difference; references encoded by the shared encoder without gradient.
Folds completed [0], skipped [], failed []. Runtime 87.0 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0128

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.999 |
| Fracture sensitivity | 0.985 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 87.0 |
| 6-class macro-F1 | 0.796 |
| Garden I-II recall | 0.545 |
| Dice, fractured femurs (missed = 0) | 0.778 |
| Dice, detected fractures only | 0.790 |
| NSD at 10 px, fractured femurs | 0.239 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.999 |
| Fracture sensitivity | 0.956 |
| Fracture specificity | 1.000 |
| 6-class accuracy (%) | 86.5 |
| 6-class macro-F1 | 0.793 |
| Garden I-II recall | 0.545 |
| Dice, fractured femurs (missed = 0) | 0.761 |
| Dice, detected fractures only | 0.796 |
| NSD at 10 px, fractured femurs | 0.233 |

## Reference tests (RefFrac, Analysis/08 section 2.6), at the OOF threshold

| Mode | OOF Dice, fractured | OOF macro-F1 | OOF Garden I-II recall | Change of P(fracture), fractured | Change of P(fracture), healthy | Healthy femurs that become fractured |
|---|---|---|---|---|---|---|
| as used | 0.778 | 0.796 | 0.545 | | | |
| self | 0.743 | 0.772 | 0.545 | -0.063 | -0.002 | 0.0% of 71 |
| random | 0.772 | 0.791 | 0.545 | -0.006 | -0.002 | 0.0% of 71 |

Pointing game of the difference map |T - R~| at 1/16 (base mode, fractured femurs: the cell with the largest difference touches the released mask): OOF 0.779 (chance 0.080, n = 136).

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
