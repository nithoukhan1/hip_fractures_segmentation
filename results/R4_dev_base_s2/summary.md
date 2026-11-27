# R4_dev_base_s2: summary

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R4`, input 768 x 512, 60 epochs, batch 16 (micro-batch 16 x 1), lr 0.0002, final weights. RefFrac R4: 3 reference(s) per femur (nearest), alignment attention (window +-0.125 of the image height), fusion difference; references encoded by the shared encoder without gradient.
Folds completed [0], skipped [], failed []. Runtime 80.8 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0007

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.999 |
| Fracture sensitivity | 0.993 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 89.4 |
| 6-class macro-F1 | 0.819 |
| Garden I-II recall | 0.636 |
| Dice, fractured femurs (missed = 0) | 0.774 |
| Dice, detected fractures only | 0.780 |
| NSD at 10 px, fractured femurs | 0.242 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.999 |
| Fracture sensitivity | 0.956 |
| Fracture specificity | 1.000 |
| 6-class accuracy (%) | 88.9 |
| 6-class macro-F1 | 0.824 |
| Garden I-II recall | 0.636 |
| Dice, fractured femurs (missed = 0) | 0.763 |
| Dice, detected fractures only | 0.798 |
| NSD at 10 px, fractured femurs | 0.240 |

## Reference tests (RefFrac, Analysis/08 §2.6), at the OOF threshold

| Mode | OOF Dice, fractured | OOF macro-F1 | OOF Garden I-II recall | Change of P(fracture), fractured | Change of P(fracture), healthy | Healthy femurs that become fractured |
|---|---|---|---|---|---|---|
| as used | 0.774 | 0.819 | 0.636 | | | |
| self | 0.658 | 0.740 | 0.364 | -0.531 | -0.000 | 0.0% of 71 |
| random | 0.769 | 0.813 | 0.636 | -0.005 | 0.000 | 1.4% of 71 |

Pointing game of the difference map |T - R~| at 1/16 (base mode, fractured femurs: the cell with the largest difference touches the released mask): OOF 0.919 (chance 0.080, n = 136).

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
