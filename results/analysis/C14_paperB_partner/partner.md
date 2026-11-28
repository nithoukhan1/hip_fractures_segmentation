# C14: R7, the patient's own other hip (RQ-B2, RQ-B3; Analysis/08 §2.5)

Thresholds: base = R7's own OOF threshold (training policy, uses the partner's label: the leak); blind and retrieved = each refitted on its own OOF predictions; random and fractured = blind's (swap tests). R3 and R4: their base thresholds.

| Mode | Threshold |
|---|---|
| R7_partner base | 0.060134 |
| R7_partner blind | 0.003126 |
| R7_partner retrieved | 0.060134 |
| R7_partner random | 0.003126 |
| R7_partner fractured | 0.003126 |
| R3_nearest_ref base | 0.000828 |
| R3_nearest_ref random | 0.000828 |
| R4_refrac base | 0.007149 |
| R4_refrac random | 0.007149 |

## oof

### R7 per mode

| Mode | Correct | macro-F1 | Dice | Garden I-II | False positives | Missed fractures |
|---|---|---|---|---|---|---|
| base | 916/1043 | 0.819 | 0.780 | 0.632 | 3 | 17 |
| blind | 916/1043 | 0.819 | 0.780 | 0.632 | 3 | 16 |
| retrieved | 917/1043 | 0.822 | 0.781 | 0.632 | 3 | 17 |
| random | 912/1043 | 0.812 | 0.779 | 0.632 | 9 | 16 |
| fractured | 864/1043 | 0.760 | 0.743 | 0.596 | 5 | 27 |

### RQ-B2: own hip vs retrieved (334 fractured femurs with a healthy partner), Dice

| A vs B | Dice A | Dice B | Difference (95% CI) | p | p (Holm) |
|---|---|---|---|---|---|
| R7 blind (partner) vs R7 retrieved | 0.783 | 0.785 | -0.001 (-0.003 to -0.000) | 0.051 | 0.103 |
| R7 blind (partner) vs R3 (retrieved, trained so) | 0.783 | 0.784 | -0.000 (-0.007 to +0.007) | 0.791 | 0.791 |

Garden I-II recall, all femurs:

| A vs B | Recall A | Recall B | Difference (95% CI) | p | p (Holm) |
|---|---|---|---|---|---|
| R7 blind vs R7 retrieved | 0.632 | 0.632 | +0.000 (+0.000 to +0.000) | 1.000 | 1.000 |
| R7 blind vs R3 | 0.632 | 0.684 | -0.053 (-0.140 to +0.035) | 0.453 | 0.906 |

### RQ-B3.1: healthy femurs whose partner is fractured (334)

- P(fracture), mean (median): blind 0.0071 (6.28e-06), retrieved 0.0093 (4.57e-06); Wilcoxon p 0.008.
- False positives: blind 3, retrieved 3 (McNemar p 1.000); base (leak) 3.

### RQ-B3.2: swap to a random healthy femur (334 fractured femurs with a healthy partner)

| Run (swap) | ΔP(fracture) mean | median | ΔDice mean | Fractures lost |
|---|---|---|---|---|
| R7 (partner → random) | +0.0058 | +0.00e+00 | +0.0009 | 0 |
| R3 (nearest → random) | +0.0003 | +0.00e+00 | -0.0001 | 0 |
| R4 (3 nearest → random) | +0.0008 | +0.00e+00 | -0.0024 | 2 |

| A vs B | Change of | Difference of the changes (95% CI) | p | p (Holm) |
|---|---|---|---|---|
| R7 (partner → random) vs R3 (nearest → random) | P(fracture) | +0.006 (-0.001 to +0.014) | 0.041 | 0.081 |
| R7 (partner → random) vs R3 (nearest → random) | Dice | +0.001 (-0.001 to +0.003) | 0.238 | 0.238 |
| R7 (partner → random) vs R4 (3 nearest → random) | P(fracture) | +0.005 (-0.004 to +0.014) | 0.342 | 0.342 |
| R7 (partner → random) vs R4 (3 nearest → random) | Dice | +0.003 (+0.001 to +0.006) | 0.003 | 0.005 |

### RQ-B3.3: a fractured reference (against blind)

- ΔP(fracture): fractured femurs -0.116, healthy femurs -0.0035.
- Dice 0.780 → 0.743 (-0.037 (-0.047 to -0.029), p <.001).
- Fractures lost 11; healthy femurs turned fractured 2.

### The leak: base (partner chosen by its label) vs blind

| Femurs | n | Correct base | Correct blind | P(fracture) base | P(fracture) blind |
|---|---|---|---|---|---|
| fractured, healthy partner | 334 | 273 | 273 | 0.9507 | 0.9507 |
| healthy, fractured partner | 334 | 331 | 331 | 0.0093 | 0.0071 |
| single | 361 | 298 | 298 | 0.9395 | 0.9395 |

## test_ensemble

### R7 per mode

| Mode | Correct | macro-F1 | Dice | Garden I-II | False positives | Missed fractures |
|---|---|---|---|---|---|---|
| base | 182/210 | 0.796 | 0.741 | 0.375 | 1 | 2 |
| blind | 183/210 | 0.808 | 0.741 | 0.438 | 1 | 1 |
| retrieved | 182/210 | 0.795 | 0.741 | 0.438 | 1 | 2 |
| random | 181/210 | 0.790 | 0.742 | 0.438 | 1 | 1 |
| fractured | 181/210 | 0.817 | 0.718 | 0.500 | 1 | 2 |

### RQ-B2: own hip vs retrieved (64 fractured femurs with a healthy partner), Dice

| A vs B | Dice A | Dice B | Difference (95% CI) | p | p (Holm) |
|---|---|---|---|---|---|
| R7 blind (partner) vs R7 retrieved | 0.731 | 0.731 | -0.000 (-0.002 to +0.002) | 0.859 | 1.000 |
| R7 blind (partner) vs R3 (retrieved, trained so) | 0.731 | 0.731 | -0.001 (-0.005 to +0.004) | 0.666 | 1.000 |

Garden I-II recall, all femurs:

| A vs B | Recall A | Recall B | Difference (95% CI) | p | p (Holm) |
|---|---|---|---|---|---|
| R7 blind vs R7 retrieved | 0.438 | 0.438 | +0.000 (-0.188 to +0.188) | 1.000 | 1.000 |
| R7 blind vs R3 | 0.438 | 0.375 | +0.062 (+0.000 to +0.188) | 1.000 | 1.000 |

### RQ-B3.1: healthy femurs whose partner is fractured (64)

- P(fracture), mean (median): blind 0.0001 (1.06e-05), retrieved 0.0031 (6.36e-06); Wilcoxon p 0.224.
- False positives: blind 1, retrieved 1 (McNemar p 1.000); base (leak) 1.

### RQ-B3.2: swap to a random healthy femur (64 fractured femurs with a healthy partner)

| Run (swap) | ΔP(fracture) mean | median | ΔDice mean | Fractures lost |
|---|---|---|---|---|
| R7 (partner → random) | +0.0060 | +0.00e+00 | +0.0000 | 0 |
| R3 (nearest → random) | +0.0002 | +0.00e+00 | -0.0004 | 0 |
| R4 (3 nearest → random) | +0.0094 | +6.00e-09 | +0.0004 | 0 |

| A vs B | Change of | Difference of the changes (95% CI) | p | p (Holm) |
|---|---|---|---|---|
| R7 (partner → random) vs R3 (nearest → random) | P(fracture) | +0.006 (-0.000 to +0.013) | 0.450 | 0.900 |
| R7 (partner → random) vs R3 (nearest → random) | Dice | +0.000 (-0.002 to +0.003) | 0.741 | 1.000 |
| R7 (partner → random) vs R4 (3 nearest → random) | P(fracture) | -0.003 (-0.012 to +0.003) | 0.703 | 0.900 |
| R7 (partner → random) vs R4 (3 nearest → random) | Dice | -0.000 (-0.002 to +0.002) | 0.527 | 1.000 |

### RQ-B3.3: a fractured reference (against blind)

- ΔP(fracture): fractured femurs -0.108, healthy femurs +0.0001.
- Dice 0.741 → 0.718 (-0.023 (-0.045 to -0.005), p 0.148).
- Fractures lost 1; healthy femurs turned fractured 0.

### The leak: base (partner chosen by its label) vs blind

| Femurs | n | Correct base | Correct blind | P(fracture) base | P(fracture) blind |
|---|---|---|---|---|---|
| fractured, healthy partner | 64 | 53 | 53 | 0.9611 | 0.9611 |
| healthy, fractured partner | 64 | 63 | 63 | 0.0031 | 0.0001 |
| single | 82 | 66 | 67 | 0.9598 | 0.9598 |

