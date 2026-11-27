# R00_reference_bank_r2: pair index and reference bank (Paper B, step B-1)

Features: `convnext_tiny.fb_in22k_ft_in1k` (ImageNet weights, frozen), input 768 x 512, 768 numbers per image; 1253 femurs, each as released and mirrored. Orientation: version 2 (protocol §8): a reference is mirrored when its pelvis side (the brighter half of the upper image) differs from the femur's. Took 154 s on cuda. Protocol: `Analysis/07_Step4_PaperB_Reference_Bank_Protocol.md`.

## Data (pair types)

| Split | Femurs | Healthy | Fractured | Pelvis on the left | Pair types |
|---|---|---|---|---|---|
| train | 1043 | 363 | 680 | 517 | fractured, healthy partner: 334; healthy, fractured partner: 334; healthy, healthy partner: 14; single: 361 |
| test | 210 | 65 | 145 | 106 | fractured, healthy partner: 64; healthy, fractured partner: 64; single: 82 |

## Checks (protocol §4)

1. **Mirror check:** the orientation rule mirrors the partner (the other hip) for 804 of 810 femurs with a partner (99.3%). Rule: >= 0.80 use the orientation rule as written; 0.60-0.80 use it and state it as a limitation; < 0.60 review the orientation rule before B-2. With the ImageNet features alone (the version-1 rule, reported only), the mirrored partner is closer for 473 (58.4%).
2. **Retrieval vs chance** (held-out femurs): mean cosine of the 3 nearest minus that of the random healthy femur = 0.547 (mean over the 5 fold models).

| Fold model | Held-out: 3 nearest | Held-out: random healthy | Held-out: nearest, median (IQR) | Test: nearest, median (IQR) |
|---|---|---|---|---|
| 0 | 0.558 | 0.047 | 0.581 (0.507-0.664) | 0.600 (0.503-0.665) |
| 1 | 0.541 | -0.008 | 0.559 (0.481-0.657) | 0.601 (0.501-0.663) |
| 2 | 0.566 | -0.007 | 0.593 (0.510-0.679) | 0.595 (0.493-0.665) |
| 3 | 0.568 | 0.018 | 0.589 (0.513-0.686) | 0.595 (0.498-0.669) |
| 4 | 0.551 | -0.001 | 0.584 (0.494-0.655) | 0.598 (0.501-0.672) |

3. **Duplicates:** no reference has the same pixels as its femur (the script stops otherwise); 0 references have cosine >= 0.995.
4. **Held-out vs test:** see the last two columns of the table above.
5. **Partner rank** (668 held-out and test femurs with a healthy partner, counted once per fold model for test femurs): the own partner, mirrored, would rank 6 (median) among the bank; rank 1 for 25.1%, top 5 for 47.5%.
6. **Bank:** fold model 0: 291 healthy (healthy, fractured partner: 267, healthy, healthy partner: 12, single: 12); fold model 1: 290 healthy (healthy, fractured partner: 268, healthy, healthy partner: 10, single: 12); fold model 2: 290 healthy (healthy, fractured partner: 268, healthy, healthy partner: 10, single: 12); fold model 3: 291 healthy (healthy, fractured partner: 267, healthy, healthy partner: 12, single: 12); fold model 4: 290 healthy (healthy, fractured partner: 266, healthy, healthy partner: 12, single: 12).
7. **Figure** `nn_fold0.png`: 00669_00442.png, 00252_00164.png, 01510_01012.png, 01195_00797.png, 01183_00788.png, 00845_00557.png (one held-out femur per class, seeded).
