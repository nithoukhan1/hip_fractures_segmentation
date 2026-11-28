# R00_reference_bank: pair index and reference bank (Paper B, step 1 of Paper B)

Features: `convnext_tiny.fb_in22k_ft_in1k` (ImageNet weights, frozen), input 768 x 512, 768 numbers per image; 1253 femurs, each as released and mirrored. Took 143 s on cuda. Protocol: `Analysis/07_Step4_PaperB_Reference_Bank_Protocol.md`.

## Data (pair types)

| Split | Femurs | Healthy | Fractured | Pair types |
|---|---|---|---|---|
| train | 1043 | 363 | 680 | fractured, healthy partner: 334; healthy, fractured partner: 334; healthy, healthy partner: 14; single: 361 |
| test | 210 | 65 | 145 | fractured, healthy partner: 64; healthy, fractured partner: 64; single: 82 |

## Checks (protocol section 4)

1. **Mirror check:** for 473 of 810 femurs with a partner (58.4%), the mirrored partner is closer than the partner as released. Rule: >= 0.80 use the orientation rule as written; 0.60-0.80 use it and state it as a limitation; < 0.60 review the orientation rule before Paper B step 2.
2. **Retrieval vs chance** (held-out femurs): mean cosine of the 3 nearest minus that of the random healthy femur = 0.539 (mean over the 5 fold models).

| Fold model | Held-out: 3 nearest | Held-out: random healthy | Held-out: nearest, median (IQR) | Test: nearest, median (IQR) |
|---|---|---|---|---|
| 0 | 0.565 | 0.066 | 0.588 (0.517-0.667) | 0.602 (0.507-0.671) |
| 1 | 0.548 | 0.007 | 0.569 (0.486-0.658) | 0.608 (0.503-0.668) |
| 2 | 0.574 | 0.008 | 0.601 (0.523-0.684) | 0.601 (0.496-0.669) |
| 3 | 0.576 | 0.033 | 0.604 (0.516-0.688) | 0.601 (0.501-0.671) |
| 4 | 0.559 | 0.016 | 0.590 (0.508-0.656) | 0.602 (0.506-0.678) |

3. **Duplicates:** no reference has the same pixels as its femur (the script stops otherwise); 0 references have cosine >= 0.995.
4. **Held-out vs test:** see the last two columns of the table above.
5. **Partner rank** (668 held-out and test femurs with a healthy partner, counted once per fold model for test femurs): the own partner would rank 6 (median) among the bank; rank 1 for 26.6%, top 5 for 48.7%.
6. **Bank:** fold model 0: 291 healthy (healthy, fractured partner: 267, healthy, healthy partner: 12, single: 12); fold model 1: 290 healthy (healthy, fractured partner: 268, healthy, healthy partner: 10, single: 12); fold model 2: 290 healthy (healthy, fractured partner: 268, healthy, healthy partner: 10, single: 12); fold model 3: 291 healthy (healthy, fractured partner: 267, healthy, healthy partner: 12, single: 12); fold model 4: 290 healthy (healthy, fractured partner: 266, healthy, healthy partner: 12, single: 12).
7. **Figure** `nn_fold0.png`: 00669_00442.png, 00252_00164.png, 01510_01012.png, 01195_00797.png, 01183_00788.png, 00845_00557.png (one held-out femur per class, seeded).
