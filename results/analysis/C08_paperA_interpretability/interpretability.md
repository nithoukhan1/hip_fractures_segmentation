# Interpretability tests: S10_m0_convnext_unet, A2_region, A3_geometry, A4_hierarchy, A5_shared_grade, A6_slgnet

Decisions at each run's OOF threshold. Femurs: OOF 1,043 (680 fractured); test ensemble 210 (145 fractured).
Intervals: cluster bootstrap over x-rays (2000 resamples); randomisation test for macro-F1 (5000 swaps of whole x-rays); seed 2026.

## 1. Oracle test: the heads get the released mask

| Run | Part | Correct: base → oracle (McNemar p) | Macro-F1: base → oracle (p) | Garden I-II recall: base → oracle | Fractured, wrong at base: fixed | Fractured, correct at base: lost | Same as metrics.json |
|---|---|---|---|---|---|---|---|
| A2_region | OOF | 921 → 930 (.122) | 0.820 → 0.832 (.152) | 0.544 → 0.544 | 13 of 117 | 9 of 563 | True |
| A2_region | test ensemble | 181 → 187 (.031) | 0.785 → 0.815 (.031) | 0.250 → 0.250 | 3 of 26 | 0 of 119 | True |
| A3_geometry | OOF | 927 → 937 (.087) | 0.838 → 0.845 (.394) | 0.684 → 0.614 | 9 of 106 | 9 of 574 | True |
| A3_geometry | test ensemble | 180 → 186 (.109) | 0.786 → 0.815 (.246) | 0.312 → 0.375 | 2 of 24 | 2 of 121 | True |
| A4_hierarchy | OOF | 916 → 921 (.405) | 0.823 → 0.825 (.715) | 0.649 → 0.649 | 14 of 122 | 9 of 558 | True |
| A4_hierarchy | test ensemble | 183 → 185 (.500) | 0.800 → 0.808 (.511) | 0.375 → 0.375 | 2 of 25 | 0 of 120 | True |
| A5_shared_grade | OOF | 918 → 924 (.307) | 0.820 → 0.826 (.393) | 0.614 → 0.614 | 15 of 120 | 9 of 560 | True |
| A5_shared_grade | test ensemble | 183 → 185 (.625) | 0.792 → 0.803 (.238) | 0.312 → 0.312 | 3 of 26 | 1 of 119 | True |
| A6_slgnet | OOF | 907 → 912 (.227) | 0.806 → 0.810 (.510) | 0.614 → 0.632 | 8 of 130 | 3 of 550 | True |
| A6_slgnet | test ensemble | 186 → 188 (.625) | 0.820 → 0.841 (.251) | 0.312 → 0.312 | 3 of 23 | 1 of 122 | True |

## 2. Region deletion vs control: A6_slgnet

| Part | Fractured femurs | ΔP(fracture): deletion / control, mean (median); Wilcoxon p | ΔP(true class): deletion / control, mean (median); Wilcoxon p | Correct decisions lost: deletion / control (McNemar p) | Wrong ones fixed: deletion / control | Same as metrics.json |
|---|---|---|---|---|---|---|
| OOF | 680 | -0.032 (-0.000) / -0.005 (-0.000); < .001 | -0.479 (-0.678) / -0.008 (-0.000); < .001 | 356 (64.7%) / 7 (1.3%) of 550 (< .001) | 25 / 3 of 130 | True |
| test ensemble | 145 | -0.039 (-0.000) / -0.002 (-0.000); < .001 | -0.538 (-0.598) / -0.002 (-0.000); < .001 | 80 (65.6%) / 0 (0.0%) of 122 (< .001) | 0 / 0 of 23 | True |

Correct decisions lost, per true class (deletion / control, of those correct at base):

| Part | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|
| OOF | 31 / 0 of 35 | 58 / 3 of 223 | 99 / 1 of 105 | 152 / 3 of 160 | 16 / 0 of 27 |
| test ensemble | 5 / 0 of 5 | 7 / 0 of 41 | 33 / 0 of 33 | 32 / 0 of 34 | 3 / 0 of 9 |

Where the control went (fractured femurs; test: each of the 5 fold models' own deletion): OOF: dimmer 10, matched 653, no_region 15, none 2; test ensemble: dimmer 10, matched 695, no_region 11, none 9.

## 3. Risk-coverage: setting aside the least confident decisions

| Run | Part | Errors | Errors among the 10% least confident (95% CI) | Error rate: all → the 90% kept | AURC (best possible) |
|---|---|---|---|---|---|
| S10_m0_convnext_unet | OOF | 121 of 1043 | 43 = 35.5% (27.3–41.9%) | 11.6% → 8.3% | 0.035 (0.007) |
| S10_m0_convnext_unet | test ensemble | 28 of 210 | 9 = 32.1% (20.6–50.0%) | 13.3% → 10.1% | 0.053 (0.010) |
| A2_region | OOF | 122 of 1043 | 42 = 34.4% (27.9–42.0%) | 11.7% → 8.5% | 0.036 (0.007) |
| A2_region | test ensemble | 29 of 210 | 11 = 37.9% (24.4–54.5%) | 13.8% → 9.5% | 0.039 (0.010) |
| A3_geometry | OOF | 116 of 1043 | 45 = 38.8% (31.5–46.8%) | 11.1% → 7.6% | 0.035 (0.006) |
| A3_geometry | test ensemble | 30 of 210 | 13 = 43.3% (31.4–61.5%) | 14.3% → 9.0% | 0.035 (0.011) |
| A4_hierarchy | OOF | 127 of 1043 | 51 = 40.2% (32.6–46.8%) | 12.2% → 8.1% | 0.049 (0.008) |
| A4_hierarchy | test ensemble | 27 of 210 | 11 = 40.7% (24.1–57.1%) | 12.9% → 8.5% | 0.028 (0.009) |
| A5_shared_grade | OOF | 125 of 1043 | 50 = 40.0% (32.5–47.2%) | 12.0% → 8.0% | 0.044 (0.008) |
| A5_shared_grade | test ensemble | 27 of 210 | 7 = 25.9% (12.5–44.0%) | 12.9% → 10.6% | 0.032 (0.009) |
| A6_slgnet | OOF | 136 of 1043 | 44 = 32.4% (25.0–39.2%) | 13.0% → 9.8% | 0.040 (0.009) |
| A6_slgnet | test ensemble | 24 of 210 | 6 = 25.0% (8.7–42.1%) | 11.4% → 9.5% | 0.029 (0.007) |

Confidence = the probability of the decided class (raw, before temperature scaling). Flagging at random would catch 10% of the errors on average.

## 4. Geometry consistency: does the region agree with the decided location and grade?

Location rule: centroid height (relative to the image) below 0.389 = neck, below 0.488 = pertrochanteric, otherwise subtrochanteric; fitted on 680 OOF fractures (released masks). Garden: region height; Evans: region area (Report 03 §2). AUCs with DeLong 95% intervals.

| Run | Part | Location: rule on the region = decided location | Garden: height vs decided grade, AUC | Evans: area vs decided grade, AUC | Garden: height vs released grade | Evans: area vs released grade | Region check (areas vs per-case file) |
|---|---|---|---|---|---|---|---|
| S10_m0_convnext_unet | OOF | 555 of 662 (83.8%) | 0.92 (0.88–0.97), n = 299 | 0.85 (0.81–0.89), n = 335 | 0.87 (0.80–0.93), n = 299 | 0.82 (0.77–0.86), n = 329 | same empty/non-empty; max area difference 0.08% |
| S10_m0_convnext_unet | test ensemble | 114 of 143 (79.7%) | 0.94 (0.87–1.00), n = 56 | 0.83 (0.74–0.93), n = 81 | 0.83 (0.68–0.99), n = 55 | 0.84 (0.75–0.93), n = 76 | same empty/non-empty; max area difference 0.00% |
| released masks, released labels | OOF | 562 of 680 (82.6%) (released location) | – | – | 0.81 (0.75–0.88), n = 311 | 0.80 (0.75–0.85), n = 333 | – |
| released masks, released labels | test ensemble | 116 of 145 (80.0%) (released location) | – | – | 0.84 (0.70–0.98), n = 57 | 0.75 (0.64–0.86), n = 76 | – |
