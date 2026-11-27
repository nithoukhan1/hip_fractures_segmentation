# C05: Does learning the fracture mask help the classifier? (S06 = A0 vs S10 = M0 = A1)

*Written 2026-11-27 from the filed runs `results/S06_convnext_tiny/` and `results/S10_m0_convnext_unet/` (register lines 7 and 8). Every number below comes from the files in this folder:*
- `stats.md` / `stats.json`: 95% intervals and paired S06 vs S10 tests (`python scripts/run_stats.py S06_convnext_tiny S10_m0_convnext_unet --compare --name C05_mask_task_s06_vs_s10`).
- `breakdown.md` / `breakdown.json`: per class, per fold and fold model, errors, comparison with the article (`python scripts/run_breakdown.py S06_convnext_tiny S10_m0_convnext_unet --name C05_mask_task_s06_vs_s10`).

**What differs between the two runs:** only M0's U-Net decoder and its mask loss (binary cross-entropy + Dice, weight 1). Everything else is the same:
- the ConvNeXt-T encoder and the 6-class head (`tests/test_nets.py` checks that the class outputs are identical with the same weights);
- the recipe (`Analysis/05_Step2_Baselines_Protocol.md`), the folds and the seeds;
- the code commit (`18e4774`).

So the comparison answers one question only. This is step A0 → A1 of Paper A.

## 1. Verdict: yes on the 1,043 out-of-fold femurs; the same direction, not significant, on the 210 test femurs

| Endpoint | S06 classifier (A0) | S10 M0 (A1) | S10 − S06 | p (Holm) |
|---|---|---|---|---|
| **OOF, 1,043 femurs:** fracture AUC | .990 (.984–.995) | .995 (.990–.998) | +.004 | .058 |
| 6-class accuracy | 86.4% (84.3–88.4) | 88.4% (86.4–90.2) | **+2.0 points** | **.026** |
| Macro-F1 (primary) | .784 (.746–.816) | .825 (.793–.855) | **+.042** | **.013** |
| Fracture vs no fracture decision | | | +0.9 points | .078 |
| **Test, 5-model ensemble, 210 femurs:** fracture AUC | 1.000 | .999 | .000 | 1.00 |
| 6-class accuracy | 84.8% (79.6–89.6) | 86.7% (82.0–91.2) | +1.9 points | 1.00 |
| Macro-F1 | .742 (.651–.816) | .779 (.695–.849) | +.037 | .88 |
| **Test, 5 fold models, mean ± SD:** 6-class accuracy | 83.5 ± 1.6% | 86.0 ± 0.5% | +2.5 points | |
| Macro-F1 | .732 ± .024 | .776 ± .017 | +.044 | |

Intervals are 95% cluster-bootstrap intervals over x-rays. Tests: DeLong (AUC), McNemar (accuracy and the binary decision), randomisation with whole x-rays swapped (macro-F1), Holm-corrected within each set. Dice is not tested, because S06 has no mask.

1. **The mask task improves classification significantly on the OOF femurs:** accuracy +2.0 points (p = .026) and macro-F1 +.042 (p = .013), after Holm correction.
2. **The gain is consistent.** The S06 and S10 models of each fold were trained on the same femurs with the same seed.
   - M0 beats S06 on all 5 held-out folds, in both accuracy and macro-F1.
   - On the test set, M0's macro-F1 is higher for all 5 fold models. Its accuracy is higher for 4 and equal for 1.
   - M0's 5 test models also vary less (accuracy SD 0.5 against 1.6 points).
3. **On the test set the gain has the same size but is not significant.** 210 femurs cannot resolve a 2-point difference. Report 04 §3 compares methods on the OOF femurs because there are 5 times more of them; the test set confirms the direction. A possible overlap of patients between folds (no patient IDs were released) would make the OOF numbers of both runs slightly optimistic, but it favours neither run in this paired comparison.

## 2. Where the gain comes from: detection and location, not grade

| Errors (as in the article) | S06 OOF | S10 OOF | S06 test ensemble | S10 test ensemble |
|---|---|---|---|---|
| Detection (fracture vs none) | 31 | 22 | 3 | 1 |
| Location | 43 | 34 | 9 | 7 |
| Grade (right location) | 68 | 65 | 20 | 20 |
| **Total** | **142** | **121** | **32** | **28** |

Recall per class, OOF pooled (test, mean of the 5 fold models, in brackets):

| Class | n OOF / test | S06 | S10 |
|---|---|---|---|
| NF No fracture | 363 / 65 | .978 (.994) | .992 (1.000) |
| 2AB Neck Garden I-II | 57 / 16 | .474 (.275) | .544 (.312) |
| 2CD Neck Garden III-IV | 254 / 41 | .921 (.980) | .909 (.990) |
| 3AB Pertroch Evans I-II | 134 / 34 | .761 (.853) | .791 (.953) |
| 3CDE Pertroch Evans III-V | 199 / 42 | .804 (.767) | .839 (.738) |
| 4 Subtrochanteric | 36 / 12 | .639 (.417) | .750 (.550) |

- With the mask, M0 misses fewer fractures and places them better. Grading within a location is barely helped: 65 of M0's 121 OOF errors and 20 of its 28 test-ensemble errors are grade errors.
- **The weakest class is still Garden I-II** (OOF recall .544; test .312, 16 femurs). It is mostly confused with Garden III-IV: 13 of its 26 OOF errors and 10 of its 11 test-ensemble errors.
- This is what the next steps target. Paper A adds region pooling, region geometry and the location → grade hierarchy (A2–A4). Paper B's primary endpoint is Garden I-II recall.

## 3. M0 also gives a mask, and it is already better than YOLO's

| Dice, fractured femurs (missed = 0) | OOF | Test, 5 fold models | Test, ensemble |
|---|---|---|---|
| S01 YOLOv8s-seg @1024 (C01) | .740 (.722–.755) | .693 ± .016 | .725 |
| S10 M0 @768 × 512 | .777 (.764–.789) | .734 ± .005 | .744 |

- This is a descriptive comparison only. The formal Dice comparison of M0 with YOLO and the three segmenters is C04, after S07–S09.
- M0 over-segments the test neck fractures just as YOLO does (predicted / true area 1.89 for Garden I-II and 1.71 for Garden III-IV, against about 1.0 OOF). The cause is the smaller test masks of the neck (C01, annotation shift).

## 4. Caveat: both ConvNeXt models are over-confident

- By epoch 60 the training loss is almost 0 for S06 and 0.05 for S10 (mostly the mask loss), so the models fit their training femurs almost perfectly.
- Their probabilities are less well calibrated than YOLO's: expected calibration error on the OOF femurs is .127 (S06) and .108 (S10), against .079 (S01).
- M0's OOF-fitted fracture threshold is 0.018 (S06: 0.123), because it scores almost every no-fracture femur near 0. The plain threshold 0.5 costs it little (OOF accuracy 88.0% against 88.4%).
- Calibration enters Paper A at A6 (logit adjustment) and must be reported for every model.

## 5. How M0 compares with the article (test set, 210 femurs)

- M0's 5 fold models reach 86.0 ± 0.5% 6-class accuracy. That equals the article's single selected YOLOv8s model (86.2%) and is above its 6-run mean (84.4 ± 2.5%).
- M0's ensemble makes 1 detection error, against 8 for the article's final model.
- Our own YOLO ensemble (S01, 88.1%, 25 errors) is still the strongest test-set classifier. It makes fewer location errors (2 against 7), while M0 is ahead on the OOF femurs (88.4% against 83.2%; 121 errors against 175).

## 6. Decision

- **The mask task stays.** M0 (= A1 = R0) is confirmed as the shared starting point. Paper A's A2–A6 and Paper B's R runs build on it unchanged.
- Targets for the next steps: grade errors within a location (Garden I-II above all) and calibration.
- C03 (classifiers S04–S06 vs S01) and C04 (segmenters S07–S10 vs S01) follow when S04, S05, S07, S08 and S09 are filed.
