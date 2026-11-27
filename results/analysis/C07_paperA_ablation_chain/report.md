# C07: Paper A's ablation chain, A0 to A6, out of fold and on the test set

*Written 2026-11-27 from these filed runs:*
- *`S06_convnext_tiny` (A0, register line 7) and `S10_m0_convnext_unet` (A1 = M0, line 8);*
- *`A2_region` … `A6_slgnet` (lines 22–26), trained from the tag `A-v1-frozen` = `f96f84c`.*

**Plan:** `Analysis/06_Step3_PaperA_Protocol.md` §7, fixed before any SLG-Net result.
- Each step is compared with the one before it, and A6 with A1.
- Out-of-fold (OOF) results come first; the test set serves as confirmation.
- p-values are adjusted by Holm within the chain.
- The primary endpoints are macro-F1 and Dice on fractured femurs.

**Where the numbers come from** (all in this folder):
- `stats.md` / `stats.json`, made by
  `python scripts/run_stats.py S06_convnext_tiny S10_m0_convnext_unet A2_region A3_geometry A4_hierarchy A5_shared_grade A6_slgnet --pairs <the 7 chain pairs> --family-holm --name C07_paperA_ablation_chain`.
  - The column "p (Holm, family)" is the Holm adjustment within the chain. It is applied for each endpoint and each scope separately: over 7 comparisons, or 6 for Dice, because A0 has no mask.
- `breakdown.md`: per-class results and error types, made by `python scripts/run_breakdown.py <the same 7 runs> --name C07_paperA_ablation_chain`.
- `folds_1-4/stats.md`: the check on folds 1–4 (§4), made by `python scripts/run_stats.py <the same 7 runs> --folds 1,2,3,4 --name C07_paperA_ablation_chain/folds_1-4`.

**How decisions are made:** each run uses its own OOF fracture threshold (Youden). The tests are:
- for macro-F1, a randomisation test that swaps whole x-rays (5,000 permutations), with a 95% cluster-bootstrap interval for the difference;
- for Dice, a Wilcoxon signed-rank test over the fractured femurs, where a missed fracture counts as 0.

## 1. The chain

**Out of fold:** 1,043 femurs, of which 680 are fractured and 57 are Garden I-II.

| Step | Run | Macro-F1 (95% CI) | Dice, fractured (95% CI) | Correct | Garden I-II recall | Errors: detection / location / grade |
|---|---|---|---|---|---|---|
| A0 | ConvNeXt-T classifier (S06) | .784 (.746–.816) | – | 901 | .474 (27/57) | 31 / 43 / 68 |
| A1 | + mask task = M0 (S10) | .825 (.793–.855) | .777 (.764–.789) | 922 | .544 (31/57) | 22 / 34 / 65 |
| A2 | + region pooling | .820 (.784–.850) | .772 (.760–.785) | 921 | .544 (31/57) | 20 / 37 / 65 |
| A3 | + region geometry | **.838** (.805–.866) | .777 (.765–.788) | **927** | **.684** (39/57) | 23 / 33 / **60** |
| A4 | + location → grade heads | .823 (.790–.852) | .770 (.756–.783) | 916 | .649 (37/57) | 19 / 36 / 72 |
| A5 | + shared grade trunk | .820 (.786–.850) | .769 (.755–.782) | 918 | .614 (35/57) | 17 / 35 / 73 |
| A6 | + logit adjustment = **SLG-Net** | .806 (.771–.836) | .774 (.761–.786) | 907 | .614 (35/57) | 25 / 36 / 75 |

**Test set:** the 5-model ensemble on 210 femurs, of which 145 are fractured and 16 are Garden I-II.

| Step | Macro-F1 (95% CI) | Dice, fractured (95% CI) | Correct | Garden I-II recall |
|---|---|---|---|---|
| A0 | .742 (.651–.816) | – | 178 | .250 |
| A1 (M0) | .779 (.695–.849) | .744 (.718–.766) | 182 | .312 |
| A2 | .785 (.705–.847) | .738 (.712–.762) | 181 | .250 |
| A3 | .786 (.703–.854) | .734 (.706–.760) | 180 | .312 |
| A4 | .800 (.719–.866) | .732 (.703–.759) | 183 | .375 |
| A5 | .792 (.712–.857) | .734 (.705–.760) | 183 | .312 |
| A6 (SLG-Net) | **.820** (.747–.879) | .737 (.711–.761) | **186** | .312 |

## 2. Each step against the one before

| Comparison | OOF macro-F1: difference (95% CI); p; Holm within the chain | OOF Dice: difference; p; Holm within the chain | Test macro-F1: difference; p | Test Dice: difference; p |
|---|---|---|---|---|
| A1 vs A0: mask task | **+.042 (+.015 to +.070); p = .003; .022** | – | +.037; p = .22 | – |
| A2 vs A1: region pooling | −.006 (−.032 to +.018); p = .64; 1.00 | **−.004; p < .001; .004** | +.006; p = .87 | −.006; p = .98 |
| A3 vs A2: geometry | +.018 (−.009 to +.047); p = .21; 1.00 | +.005; p = .14; .41 | +.001; p = .96 | −.004; p = .82 |
| A4 vs A3: hierarchy | −.015 (−.040 to +.011); p = .27; 1.00 | −.007; p = .047; .24 | +.014; p = .60 | −.002; p = .91 |
| A5 vs A4: shared trunk | −.003 (−.027 to +.021); p = .82; 1.00 | −.001; p = .35; .71 | −.008; p = .88 | +.001; p = .57 |
| A6 vs A5: logit adjustment | −.014 (−.039 to +.010); p = .25; 1.00 | +.005; p = .79; .79 | +.028; p = .12 | +.003; p = .58 |
| **A6 vs A1: SLG-Net vs M0** | −.020 (−.047 to +.008); p = .17; 1.00 | −.002; p = .08; .30 | +.041; p = .21 | −.006; p = .66 |

No test-set difference reaches p < .05, even before any adjustment.

## 3. What the chain shows
1. **The mask task is the one step that clearly helps.** A1 beats A0 by +.042 macro-F1 out of fold, and this holds after Holm within the chain (p = .022). This repeats C05.
2. **None of SLG-Net's own components raises macro-F1 out of fold.** Every step from A2 to A6 moves it by less than .02, and none of these changes is significant.
   - **Geometry (A3) gives the best out-of-fold model.** It has macro-F1 .838, the fewest grade errors (60) and the highest Garden I-II recall: 39 of 57 femurs, against 31 for M0. Even so, it is not significantly better than A2 (p = .21).
   - **The hierarchy, the shared trunk and logit adjustment (A4–A6)** each lower out-of-fold macro-F1 a little, mostly through grade errors (60, then 72, 73 and 75).
   - **Logit adjustment did not raise the rare-class recalls out of fold.** Garden I-II stays at .614 (A5 .614), and subtrochanteric fractures go from .722 to .750 (26 → 27 of 36).
   - **Region pooling (A2) lowers Dice by .004.** This is small but consistent over the 680 fractured femurs (Holm p = .004), because the decision losses also reach the mask. No later step changes Dice significantly.
3. **SLG-Net (A6) compared with M0 (A1):**
   - Out of fold: macro-F1 −.020 (p = .17), Dice −.002 (p = .08), and 15 fewer femurs correct.
   - On the test set: macro-F1 +.041 (p = .21) and 4 more femurs correct.
   - The protocol makes the out-of-fold result primary. So **A6 is not better than M0**, and its higher test macro-F1 is not a confirmed gain: it runs against the out-of-fold result and is far from significant on 210 femurs.
   - No equivalence margin was fixed in advance, so "as good as M0" cannot be claimed either. The out-of-fold interval (−.047 to +.008) allows a loss of up to about .05.
4. **A3 was not the pre-specified final model.** The protocol named A6 as SLG-Net before any result. Choosing A3 now, on the same out-of-fold femurs, would be a post-hoc selection, so A3's numbers can only be reported as exploratory.

## 4. Check on folds 1–4 (Analysis/06 §3, rule 5)
The fold-0 development used only fold 0's femurs. The same runs, scored on the 835 femurs of folds 1–4 (thresholds fitted again on them):

| Run | Macro-F1, folds 1–4 | Macro-F1, all 5 folds | Dice, folds 1–4 | Dice, all 5 folds |
|---|---|---|---|---|
| A0 | .786 | .784 | – | – |
| A1 (M0) | .828 | .825 | .784 | .777 |
| A2 | .823 | .820 | .776 | .772 |
| A3 | .836 | .838 | .777 | .777 |
| A4 | .832 | .823 | .777 | .770 |
| A5 | .815 | .820 | .774 | .769 |
| A6 | .808 | .806 | .779 | .774 |

Every value moves by .009 or less, and A3 and M0 stay ahead of A6. **The fold-0 development did not inflate the final results.**

## 5. Calibration (reported only)
- **Temperatures** fitted on the OOF femurs: A0 3.43, A1 3.14, A2 2.74, A3 2.64, A4 3.21, A5 3.38, A6 3.41. Every model is over-confident.
- **ECE out of fold** falls from .097–.127 to .029–.046 after scaling.
- **ECE on the test ensemble:** before scaling, A4 (.078) and A6 (.081) are the best calibrated. Scaling does not help A6 there (.081 → .083), while it does for M0 (.103 → .064).

## 6. What this means, and what comes next
- **The pre-registered answer to "does each component help?" is no**, except for the mask task. SLG-Net is not more accurate than M0 out of fold, and its test-set advantage is not significant.
- **Whether SLG-Net is worth reporting** then rests on what its design is for: decisions that depend on the fracture region. C08 measures this with the oracle, deletion and risk–coverage tests.
- **C09** places A6 against every baseline, including the base article's YOLOv8 (S01).
- How to frame Paper A in light of this is a decision for the person. It is raised with the C07–C09 results, not decided here.
