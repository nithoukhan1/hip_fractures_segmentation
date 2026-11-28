# SLG-Net ablation analysis: Paper A's ablation chain, ConvNeXt-T classifier to SLG-Net, out of fold and on the test set

*Written 2026-11-27 from these filed runs:*
- *`S06_convnext_tiny` (ConvNeXt-T classifier, register line 7) and `S10_m0_convnext_unet` (ConvNeXt U-Net, line 8);*
- *`A2_region` … `A6_slgnet` (lines 22–26), trained from the tag `A-v1-frozen` = `f96f84c`.*

**Plan:** `Analysis/06_Step3_PaperA_Protocol.md` section 7, fixed before any SLG-Net result.
- Each step is compared with the one before it, and SLG-Net with ConvNeXt U-Net.
- Out-of-fold (OOF) results come first; the test set serves as confirmation.
- p-values are adjusted by Holm within the chain.
- The primary endpoints are macro-F1 and Dice on fractured femurs.

**Where the numbers come from** (all in this folder):
- `stats.md` / `stats.json`, made by
  `python scripts/run_stats.py S06_convnext_tiny S10_m0_convnext_unet A2_region A3_geometry A4_hierarchy A5_shared_grade A6_slgnet --pairs <the 7 chain pairs> --family-holm --name C07_paperA_ablation_chain`.
  - The column "p (Holm, family)" is the Holm adjustment within the chain. It is applied for each endpoint and each scope separately: over 7 comparisons, or 6 for Dice, because ConvNeXt-T classifier has no mask.
- `breakdown.md`: per-class results and error types, made by `python scripts/run_breakdown.py <the same 7 runs> --name C07_paperA_ablation_chain`.
- `folds_1-4/stats.md`: the check on folds 1–4 (section 4), made by `python scripts/run_stats.py <the same 7 runs> --folds 1,2,3,4 --name C07_paperA_ablation_chain/folds_1-4`.

**How decisions are made:** each run uses its own OOF fracture threshold (Youden). The tests are:
- for macro-F1, a randomisation test that swaps whole x-rays (5,000 permutations), with a 95% cluster-bootstrap interval for the difference;
- for Dice, a Wilcoxon signed-rank test over the fractured femurs, where a missed fracture counts as 0.

## 1. The chain

**Out of fold:** 1,043 femurs, of which 680 are fractured and 57 are Garden I-II.

| Step | Run | Macro-F1 (95% CI) | Dice, fractured (95% CI) | Correct | Garden I-II recall | Errors: detection / location / grade |
|---|---|---|---|---|---|---|
| ConvNeXt-T classifier | ConvNeXt-T classifier | .784 (.746–.816) | – | 901 | .474 (27/57) | 31 / 43 / 68 |
| ConvNeXt U-Net | + mask task = ConvNeXt U-Net | .825 (.793–.855) | .777 (.764–.789) | 922 | .544 (31/57) | 22 / 34 / 65 |
| Region-pooling model | + region pooling | .820 (.784–.850) | .772 (.760–.785) | 921 | .544 (31/57) | 20 / 37 / 65 |
| Geometry model | + region geometry | **.838** (.805–.866) | .777 (.765–.788) | **927** | **.684** (39/57) | 23 / 33 / **60** |
| Hierarchy model | + location → grade heads | .823 (.790–.852) | .770 (.756–.783) | 916 | .649 (37/57) | 19 / 36 / 72 |
| Shared-trunk model | + shared grade trunk | .820 (.786–.850) | .769 (.755–.782) | 918 | .614 (35/57) | 17 / 35 / 73 |
| SLG-Net | + logit adjustment = **SLG-Net** | .806 (.771–.836) | .774 (.761–.786) | 907 | .614 (35/57) | 25 / 36 / 75 |

**Test set:** the 5-model ensemble on 210 femurs, of which 145 are fractured and 16 are Garden I-II.

| Step | Macro-F1 (95% CI) | Dice, fractured (95% CI) | Correct | Garden I-II recall |
|---|---|---|---|---|
| ConvNeXt-T classifier | .742 (.651–.816) | – | 178 | .250 |
| ConvNeXt U-Net | .779 (.695–.849) | .744 (.718–.766) | 182 | .312 |
| Region-pooling model | .785 (.705–.847) | .738 (.712–.762) | 181 | .250 |
| Geometry model | .786 (.703–.854) | .734 (.706–.760) | 180 | .312 |
| Hierarchy model | .800 (.719–.866) | .732 (.703–.759) | 183 | .375 |
| Shared-trunk model | .792 (.712–.857) | .734 (.705–.760) | 183 | .312 |
| SLG-Net | **.820** (.747–.879) | .737 (.711–.761) | **186** | .312 |

## 2. Each step against the one before

| Comparison | OOF macro-F1: difference (95% CI); p; Holm within the chain | OOF Dice: difference; p; Holm within the chain | Test macro-F1: difference; p | Test Dice: difference; p |
|---|---|---|---|---|
| ConvNeXt U-Net vs ConvNeXt-T classifier: mask task | **+.042 (+.015 to +.070); p = .003; .022** | – | +.037; p = .22 | – |
| Region-pooling model vs ConvNeXt U-Net: region pooling | −.006 (−.032 to +.018); p = .64; 1.00 | **−.004; p < .001; .004** | +.006; p = .87 | −.006; p = .98 |
| Geometry model vs the region-pooling model: geometry | +.018 (−.009 to +.047); p = .21; 1.00 | +.005; p = .14; .41 | +.001; p = .96 | −.004; p = .82 |
| Hierarchy model vs the geometry model: hierarchy | −.015 (−.040 to +.011); p = .27; 1.00 | −.007; p = .047; .24 | +.014; p = .60 | −.002; p = .91 |
| Shared-trunk model vs the hierarchy model: shared trunk | −.003 (−.027 to +.021); p = .82; 1.00 | −.001; p = .35; .71 | −.008; p = .88 | +.001; p = .57 |
| SLG-Net vs the shared-trunk model: logit adjustment | −.014 (−.039 to +.010); p = .25; 1.00 | +.005; p = .79; .79 | +.028; p = .12 | +.003; p = .58 |
| **SLG-Net vs ConvNeXt U-Net: SLG-Net vs ConvNeXt U-Net** | −.020 (−.047 to +.008); p = .17; 1.00 | −.002; p = .08; .30 | +.041; p = .21 | −.006; p = .66 |

No test-set difference reaches p < .05, even before any adjustment.

## 3. What the chain shows
1. **The mask task is the one step that clearly helps.** ConvNeXt U-Net beats ConvNeXt-T classifier by +.042 macro-F1 out of fold, and this holds after Holm within the chain (p = .022). This repeats the mask-task analysis.
2. **None of SLG-Net's own components raises macro-F1 out of fold.** Every step from the region-pooling model to SLG-Net moves it by less than .02, and none of these changes is significant.
   - **Geometry (geometry model) gives the best out-of-fold model.** It has macro-F1 .838, the fewest grade errors (60) and the highest Garden I-II recall: 39 of 57 femurs, against 31 for ConvNeXt U-Net. Even so, it is not significantly better than the region-pooling model (p = .21).
   - **The hierarchy, the shared trunk and logit adjustment (hierarchical SLG-Net models)** each lower out-of-fold macro-F1 a little, mostly through grade errors (60, then 72, 73 and 75).
   - **Logit adjustment did not raise the rare-class recalls out of fold.** Garden I-II stays at .614 (shared-trunk model .614), and subtrochanteric fractures go from .722 to .750 (26 → 27 of 36).
   - **Region pooling (region-pooling model) lowers Dice by .004.** This is small but consistent over the 680 fractured femurs (Holm p = .004), because the decision losses also reach the mask. No later step changes Dice significantly.
3. **SLG-Net compared with ConvNeXt U-Net:**
   - Out of fold: macro-F1 −.020 (p = .17), Dice −.002 (p = .08), and 15 fewer femurs correct.
   - On the test set: macro-F1 +.041 (p = .21) and 4 more femurs correct.
   - The protocol makes the out-of-fold result primary. So **SLG-Net is not better than ConvNeXt U-Net**, and its higher test macro-F1 is not a confirmed gain: it runs against the out-of-fold result and is far from significant on 210 femurs.
   - No equivalence margin was fixed in advance, so "as good as ConvNeXt U-Net" cannot be claimed either. The out-of-fold interval (−.047 to +.008) allows a loss of up to about .05.
4. **Geometry model was not the pre-specified final model.** The protocol named SLG-Net as SLG-Net before any result. Choosing the geometry model now, on the same out-of-fold femurs, would be a post-hoc selection, so the geometry model's numbers can only be reported as exploratory.

## 4. Check on folds 1–4 (Analysis/06 section 3, rule 5)
The fold-0 development used only fold 0's femurs. The same runs, scored on the 835 femurs of folds 1–4 (thresholds fitted again on them):

| Run | Macro-F1, folds 1–4 | Macro-F1, all 5 folds | Dice, folds 1–4 | Dice, all 5 folds |
|---|---|---|---|---|
| ConvNeXt-T classifier | .786 | .784 | – | – |
| ConvNeXt U-Net | .828 | .825 | .784 | .777 |
| Region-pooling model | .823 | .820 | .776 | .772 |
| Geometry model | .836 | .838 | .777 | .777 |
| Hierarchy model | .832 | .823 | .777 | .770 |
| Shared-trunk model | .815 | .820 | .774 | .769 |
| SLG-Net | .808 | .806 | .779 | .774 |

Every value moves by .009 or less, and the geometry model and ConvNeXt U-Net stay ahead of SLG-Net. **The fold-0 development did not inflate the final results.**

## 5. Calibration (reported only)
- **Temperatures** fitted on the OOF femurs: ConvNeXt-T classifier 3.43, ConvNeXt U-Net 3.14, the region-pooling model 2.74, the geometry model 2.64, the hierarchy model 3.21, the shared-trunk model 3.38, SLG-Net 3.41. Every model is over-confident.
- **ECE out of fold** falls from .097–.127 to .029–.046 after scaling.
- **ECE on the test ensemble:** before scaling, the hierarchy model (.078) and SLG-Net (.081) are the best calibrated. Scaling does not help SLG-Net there (.081 → .083), while it does for ConvNeXt U-Net (.103 → .064).

## 6. What this means, and what comes next
- **The pre-registered answer to "does each component help?" is no**, except for the mask task. SLG-Net is not more accurate than ConvNeXt U-Net out of fold, and its test-set advantage is not significant.
- **Whether SLG-Net is worth reporting** then rests on what its design is for: decisions that depend on the fracture region. The SLG-Net interpretability analysis measures this with the oracle, deletion and risk–coverage tests.
- **SLG-Net main-table analysis** places SLG-Net against every baseline, including the base article's YOLOv8 (YOLOv8s-seg).
- How to frame Paper A in light of this is a decision for the person. It is raised with the SLG-Net analyses results, not decided here.
