# Reference analysis: Does a healthy reference help, and does retrieval matter? (reference question)

*Written 2026-11-28 from these filed runs (all from tag `B-v1-frozen` = 1c88912, 5/5 folds):*
- *`S10_m0_convnext_unet` (ConvNeXt U-Net, register line 8);*
- *`R2_random_ref` (line 33), `R3_nearest_ref` (line 32), `R4_refrac` (RefFrac, line 31);*
- *for the reference tests (section 4) also `R1_aasn` (35), `R5_concat` (36), `R6a_no_align` (37), `R6b_affine` (38) and `R7_partner` (34).*

**Plan:** `Analysis/08_Step5_PaperB_RefFrac_Protocol.md` section 5 (reference analysis) and section 2.6 (the reference tests). Primary endpoints, fixed before any R run (Report 04 section 5.5): **mean Dice on fractured femurs** and **Garden I-II recall** (57 out-of-fold and 16 test femurs). Subgroups: the smallest third of masks; paired vs single femurs.

**The chain of comparisons** (difference A − B): the random-reference model vs ConvNeXt U-Net (does *any* healthy reference help?), the nearest-reference model vs the random-reference model (does *retrieval* help?), RefFrac vs the nearest-reference model (do 3 references beat 1?), and RefFrac vs ConvNeXt U-Net (RefFrac overall). Holm across these 4 pairs, each endpoint and scope separately.

**Where the numbers come from** (all in this folder):
- `stats.md` / `stats.json`: intervals, calibration and the paired tests of Dice, macro-F1, accuracy, AUC and fracture detection, made by
  `python scripts/run_stats.py S10_m0_convnext_unet R2_random_ref R3_nearest_ref R4_refrac --pairs R2_random_ref:S10_m0_convnext_unet,R3_nearest_ref:R2_random_ref,R4_refrac:R3_nearest_ref,R4_refrac:S10_m0_convnext_unet --family-holm --name C13_paperB_reference`.
- `g12.md`: Garden I-II recall (exact McNemar on the Garden I-II femurs, bootstrap interval, Holm across the 4 pairs), made by
  `python scripts/run_refstats.py g12 <the same 4 runs> --pairs <the same 4 pairs> --name C13_paperB_reference`.
- `subgroups.md`: Dice in the subgroups, `python scripts/run_refstats.py subgroups <the same> --name C13_paperB_reference`.
- `reftests.md`: the reference tests of every R run, `python scripts/run_refstats.py reftests R1_aasn R2_random_ref R3_nearest_ref R4_refrac R5_concat R6a_no_align R6b_affine R7_partner --name C13_paperB_reference`.
- The check on folds 1–4 (rule 5 of section 3): `folds_1-4/stats.md` (`run_stats.py … --folds 1,2,3,4`) and `g12_folds_1-2-3-4.md` (`run_refstats.py g12 … --folds 1,2,3,4`); thresholds refitted on those femurs.
- `breakdown.md`: per class and error type, `python scripts/run_breakdown.py <the same 4 runs> --name C13_paperB_reference`.

## 1. The four models

| Model | OOF Dice, fractured (95% CI) | OOF Garden I-II recall (57) | OOF macro-F1 | OOF correct /1043 | Test Dice | Test Garden I-II (16) | Test macro-F1 | Test correct /210 |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt U-Net (no reference) | .777 (.764–.789) | .544 | .825 | 922 | **.744** | .312 | .779 | **182** |
| Random-reference model (a random healthy femur) | **.781** (.769–.792) | .614 | .814 | 916 | .741 | **.438** | **.804** | **182** |
| Nearest-reference model (the nearest healthy femur) | .775 (.763–.788) | .684 | .826 | 918 | .742 | .375 | .765 | 177 |
| RefFrac (the 3 nearest) | .777 (.764–.789) | **.702** | **.826** | 920 | .739 | .312 | .779 | 180 |

## 2. Primary endpoints (paired, same femurs)

**Dice on fractured femurs** (Wilcoxon, `stats.md`):

| A vs B | OOF difference; p; Holm | Test difference; p; Holm |
|---|---|---|
| Random-reference model vs ConvNeXt U-Net | +.004; .61; 1.00 | −.003; .23; .77 |
| Nearest-reference model vs the random-reference model | −.005; .49; 1.00 | +.001; .51; 1.00 |
| RefFrac vs the nearest-reference model | +.001; .84; 1.00 | −.003; .76; 1.00 |
| RefFrac vs ConvNeXt U-Net | +.000; .25; .98 | −.005; .19; .77 |

**Garden I-II recall** (`g12.md`; "only A / only B" = Garden I-II femurs that only one of the two gets right):

| A vs B | OOF difference (95% CI) | Only A / only B | p (exact); Holm | Test difference | Only A / only B | p |
|---|---|---|---|---|---|---|
| Random-reference model vs ConvNeXt U-Net | +.070 (−.035 to +.175) | 7 / 3 | .34; 1.00 | +.125 | 2 / 0 | .50 |
| Nearest-reference model vs the random-reference model | +.070 (−.053 to +.176) | 8 / 4 | .39; 1.00 | −.062 | 0 / 1 | 1.00 |
| RefFrac vs the nearest-reference model | +.018 (−.053 to +.088) | 3 / 2 | 1.00; 1.00 | −.062 | 1 / 2 | 1.00 |
| RefFrac vs ConvNeXt U-Net | **+.158 (+.035 to +.281)** | 12 / 3 | **.035**; .14 | .000 | 1 / 1 | 1.00 |

**Secondary endpoints:** nothing differs in macro-F1, accuracy, AUC or fracture detection, out of fold or on the test set (every Holm p ≥ .27). The largest test difference is the nearest-reference model below the random-reference model in macro-F1 (−.039, p .067).

**Folds 1–4 check:** Dice moves by ≤ .007 (ConvNeXt U-Net .784, the random-reference model .783, the nearest-reference model .780, RefFrac .778). Garden I-II recall on those 46 femurs: ConvNeXt U-Net .543, the random-reference model .587, the nearest-reference model .717, RefFrac .652. RefFrac vs ConvNeXt U-Net becomes +.109 (9 / 4, p .27). The nearest-reference model vs the random-reference model +.130 (8 / 2, p .11). The ordering "every reference model above ConvNeXt U-Net" holds, but the size of RefFrac's gain shrinks, so the fold-0 development did not create the effect but it is small.

## 3. Subgroups (`subgroups.md`)

| Subgroup (fractured femurs) | n OOF / test | ConvNeXt U-Net | Random-reference model | Nearest-reference model | RefFrac | Any pair after Holm? |
|---|---|---|---|---|---|---|
| Smallest third of masks | 227 / 49 | .712 / .661 | .724 / .653 | .712 / .659 | .713 / .656 | OOF none. Test: the random-reference model < ConvNeXt U-Net by .007 (Holm .026), the nearest-reference model > the random-reference model by .006 (.047) |
| Larger two thirds | 453 / 96 | .809 / .786 | .809 / .785 | .807 / .784 | .809 / .781 | none |
| With a healthy partner | 334 / 64 | .784 / .731 | .790 / .727 | .784 / .731 | .780 / .730 | none |
| Single femur | 346 / 81 | .769 / .754 | .772 / .751 | .768 / .750 | .774 / .746 | none |

(Values: OOF / test Dice.) A reference does not help small masks, and it helps femurs with or without a partner equally (not at all). The two test-set differences in small masks are under .01 Dice and point in opposite directions out of fold, so they are noise.

## 4. Does the model use the reference? The reference tests (`reftests.md`)

Each run predicted the same femurs again with other references. Each mode is read at the threshold of the mode it replaces.

| Run | Self (the femur as its own reference): OOF Dice | Garden I-II | Fractures lost | Random healthy reference: OOF Dice | Fractures lost | Pointing game, OOF (chance .079) |
|---|---|---|---|---|---|---|
| AASN-style model AASN | .781 → .768 | .579 → .561 | 11 / 668 | .781 → .781 | 1 | .42 |
| Random-reference model | .781 → .705 | .614 → .526 | 64 / 665 | (its own mode) | 0 | .68 |
| Nearest-reference model | .775 → .756 | .684 → .614 | 17 / 669 | .775 → .775 | 2 | .55 |
| **RefFrac** | **.777 → .536** | **.702 → .158** | **201 / 663** | .777 → .775 | 2 | .53 |
| Concatenation model | .780 → .733 | .561 → .421 | 34 / 662 | .780 → .779 | 1 | .35 |
| No-alignment model no alignment | .777 → .222 | .561 → .000 | 477 / 662 | .777 → .773 | 5 | .48 |
| Affine-alignment model | .776 → .133 | .561 → .000 | 553 / 660 | .776 → .772 | 4 | .39 |

- **Self:** every self change in Dice is significant out of fold (Holm < .001). P(fracture) of fractured femurs falls (RefFrac −.586) while that of healthy femurs moves by at most .014, and no healthy femur turns fractured. So the models read the *difference* from the reference: with no difference, they call the femur healthy. RefFrac depends on this far more than the first three reference runs and the concatenation model.
- **Random:** swapping the retrieved references for a random healthy femur changes almost nothing (OOF Dice ≤ .004, every Holm p ≥ .23; at most 11 of ~357 healthy femurs flip). The model needs *a healthy* femur, not a *matched* one.
- **Pointing game:** the largest cell of |target − reference| touches the released mask in 35–68% of fractured femurs, 4–9 times chance. The random-reference model's random reference points best (.68).
- **Test set:** the same pattern, weaker for self (RefFrac .739 → .680, 13 of 143 fractures lost; the no-alignment model and the affine-alignment model still collapse).
- Self is a pattern the model never saw in training, so these drops show dependence, not the size of a benefit.

## 5. What the reference analysis shows
1. **Reference question, Dice (primary): no.** A healthy reference does not improve segmentation. Every Dice difference is ≤ .005, out of fold and on the test set, overall and in every subgroup.
2. **Reference question, Garden I-II recall (primary): a trend that does not survive.** Out of fold, every reference model finds more Garden I-II fractures than ConvNeXt U-Net, ordered ConvNeXt U-Net .544 < the random-reference model .614 < the nearest-reference model .684 < RefFrac .702. RefFrac vs ConvNeXt U-Net is +.158 (12 vs 3 femurs, exact p .035) but Holm across the 4 pairs gives .14. On the 16 test femurs there is no gain (RefFrac .312 = ConvNeXt U-Net). Folds 1–4 keep the order but shrink RefFrac's gain to +.109.
3. **Does retrieval matter? No.** nearest-reference model vs the random-reference model is not significant on any endpoint. Swapping a model's retrieved references for random ones at prediction time changes nothing.
4. **The models do use the reference.** Taking the reference away (self) removes RefFrac's fracture decisions (201 of 663 found fractures lost, Garden I-II recall .702 → .158). What they use is "different from a healthy femur", not "different from a femur like me".
5. **For the paper:** Paper B's pre-registered primary hypothesis is not confirmed. What can be reported honestly is (a) a negative result on Dice, (b) a Garden I-II trend that is consistent in direction out of fold but not significant after correction and absent on the test set, and (c) a clear mechanistic finding: the comparison is used, but its *match* is not. See the own-hip analysis (own hip), the RefFrac component analysis (alignment, fusion, AASN) and the RefFrac main-table analysis (main table).
