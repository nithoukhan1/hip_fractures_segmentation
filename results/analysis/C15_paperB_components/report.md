# RefFrac component analysis: RefFrac's components: alignment, fusion, and the published baseline AASN (alignment-and-fusion question)

*Written 2026-11-28 from these filed runs (tag `B-v1-frozen` = 1c88912, 5/5 folds):* *`R4_refrac` (RefFrac: 3 nearest references, local cross-attention alignment, difference fusion; register line 31), `R1_aasn` (AASN adapted: nearest reference, global affine, concatenation + margin loss; line 35), `R5_concat` (RefFrac with concatenation; line 36), `R6a_no_align` (RefFrac without alignment; line 37), `R6b_affine` (RefFrac with global affine alignment; line 38).*

**Plan:** `Analysis/08_Step5_PaperB_RefFrac_Protocol.md` section 5 (RefFrac component analysis) and sections 2.2–2.4. Each comparison changes one component of RefFrac, except the AASN-style model, which is the closest published method (Chen et al., ECCV 2020) adapted to femur crops.

**Where the numbers come from** (all in this folder):
- `stats.md` / `stats.json`: `python scripts/run_stats.py R4_refrac R1_aasn R5_concat R6a_no_align R6b_affine --pairs R4_refrac:R1_aasn,R4_refrac:R5_concat,R4_refrac:R6a_no_align,R4_refrac:R6b_affine --family-holm --name C15_paperB_components`. Holm across the 4 pairs, each endpoint and scope separately.
- `g12.md`: Garden I-II recall, `python scripts/run_refstats.py g12 <the same 5 runs> --pairs <the same 4 pairs> --name C15_paperB_components`.
- `breakdown.md`: per class and error type.
- The reference tests of these runs (self, random) and the pointing game: `../C13_paperB_reference/reftests.md` and the reference analysis, section 4.

## 1. The five models

| Model | What differs from RefFrac | Minutes (T4 x2) | OOF Dice (95% CI) | OOF Garden I-II | OOF macro-F1 | OOF correct | Test Dice | Test Garden I-II | Test macro-F1 | Test correct |
|---|---|---|---|---|---|---|---|---|---|---|
| **RefFrac** | – | 242 | .777 (.764–.789) | **.702** | .826 | 920 | .739 | .312 | .779 | 180 |
| AASN-style model AASN (adapted) | 1 reference, affine, concatenation, margin loss | 178 | **.781** (.769–.792) | .579 | .820 | 915 | .738 | **.438** | **.798** | **181** |
| Concatenation model | concatenation instead of difference | 242 | .780 (.768–.792) | .561 | **.827** | **925** | .740 | .188 | .741 | 178 |
| No-alignment model | no alignment | 230 | .777 (.764–.789) | .561 | .807 | 908 | .741 | .312 | .788 | 180 |
| Affine-alignment model | global affine instead of local cross-attention | 266 | .776 (.763–.788) | .561 | .814 | 915 | **.742** | .312 | .792 | **181** |

## 2. RefFrac against each variant (paired)

| RefFrac vs | OOF Dice: diff; p (Holm) | OOF macro-F1: diff; p (Holm) | OOF Garden I-II: diff (95% CI); only RefFrac / only other; p; Holm | Test Dice; p | Test macro-F1; p | Test Garden I-II |
|---|---|---|---|---|---|---|
| AASN-style model AASN | −.004; 1.00 | +.006; 1.00 | +.123 (+.035 to +.228); 8 / 1; .039; .077 | +.001; .09 | −.019; .44 | −.125 (0 / 2) |
| Concatenation model | −.004; .33 | −.002; 1.00 | +.140 (+.035 to +.246); 9 / 1; .021; .064 | −.001; .96 | +.038; .17 | +.125 (2 / 0) |
| No-alignment model no alignment | +.000; 1.00 | +.019; .69 | **+.140 (+.053 to +.228); 8 / 0; .008; .031** | −.002; .93 | −.009; .65 | .000 (0 / 0) |
| Affine-alignment model | +.001; 1.00 | +.012; .97 | +.140 (+.035 to +.263); 10 / 2; .039; .077 | −.004; .37 | −.013; .64 | .000 (1 / 1) |

Dice, macro-F1, accuracy and AUC: no difference survives Holm, out of fold or on the test set (smallest family-Holm p .33). Garden I-II, test set: no difference (16 femurs; every p ≥ .50).

## 3. What the reference tests add (reference analysis, section 4)

| Model | Self test: OOF Dice; fractures lost | Pointing game, OOF (chance .079) |
|---|---|---|
| RefFrac | .777 → .536; 201 / 663 | .53 |
| AASN-style model AASN | .781 → .768; 11 / 668 | .42 |
| Concatenation model | .780 → .733; 34 / 662 | .35 |
| No-alignment model no alignment | .777 → .222; 477 / 662 | .48 |
| Affine-alignment model | .776 → .133; 553 / 660 | .39 |

- **Difference vs concatenation (RefFrac vs the concatenation model):** with the |T − R̃| term, the model leans on the comparison much more (201 vs 34 fractures lost without it) and points to the fracture more often (.53 vs .35). Its accuracy is the same.
- **Alignment (RefFrac vs the no-alignment model, the affine-alignment model):** without alignment, or with a global affine one, the model depends on the reference even more (it collapses in the self test), yet points less well (.48, .39). Local cross-attention alignment gives the best pointing among the difference models with 3 references.
- **AASN (AASN-style model)** hardly uses its reference (11 fractures lost) and points less well (.42). Its Garden I-II recall is lower than RefFrac's out of fold.

## 4. What the RefFrac component analysis shows
1. **No component changes Dice or macro-F1.** All five models are within .005 Dice and .02 macro-F1 out of fold.
2. **Garden I-II recall is where RefFrac stands out out of fold:** +.12 to +.14 against every variant. Against the no-alignment model (no alignment) it survives Holm (8 vs 0 femurs, Holm .031); against the AASN-style model, the concatenation model and the affine-alignment model Holm gives .06–.08. On the 16 test femurs none of this is seen.
3. **The components change how the reference is used**, not how well the model scores: the difference fusion and local alignment make the model rely on, and point with, the comparison.
4. **Against the published idea (AASN adapted, the AASN-style model):** RefFrac is not better in Dice or macro-F1; it finds more Garden I-II fractures out of fold (+.123, Holm .077) but fewer on the test set (5 vs 7 of 16).
5. **For the paper:** the ablation supports the design only on the secondary, mechanistic side (dependence, pointing) and on an out-of-fold Garden I-II trend. It does not support a claim that RefFrac's components improve accuracy.
