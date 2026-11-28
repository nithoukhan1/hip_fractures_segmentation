# C08: Paper A's interpretability tests: oracle, region deletion, risk–coverage and geometry consistency

*Written 2026-11-27 from the filed runs `A2_region` … `A6_slgnet` (register lines 22–26) and M0 (`S10_m0_convnext_unet`, line 8). Updated 2026-11-28 with A6's geometry (§4) and with `A1_m0_deletion` (line 39), M0 retrained with the deletion test (§2b, the rule of Analysis/06 §13).*

**Plan:** `Analysis/06_Step3_PaperA_Protocol.md` §5 (tests 1, 2, 3 and 5 of Report 04 §2), fixed before any SLG-Net result.

**Where the numbers come from:**
- `interpretability.md` / `interpretability.json` in this folder, made by
  `python scripts/run_interpretability.py S10_m0_convnext_unet A2_region A3_geometry A4_hierarchy A5_shared_grade A6_slgnet A1_m0_deletion --geometry S10_m0_convnext_unet=results/analysis/C08_paperA_interpretability/geometry_S10_m0_convnext_unet.csv --geometry A6_slgnet=results/analysis/C08_paperA_interpretability/geometry_A6_slgnet.csv --compare-deletion A6_slgnet:A1_m0_deletion --name C08_paperA_interpretability` (re-run 2026-11-28 with A6's geometry and then with `A1_m0_deletion`; every earlier number is unchanged).
- `m0_replicate/stats.md`: the new M0 run against S10, `python scripts/run_stats.py A1_m0_deletion S10_m0_convnext_unet --pairs A1_m0_deletion:S10_m0_convnext_unet --name C08_paperA_interpretability/m0_replicate`.
- `geometry_<run>.csv` holds the geometry of each femur's predicted region, read from the run's saved probability maps.
  - The maps are in the Kaggle work folder, which is not in the repository. Each file was made on the person's PC with `python scripts/map_geometry.py <the run's output zip> --run <run> --out ...`.
  - These regions reproduce the areas in the run's per-case files: out of fold to within 0.08% (the maps are saved with 8 bits), and exactly on the test set.
- Every change and share in §1–§2 is recomputed from the per-case files. Each one equals what the run itself wrote in `metrics.json` (column "Same as metrics.json").

**How decisions are made:** at each run's own OOF threshold. Out of fold there are 1,043 femurs (680 fractured); the test ensemble has 210 (145 fractured).

## 1. Oracle test: the heads get the released mask (A2–A6)

| Run | Correct, OOF: base → oracle (McNemar p) | Macro-F1, OOF: base → oracle (p) | Fractured, OOF: wrong ones fixed / correct ones lost | Correct, test: base → oracle (p) |
|---|---|---|---|---|
| A2 region pooling | 921 → 930 (.12) | .820 → .832 (.15) | 13 of 117 / 9 of 563 | 181 → 187 (.031) |
| A3 + geometry | 927 → 937 (.087) | .838 → .845 (.39) | 9 of 106 / 9 of 574 | 180 → 186 (.11) |
| A4 + hierarchy | 916 → 921 (.41) | .823 → .825 (.72) | 14 of 122 / 9 of 558 | 183 → 185 (.50) |
| A5 + shared trunk | 918 → 924 (.31) | .820 → .826 (.39) | 15 of 120 / 9 of 560 | 183 → 185 (.63) |
| A6 SLG-Net | 907 → 912 (.23) | .806 → .810 (.51) | 8 of 130 / 3 of 550 | 186 → 188 (.63) |

- **Better masks would help only a little.** With the released mask, the flat heads (A2, A3) gain 9–10 correct femurs out of fold and 6 on the test set; this is significant only for A2 on the test set (p = .031). The hierarchical heads (A4–A6) gain 2–6.
- **For SLG-Net, the remaining errors are mostly not mask errors.** The released mask fixes 8 of its 130 wrong fractured femurs and breaks 3 of its 550 correct ones.

## 2. Region deletion vs control (A6)

The predicted region is blurred away (deletion), or the same shape is blurred at another place on bone (control).

| Part | Correct fractured decisions lost: deletion / control (McNemar p) | Change of P(true class): deletion / control, mean (Wilcoxon p) | Change of P(fracture): deletion / control, mean (Wilcoxon p) |
|---|---|---|---|
| OOF (680 fractured, 550 correct) | **356 (64.7%) / 7 (1.3%)** (p < .001) | −.479 / −.008 (p < .001) | −.032 / −.005 (p < .001) |
| Test ensemble (145 fractured, 122 correct) | **80 (65.6%) / 0** (p < .001) | −.538 / −.002 (p < .001) | −.039 / −.002 (p < .001) |

1. **SLG-Net's location and grade decisions depend on the fracture region.** Removing the region loses about two thirds of the correct decisions, while removing an equal area elsewhere loses almost none.
2. **The fracture / no-fracture decision does not depend on it.** P(fracture) hardly moves (median change −.000). After deletion the femur is still called fractured, but it falls into the most common class, neck Garden III-IV: 433 of the 680 fractured femurs are called that, against 241 before (confusion matrices in `results/A6_slgnet/metrics.json`, `tests`).
3. **Per class (OOF, correct decisions lost: deletion / control):**

   | Class | Lost: deletion / control |
   |---|---|
   | Pertrochanteric Evans I-II | 99 / 1 of 105 |
   | Pertrochanteric Evans III-V | 152 / 3 of 160 |
   | Neck Garden I-II | 31 / 0 of 35 |
   | Subtrochanteric | 16 / 0 of 27 |
   | Neck Garden III-IV | 58 / 3 of 223 |

   Garden III-IV loses least because it is where deleted femurs end up. For the same reason, deletion turns 25 of the 130 wrong fractured decisions into right ones; the control turns 3.
4. **Where the control was placed** (OOF fractured femurs):
   - 653 of 680 on tissue at least 0.8 × as bright as the region;
   - 10 on dimmer tissue;
   - 2 with no place free of overlap;
   - 15 femurs had no predicted region, so nothing was deleted.
5. ~~**Limit:** M0 was not run with this test, so these numbers show that SLG-Net's decisions use the region, not that they use it more than M0's do.~~ Resolved 2026-11-28 by §2b.

## 2b. SLG-Net against M0 under deletion (Analysis/06 §13, fixed before the M0 run)

**The M0 run.** `A1_m0_deletion` is M0 trained again with S10's settings and seed, plus the deletion test. It replicates S10 (`m0_replicate/stats.md`): out of fold 921 vs 922 correct, macro-F1 .825 vs .825, Dice .778 vs .777, no difference significant; test 179 vs 182 correct, macro-F1 .783 vs .779. Its Garden I-II recall is .614 out of fold (S10 .544): with 57 femurs, a retrained M0 moves 4 femurs by chance.

**M0 alone** (same test as §2): deletion loses 274 of its 562 correct fractured decisions out of fold (48.8%), the control 8 (1.4%); on the test set 62 of 118 (52.5%) vs 1. Unlike SLG-Net, M0's P(fracture) also falls (mean −.199 vs A6 −.032).

**The paired comparison** (fractured femurs that both classify correctly at base):

| Part | Both correct | Lost: SLG-Net / M0 | Only SLG-Net lost / only M0 lost | Difference (95% CI) | p (mid-p) | Controls lost: SLG-Net / M0 |
|---|---|---|---|---|---|---|
| OOF | 525 | **337 (64.2%) / 253 (48.2%)** | 117 / 33 | **+16.0 points (+11.6 to +20.4)** | < .001 | 7 / 5 |
| Test ensemble | 116 | 76 (65.5%) / 61 (52.6%) | 19 / 4 | +12.9 points (+5.2 to +20.7) | .002 | 0 / 1 |

- Change of P(true class) under deletion, all fractured femurs: SLG-Net −.479 vs M0 −.442 out of fold (Wilcoxon p .044); −.538 vs −.476 on the test set (p < .001).
- Each model's own control loses 1.3% (SLG-Net) and 1.4% (M0) out of fold, below the 5% limit.

**Verdict by the rule of §13: SLG-Net's location and grade decisions depend on the fracture region more than M0's.** Out of fold it loses 16 points more (p < .001), and the test set points the same way (+12.9 points, p .002).

What this means: M0 also uses the region a lot (about half its correct decisions are lost), because its decoder and head share the encoder that sees the fracture. SLG-Net's explicit region pooling makes the dependence clearly stronger, and it confines it to location and grade: SLG-Net's fracture decision stays global, while M0's fracture probability also drops.

## 3. Risk–coverage: does low confidence flag the errors? (test 5)

The confidence of a decision is the probability of the decided class. Flagging the 10% least confident femurs at random would catch 10% of the errors.

| Run | OOF: errors among the 10% least confident (95% CI) | OOF error rate: all → the 90% kept | Test: errors among the 10% least confident | Test error rate: all → kept |
|---|---|---|---|---|
| M0 | 43 of 121 = 35.5% (27.3–41.9%) | 11.6% → 8.3% | 9 of 28 = 32.1% | 13.3% → 10.1% |
| A2 | 42 of 122 = 34.4% (27.9–42.0%) | 11.7% → 8.5% | 11 of 29 = 37.9% | 13.8% → 9.5% |
| A3 | 45 of 116 = 38.8% (31.5–46.8%) | 11.1% → 7.6% | 13 of 30 = 43.3% | 14.3% → 9.0% |
| A4 | 51 of 127 = 40.2% (32.6–46.8%) | 12.2% → 8.1% | 11 of 27 = 40.7% | 12.9% → 8.5% |
| A5 | 50 of 125 = 40.0% (32.5–47.2%) | 12.0% → 8.0% | 7 of 27 = 25.9% | 12.9% → 10.6% |
| A6 | 44 of 136 = 32.4% (25.0–39.2%) | 13.0% → 9.8% | 6 of 24 = 25.0% | 11.4% → 9.5% |

- **For every model, the least confident 10% hold a third or more of the errors** (32–40% out of fold), so confidence is useful for sending cases to a reader.
- **SLG-Net's confidence is not better at this than M0's** (32.4% vs 35.5% out of fold, with overlapping intervals). A3–A5 catch a little more (39–40%), also within the intervals.

## 4. Geometry consistency: does the region agree with the decided location and grade? (test 3)

The released masks show that region height separates the Garden grades, area separates the Evans grades, and the centroid height gives the location (Report 03 §2). The test asks whether a model's decisions relate to its own predicted region in the same way.
- **Location rule:** two cut points on the centroid's distance from the top of the image (as a share of its height), fitted on the 680 OOF fractures' released masks:
  - neck when the share is below 0.389;
  - pertrochanteric when it is from 0.389 to 0.488;
  - subtrochanteric when it is above 0.488.
- **The released masks reproduce Report 03:**
  - Garden height AUC .81 (OOF) and .84 (test);
  - Evans area AUC .80 and .75;
  - the location rule is right for 82.6% (OOF, fitted there) and 80.0% (test).

| Model | Part | Location: rule on the region = decided location | Garden: region height vs decided grade, AUC (95% CI) | Evans: region area vs decided grade, AUC (95% CI) |
|---|---|---|---|---|
| M0 | OOF | 555 of 662 (83.8%) | .92 (.88–.97) | .85 (.81–.89) |
| M0 | test | 114 of 143 (79.7%) | .94 (.87–1.00) | .83 (.74–.93) |
| Released masks vs released labels | OOF / test | 82.6% / 80.0% | .81 / .84 | .80 / .75 |
| **A6** | OOF | 547 of 664 (82.4%) | .87 (.81–.93) | .85 (.81–.89) |
| **A6** | test | 110 of 143 (76.9%) | .98 (.96–1.00) | .88 (.80–.96) |

- **M0's decisions already agree with its own regions**, even though its head never sees the region's geometry. Its decided Garden grade follows its region's height more closely (AUC .92) than the released grades follow the released masks (.81).
- **A6's decisions agree with its regions about as well as M0's do, not better** (added 2026-11-28, from `geometry_A6_slgnet.csv`, made on the PC from `A6_slgnet.zip`):
  - out of fold, location 82.4% vs M0's 83.8%, Garden height AUC .87 vs .92 (the intervals overlap), Evans area .85 vs .85;
  - on the test set, A6 is higher for the grades (.98 vs .94; .88 vs .83) and lower for location (76.9% vs 79.7%), on 56–143 femurs;
  - its regions carry the released grades as well as M0's do: height vs the released Garden grade .84 (M0 .87), area vs the released Evans grade .81 (M0 .82).
- **Region check:** A6's regions reproduce the areas in its per-case files to within 0.09% out of fold, and exactly on the test set.

## 5. What C08 shows
1. **SLG-Net's location and grade decisions are tied to the predicted fracture region** (deletion −65% vs control −1%), which is what the design set out to achieve. Its fracture / no-fracture decision comes from the whole image.
2. **Its remaining errors are mostly not caused by imperfect masks** (oracle: 8 fixed, 3 broken).
3. **This dependence does not buy accuracy** (C07), and SLG-Net's confidence flags its errors no better than M0's.
4. **The geometry test does not separate SLG-Net from M0:** both models' decided locations and grades agree with their own regions to a similar degree (out of fold, Garden AUC .87 vs .92, Evans .85 vs .85, location 82% vs 84%).
5. ~~No comparison with M0 is possible for the deletion test.~~ **Updated 2026-11-28 (§2b): SLG-Net's decisions depend on the fracture region significantly more than M0's** (64.2% vs 48.2% of the jointly correct fractured decisions lost, p < .001; test set +12.9 points, p .002), by the rule fixed before M0's run. This is SLG-Net's one property that M0 does not match, and the basis of its interpretability claim; accuracy (C07) and the geometry test (§4) do not separate the two models.
