# SLG-Net development analysis: Paper A's fold-0 development (three development questions) against ConvNeXt U-Net, decided by the fixed rule

*Written 2026-11-27 from the filed runs `results/A6_dev_base/`, `A6_dev_no_tf/`, `A6_dev_detach/`, `A6_dev_scale32/` (register lines 13–16) and ConvNeXt U-Net's fold-0 model in `results/S10_m0_convnext_unet/` (register line 8). Rule and questions: `Analysis/06_Step3_PaperA_Protocol.md` section 4. Every number below comes from the files in this folder:*
- `stats.md` / `stats.json`: `python scripts/run_stats.py S10_m0_convnext_unet A6_dev_base A6_dev_no_tf A6_dev_detach A6_dev_scale32 --compare --folds 0 --name C06_paperA_dev_fold0`.
  - Only fold 0's 208 held-out femurs (139 x-rays) are used.
  - Each run's fracture threshold is fitted again on those femurs (Youden).
  - The test set was not predicted by the development runs and is not used.
- The per-class counts of section 3 come from the same per-case files and the same thresholds. They are the confusion matrices of the five runs on fold 0.
- The oracle test is in each run's `metrics.json` (`tests.oracle`).

**The four runs** are SLG-Net trained on folds 1–4 and scored on fold 0, with ConvNeXt U-Net's recipe and seed (Step 2). Each changes one setting from the base:

| Run | Teacher forcing | Mask | Region pooling | Code | Time (one T4) |
|---|---|---|---|---|---|
| `A6_dev_base` | 30 epochs | end-to-end | 1/16 + 1/32 | `f342b8a` | 49.3 min |
| `A6_dev_no_tf` (teacher-forcing question) | **none** | end-to-end | 1/16 + 1/32 | `f342b8a` | 51.9 min |
| `A6_dev_detach` (detached-mask question) | 30 epochs | **detached** | 1/16 + 1/32 | `f342b8a` | 49.9 min |
| `A6_dev_scale32` (pooling-scale question) | 30 epochs | end-to-end | **1/32 only** | `f342b8a` | 50.6 min |

## 1. Results on fold 0 (208 femurs)

| Run | Macro-F1 (95% CI) | Change vs base | 6-class accuracy | Dice, fractured femurs (missed = 0) | Change vs base | Garden I-II recall (11 femurs) | Threshold |
|---|---|---|---|---|---|---|---|
| ConvNeXt U-Net (fold 0) | .822 (.735–.889) | | 184/208 (88.5%) | .749 (.710–.783) | | .545 | .0002 |
| A6_dev_base | **.776** (.675–.853) | | 184/208 | .754 (.718–.785) | | .545 | .0217 |
| A6_dev_no_tf (teacher-forcing question) | .783 (.689–.861) | +.008 | 184/208 | .754 | −.000 | .727 | .0004 |
| A6_dev_detach (detached-mask question) | .794 (.689–.876) | +.019 | 184/208 | .754 | .000 | .727 | .0016 |
| A6_dev_scale32 (pooling-scale question) | **.827** (.727–.897) | **+.052** | 185/208 (88.9%) | .760 (.729–.786) | +.006 | .818 | .0017 |

Intervals: 95% cluster bootstrap over x-rays.

**Paired tests** (`stats.md`, Holm within each pair; none significant):
- Base vs ConvNeXt U-Net: macro-F1 −.047 (p = .25), accuracy equal, Dice +.005 (p = .65).
- Base vs the pooling-scale question: macro-F1 −.052 (p = .18).
- Pooling-scale question vs ConvNeXt U-Net: macro-F1 +.005 (p = .87), Dice +.011 (p = .21).

The macro-F1 values to 4 decimals are ConvNeXt U-Net .8220, base .7755, the teacher-forcing question .7832, the detached-mask question .7945 and the pooling-scale question .8274. So the detached-mask question's gain is +.0190, and the pooling-scale question's is +.0519.

## 2. Decisions by the fixed rule (section 4)

| Rule | Result | Decision |
|---|---|---|
| 1. Go / no-go: base macro-F1 must reach .79 (ConvNeXt U-Net − .03) | .776 | **Fails: the design is reviewed before any 5-fold run** (section 3) |
| 2. The teacher-forcing question forcing: alternative needs ≥ +.02 macro-F1 and a Dice drop ≤ .01 | +.008 | Base stays: **teacher forcing over 30 epochs** |
| 2. The detached-mask question mask: same | +.019 (+.0190) | Base stays: **end-to-end** (short of .02 by .001) |
| 2. The pooling-scale question at 1/32 only: same | +.052; Dice +.006 | **Changes: region pooling at 1/32 only** |
| 3. Two winners → run their combination | one winner | No combination run is needed |
| 4. Garden I-II recall: reported only | .545 → .727–.818 for the alternatives | Not used (11 femurs) |

## 3. The design review (rule 1)

**Where the base loses.** It does not lose on accuracy: four of the five runs get exactly 184 of 208 femurs right.
- Each pair of runs differs on 8–16 femurs, about half going each way. Base vs ConvNeXt U-Net: 8 femurs right only in the base, 8 right only in ConvNeXt U-Net.
- What moves macro-F1 is which femurs are wrong. Almost all of the gap is in the two smallest classes:

| Class (fold-0 femurs) | ConvNeXt U-Net | Base | Pooling-scale question (1/32) |
|---|---|---|---|
| Subtrochanteric (7): right / F1 | 5 / .833 | 3 / .545 | 4 / .727 |
| Neck Garden I-II (11): right / F1 | 6 / .571 | 6 / .522 | 9 / .720 |
| Pertrochanteric Evans III-V (40): right / F1 | 31 / .861 | 35 / .909 | 31 / .849 |
| No fracture (72): right | 70 | 71 | 72 |

- One subtrochanteric femur is about .03 of macro-F1 on fold 0. The base's shortfall (−.047) is mostly two subtrochanteric femurs: `00845_00557`, called Evans I-II, and `01420_00944`, missed as no fracture.
- A lower Garden I-II precision adds a little: 12 femurs are called Garden I-II against ConvNeXt U-Net's 10, with the same 6 right. A better Evans III-V score makes up a little.
- With 7 and 11 femurs in these classes, the differences are within noise (p ≥ .18). By the rule, the base still fails.

**What the review finds.**
1. The same heads with region pooling at 1/32 only (pooling-scale question) reach .827, ConvNeXt U-Net's level (+.005), with Dice +.011. So the heads themselves do not hurt. The 1/16 region features are the likely cause: they are less semantic and finer (48 × 32 cells) and the smallest classes have the fewest training examples (29 subtrochanteric fractures in folds 1–4).
2. **Oracle test** (the heads get the released mask instead of the predicted one):

| Run | Macro-F1 as used → oracle | Accuracy as used → oracle | Wrong fractured femurs fixed by the oracle | Correct ones lost |
|---|---|---|---|---|
| Base | .776 → .839 (+.063) | 184 → 190 | 6 of 23 (26%) | 0 of 113 |
| Teacher-forcing question no teacher forcing | .783 → .791 (+.007) | 184 → 185 | 1 of 23 | 0 |
| Detached-mask question | .794 → .838 (+.044) | 184 → 186 | 2 of 24 | 0 |
| Pooling-scale question 1/32 only | .827 → .810 (−.017) | 185 → 183 | 0 of 23 | 2 of 113 |

   - With teacher forcing, the base heads clearly use the region: a better mask fixes a quarter of their wrong fractured femurs.
   - Without teacher forcing (teacher-forcing question), they hardly use it: the oracle mask changes 1 femur. This supports keeping teacher forcing.
   - With pooling at 1/32 only, the better mask does not help on fold 0. The final SLG-Net run will show, with the deletion test on all 1,043 femurs (SLG-Net interpretability analysis), how much the 1/32 heads depend on the region. This is a point to watch, not a reason to override the rule: fold 0 has 23 wrong fractured femurs.
3. **Calibration** (reported only; `stats.md`): every run is over-confident by about the same factor. Temperature 3.1–3.4, OOF ECE .10–.13 before, .04–.07 after.
   - ConvNeXt U-Net's 5-fold temperature is 3.14; fitted on fold 0 alone it is 3.22.
   - A6_dev_detach has one femur whose true class is written as probability 0 (below the 8 decimals of the per-case file). Like every written 0 of a network, it is read as 10⁻¹²; `scripts/run_stats.py` changed for this on 2026-11-27, and before that change its temperature was "not applicable".
   - The fold-0 thresholds are very low (.0002–.022) for the same reason: almost every no-fracture femur gets a fracture probability near 0, and Youden ties go to the lower threshold.

**Review conclusion.** The base setting failed the go/no-go because of its 1/16 region features. The pooling-scale question removes them, and with that setting the heads match ConvNeXt U-Net on fold 0 (.827, above the .79 line). The proposed settings for the freeze (Paper A milestone 2, tag `A-v1-frozen`) are therefore:
- teacher forcing over 30 epochs;
- an end-to-end mask;
- region pooling at 1/32 only.

The person decides whether to freeze on this basis or to check the pooling-scale question choice with a second seed first (section 5).

## 4. Caveats

- **Fold 0 is small** (208 femurs; 7 subtrochanteric, 11 Garden I-II). Every difference in this report is within noise, as the protocol expected; the rule only sets defaults.
- **The choice of 1/32 was made on the same 208 femurs** that showed the base's shortfall. The protocol's safeguards are:
  - the final runs are judged on all 1,043 out-of-fold femurs and the test set;
  - results are also reported on folds 1–4 only (Analysis/06 section 3, point 5), which no development run has seen.
- **One seed per setting.** Seed-to-seed variation on fold 0 was not measured. It could be as large as these differences.
- **Interactions were not tested.** The three development questions each change one setting of the base.

## 5. Decision

1. Teacher forcing and the end-to-end mask stay, and region pooling moves to 1/32 only.
2. The design review finds no fault in the heads at that setting.
3. `kaggle/runs.json` holds the five final runs with these settings: `A2_region`, `A3_geometry`, `A4_hierarchy`, `A5_shared_grade` and `A6_slgnet` (5 folds and the test set, about 2.6 h each on T4 x2).
4. They start from the tag `A-v1-frozen` once the person approves the freeze. The alternative is to repeat `A6_dev_base` and `A6_dev_scale32` with a second seed first (one session, about 55 min), with the same rule applied to the mean of the two seeds.
