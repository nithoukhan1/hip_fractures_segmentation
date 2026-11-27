# C11: Paper A's second-seed check of the base and of region pooling at 1/32 only

*Written 2026-11-27 from the filed runs `results/A6_dev_base_s2/` and `results/A6_dev_scale32_s2/` (register lines 20–21, code `f96f84c`, seed 2027), the first-round runs `results/A6_dev_base/` and `results/A6_dev_scale32/` (lines 13 and 16, seed 2026), and M0's fold-0 model in `results/S10_m0_convnext_unet/`.*
- **Rule:** `Analysis/06_Step3_PaperA_Protocol.md` §10, fixed before either run was launched.
- **Numbers:** `stats.md` / `stats.json` in this folder: `python scripts/run_stats.py S10_m0_convnext_unet A6_dev_base A6_dev_scale32 A6_dev_base_s2 A6_dev_scale32_s2 --compare --folds 0 --name C11_paperA_seed2`.
- **Scoring is the same as C06:** fold 0's 208 held-out femurs (139 x-rays), with each run's threshold fitted again on them. The test set is not used.
- **This analysis was written before any result of the five final runs was opened.** The final runs were filed only after the rule below was applied.

## 1. Results on fold 0 (208 femurs)

| Run | Seed | Macro-F1 (95% CI) | Dice, fractured femurs (missed = 0) | 6-class accuracy | Garden I-II recall (11) | Oracle: accuracy, macro-F1 |
|---|---|---|---|---|---|---|
| M0 (S10, fold 0) | 2026 | .822 (.735–.889) | .749 (.710–.783) | 184/208 | .545 | – |
| `A6_dev_base` | 2026 | .776 (.675–.853) | .754 (.718–.785) | 184/208 | .545 | 190, .839 |
| `A6_dev_base_s2` | 2027 | .771 (.668–.847) | .744 (.703–.780) | 180/208 | .545 | 186, .839 |
| `A6_dev_scale32` | 2026 | .827 (.727–.897) | .760 (.729–.786) | 185/208 | .818 | 183, .810 |
| `A6_dev_scale32_s2` | 2027 | .827 (.728–.896) | .770 (.740–.795) | 185/208 | .727 | 189, .859 |

**Mean of the two seeds:**

| Setting | Macro-F1 | Dice, fractured femurs | 6-class accuracy |
|---|---|---|---|
| Base (pooling at 1/16 + 1/32) | .773 | .749 | 182/208 |
| Pooling at 1/32 only | **.827** | **.765** | 185/208 |
| Difference | **+.054** | **+.016** | +3 femurs |

## 2. Decision by the fixed rule (§10)

| Condition for keeping pooling at 1/32 only | Value | Result |
|---|---|---|
| Mean macro-F1 at least .02 above the base | +.054 | passes |
| Mean Dice no more than .01 below the base | +.016 (higher) | passes |
| Mean macro-F1 at least .79 | .827 | passes |

**All three conditions hold: pooling at 1/32 only stays, and the freeze `A-v1-frozen` stands.** The five final runs, trained from that tag, are valid final runs, and their analysis (C07–C09) may start.

## 3. Seed noise (rule point 3)

| Setting | Macro-F1, seed 2027 − seed 2026 | Dice, seed 2027 − seed 2026 | Accuracy |
|---|---|---|---|
| Base | −.004 | −.010 | 180 vs 184 |
| Pooling at 1/32 only | −.001 | +.010 | 185 vs 185 |

- Macro-F1 moved by less than .005 between seeds, for both settings. So the .052 and .056 gaps between the settings, one per seed, are about ten times the seed noise. The margin of .02 in §4 was conservative for macro-F1.
- Dice moved by .010 between seeds, in opposite directions for the two settings. A single-seed Dice difference of .01 on fold 0 is therefore within noise, which is why §4 allowed a Dice drop of up to .01.
- The paired tests within each seed are not significant (base vs 1/32: macro-F1 p = .18 for both seeds). As in C06, fold 0 is too small for a test to decide; the rule decides on the size of the effect, fixed in advance.

## 4. A point from C06 that the second seed resolves
C06 noted that with pooling at 1/32 only, the released mask (oracle test) did not help on fold 0: 183 instead of 185 correct, with no wrong fractured femur fixed. With seed 2027 the oracle mask helps, as it does for the base: 189 instead of 185 correct, and 5 of the 23 wrong fractured femurs (22%) are fixed. So the C06 observation was seed noise, and the 1/32 heads do use the region. The deletion test of the final A6 (C08) measures this on all 1,043 femurs.

## 5. Calibration (reported only)
Temperatures fitted on fold 0: M0 3.22, base 3.14 and 3.30, pooling at 1/32 3.25 and 3.22. All are over-confident by the same factor as M0.
