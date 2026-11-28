# RefFrac development analysis: Paper B's fold-0 development (whole-reference question, the margin-loss question and a second seed) against ConvNeXt U-Net, decided by the fixed rule

*Written 2026-11-27 from the filed runs `results/R4_dev_base/`, `R4_dev_base_s2/`, `R4_dev_global/`, `R4_dev_margin/` (register lines 27–30, code `2619b39`) and ConvNeXt U-Net's fold-0 model in `results/S10_m0_convnext_unet/` (register line 8).*
- **Rule and questions:** `Analysis/08_Step5_PaperB_RefFrac_Protocol.md` section 4, fixed before any R run.
- **Numbers:** `stats.md` / `stats.json` in this folder: `python scripts/run_stats.py S10_m0_convnext_unet R4_dev_base R4_dev_base_s2 R4_dev_global R4_dev_margin --compare --folds 0 --name C12_paperB_dev_fold0`.
  - Only fold 0's 208 held-out femurs (139 x-rays) are used, 136 of them fractured.
  - Each run's fracture threshold is fitted again on those femurs (Youden).
  - The development runs did not predict the test set.
- **The reference tests and the pointing game** come from each run's `metrics.json` (`tests`, `reference`) and its per-case files (`per_case_oof_self.csv`, `per_case_oof_random.csv`, `per_case_oof_references.csv`).

**The four runs** are RefFrac trained on folds 1–4 and scored on fold 0, with ConvNeXt U-Net's recipe:

| Run | References | Attention | Margin loss | Seed | Time (one T4) |
|---|---|---|---|---|---|
| `R4_dev_base` | the 3 nearest healthy femurs | local, ±1/8 of the image height | none | 2026 | 83.2 min |
| `R4_dev_base_s2` | same | same | none | **2027** | 80.8 min |
| `R4_dev_global` (whole-reference question) | same | **whole reference** | none | 2026 | 87.0 min |
| `R4_dev_margin` (margin-loss question) | same | local | **0.5** | 2026 | 82.0 min |

## 1. Results on fold 0 (208 femurs)

| Run | Dice, fractured femurs (missed = 0) (95% CI) | Macro-F1 (95% CI) | 6-class accuracy | Fracture sensitivity / specificity | Garden I-II recall (11) | Threshold |
|---|---|---|---|---|---|---|
| ConvNeXt U-Net (fold 0) | .749 (.710–.783) | .822 (.735–.889) | 184/208 | .985 / .972 | .545 | .0002 |
| `R4_dev_base` | .765 (.735–.792) | .817 (.715–.889) | 183/208 | .978 / 1.000 | .636 | .0001 |
| `R4_dev_base_s2` | .774 (.747–.798) | .819 (.717–.890) | 186/208 | .993 / .986 | .636 | .0007 |
| **Base (mean of the two seeds)** | **.770** | **.818** | 184.5/208 | | .636 | |
| `R4_dev_global` (whole-reference question) | .778 (.753–.800) | .796 (.688–.872) | 181/208 | .985 / .986 | .545 | .0128 |
| `R4_dev_margin` (margin-loss question) | .755 (.721–.786) | .829 (.724–.897) | 185/208 | .956 / 1.000 | .727 | .9745 |

Intervals: 95% cluster bootstrap over x-rays.

**Paired tests** (`stats.md`; Holm within each pair; none is significant):
- Base vs ConvNeXt U-Net: Dice +.017 (p = .73) and +.025 (p = .32) for the two seeds; macro-F1 −.005 (p = .86) and −.003 (p = .92).
- Whole-reference question vs base (seed 2026): Dice +.013 (p = .08), macro-F1 −.021 (p = .35).
- Margin-loss question vs base (seed 2026): Dice −.010 (p = .88), macro-F1 +.013 (p = .73).

## 2. Decisions by the fixed rule (section 4)

The **seed noise** N is the Dice difference between the two seeds of the base: .774 − .765 = **.009**.

| Rule | Value | Decision |
|---|---|---|
| 1. Go / no-go: the base needs Dice ≥ .739 (ConvNeXt U-Net − .01) and macro-F1 ≥ .79 (ConvNeXt U-Net − .03) | Dice .770, macro-F1 .818 | **Passes:** the design goes ahead to the final runs |
| 2. The whole-reference question reference: needs Dice ≥ base + .01 and > base + N, and macro-F1 ≥ base − .02 | Dice +.009; macro-F1 −.022 | **Base stays: the local window** (the Dice gain is below .01 and within the seed noise; macro-F1 also falls by more than .02) |
| 2. The margin-loss question loss: same | Dice −.014; macro-F1 +.012 | **Base stays: no margin loss** |
| 3. Two winners → run their combination | no winner | No combination run is needed |
| 4. Reported, not used | Garden I-II recall .636 (base) vs .545 (ConvNeXt U-Net), 11 femurs | Not used |
| 5. Does the base use its references? `self` must change Dice by ≥ .005 or P(fracture) by ≥ .01 | Dice −.090, P(fracture) −.49 (means of the two seeds) | **Yes, strongly** (section 3) |

**So no setting changes.** The settings of `Analysis/08` sections 1–2 are frozen as written: RefFrac uses the 3 nearest healthy references, local cross-attention within ±1/8 of the image height, difference fusion and no margin loss. The final the eight reference runs use them (section 5).

## 3. Do the models use their references? (rule point 5 and section 2.6)

Each run predicted fold 0 twice more: with the target itself as its reference (`self`, so the difference T − R̃ is zero), and with a random healthy femur instead of the 3 nearest (`random`). Changes are against the run's own predictions, at its own threshold, on the 136 fractured femurs:

| Run | `self`: Dice | `self`: P(fracture) | `self`: correct decisions lost | `random`: Dice | `random`: P(fracture) | Pointing game: hit rate (chance) |
|---|---|---|---|---|---|---|
| `R4_dev_base` | .765 → .701 (−.065) | −.457 | 7.2% | .761 (−.004) | +.004 | 64.0% (8.0%) |
| `R4_dev_base_s2` | .774 → .658 (−.116) | −.531 | 14.8% | .769 (−.005) | −.005 | 91.9% (8.0%) |
| `R4_dev_global` | .778 → .744 (−.035) | −.063 | 3.6% | .772 (−.006) | −.006 | 77.9% (8.0%) |
| `R4_dev_margin` | .755 → .318 (−.438) | −.351 | 55.8% | .745 (−.011) | +.004 | 54.4% (8.0%) |

- **The base relies on the difference from its references.** With its own image as the reference, its fracture probability on fractured femurs falls by about half, and it misses 11 and 17 more fractures (seeds 2026 and 2027). Healthy femurs are unaffected (P(fracture) changes by less than .0001, no new false positive).
- **Which healthy femur is the reference hardly matters at prediction time.** For the base, a random healthy femur instead of the 3 nearest changes Dice by .004–.005 and P(fracture) by under .01. This is a first hint for the reference question (does retrieval help?), but only a hint: the reference question compares models trained with random and with retrieved references (random-reference model vs the nearest-reference model, the reference analysis).
- **The difference map points at the fracture.** For the base, the cell of largest difference ‖T − R̃‖ at 1/16 touches the fracture mask in 64% and 92% of fractured femurs (two seeds), against 8% by chance. The two seeds differ a lot here, so the pointing game must be judged on the final 5-fold runs.
- The margin loss (margin-loss question) makes the model depend on the difference far more (`self` loses 56% of its correct decisions), without better Dice. The whole-reference attention (whole-reference question) depends on it least.

## 4. Seed noise
- Dice: .765 vs .774 (N = .009); macro-F1: .817 vs .819 (.002); accuracy 183 vs 186.
- So on fold 0, a Dice difference below about .01 between two single runs is within the noise of one seed, which is why rule 2 asks for more than both .01 and N.

## 5. What happens next: the freeze (Paper B milestone 2) and the final runs
- **Freeze (`Analysis/08` section 3.3):** the final the eight reference runs are written into `kaggle/runs.json` with the settings of sections 1–2, unchanged. After that pull request is merged, you create the tag `B-v1-frozen` on `main` (GitHub → Releases), as for Paper A. Every final R run is launched with `BRANCH = "B-v1-frozen"`.
- **The training code of the tag is the code that trained these development runs** (`2619b39`); the pull request adds results, analyses, documents and run entries only.
- **Compute, from the measured development runs** (`Analysis/08` section 8):
  - an RefFrac fold takes about 81 min on one T4 (81 s per epoch; ConvNeXt U-Net: 45 s), so RefFrac, the concatenation model, the no-alignment model and the affine-alignment model take about 4.2 h each on T4 x2;
  - AASN-style model, the random-reference model, the nearest-reference model and the own-hip model use one reference and should take about 3 h each (an estimate: no k = 1 run has been timed on a T4 yet);
  - about 29 h of session time in all, a little more than the 26 h estimated in `Analysis/08` section 5.

## 6. Calibration (reported only)
Temperatures fitted on fold 0: ConvNeXt U-Net 3.22, base 3.77 and 3.36, the whole-reference question 3.35, the margin-loss question 2.96. All are over-confident, like ConvNeXt U-Net.
