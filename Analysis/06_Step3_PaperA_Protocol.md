# Report 06: Step 3 protocol, Paper A (SLG-Net): runs A2–A6, fold-0 development and the interpretability tests

**Date:** 2026-11-27 · **Code:** `scripts/train_net.py --net slg:convnext_tiny.fb_in22k_ft_in1k:<A2..A6>` with `src/femurseg/slg.py` · **Runs:** `kaggle/runs.json`
**Why this report:** it fixes, before any SLG-Net run exists, what each run of Paper A adds, how the fold-0 development is run and decided, what is then frozen, and how the final runs and the interpretability tests are scored. Paper A's methods section is written from it. The design comes from Report 03 §3 and §7, and the run plan from Report 04 §4.

**Starting point (analysis C05, 2026-11-27):**
- M0 (S10 = A1) beats the plain ConvNeXt-T classifier (S06 = A0) on the 1,043 out-of-fold femurs: macro-F1 .784 → .825 (Holm p = .013).
- Most of M0's remaining errors are grade errors at the right location (65 of 121 OOF errors).
- Garden I-II is the weakest class (OOF recall .544), and M0 is over-confident (ECE .108).

SLG-Net targets exactly these three problems.

---

## 1. The runs of Paper A

| Run | Network | What it adds | Question (Report 03 §7.2) |
|---|---|---|---|
| A0 | = S06 (filed, register line 7) | ConvNeXt-T image classifier | Starting point |
| A1 | = M0 = S10 (filed, register line 8) | + U-Net decoder and mask loss; flat head on the global context | Does segmentation as an extra task help? **C05: yes** |
| A2 | `slg:…:A2` | + region pooling (C2) | Does looking inside the predicted region help? |
| A3 | `slg:…:A3` | + region geometry (C3) | Does the region's shape and position help? |
| A4 | `slg:…:A4` | + location → grade hierarchy (C4), one grade trunk per location | Does deciding location first, then grade, help? |
| A5 | `slg:…:A5` | + shared severity trunk (C5) | Does sharing one grade trunk help Garden I-II? |
| A6 | `slg:…:A6` | + logit adjustment (C6) = **SLG-Net** | Does it help the rare classes? |

**One recipe.** Every run uses the Step 2 recipe unchanged (Report 05 §2):
- input 768 × 512, the same augmentations;
- AdamW (learning rate 2 × 10⁻⁴, weight decay 0.05), 2 warm-up epochs then cosine decay, **60 epochs, final weights**;
- batch 16 (micro-batch set by the pre-flight), fp16, seed 2026 + fold.

Only the heads, their losses and teacher forcing differ from M0. Each variant adds exactly one component to the one before it, so each step of the chain answers one question.

## 2. The components, exactly

### 2.1 Shared by A2–A6
1. **Encoder, decoder and mask:** M0's (`segmentation_models_pytorch` U-Net with the `tu-convnext_tiny` encoder). The mask loss is BCE + Dice with weight 1 on every femur; healthy femurs teach an empty mask. The predicted region is fracture probability ≥ 0.5.
2. **Global context c:** exactly M0's. It is the average of the 1/32 encoder features, normalised by the classifier's pretrained LayerNorm (768 numbers).
3. **Mask given to the heads, m:** the predicted fracture-probability map at the input size. During training it is sometimes replaced by the released mask (teacher forcing, point 5).
4. **Region features r (C2):** taken at 1/16 (384 channels) and 1/32 (768 channels).
   - The encoder features are averaged inside m, after m is averaged onto each feature grid. Each average is LayerNorm-ed.
   - The result is multiplied by the **mask confidence q**, the mean of the 500 highest pixel probabilities of m.
   - A femur without a predicted region therefore has q ≈ 0 and r ≈ 0: the heads then know there is no region to look at.
5. **Teacher forcing (C7):** at every training step, each femur's heads get the released mask with probability p_tf, and the femur's own predicted mask otherwise.
   - p_tf falls linearly from 1 to 0 over the first **30 epochs** (half of training), then stays 0.
   - Predictions always use the predicted mask, except in the oracle test (§5).
6. **End-to-end:** the decision losses also reach the mask, through r, q and the geometry. Development question D2 (§4) tests the alternative, a detached mask.

### 2.2 A2 and A3: a flat 6-class head
- **A2:** one linear 6-class layer on dropout([c, r]) (1,920 numbers). Loss: cross-entropy + the mask loss, as in M0.
- **A3:** adds a geometry embedding e_g (64 numbers) to the head's input: [c, r, e_g]. Here e_g = GELU(linear(g)), where g holds 8 numbers computed from m (all differentiable):
  - q;
  - the mean probability inside the region;
  - log₁₀ of the area fraction;
  - the centroid (y and x);
  - height and width (4 standard deviations, the full axes of the equivalent ellipse);
  - the y–x correlation (orientation).

  These numbers come from our EDA: mask height separates Garden grades (AUC .81 train / .84 test), area separates Evans grades (AUC .80 / .75), and centroid height gives the location in 80% of test femurs (Report 03 §2).

### 2.3 A4–A6: hierarchical heads and the chain rule (C4–C6)

**Three heads:**
- **Fracture:** P(fracture) = σ(linear([c, e_g])).
- **Location:** P(location | fracture) = softmax(linear([c, r, e_g])) over neck, pertrochanteric and subtrochanteric.
- **Grade:** P(high grade | location), for the neck (Garden III-IV vs I-II) and the pertrochanteric region (Evans III-V vs I-II). The release does not grade subtrochanteric fractures.
  - **A4:** one trunk per location: linear (1,216 → 256), GELU, linear → 1.
  - **A5, A6:** one shared trunk of the same size, which also receives a learned 16-number embedding of the location. Low-displacement neck fractures can then borrow strength from the stable pertrochanteric ones.

**6-class probabilities** come from the chain rule of Report 03 §3.3. They are always valid and consistent. The decision rule of every Step 2 model applies unchanged:
- score = P(fracture);
- class = the most probable fracture class;
- the threshold is fitted on the OOF femurs (Report 05 §3).

**Loss** (weights fixed now, Report 03 §3.4):
- fracture BCE, on all femurs;
- location cross-entropy, on fractured femurs;
- grade BCE, on neck and pertrochanteric fractures, at their own location;
- 0.5 × the negative log-likelihood of the chain-rule probability, on all femurs;
- the mask loss.

**A6, logit adjustment (Menon et al. 2021):** applied during training only.
- τ = 1 times the training fold's log-priors is added to the location logits (log P(location | fracture)) and to the grade logits (log-odds of a high grade at each location).
- Predictions use the plain logits. These correspond to class-balanced posteriors at those nodes, so Garden I-II, Evans I-II and subtrochanteric fractures gain weight.
- The fracture node is not adjusted, because its threshold is fitted on the OOF femurs anyway.
- For the fold-0 model, the offsets are −0.78, −0.72 and −2.93 (neck, pertrochanteric, subtrochanteric), and +1.48 (neck) and +0.40 (pertrochanteric) for a high grade. They come from 249 neck, 266 pertrochanteric and 29 subtrochanteric training fractures, of which 46 are Garden I-II and 107 Evans I-II.

**Size:** M0 has 31.93 M parameters, A2 31.94 M, A3 31.95 M, A4 32.56 M, and A5 and A6 32.26 M. The heads cost almost nothing to run, so a fold takes about as long as M0's: 44.5 min on one T4 for fold 0 (S10).

## 3. Rules that keep Paper A's numbers honest (Report 04 §3.4)
1. **Development uses fold 0 only.** The development runs predict fold 0's 208 held-out femurs and never the test set (`--no-test`).
2. **Fixed now, before any SLG-Net result:**
   - the variants;
   - the recipe and the loss weights;
   - τ = 1;
   - the teacher-forcing schedule;
   - the region-pooling scales.

   The only exceptions are the three development questions in §4.
3. **Freeze (milestone M2-A):** the settings chosen in §4 are written into `kaggle/runs.json` and tagged `A-v1-frozen`. Every final run (A2–A6, 5 folds, test set) starts from that tag.
4. **Thresholds and the calibration temperature (§6)** come from the pooled OOF predictions only.
5. **Check on folds 1–4:** final results are also reported on folds 1–4 only. If those agree with all 5 folds, the fold-0 development did not inflate anything.

## 4. Fold-0 development: three questions, two Kaggle sessions

| Question | Base setting (A6_dev_base) | Alternative run | Changed argument |
|---|---|---|---|
| D1 Is teacher forcing needed? | Teacher forcing over 30 epochs | `A6_dev_no_tf` | `--tf-epochs 0` |
| D2 Should the decision losses reach the mask? | Yes (end-to-end) | `A6_dev_detach` | `--slg-detach` |
| D3 Region pooling at 1/16 + 1/32, or at 1/32 only? | 1/16 + 1/32 | `A6_dev_scale32` | `--slg-scales 32` |

**How the runs are launched:** each development run is A6 trained on fold 0 only, about 50 min on one T4, with the oracle test. Kaggle gives two T4s per session, so two runs share one session (a *parallel group* of `kaggle/run.py`):

| RUN to launch | GPU 0 | GPU 1 | Session time |
|---|---|---|---|
| `A_dev_round1` | `A6_dev_base` | `A6_dev_no_tf` | ≈ 55 min |
| `A_dev_round2` | `A6_dev_detach` | `A6_dev_scale32` | ≈ 55 min |

**Reference:** M0's own fold-0 model, from the filed run S10 (same fold, recipe and seed), so no new run is needed. With its threshold fitted on fold 0 alone, it scores:
- macro-F1 .822 (95% interval .735–.889);
- accuracy 88.5% (184 of 208);
- Dice on fractured femurs .749;
- Garden I-II recall .545, from only 11 femurs.

**Decision rule, fixed now.** Analysis **C06** compares the four runs with M0 on fold 0's 208 femurs (`scripts/run_stats.py --folds 0`, each threshold fitted again on those femurs).
1. **Go / no-go:** if `A6_dev_base` has a fold-0 macro-F1 below .79 (M0 − .03), the heads are hurting. The design is then reviewed before any 5-fold run.
2. **Each question:** the base setting stays unless the alternative improves fold-0 macro-F1 by at least .02, *and* lowers Dice on fractured femurs by no more than .01.
3. **Two winners:** if two alternatives pass, their combination is run on fold 0 before freezing.
4. **Garden I-II recall is reported but not used**, because fold 0 holds only 11 of these femurs.

**Why so conservative:** fold 0 is small. On fold 0 alone, even the M0 vs S06 gain (+.026 macro-F1, significant on all 1,043 femurs) has p = .51. So the rule changes a default only for a clear effect, and it is not a search.

## 5. Interpretability tests (Report 04 §2), written by the training runs themselves

`--extra-tests oracle,deletion` predicts every held-out and test femur again in these modes. Each mode writes its own per-case files (`per_case_oof_<mode>.csv`, …) and its metrics under `tests` in `metrics.json`. All modes use the run's own OOF threshold.

1. **Oracle (test 1):** the heads get the released mask instead of the predicted one. The network's own mask and its Dice do not change. If grading improves, the decision really depends on the region, and the gain measures how much better masks would help.
2. **Region deletion (test 2):**
   - **Deletion:** the predicted region (probability ≥ 0.5) is dilated by 8 px and replaced by the image blurred with a Gaussian of σ = 16 px (both at the 768 × 512 input, so 16 and 32 px at full size). This removes the fracture line but keeps the local brightness, so no artificial edge appears.
   - **Control:** the same region shape is blurred at another place: a random position, seeded by the file name, without overlap with the region and on tissue at least 0.8 × as bright (on bone rather than background). If no such place is found in 200 tries, the brightest place without overlap is used. A per-femur record (`folds/fold<k>/oof_deletion_info.csv`) says which happened, and `metrics.json` counts them (`control_placement_oof`).
   - **Reported**, on fractured femurs: the change in P(fracture) and in P(true class), and the share of correct decisions lost. Deletion and control are compared by a Wilcoxon test on the paired per-femur changes.
   - If the model decides from the region, deletion must hurt far more than the control.
3. **Geometry consistency (test 3) and risk–coverage (test 5):** computed at the analysis stage (C08) from the per-case files and the saved probability maps, with no extra run.

The development runs record the oracle test only. The final A2–A5 runs record the oracle test, and the final A6 run records both tests.

## 6. Calibration (before freezing, Week 5 of the plan)
- **Temperature scaling:** one temperature for the 6-class probabilities, fitted on the pooled OOF predictions by negative log-likelihood, then applied unchanged to the test set.
- **Reported:** ECE, Brier score and NLL, before and after, for every model (A0–A6 and the baselines).
- It is an analysis step and needs no new training. **Implemented 2026-11-27 in `scripts/run_stats.py`** (section "Calibration" of `stats.md`, key `calibration` of `stats.json`):
  - one temperature per run, fitted on its OOF femurs (with `--folds 0`, on fold 0's femurs only), then applied unchanged to the test ensemble and to each fold model on the test set;
  - ECE uses 15 equal-width bins of the top-class probability;
  - the temperature never changes a femur's most probable class, and no threshold or decision uses it.
- **Not applicable to the YOLO baselines (S01–S03):** YOLO gives probability 0 to every class without a detected instance, and 73–99 of the 1,043 OOF femurs get probability 0 for their true class. The NLL that fits the temperature is then undefined, so their ECE and Brier are reported before calibration only.
- **First values** (filed runs, no new training): S06 temperature 3.43, OOF ECE .127 → .046, test-ensemble ECE .102 → .066; S10 (M0) temperature 3.14, OOF ECE .108 → .040, test-ensemble ECE .103 → .064. Both ConvNeXt models are over-confident by about the same factor.

## 7. Final runs, analyses and compute

**Final runs (after M2-A, from the tag):**
- A2, A3, A4 and A5: 5 folds and the test set, with the oracle test.
- A6: 5 folds and the test set, with the oracle and deletion tests.
- Each run takes about 2.4 h on T4 x2 (S10 took 2.3 h), so about 12 h in all.

**Analyses:**

| Analysis | Runs | Question |
|---|---|---|
| C06 | `A6_dev_*` vs M0 (fold 0) | Development decisions D1–D3 (§4) |
| C07 | A0 … A6 | Ablation chain: each step vs the one before, and A6 vs A1. OOF first, test as confirmation. Holm within the chain. Primary endpoints: macro-F1 and Dice on fractured femurs |
| C08 | A6 (and A2–A5 oracle) | Interpretability: oracle, deletion vs control, geometry consistency, risk–coverage |
| C09 | A6 vs S01, S04–S10 | Paper A's main table: SLG-Net against every baseline |

**Compute:**
- development: about 2 h of session time;
- final runs: about 12 h;
- the sensitivity runs of Report 04 §4 (EfficientNetV2-S; input 1152 × 768): about 8 h, later.

This is well inside the 35–38 GPU-hours planned for Track A (Report 04 §6).

## 8. Known limitations, stated now
- **Small development set:** fold 0 has 208 femurs, so small differences are noise. Hence the default-favouring rule of §4.
- **Interactions not tested:** D1–D3 each test one change against the base, not every combination.
- **Logit adjustment trades accuracy for rare-class recall:** macro-F1 (primary) and accuracy are both reported.
- **The deletion control** can land on soft tissue in crops that show little bone. The brightness condition reduces this, and the report states how often it happened.
- **The train/test annotation shift (C01)** affects every model's test Dice in the same way. As before, Dice is reported per location, out of fold and on the test set side by side.

---

## 9. Addendum (2026-11-27): result of the fold-0 development (C06) and the settings proposed for the freeze

The two development rounds ran on Kaggle (commit `f342b8a`, about 50 min each on one T4; register lines 13–16). Analysis C06 (`results/analysis/C06_paperA_dev_fold0/report.md`) applied the rule of §4 on fold 0's 208 femurs:

| Rule | Result | Decision |
|---|---|---|
| Go / no-go (base ≥ .79) | `A6_dev_base` macro-F1 .776 (M0 .822) | **Fails: design review** |
| D1 teacher forcing | `A6_dev_no_tf` +.008 | Keep teacher forcing over 30 epochs |
| D2 end-to-end mask | `A6_dev_detach` +.019 (short of .02 by .001) | Keep the end-to-end mask |
| D3 region pooling at 1/32 only | `A6_dev_scale32` +.052, Dice +.006 | **Pool at 1/32 only** |
| Two winners | one | No combination run |

**Design review (rule 1).**
- Accuracy is the same for the base and M0 (184 of 208).
- The base's shortfall comes from the two smallest classes: two of the 7 subtrochanteric femurs and a lower Garden I-II precision. It is not significant (p = .25).
- The same heads with pooling at 1/32 only reach .827, M0's level. So the 1/16 region features, not the heads, caused the shortfall.
- The oracle test shows that, with teacher forcing, the base heads use the region: the released mask fixes 6 of their 23 wrong fractured femurs. The 1/32-only heads did not gain from the released mask on fold 0. The deletion test of the final A6 (C08) will show how much they depend on the region.

**Settings proposed for the freeze (M2-A):** teacher forcing over 30 epochs, an end-to-end mask, and region pooling at 1/32 only (`--slg-scales 32`). Everything else is as fixed in §1–§3.
- The five final runs are in `kaggle/runs.json`, about 2.6–2.8 h each on T4 x2, about 13 h in all:
  - `A2_region`, `A3_geometry`, `A4_hierarchy` and `A5_shared_grade`, with the oracle test;
  - `A6_slgnet`, with the oracle and deletion tests.
- They start from the tag `A-v1-frozen` once the person approves the freeze.
- The alternative offered to the person is to repeat `A6_dev_base` and `A6_dev_scale32` with a second seed first (one session, about 55 min) and apply the same rule to the mean of the two seeds.

**Change to the analysis code (no effect on any decision).** `scripts/run_stats.py` now fits the calibration temperature for network runs whose per-case files write a true-class probability as 0.
- The 0 is below their 8 decimals, and it is read as 10⁻¹² like every other written 0.
- One femur of `A6_dev_detach` and one of S05 had such a 0.
- YOLO runs stay "not applicable", because there a 0 means that no instance of the class was detected.

## 10. Addendum (2026-11-27): freeze approved, and a second seed for the base and 1/32 pooling

Written on 2026-11-27, before either second-seed run was launched. §1–§9 are unchanged.

**The person's decision (2026-11-27).** The freeze of §9 is approved, and the five final runs start. The second-seed check offered as the alternative in §9 runs as well, at the same time.

**Tag `A-v1-frozen`.**
- The tag goes on `main` after the pull request of Step 16 is merged. The person creates it on GitHub (Releases), because this project's Claude sessions cannot push tags.
- The training code at that commit is identical to commit `f342b8a`, which trained the development runs: `scripts/train_net.py` and `src/femurseg/slg.py`, `nets.py`, `torchdata.py`, `runner.py`, `data.py`.
- Since `f342b8a` only these changed: the evaluation (a refactor with identical numbers, plus calibration functions used by `scripts/run_stats.py`), clearer error messages, and the dataset check of `kaggle/run.py`. The rest is Paper B code. No setting of §1–§9 changed.
- The final runs are launched with `BRANCH = "A-v1-frozen"` (`kaggle/README.md`).

**Second-seed runs** (`kaggle/runs.json`, parallel group `A_dev_seed2`, one session on T4 x2, about 55 min, launched from `main`):

| Run | Same as | Only difference |
|---|---|---|
| `A6_dev_base_s2` | `A6_dev_base` (register line 13) | `--seed 2027` instead of the default 2026 |
| `A6_dev_scale32_s2` | `A6_dev_scale32` (register line 16) | `--seed 2027` |

- The seed sets the random start of the new layers, the order of the training images and the augmentation (`torch`, `numpy` and the data loader, each seeded with seed + fold).
- As in round 1 and round 2: fold 0 only, 60 epochs, no test-set predictions, with the oracle test.

**Rule, fixed now** (analysis C11, `results/analysis/C11_paperA_seed2/`). It uses the same scoring as C06: fold 0's 208 femurs, with the threshold refitted on fold 0. Each setting's result is the mean of its two seeds.
1. **Pooling at 1/32 only stays in the freeze if all three conditions hold:**
   - its mean macro-F1 is at least .02 above the base's (the margin of §4);
   - its mean Dice on fractured femurs (missed = 0) is no more than .01 below the base's;
   - its mean macro-F1 is at least .79 (the go/no-go of §4).
2. **If any condition fails, the freeze goes back to the person before any final-run result is analysed.**
   - Final runs that are already running finish, but their results are not opened until the person decides.
   - If the person then changes the frozen settings, the five final runs are repeated from a new tag (about 13 h).
3. **Seed noise is reported:** the difference between the two seeds of each setting, for macro-F1 and Dice. It shows how large a single-seed difference on fold 0 can be by chance, the yardstick behind the .02 margin of §4.
4. **No other decision of C06 is reopened.** Teacher forcing and the end-to-end mask stay as decided in §9.

## 11. Addendum (2026-11-27): result of the second-seed check (C11): the freeze stands

Written on 2026-11-27, after the second-seed runs and before any result of the five final runs was opened. §1–§10 are unchanged.

- **Runs:** `A6_dev_base_s2` and `A6_dev_scale32_s2` (register lines 20–21), launched from `main` = `f96f84c`, 49.4 min on T4 x2. Both trained fold 0 completely.
- **Result** (`results/analysis/C11_paperA_seed2/report.md`), means of the two seeds on fold 0:
  - macro-F1: base .773, pooling at 1/32 only **.827** (+.054);
  - Dice on fractured femurs: base .749, 1/32 **.765** (+.016).
- **Rule of §10:** all three conditions hold (+.054 ≥ .02; Dice not lower; .827 ≥ .79). **Pooling at 1/32 only stays, and the freeze `A-v1-frozen` stands.**
- **Seed noise:** macro-F1 moved by .004 (base) and .001 (1/32) between seeds; Dice by .010 for both.
- **The five final runs** (`A2_region` … `A6_slgnet`, register lines 22–26) were launched from the tag `A-v1-frozen` = `f96f84c`; each log shows that commit, and each trained all 5 folds (2.4–2.9 h). They were filed after this result, and their analysis (C07–C09) follows §7.

## 12. Addendum (2026-11-27): results of the final runs (C07–C09)

Written on 2026-11-27, after the five final runs were filed and C11 had confirmed the freeze (§11). §1–§11 are unchanged.

**How the multiple comparisons were adjusted.**
- §7 says "Holm within the chain" without naming the set. C07 adjusts each endpoint and each scope (OOF, test) separately, across the 7 comparisons of the chain: 6 for Dice, because A0 has no mask. This is `scripts/run_stats.py --pairs … --family-holm`, column "p (Holm, family)".
- §7 fixed no adjustment for C09. It uses the same Holm across its 8 comparisons, the cautious choice.

**C07, the ablation chain** (`results/analysis/C07_paperA_ablation_chain/report.md`; out of fold, 1,043 femurs):
- **Only the mask task helps significantly:** A1 vs A0, macro-F1 +.042 (Holm within the chain p = .022).
- **From A2 to A6, no step changes macro-F1 significantly.**
  - A3 (geometry) has the best out-of-fold macro-F1 (.838) and Garden I-II recall (39 of 57). It was not the pre-specified final model, so its numbers are exploratory only.
  - Region pooling (A2) lowers Dice by .004 (Holm p = .004). No later step changes Dice significantly.
- **SLG-Net (A6) vs M0 (A1):**
  - out of fold: macro-F1 .806 vs .825 (−.020, 95% CI −.047 to +.008, p = .17), Dice .774 vs .777 (p = .08);
  - on the test set: A6 is ahead (macro-F1 +.041, p = .21; 186 vs 182 femurs correct), not significantly.
  - **The pre-registered answer: SLG-Net is not more accurate than M0.** No equivalence margin was fixed, so equivalence is not claimed either.
- **Folds 1–4 only (§3, rule 5):** every value is within .009 of its 5-fold value, so the fold-0 development did not inflate the results.
- **Calibration (§6):** temperatures 2.6–3.4; out-of-fold ECE falls from .097–.127 to .029–.046.

**C08, interpretability** (`results/analysis/C08_paperA_interpretability/report.md`):
- **Oracle (test 1):** the released mask fixes 8 of A6's 130 wrong fractured femurs and breaks 3 of its 550 correct ones (out of fold 907 → 912 correct, p = .23). A2–A5 gain 5–10. Better masks would help little.
- **Deletion (test 2, A6):**
  - deleting the predicted region loses 356 of the 550 correct fractured decisions (64.7%); the control loses 7 (1.3%), McNemar p < .001;
  - on the test set: 80 of 122 against 0;
  - P(fracture) hardly moves (mean −.032). So the location and grade decisions depend on the region, and the fracture decision does not.
- **Risk–coverage (test 5):** for every model, the 10% least confident femurs hold 32–40% of the errors (A6 32.4%, M0 35.5%). SLG-Net's confidence is not better at flagging errors.
- **Geometry (test 3):**
  - done for M0: the location rule agrees with M0's decided location for 83.8% of femurs; region height vs the decided Garden grade AUC .92; region area vs the decided Evans grade AUC .85;
  - **A6 waits for its probability maps**, which are only in its Kaggle output (`A6_slgnet.zip`). Then `scripts/map_geometry.py` makes its geometry file, and C08 is re-run with it.
- **Limit:** M0 was not run with the deletion test. So these results show that SLG-Net's decisions use the region, not that they use it more than M0's do.

**C09, the main table** (`results/analysis/C09_paperA_main_table/report.md`; Holm across 8 comparisons):
- **Out of fold, A6 beats:**
  - the base article's model S01: macro-F1 +.061 (Holm p = .006), Dice +.034, accuracy +3.7 points;
  - EfficientNet-B4: macro-F1 +.050 (p = .020);
  - U-Net, DeepLabV3+ and SegFormer on Dice.
- **Against DenseNet-121, ConvNeXt-T, U-Net, DeepLabV3+, SegFormer and M0,** the macro-F1 differences lie between −.020 and +.022, none significant.
- **Test set:** no macro-F1 difference survives Holm. A6 has the most correct femurs (186 of 210); S01 has the highest test macro-F1 (.845).
- **Against the article's final model (test set):** accuracy 88.6% vs 86.2%, macro-F1 .820 vs .814, Garden I-II recall .312 vs .500.

**What follows.**
- How Paper A presents SLG-Net is the person's decision, raised with these results (`PROJECT_TRACKER.md`, open questions).
- The sensitivity runs of §7 (about 8 h) wait for that decision.
- A6's geometry test is completed when its maps arrive.
