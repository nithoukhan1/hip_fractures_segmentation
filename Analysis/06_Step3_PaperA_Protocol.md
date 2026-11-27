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
- **Reported:** ECE and Brier, before and after, for every model (A0–A6 and the baselines).
- It is an analysis step and needs no new training. It will be added to `run_stats.py` and `run_breakdown.py` before M2-A.

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
