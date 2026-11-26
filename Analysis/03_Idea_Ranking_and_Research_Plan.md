# Report 03: Ranking the directions (D1–D9) and the research plan

**Date:** 2026-11-26
**Builds on:** `Analysis/01_Base_Article_and_EDA_Report.md` (article + EDA), `Analysis/02_How_to_Use_the_Dataset.md` (D1–D9), `research/01_prior_use_and_sota_review.md` (who has used the data).
**Status:** options analysis.

> **Decision, 26 Nov 2026:** you chose to run **Option A and Option B in parallel, as two papers**.
> - The adopted plan is **`Analysis/04_Execution_Plan_A_and_B.md`**. It replaces §8 (compute) and §9 (timeline) below.
> - It adds the no-clinician constraint, the full Paper B plan and the validation rules.
> - Run names changed there: baselines B1–B9 below are **S1–S9** (shared), and Paper B runs are **R1–R7**.

---

## 0. Summary in five lines

1. I scored all nine directions on novelty, evidence in our data, chance of a clear result, Kaggle feasibility and reviewer value (Section 2).
2. The strongest core is **D3 (location → grade reasoning) built on D2 (a real segmentation decoder)**. Support pieces D1 (rare classes), D7 (uncertainty) and D9 (benchmark) are cheap and make the paper complete.
3. **Option A, "SLG-Net" (Segment, Locate, Grade), is my recommendation.** The network first outlines the fracture region, then uses that region to decide the location, then the grade. It attacks the paper's main weaknesses directly: 15 of its 29 test errors are wrong grades at the right location, and its Dice falls at high resolution.
4. **Option B, "RefFrac" (compare with a healthy reference femur, D4),** is the most novel but riskiest: a healthy contralateral femur exists for only 44% of test fractures, and the data has a built-in shortcut. It fits better as a second paper on the same code.
5. Core compute for Option A is about **70–80 Kaggle GPU-hours** (about 3 weeks of free quota), and the plan reaches **submission in about 12 weeks** (late February 2027) with a 2-week buffer.

---

## 1. Step 1: How the directions were scored

A Q2/Q3 reviewer asks five questions, so each direction gets a 1–5 score on each.

| Criterion | Weight | What a 5 means |
|---|---|---|
| **N** Novelty | 0.25 | Not done on this data **and** not standard in hip-fracture AI |
| **E** Evidence in our data | 0.15 | Our EDA or the base paper shows the problem clearly |
| **G** Chance of a clear result | 0.20 | Very likely to give a measurable, reportable gain or finding |
| **F** Feasibility | 0.15 | Fits Kaggle's free GPUs and a ~12-week plan |
| **R** Reviewer and clinical value | 0.25 | Matches clinical reasoning and makes a convincing story |

Novelty and reviewer value carry the most weight, because the citing literature is full of "YOLO + one attention block" papers that reviewers now see as weak (literature review, Section 4).

## 2. Step 2: Scores for D1–D9, with the evidence behind each

| Rank | Dir. | Direction | N | E | G | F | R | **Score** | Role in the paper |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **D3** | Location → grade (hierarchical) grading guided by the mask | 4 | 5 | 3.5 | 4 | 5 | **4.30** | **Core** |
| 2 | D7 | Uncertainty and calibration | 2.5 | 4 | 5 | 5 | 4 | 3.98 | Support (cheap, always gives results) |
| 3 | **D2** | Dedicated segmentation decoder + per-class Dice/HD95/NSD | 3 | 5 | 4 | 4 | 3.5 | **3.78** | **Core (backbone of the method)** |
| 4 | D4 | Compare with the healthy contralateral femur | 4.5 | 4 | 2.5 | 2.5 | 4 | 3.60 | Option B / second paper |
| 5 | D9 | Independent benchmark of modern models | 2.5 | 4 | 5 | 4 | 3 | 3.58 | Support (needed anyway) |
| 6 | D1 | Rare-class (long-tail) learning | 2 | 5 | 3 | 5 | 3 | 3.35 | Support, built into the hierarchy |
| 7 | D8 | External test on FracAtlas hips | 3 | 4 | 2.5 | 2.5 | 4.5 | 3.35 | Optional, if time allows |
| 8 | D6 | SAM / MedSAM with adapters | 3 | 2 | 2.5 | 3 | 3.5 | 2.88 | One extra baseline (optional) |
| 9 | D5 | Label-noise / annotation-shift-aware training | 2.5 | 3 | 2.5 | 3.5 | 3 | 2.85 | Analysis only (report, don't optimise) |

Score = 0.25·N + 0.15·E + 0.20·G + 0.15·F + 0.25·R. The scores are a structured judgement, not a measurement. D7 ranks high because it is low-risk, not because it is novel, so it supports the core rather than being one.

### Why each direction got its score

**D3, location → grade (rank 1).**
- *Evidence in the base paper:* of its 29 test errors, 8 are fracture/no-fracture errors (derived from Se 95.17% and Sp 98.46%), **15 are the wrong grade at the right location, and 6 are the wrong location**. Grade errors are the biggest error type.
- *Evidence in the labels themselves:* `data.yml` codes are `2AB, 2CD, 3AB, 3CDE, 4`. The digit is the location (2 neck, 3 pertrochanteric, 4 subtrochanteric) and the letters are the grade. The labels are hierarchical, but the base model learns them as 6 flat classes.
- *New evidence from our data (computed today from the ground-truth masks):*
  - Mask **height alone** separates Garden I–II from III–IV with AUC **0.81** (train, n=311) and **0.84** (test, n=57).
  - Mask **area alone** separates Evans I–II from III–V with AUC **0.80** (train, n=333) and **0.75** (test, n=76).
  - A small decision tree (depth 3) on the mask **centroid height alone**, fitted on the training masks, predicts the location of test fractures with **80%** accuracy (neck 82%, pertrochanteric 78%, subtrochanteric 83% recall).
  - So the fracture region's geometry carries real grade and location information, which a flat classifier never uses explicitly.
- *Literature:* hierarchical classification of proximal femur fractures exists, but only as cascades of image classifiers without fracture masks: Tanzi et al. 2020 (AO/OTA cascade, 3-class accuracy 0.86, 5-class 0.81) and Jiménez-Sánchez et al. 2019 (box localisation, then A vs B). Garden-only grading reaches 90.6% on private data with CT reference (Yang et al. 2025). **None of the papers we could read combines a fracture-region mask with a location → grade decision in one network** (checked: base paper and its 25 citers, Tanzi 2020, Jiménez-Sánchez 2019, Yilmaz 2025, Zheng 2025, Wang 2025, FMC-Net 2026).
- *Risk:* the released grades are already grouped into two levels per location (I–II vs III–IV; I–II vs III–V), so "ordinal" modelling reduces to a binary decision inside each location. We therefore frame it as a **hierarchical, conditional** model, not as an ordinal-regression paper.

**D2, segmentation decoder (rank 3, but the backbone of the method).**
- *Evidence:* the paper's Dice is 0.77 with a very wide 95% CI (0.56–0.98). Dice gets **worse** at higher resolution (0.780 at 640 px vs 0.755 at 1536 px for YOLOv8-small), because YOLO's mask head builds masks from low-resolution prototypes. No per-class Dice or boundary metric is reported.
- *Ceiling:* the one pixel-identical image pair in train has two masks with IoU 0.70, which is **Dice ≈ 0.82**. That is a rough inter-annotator ceiling from a single pair, so realistic Dice gains are a few points, not 0.10. We must say this honestly; it also protects us from reviewers expecting Dice 0.90.
- *Literature:* no dedicated segmentation model has been reported on hip fracture-region masks (review §5b). Auxiliary classification improved fracture segmentation in rib CT (Dice 62.2 → 64.6; Harini et al. 2024), which supports the multi-task design.

**D4, contralateral femur (rank 4).**
- *Evidence:* 398 x-rays hold one fractured and one healthy femur; no x-ray has two fractures.
- *Literature:* left-right comparison works for **pelvic** fracture detection on full pelvic x-rays (AASN, a Siamese network with a spatial transformer, AUC 0.977; Chen et al., ECCV 2020). None of the hip papers we could read uses the contralateral femur for segmentation or grading.
- *Why it is not the core:*
  1. **Coverage:** only 64 of 145 test fractures (44%) have a healthy partner. The method needs a fallback for the other 56%.
  2. **Shortcut:** in a pair, exactly one femur is fractured, so a model can learn "if the other side looks broken, this side is healthy". That inflates results and reviewers will catch it.
  3. **Alignment:** the two crops were cut and resized independently, so they are not registered.
- High novelty, medium-low chance of a clean result. It is Option B below.

**D9, benchmark (rank 5).** Nobody has trained anything on this dataset except the original authors. A careful benchmark under one protocol is a contribution, but alone it is Q3/Q4. We need the baselines anyway, so it comes for free.

**D1, rare classes (rank 6).** Garden I–II has 57 training femurs and subtrochanteric 36, and both are the base model's worst classes. Standard long-tail fixes (logit adjustment, Menon et al. 2021) are not new on their own, so D1 is built into the hierarchy instead (Section 3.4).

**D7, uncertainty and calibration (rank 2 by score, support role).** The base paper's title promises interpretability but measures no calibration or uncertainty. Training 5 fold-models gives a deep ensemble for free (Lakshminarayanan et al. 2017), and temperature scaling (Guo et al. 2017) is cheap.

**D8, FracAtlas external test (rank 7).** Reviewers like external validation, but FracAtlas has only 63 fractured hip studies, different labels (no Garden/Evans) and full images that we would have to crop ourselves. Kept as an optional extra.

**D6, SAM/MedSAM (rank 8).** "SAM + adapter on dataset X" papers are common, and the masks here are loose regions rather than crisp objects, which plays to SAM's weak side. Worth one baseline row if compute allows.

**D5, label noise (rank 9).** The train/test annotation shift (test neck masks 35–40% smaller) is real, but designing the method around it would use knowledge taken from the test set. We **report and analyse** it (predicted vs reference area per split) instead of optimising for it.

---

## 3. Step 3: Option A (recommended): SLG-Net, "Segment, Locate, Grade"

### 3.1 The idea in plain words
A radiologist reads a hip fracture in three steps: *where is the break* (outline it), *which part of the femur is it in* (location), and *how bad is it* (grade: displaced or not, stable or not). SLG-Net copies that order. It first segments the fracture region. It then uses **what is inside the region** (its image features) and **the region's shape and position** (its geometry) to decide the location and, conditioned on the location, the grade. The mask is no longer a by-product: it is the evidence the decision is built on, which makes the model interpretable by construction.

### 3.2 Research questions (each becomes a results subsection)
- **RQ1:** Does a dedicated high-resolution segmentation decoder outline fracture regions better than YOLOv8-seg (per-class Dice, HD95, NSD)?
- **RQ2:** Does using the segmented region to make the decision improve fine-grained grading over a flat classifier with the same backbone? *(This is the question the base paper raised but never tested.)*
- **RQ3:** Does the location → grade hierarchy reduce grade errors and help the rare classes (Garden I–II, subtrochanteric)?
- **RQ4:** Are the predictions calibrated, and can the model's uncertainty flag the cases it gets wrong?

### 3.3 Architecture sketch

```
 femur crop, resized to 768 x 512 (keeps the 3:2 portrait shape)
            |
   +--------v---------+
   |  Encoder         |  ConvNeXt-Tiny, ImageNet weights (EfficientNetV2-S as a check)
   |  1/4 1/8 1/16 1/32 features
   +--------+---------+
            |------------------------> Segmentation decoder (U-Net style, full resolution)
            |                                   |
            |                                   v
            |                     fracture-region probability map  M
            |                                   |
            |      +----------------------------+----------------------------+
            |      v                            v                            v
            |  region features r          region geometry g             mask statistics
            |  (encoder features          (soft area, centroid,         (max and mean of M)
            |   averaged inside M)         height, width, spread)
            v      |                            |
   global context c (average of 1/32 features)  |
            |      |                            |
            v      v                            v
   Head 1  P(fracture)                <- c, g, mask statistics
   Head 2  P(location | fracture)     <- r, c, g       (neck / pertrochanteric / subtrochanteric)
   Head 3  P(high grade | location)   <- r, g, location embedding  (one shared "severity" trunk)
            |
            v
   6-class probabilities by the chain rule, the mask, and an uncertainty score
```

Chain rule (always gives valid, consistent 6-class probabilities):
- P(no fracture) = 1 − p_f
- P(neck, Garden I–II) = p_f · p_loc(neck) · (1 − p_sev(neck)), and P(neck, Garden III–IV) = p_f · p_loc(neck) · p_sev(neck)
- Pertrochanteric Evans I–II and III–V: the same, with p_loc(pertroch) and p_sev(pertroch)
- P(subtrochanteric) = p_f · p_loc(subtroch) (the release does not grade these)

### 3.4 Components, and which weakness each one answers

| # | Component | Weakness it targets (evidence) | How we prove it works |
|---|---|---|---|
| C1 | Full-resolution U-Net-style decoder, Dice + BCE loss | Coarse YOLO masks; Dice falls with resolution (G1) | Per-class Dice, HD95, NSD vs YOLOv8-seg and U-Net family |
| C2 | Mask-weighted region pooling (the heads look inside the predicted region) | Grade errors at the right location (15 of 29) | Ablation: global pooling vs region pooling |
| C3 | Differentiable region geometry vector g | Geometry alone gives AUC 0.75–0.84 for grade and 80% for location | Ablation: with vs without g |
| C4 | Hierarchical heads (fracture → location → grade), trained conditionally | Flat 6-way labels ignore the label hierarchy (G3) | Ablation: flat 6-way head vs hierarchy; error-type table |
| C5 | Shared "severity" trunk for the neck and pertrochanteric grades | Rare Garden I–II can borrow strength from "stable" Evans I–II (both low displacement) | Ablation: separate vs shared grade heads; Garden I–II recall |
| C6 | Logit adjustment at each node (Menon et al. 2021) | Rare classes: 57 and 36 training femurs (G2) | Balanced accuracy, per-class recall |
| C7 | Teacher forcing: the heads start from the released (ground-truth) mask and move to the predicted mask during training | Bad early masks would confuse the heads | Training stability; "oracle mask" analysis |
| C8 | 5-fold deep ensemble + temperature scaling + mask uncertainty | No calibration or uncertainty (G7) | ECE, Brier, reliability plots, risk–coverage curves |

**Losses:** L = L_seg (Dice + BCE, on all femurs; healthy femurs teach an empty mask) + L_fracture (BCE) + L_location (cross-entropy, fractured femurs only) + L_grade (BCE, neck and pertrochanteric fractures only) + 0.5 · L_joint (negative log-likelihood of the chain-rule 6-class probability, keeps the parts consistent).

**What is novel, stated the way the paper will state it:**
1. The first network that uses a predicted fracture-region mask as the explicit intermediate step for location → grade decisions on hip x-rays (earlier hierarchical work used classifier cascades or boxes, never masks).
2. A differentiable region-geometry descriptor, motivated by our finding that mask geometry alone carries grade information (AUC up to 0.84).
3. A controlled answer to "does segmentation help grading?", including an **oracle experiment** (heads fed the released ground-truth mask), which shows how much better grading could get with perfect masks.
4. The first independent benchmark on HUSJA-UA, with the first per-class segmentation metrics, 5-fold out-of-fold statistics and calibrated uncertainty.

### 3.5 Risks and how we handle them

| Risk | Likelihood | Mitigation |
|---|---|---|
| Gains over a strong flat classifier are small (1–3 points) and not significant on 210 test femurs | Medium | Primary endpoints are fixed in advance (Section 7.5). Paired tests also on the **1,043 out-of-fold predictions**, which give 5× more statistical power than the test set. Report honest CIs. |
| Predicted masks are poor on hard cases and mislead the heads | Medium | Global context branch, teacher forcing, mask-confidence gating, and the oracle analysis to quantify the effect |
| Reviewers call it "just multi-task learning" | Medium | Clinical-reasoning framing, geometry evidence, ablation of each part, error-type analysis |
| Train/test annotation shift lowers everyone's test Dice | Certain | Same for all models. We report predicted vs reference area per split and explain it; no tuning on test |
| Compute overrun | Low | Development on fold 0 only; 5 folds only for frozen configurations |

**Safety net:** even if the hierarchy gives only modest gains, the paper still has four solid contributions: a better segmentation decoder (RQ1), an honest answer to whether segmentation helps grading (RQ2), calibrated uncertainty (RQ4) and the first independent benchmark. That is enough for a Q3 journal at worst.

---

## 4. Step 4: Option B (alternative, higher risk): RefFrac, "compare with a healthy reference"

### 4.1 The idea
Radiologists compare the painful hip with the other side. RefFrac gives the network a **healthy reference femur** next to the target femur: the mirrored contralateral femur from the same x-ray when it exists, otherwise the most similar healthy femur retrieved from the training set. A shared (Siamese) encoder processes both. A cross-attention module aligns them and computes "what is different here", and the decoder and heads use that difference to segment and classify.

### 4.2 What it needs to be credible
1. **Shortcut control:** during training the reference is **always healthy** (a healthy target never gets its fractured partner as the reference). A "swap test" shows the model does not just look at the reference.
2. **Fallback for the 56% of fractures without a partner:** the retrieved healthy reference, evaluated separately for "true partner" and "retrieved" cases.
3. **Alignment:** a light spatial transformer (as in AASN) or attention-based matching, because the crops are not registered.

### 4.3 Honest assessment
- **Strengths:** clearly new for hip fracture segmentation and grading. Clinically intuitive. Uses a property of this dataset nobody has touched.
- **Weaknesses:** a usable partner exists for only 44% of test fractures. The test "no fracture" class is almost entirely partners of fractured hips (64 of 65), so detection results are easy to inflate by accident. The extra gain over a strong supervised model is uncertain. Roughly 1.5× the compute of Option A and 3–4 more weeks.

## 5. Step 5: Side-by-side comparison

| | **A: SLG-Net (recommended)** | B: RefFrac |
|---|---|---|
| Core directions | D3 + D2, with D1, D7, D9 | D4 + D2, with D7, D9 |
| Novelty | Medium-high | High |
| Evidence in our data | Strong (15 of 29 errors are grade errors; geometry AUC up to 0.84) | Moderate (partner for 44% of test fractures) |
| Chance of a clear positive result | High | Medium-low |
| Main reviewer risk | "Multi-task is common", answered by ablations and the oracle study | Shortcut and coverage concerns |
| Core Kaggle GPU-hours | ~70–80 | ~105–120 |
| Time to submission | ~12 weeks (+2 buffer) | ~15–16 weeks (+2 buffer) |
| Realistic tier | Q2, with Q3 as fallback | Q2 if it works; hard to place if it does not |

**My recommendation:** do **Option A now as Paper 1**. Then reuse its code, splits and baselines for **Option B as Paper 2**. B becomes much cheaper once A exists, and two papers from one dataset is a good PhD outcome.

---

## 6. Step 6: The data protocol (already prepared, same for both options)

1. **Official test set (210 femurs) is frozen.** It is used once per final configuration and never for tuning.
2. **Train (1,043 femurs) is split into 5 folds,** grouped by x-ray ID so both femurs of one radiograph stay together, and stratified by the 6 classes. The duplicate image pair is merged into one group (both are in fold 0).
3. **Done today:** `scripts/make_splits.py` ran on your local data and wrote `splits/folds_v1.csv`. Each fold has 208–209 femurs from 139–141 x-rays (close to the paper's 201-femur validation set), with 11–12 Garden I–II and 7–8 subtrochanteric cases per fold. The class counts match the paper's Table 1 exactly. Fingerprint `733a70aa…`, which lets us confirm the Kaggle copy is identical.
4. **Each configuration trains 5 models** (fold k held out). Training length is fixed in advance, and thresholds and calibration come from the pooled out-of-fold predictions (rules in Report 04 §3.4). We report (a) out-of-fold results on all 1,043 training femurs, (b) test mean ± SD over the 5 models, and (c) the 5-model ensemble on test (headline number).
5. **Pre-processing:** resize to 768×512, random horizontal flip (so left and right femurs need no mirroring), small rotation and scale, brightness, contrast and gamma, and mild noise or blur to mimic old CR images. No vertical flips (anatomy).

## 7. Step 7: Experiment plan (Option A)

### 7.1 Baselines (all trained by us on the same 5 folds)
| ID | Model | Why it is there |
|---|---|---|
| B1 | YOLOv8s-seg @ 1024 px | The paper's final model (our reproduction; their code is not public) |
| B2 | YOLOv8s-seg @ 640 px | The paper's best-Dice setting (small @640: test Dice .780; corrected 2026-11-26, earlier drafts said nano) |
| B3 | YOLO11s-seg @ 1024 px (run S03; changed from 640 px on 2026-11-26 because S01 vs S02 showed no difference, and 1024 px matches S01) | Newer YOLO generation |
| B4 | DenseNet121 classifier | The paper's best classifier (AUC 0.993) |
| B5 | EfficientNet-B4 classifier | Paper baseline |
| B6 | ConvNeXt-T classifier | **Same backbone as ours**, isolates the effect of our heads |
| B7–B9 | U-Net (ResNet34), DeepLabV3+, SegFormer-B2 | Standard segmentation models for RQ1 |
| (opt.) | MedSAM + LoRA, nnU-Net 2D | Foundation-model (D6) and self-configuring baselines, if compute allows |

### 7.2 Ablations (each removes or swaps one component; 5 folds each)
| ID | Variant | Tests |
|---|---|---|
| A0 | ConvNeXt-T + flat 6-way head (= B6) | Starting point |
| A1 | + segmentation decoder, flat head on global features | Does segmentation as an extra task help? |
| A2 | + region pooling (C2) | Does looking inside the region help? |
| A3 | + geometry vector (C3) | Does region shape help? |
| A4 | + hierarchical heads (C4) | Does location → grade help? |
| A5 | + shared severity trunk (C5) | Does sharing help Garden I–II? |
| A6 | + logit adjustment (C6) = **full SLG-Net** | Rare-class effect |
| Oracle | A6 with the released (ground-truth) mask at test time | Upper bound: how much better with perfect masks |
| Sens. | Second backbone (EfficientNetV2-S); higher input (1152×768) | Gains are not tied to one setting |

### 7.3 Metrics (chosen following *Metrics Reloaded*, Maier-Hein et al. 2024)
- **Detection:** AUC, sensitivity, specificity, binary accuracy (threshold fixed on the out-of-fold predictions).
- **6-class:** accuracy (to compare with the paper's 86.2%), **macro-F1**, balanced accuracy, per-class recall and precision, confusion matrix, and the **error-type split** (detection / location / grade errors, as in the paper's discussion).
- **Segmentation (fractured femurs only; a missed fracture counts as Dice 0):** Dice and IoU per class, **HD95**, **NSD** (boundary tolerance fixed in pixels), predicted-to-reference area ratio, and false-positive mask rate on healthy femurs. The base paper never stated how these cases were counted, so we state it.
- **Calibration and uncertainty:** ECE, Brier score, reliability diagrams, risk–coverage curves (AURC), and the correlation between mask uncertainty and per-case Dice.

### 7.4 Statistics
- 95% CIs by stratified bootstrap (2,000 resamples) for every metric.
- Paired tests of SLG-Net vs each baseline: **DeLong** (AUC), **exact McNemar** (accuracy), paired bootstrap (macro-F1, Dice), **Wilcoxon signed-rank** (per-case Dice and HD95), with **Holm** correction across baselines.
- Run every test twice: on the frozen test set (n=210) and on the out-of-fold predictions (n=1,043, much more power).

### 7.5 Pre-specified primary endpoints (written down now, before any result)
1. 6-class **macro-F1** (it weights rare classes fairly; accuracy is reported too, for comparison with the paper).
2. Mean **Dice on fractured femurs**.
All other metrics are secondary. Fixing this in advance protects against "metric shopping" and reviewers value it.

### 7.6 Reporting
Follow the **CLAIM 2024** checklist (Tejani et al. 2024). Release code, split file and trained-model weights with the paper.

## 8. Step 8: Kaggle compute estimate (Option A)

Hours per run are my estimates for a single T4 or P100 with mixed precision. Week 1 measures the real numbers, and this table gets updated.

| Block | Runs | Hours per run | GPU-hours |
|---|---|---|---|
| Environment check + image cache | 1 | 0.3 | 0.3 |
| B1 YOLOv8s-seg @1024 | 5 | 1.5–2.0 | 8–10 |
| B2 YOLOv8s-seg @640, B3 YOLO11s-seg @1024 | 10 | 0.5–0.8 | 6–7 |
| B4–B6 classifiers | 15 | 0.3–0.4 | 5–6 |
| B7–B9 segmenters | 15 | 0.4–0.5 | 6–8 |
| SLG-Net development (fold 0 only) | ~25 | 0.5 | 12–15 |
| SLG-Net final | 5 | 0.6 | 3 |
| Ablations A0–A5 (A6 = final) | 30 | 0.5 | 15–18 |
| Sensitivity runs | 10 | 0.6–1.2 | 9 |
| Ensembles, test-time augmentation, uncertainty | n/a | n/a | 2 |
| **Core total** | | | **≈ 70–80** |
| Optional: MedSAM-LoRA, nnU-Net 2D, FracAtlas | | | +15–30 |

**What this means in practice:**
- Kaggle's free GPU quota is commonly reported as about **30 hours per week**, with a **12-hour limit per session** (your exact quota is shown in your Kaggle account settings). The core plan uses roughly 3 weeks of quota, spread over 7 weeks of experiments, so there is slack.
- Each "Save Version" run chains several folds (for example 5 folds × 0.5 h ≈ 2.5 h). The YOLO @1024 runs go 2–3 folds per session.
- **Efficiency rules:**
  1. Resize all images once at session start (about 1 minute).
  2. Use mixed precision.
  3. Develop on fold 0 only.
  4. Run all 5 folds only for frozen configurations.
  5. If the T4×2 accelerator counts as one session, run two folds in parallel, one per GPU.

## 9. Step 9: Timeline (step by step, with milestones)

| Step | When | What happens | Milestone / output |
|---|---|---|---|
| 0 | 26–29 Nov | You choose the option. Repo goes live. First Kaggle run (environment check) | Decision logged; `env_report.json` |
| 1 | Week 1: 30 Nov–6 Dec | Data loader, metrics module, YOLO baselines B1–B2 on Kaggle | First per-class Dice ever reported on HUSJA-UA |
| 2 | Week 2: 7–13 Dec | Baselines B3–B9, baseline table v1, error analysis | **M1: baseline table** |
| 3 | Weeks 3–4: 14–27 Dec | Build SLG-Net (decoder, region pooling, geometry, hierarchy), unit tests, fold-0 development | Working model |
| 4 | Week 5: 28 Dec–3 Jan | Rare-class pieces and calibration, then freeze the configuration (git tag) | **M2: frozen method** |
| 5 | Weeks 6–7: 4–17 Jan | 5-fold final runs, ablations, sensitivity runs | **M3: all runs done** |
| 6 | Week 8: 18–24 Jan | Statistics, uncertainty analysis, figures, error and failure analysis | **M4: results package** |
| 7 | Weeks 9–11: 25 Jan–14 Feb | Manuscript: methods and results first, then introduction and discussion; CLAIM checklist; internal review | Full draft |
| 8 | Week 12: 15–21 Feb | Journal choice (check JCR), formatting, code release, cover letter, submission | **M5: submitted** |
| Buffer | 2 weeks | Holidays, re-runs, reviewer-style self-check | Latest submission about 7 Mar 2027 |

Option B would add about 3–4 weeks (Steps 3–5 take longer).

## 10. Step 10: Candidate journals (confirm quartiles in JCR first; they change every June)

- **Journal of Imaging Informatics in Medicine** (Springer): recent hip-fracture deep-learning papers appeared here (Zhao 2026, Mohaidat 2026).
- **BMC Medical Imaging** (open access).
- **Quantitative Imaging in Medicine and Surgery**: published TRFC-net (trochanteric fracture classification, 2026).
- **Diagnostics** or **Bioengineering** (MDPI): Wang 2025 (segmentation-guided neck-fracture detection) appeared in Bioengineering.
- **International Journal of Imaging Systems and Technology** (Wiley); **Medical & Biological Engineering & Computing** (Springer).
- Stretch targets: **Insights into Imaging** (the base paper's journal), **Computers in Biology and Medicine**.

I'll check each journal's current JCR quartile and scope at Step 8, when we know how strong the results are.

## 11. Decision needed from you

Pick one:
1. **Option A, SLG-Net** (recommended). Start Step 1 right away.
2. **Option B, RefFrac.**
3. **A now, B as a second paper later** (same as 1, plus B is recorded in the tracker as Paper 2).

Steps 0–2 (environment check, data loader, metrics, all baselines) are identical for both options, so no time is lost while you decide.

---

## References (checked in this session or in the earlier literature review)

1. González G, et al. Classification and segmentation of hip fractures in x-rays: highlighting fracture regions for interpretable diagnosis. *Insights Imaging* 2025;16:86. doi:10.1186/s13244-025-01958-y
2. Tanzi L, Vezzetti E, Moreno R, Aprato A, Audisio A, Massè A. Hierarchical fracture classification of proximal femur X-ray images using a multistage deep learning approach. *Eur J Radiol* 2020;133:109373. doi:10.1016/j.ejrad.2020.109373
3. Jiménez-Sánchez A, et al. Precise proximal femur fracture classification for interactive training and surgical planning. arXiv:1902.01338 (2019)
4. Yang et al. Garden classification of femoral neck fracture using deep-learning algorithm. *Sci Rep* 2025. https://www.nature.com/articles/s41598-025-27766-4
5. Wang et al. *Bioengineering* 2025;12(5):466 (femoral-neck segmentation guidance for detecting undisplaced neck fractures). https://mdpi.com/2306-5354/12/5/466
6. Chen H, Wang Y, Zheng K, et al. Anatomy-aware Siamese network: exploiting semantic asymmetry for accurate pelvic fracture detection in X-ray images. *ECCV* 2020. arXiv:2007.01464
7. Yilmaz A, et al. An automated hip fracture detection, classification system on pelvic radiographs and comparison with 35 clinicians. *Sci Rep* 2025;15:16001. doi:10.1038/s41598-025-98852-w
8. Zheng Z, et al. Deep learning for automated hip fracture detection and classification: achieving superior accuracy. *Bone Joint J* 2025;107-B(2):213–220. doi:10.1302/0301-620X.107B2.BJJ-2024-0791.R1
9. Li C, Chen P, Liu Q, Ling C. FMC-Net: an anatomy-aware multi-cue lightweight network for fracture classification in radiographs. *Front Med* 2026;13:2034134. doi:10.3389/fmed.2026.2034134
10. Ahmed et al. Beyond prompt-only segmentation: foundation model-guided trust-region label refinement for single-stage bone fracture instance segmentation. *Eng Appl Artif Intell* 2026;115849. doi:10.1016/j.engappai.2026.115849
11. Harini G, Farooq A, Mishra D. Leveraging auxiliary classification for rib fracture segmentation. *ICVGIP* 2024. doi:10.1145/3702250.3702284
12. Shi X, Cao W, Raschka S. Deep neural networks for rank-consistent ordinal regression based on conditional probabilities (CORN). *Pattern Anal Appl* 2023. doi:10.1007/s10044-023-01181-9
13. Menon AK, et al. Long-tail learning via logit adjustment. *ICLR* 2021
14. Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS* 2017
15. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *ICML* 2017
16. Maier-Hein L, Reinke A, et al. Metrics reloaded: recommendations for image analysis validation. *Nat Methods* 2024;21:195–212
17. Tejani AS, et al. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 update. *Radiol Artif Intell* 2024
18. Abedeen I, et al. FracAtlas: a dataset for fracture classification, localization and segmentation of musculoskeletal radiographs. *Sci Data* 2023. doi:10.1038/s41597-023-02432-4
19. Ma J, et al. Segment anything in medical images (MedSAM). *Nat Commun* 2024
20. Isensee F, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nat Methods* 2021
21. To read in full before writing the related-work section: Jiménez-Sánchez et al., "Medical-based deep curriculum learning for improved fracture classification" (MICCAI 2019); Kazi et al., "Automatic classification of proximal femur fractures based on attention models" (MLMI 2017)

Earlier-review sources (OccuNet *Radiology* 2026, Shen *Front Med* 2026, Zhao *JIIM* 2026, Mohaidat *JIIM* 2026, Nie *QIMS* 2026) are listed in `research/01_prior_use_and_sota_review.md`.
