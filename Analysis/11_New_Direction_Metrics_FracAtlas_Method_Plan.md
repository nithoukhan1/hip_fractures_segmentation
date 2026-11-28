# Report 11: New direction: standard metrics, the FracAtlas external test, and a method that must beat our own baselines

**Date:** 2026-11-28 · **Status:** approved by the person on 2026-11-28. The fold-0 screening rules (section 7) were fixed before any screening run.

**The person's requirements (2026-11-28):**
1. The proposed method must beat **our own trained baselines**, including ConvNeXt U-Net, DenseNet-121, SegFormer-B2 and the rest.
2. It must generalise to another dataset, **FracAtlas**.
3. It must be explainable.
4. Only **standard** segmentation and classification metrics.

---

## 1. Is our comparison with the published method fair?

There are two different comparisons, and they need to be kept apart.

| Comparison | Same training femurs? | Same test femurs? | Same code and metrics? | Fair? |
|---|---|---|---|---|
| **Our YOLOv8s-seg** (our own retraining of the published method) vs **ConvNeXt U-Net** and every other baseline | Yes: the same 5 folds of the 1,043 training femurs | Yes: the released 210 test femurs | Yes: the same evaluation code | **Yes, like-for-like** |
| **The article's own reported numbers** vs ours | No: the authors built their own validation set (the released validation folder is empty) and chose their model and seed from many runs | Yes: the same released 210 test femurs | No: their own code | **Only roughly.** Their 6-class test accuracy is 86.2% (181 of 210). Our retrained YOLOv8s-seg gets 185 of 210 (88.1%), so our reproduction is at least as good as theirs |

So when we say "ConvNeXt U-Net beats the published method", we mean that it beats **our faithful retraining** of it, with the same data, folds, test set and code. That is the comparison a reviewer accepts. The article's own numbers are quoted for reference only.

What is true today (out of fold = the 1,043 training femurs, each predicted by the fold model that never saw it):

| Model | Out of fold: accuracy / macro-F1 / Dice | Test (210): correct / macro-F1 / Dice |
|---|---|---|
| YOLOv8s-seg (published method, retrained by us) | 83.2% / .744 / .740 | 185 / .845 / .725 |
| ConvNeXt U-Net (ours) | 88.4% / .825 / .777 | 182 / .779 / .744 |

ConvNeXt U-Net is clearly better out of fold (all p < .002). On the test set the two are tied: 182 vs 185 correct is not a significant difference.

## 2. Metrics: what we use now, and the standard-only set

### 2.1 What the reports use now

| Group | Metrics now | Standard? |
|---|---|---|
| Fracture yes/no | AUC, sensitivity, specificity | Yes |
| 6-class type and grade | accuracy, macro-F1, per-class precision, recall and F1, confusion matrix | Yes |
| | balanced accuracy, Cohen's kappa | Standard, but redundant here |
| | "Garden I-II recall" as a separate endpoint | It is simply the recall of one class; reporting it on its own looks custom |
| | error breakdown (detection / location / grade) | Our own description |
| Segmentation | Dice on fractured femurs (missed = 0), IoU, HD95 | Yes |
| | NSD (normalised surface distance), "Dice on detected femurs" | NSD is used in medical challenges but is less familiar; "Dice on detected" is ours |
| Calibration | ECE, Brier score, temperature | Standard, but not needed for the claims |
| Explainability | region deletion, oracle, risk–coverage, pointing game, self and swap tests | Deletion and the pointing game are standard explainability tests; the others are ours |

### 2.2 Proposed standard-only set (for the paper and all new reports)

**Classification**, computed for both tasks: fracture yes/no, and the 6 classes.
- Accuracy.
- Precision, recall (sensitivity) and F1: per class, and macro-averaged.
- Specificity: for the fracture yes/no task.
- AUC (area under the ROC curve): binary for fracture yes/no; one-vs-rest macro-averaged for the 6 classes.
- Confusion matrix.

**Segmentation**, on fractured femurs:
- Dice coefficient;
- IoU (Jaccard index);
- HD95 (95th-percentile Hausdorff distance, in pixels).

**Explainability** (standard tests from the literature):
- the pointing game: does the most important point fall inside the annotated fracture?
- deletion: how much does the prediction drop when the highlighted region is removed?

**Statistics:**
- 95% confidence intervals by bootstrap over x-rays;
- McNemar's test for accuracy;
- DeLong's test for AUC;
- Wilcoxon's test for Dice.

**Dropped from the main text:** balanced accuracy, kappa, NSD, "Dice on detected", ECE/Brier/temperature, the error breakdown, and the oracle, risk–coverage, self and swap tests. Garden I-II recall stays, but only as one row of the per-class recall table. The filed results are not changed. Every dropped number stays in the existing reports and can go to a supplement if a reviewer asks.

## 3. FracAtlas: what is on your PC and what it can test

Folder: `FracAtlas (FigShare)\FracAtlas\`. It holds 4,083 radiographs (717 fractured), a `dataset.csv` with flags per image, fracture annotations (COCO polygons, YOLO, Pascal VOC, VGG) and the authors' train/valid/test split of the fractured images. I read it without changing anything.

| What | Count |
|---|---|
| All images / fractured | 4,083 / 717 |
| Images flagged "hip" / fractured "hip" images | 338 / 63 |
| Fracture polygons on the 63 fractured hip images | 70 |
| Image size of the fractured hip images | 48 are only 373 × 454 px; 10 are 2304 × 2880 |
| Labels | fractured yes/no, number of fractures, body part, view, hardware. **No fracture type, no Garden or Evans grade, no left/right** |

**I looked at all 63 fractured "hip" images with their annotations.** The "hip" flag means "the image shows the hip region", not "a hip fracture":

| Group | Images | Usable for our task? |
|---|---|---|
| Adult proximal femur fracture (neck, trochanteric or subtrochanteric region), no implant | **about 22** (one has fractures on both sides), from roughly 14–15 patients (several patients have 2 views) | **Yes:** this is the external test set |
| Post-operative: plate, nail, screws or hip prosthesis | about 22 | No: HUSJA-UA has no operated hips. At most a separate stress test |
| Children: whole-body or leg images, mostly femoral shaft or distal femur | about 12 | No: other anatomy and age |
| Adult femoral shaft, not proximal | 2 | No |
| Forearm (wrongly flagged as hip) | 4 | No |
| Other (pelvis) | 1 | No |

The 22 usable images are 2180, 2337, 2338, 2340, 2341, 2540–2545, 2549, 2701, 2768, 3341, 3342, 3766, 3767, 3892, 3893, 4333 and 4334 (all `IMG000….jpg`). The final list will be fixed by written rules before any model sees an image.

**What FracAtlas can and cannot test:**

| Task | Possible on FracAtlas? | Metrics |
|---|---|---|
| Fracture yes/no per femur | **Yes**, using the about 25 fractured femurs above, plus healthy femurs (the other side of these images, and adult hip images without fracture or implant; the 275 non-fractured hip images still need screening with the same rules) | AUC, sensitivity, specificity, accuracy, F1 |
| Fracture localisation | **Yes**: FracAtlas outlines each fracture tightly | Pointing game, overlap of the predicted region with the outline |
| Fracture segmentation | **Yes, with a caveat**: FracAtlas outlines the fracture line tightly, while HUSJA-UA marks a looser region, so Dice and IoU will be low for every model. The comparison between models is still fair | Dice, IoU, HD95 |
| Fracture type (neck vs trochanteric vs subtrochanteric) and grade (Garden, Evans) | **No:** FracAtlas has no such labels, and we have no clinician to add them | — |

**Problems to solve before the test (honest risks):**
1. **Femur crops.** Our models read one femur, cropped and padded like the HUSJA-UA images (portrait, 1024 × 1536, contrast windowed). FracAtlas has whole hip or pelvis images. We need a crop rule for each femur, fixed before testing: for example a box around each proximal femur, drawn once and checked visually.
2. **Low resolution.** Most images are 373 × 454 px, so a femur crop is only about 150 × 230 px, against 1024 × 1536 in HUSJA-UA. That is a large domain shift, and it will lower every model's results.
3. **Small numbers.** With about 25 fractured femurs, a sensitivity of 80% has a 95% interval of roughly 60–92%. The external test can show that a model generalises, or fails to, but it cannot separate two good models.
4. **Zero-shot only.** We do not fine-tune on FracAtlas: there are too few cases, and fine-tuning would remove the "unseen dataset" point. Every model (the baselines and ours) runs as trained on HUSJA-UA, with its HUSJA-UA threshold.

**How the external test runs:** a short Kaggle job (about 0.5 h) loads the saved fold models of each run from their Kaggle outputs, predicts every FracAtlas femur crop, and writes the same per-case files as now. If a run's weights are no longer in its Kaggle output, that run must be retrained.

## 4. A method that can beat our own baselines: candid assessment and plan

### 4.1 Why this is hard
- ConvNeXt U-Net is already the best of our 10 baselines out of fold, and neither SLG-Net nor RefFrac beat it.
- **What "beat" can mean on this data:**
  - out of fold (1,043 femurs), a paired test can detect a gain of about 2 accuracy points (22 femurs);
  - on the 210-femur test set it needs about 4–5 points (9–10 femurs), more than any change we have seen so far.
  - Realistically, "outperforms our baselines" can be shown out of fold, with the test set going the same way. A significant test-set win is unlikely for any method.
- **Where the errors are:** 65 of ConvNeXt U-Net's 121 out-of-fold errors are grade errors: the right location, but the wrong Garden or Evans grade. These depend on fine detail, while the model sees the femur at half its resolution (768 × 512 instead of 1024 × 1536).

### 4.2 Candidates (each is tested on fold 0 first; only the winner gets the full 5-fold run)

| # | Candidate | Idea | Why it might work | Kaggle hours (fold 0 / full 5 folds + test) | Chance of a significant out-of-fold gain |
|---|---|---|---|---|---|
| 1 | **Segment-and-zoom** (recommended as the proposed method) | ConvNeXt U-Net finds the fracture region; a second branch reads a crop around that region at **full resolution** and makes the grade decision from it plus the whole-femur view | It targets the grade errors directly, at little extra cost. The zoomed crop is also the explanation: the decision is made from the region shown to the user | ~1 / ~3.5 | Moderate |
| 2 | Full resolution throughout (1536 × 1024) | Same model, native resolution | Simplest way to give the model the fine detail | ~2 / ~9 | Moderate; memory and time risk on a T4 |
| 3 | Stronger or radiograph-pretrained encoder (ConvNeXt-S, or an encoder pretrained on X-rays) | Better features | Usually gives 1–2 points | ~1 / ~4 | Low to moderate |
| 4 | Ordinal grade loss (Garden I-II < III-IV, Evans stable < unstable) | Grades are ordered, so the loss should know that | Cheap, can be added to any candidate | +0 | Low on its own |

**Recommended route:**
1. Fold-0 development of candidates 1, 2 and 3 (candidate 4 added to the best one), two seeds each for the winner. About 8–10 h.
2. The decision rule, written before the runs: a candidate goes forward only if its fold-0 macro-F1 beats ConvNeXt U-Net's by at least .02 (mean of two seeds), and Dice does not fall by more than .01.
3. The winner: 5 folds + test (about 4–9 h), and a second seed (the same again) to show the gain is not seed noise.
4. The claim "outperforms our baselines" needs, out of fold:
   - McNemar p < .05 on accuracy;
   - a significant gain in macro-F1;
   - no loss in Dice;
   - against ConvNeXt U-Net, and so against every other baseline. The test set must go the same way.
5. The FracAtlas external test for every model (about 0.5 h).

**Total:** about 20–30 Kaggle hours, so 1–2 weeks of quota. **Risk:** no candidate passes the fold-0 rule. In that case we would know within the first 8–10 h, before spending on full runs.

### 4.3 Explainability, with standard tests
- **Our method:** the predicted fracture mask and the zoomed region are the explanation, because the decision is made from them.
- **Baselines** (classifiers): Grad-CAM heat maps.
- **For every model, measured the same way:**
  - the pointing game against the HUSJA-UA masks and the FracAtlas outlines;
  - the deletion test.
- What we already have counts here: SLG-Net's decisions depend on the fracture region more than ConvNeXt U-Net's (64% vs 48% of correct decisions lost when the region is removed, p < .001).

## 5. What happens to SLG-Net and RefFrac
- They stay in the paper as tested extensions that did not improve accuracy. They support the explainability story: grounding in the region is measurable.
- The new proposed method (candidate 1, if it passes) becomes the main contribution.
- The optional second-backbone check is replaced by this plan.

## 6. Steps, in order (after your approval)
1. **Rules first** (no GPU):
   - the FracAtlas inclusion rules and crop rule;
   - the fold-0 decision rule;
   - the standard-metrics script, applied to all filed runs.
2. **FracAtlas preparation** (no GPU):
   - screen the 338 hip images with the rules;
   - make the femur crops;
   - a visual check sheet for you.
3. **Fold-0 development** of the candidates (Kaggle, about 8–10 h).
4. **The winner's** 5-fold and second-seed runs (Kaggle, about 8–18 h).
5. **FracAtlas zero-shot test** of all models (Kaggle, about 0.5 h).
6. **Tables** with the standard metrics; explainability figures; then the manuscript.

## 7. Fold-0 screening rules (fixed 2026-11-28, before any screening run)

### 7.1 The three candidates, as built
| Candidate | Run names (first seed / second seed) | Network | Input | What differs from ConvNeXt U-Net |
|---|---|---|---|---|
| 1. Segment-and-zoom | `N_segzoom_fold0` / `N_segzoom_fold0_seed2` | ConvNeXt U-Net + a second ConvNeXt-T encoder for the crop (`zoom:` in the training script) | 1536 × 1024, the full resolution of the release | The ConvNeXt U-Net part sees the image averaged down by 2, exactly its usual 768 × 512 input. The crop is a 640 × 640 full-resolution square around the predicted mask, resampled to 512 × 512. The decision is made from the whole-femur features and the crop features together |
| 2. Higher resolution | `N_res1152_fold0` / `N_res1152_fold0_seed2` | ConvNeXt U-Net unchanged | 1152 × 768 | Only the input size |
| 3. Larger encoder | `N_convnext_s_fold0` / `N_convnext_s_fold0_seed2` | ConvNeXt U-Net with ConvNeXt-S (ImageNet-22k weights) | 768 × 512 | Only the encoder |

**Change from section 4.2, stated before the runs:** candidate 2 is screened at 1152 × 768, not at 1536 × 1024. The full size would take about 3 h for fold 0 and about 14 h for 5 folds on the T4s, and the segment-and-zoom network already tests what full resolution adds where it matters. If 1152 × 768 passes, the full size is tried next.

Candidate 4 (the ordinal grade loss) is not screened on its own. It will be added to the winner.

**Segment-and-zoom, in detail:**
1. **Locate:** the ConvNeXt U-Net predicts the fracture mask as always. The crop centre is the probability-weighted centre of the predicted fracture pixels (probability ≥ 0.5). A femur without such pixels, which is most healthy femurs, gets its crop at its most suspicious place (the 500 most probable pixels). Every femur therefore gets exactly one crop.
2. **Training:** in the first 30 epochs the crop is sometimes centred on the released mask instead. The chance of this falls from 1 to 0 (teacher forcing, as in SLG-Net). The centre also moves by up to ±32 pixels, so that the network does not learn to expect the fracture in the exact middle.
3. **Zoom:** a second ConvNeXt-T, with its own ImageNet weights, reads the crop.
4. **Decide:** the whole-femur features and the crop features are joined, normalised and given to one linear 6-class layer. The ConvNeXt U-Net's own 6-class head is kept as an auxiliary output (weight 0.5 in the loss), so the whole-femur branch stays a full ConvNeXt U-Net.
5. **Explainability:** every femur's crop window is saved. The pointing game of the crop (does the window hold the centre of the released mask?) is part of every run's results.

**Everything else is the ConvNeXt U-Net's:** 60 epochs, effective batch 16, AdamW with learning rate 2e-4 and a cosine schedule, the same augmentations, folds and loss for the mask, and final weights (no early stopping). The first seed is 2026, the second 2027.

### 7.2 The base to beat, computed now from the filed runs
The base is the two ConvNeXt U-Net runs we already have: the ConvNeXt U-Net and its second run, the same settings trained twice. They are evaluated on fold 0's 208 held-out femurs, with the fracture threshold fitted on fold 0, as every candidate will be (`scripts/run_stats.py --folds 0`).

| Fold 0, 208 femurs | ConvNeXt U-Net | Second ConvNeXt U-Net run | Mean = the base |
|---|---|---|---|
| Accuracy (6 classes) | 184/208 = .885 | 184/208 = .885 | .885 |
| Macro-F1 (6 classes) | .822 | .811 | **.817** |
| Dice, fractured femurs (a missed fracture counts 0) | .749 | .761 | **.755** |

The two identical runs differ by .011 in macro-F1. That is the size of run-to-run noise on one fold, and why the rule uses the mean of two seeds.

### 7.3 The pass rule
A candidate passes the screening if both hold, for the mean of its two seeds on fold 0:
- **macro-F1 ≥ .837** (the base + .02);
- **Dice of fractured femurs ≥ .745** (the base − .01).

**Second seeds:**
- The segment-and-zoom network always gets its second seed, because it is the recommended method.
- Candidates 2 and 3 get a second seed only if their first seed reaches macro-F1 ≥ .827 (the base + .01). Below that, the second seed would need a gain of more than .03 to pass, which is very unlikely.

**After the screening:**
- If more than one candidate passes, the one with the higher mean macro-F1 goes forward. If they are within .01, the cheaper one goes forward.
- The winner gets 5 folds and the test set. The final claim needs the out-of-fold rule of section 4.2, step 4.
- If no candidate passes, we stop and report that to the person before any further runs.

**What we report for every screening run** (the standard set of section 2.2): accuracy, macro-F1, per-class precision, recall and F1, specificity, AUC, Dice, IoU, HD95, and for the segment-and-zoom network, the pointing game of its crop.

### 7.4 Kaggle sessions (two runs per session, one per T4)
| Session (RUN name) | GPU 0 | GPU 1 | Expected time |
|---|---|---|---|
| `N_screen_round1` | segment-and-zoom, seed 2026 | 1152 × 768, seed 2026 | about 1.8 h |
| `N_screen_round2` | ConvNeXt-S, seed 2026 | segment-and-zoom, seed 2027 | about 1.6 h |
| `N_screen_round3` (only if section 7.3 asks for these seeds) | 1152 × 768, seed 2027 | ConvNeXt-S, seed 2027 | about 1.8 h |

Screening total: about 3.5–5 Kaggle hours.

