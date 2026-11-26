# Report 01: Base Article Deep Reading and Dataset EDA

**Project:** Segmentation Project (HUSJA-UA Femoral Fracture dataset)
**Date:** 2026-11-26
**Base article:** González G, Galant J, Salinas JM, Benítez E, Sánchez-Valverde MD, Calbo J, Cerrolaza N. *Classification and segmentation of hip fractures in x-rays: highlighting fracture regions for interpretable diagnosis.* Insights into Imaging (2025) 16:86. doi:10.1186/s13244-025-01958-y (PubMed 40232323)
**EDA code and outputs:** `EDA/eda_femoral.py`, `EDA/per_image_stats.csv`, `EDA/summary.json`, `EDA/figures/`
**Related:** `research/01_prior_use_and_sota_review.md` (literature thread)

---

## Part 1. The base article, section by section

### 1.1 Objective and claim
- Build one AI model that **detects, grades and segments** proximal femoral fractures in hip x-rays, and compare it with classification CNNs.
- Main claim: the first method that **segments** femoral fractures (as opposed to Grad-CAM heatmaps or bounding boxes), while also grading them.

### 1.2 Clinical motivation (Introduction)
- 1-year mortality after hip fracture fell from about 30% to 22%, but up to 50% lose functional independence.
- 4–12% of clinically relevant fractures are missed on initial radiographs. Sensitivity is about 69–73% for GPs and first-year residents versus about 96% for attending radiologists and orthopaedists.
- Prior AI work:
  1. Binary fracture / no fracture CNNs, AUC 0.91–0.99, no location or type.
  2. Multiclass location (neck / pertrochanteric / subtrochanteric), sometimes displaced vs undisplaced (Jiménez-Sánchez 89%, Murphy 92%, Yu, Kitamura).
  3. Localisation by Grad-CAM, point networks (Cheng 2021) or boxes (Liu 2022).
  4. Grading on clinical scales: only Mutasa 2020 (Garden I/II vs III/IV, 86%) and Krogue 2020 (displaced vs non-displaced).
- Their stated novelty: grading + segmentation in one network, adding subtrochanteric and splitting pertrochanteric into Evans I–II vs III–V.

### 1.3 Data collection
- Single centre: Hospital Universitario de San Juan de Alicante, IRB 21/060, retrospective, consent waived.
- HIS queried for ICD codes of femoral fracture, 2010–2021; images pulled from PACS and anonymised via BIMCV.
- Retrieved 10,300 images from 2,609 patients (the abstract says 10,308 / 2,618: a small internal inconsistency).
- 5 devices, 3 vendors (Philips DigitalDiagnost C90, Bucky Diagnost FS, Essenta DR; Agfa DR 100; Canon M1 mobile). DX 70.9%, CR 29.1%.
- 986 x-rays (846 studies, 707 patients) randomly selected for annotation, so class frequencies reflect clinical prevalence (which makes Garden I–II and subtrochanteric rare).

### 1.4 Reference standard (annotation)
- Two radiologists (3 and 20 years' experience) drew **a polygon around each fracture**, gave a location and a grade (Garden for neck, Evans for pertrochanteric; subtrochanteric not graded, although Seinsheimer exists).
- **Test images only** were reviewed by a committee (the two readers plus a 20-year MSK radiologist). Train and validation were not committee-reviewed. This matters: see EDA finding 2.6.
- Tool: OHIF viewer inside BIMCV.
- Labels collapsed to 6 classes per femur: no fracture; Neck Garden I–II; Neck Garden III–IV; Pertroch Evans I–II; Pertroch Evans III–V; Subtrochanteric.

### 1.5 Pre-processing
- A YOLOv8 detector found femurs (semi-automatic, manual QC). 1,586 femurs detected, 3 added by hand; 49 rejected (too close to the edge), 1 duplicate, 5 false; 80 dropped because only one femur in the image was annotated. **Final: 1,454 femurs.**
- Each detection got a **100% margin**, then was cropped and resized to a canonical size keeping isotropic pixels. Areas outside the original image become black (EDA confirms about 33% black pixels per ROI).
- Intensity windowed between the 2nd and 98th percentiles and quantised to 8 bit (because YOLOv8 needs 8-bit input). EDA confirms about 8.5% saturated white pixels per ROI.

### 1.6 Model
- YOLOv8-seg (Ultralytics), backbone + neck + decoupled head; each predicted mask is tied to a class label, so classification is "class of the highest-confidence instance" and detection is "any instance present".
- 100 epochs, checkpoint with best validation loss.
- Grid: model size {nano, small, medium} × imgsz {320, 640, 1024, 1536}, **6 seeds each** (72 runs). Final model chosen as small @1024 by validation AUC/accuracy, then best-validation-accuracy seed.
- Hyper-parameters, augmentation and the confidence threshold for YOLO are **not reported**.

### 1.7 Baselines
- VGG16, DenseNet121, InceptionV3, ResNet50V2, EfficientNetB4 (ImageNet init, GAP, dropout 0.2, dense 6). Adam, categorical CE, 100 epochs, LR halved on 5-epoch plateau, flips/rotation/translation/zoom/brightness. TensorFlow. Each trained 6 times.
- Input size for baselines not reported precisely ("small image sizes"); compared against YOLO nano @320.

### 1.8 Evaluation and statistics
- Binary detection: ROC AUC (DeLong CIs and comparisons, pROC).
- 6-class accuracy with binomial CIs (caret), confusion matrix; pairwise Durbin test.
- Segmentation: mean Dice between annotated and predicted region. **How false-positive and false-negative masks enter the Dice, and whether no-fracture femurs are included, is not stated.**

### 1.9 Results (numbers to beat)
| Split (femurs) | Train | Val | Test |
|---|---|---|---|
| Patients / studies | 505 / 609 | 97 / 115 | 105 / 122 |
| Femurs | 1043 | 201 | 210 |
| No fracture | 363 | 67 | 65 |
| Neck G I–II | 57 | 11 | 16 |
| Neck G III–IV | 254 | 42 | 41 |
| Pertro E I–II | 134 | 37 | 34 |
| Pertro E III–V | 199 | 34 | 42 |
| Subtroch | 36 | 10 | 12 |

Grid on test (mean ± sd of 6 seeds):
| | 320 | 640 | 1024 | 1536 |
|---|---|---|---|---|
| AUC nano/small/medium | .978/.979/.980 | .983/.985/.985 | .985/.986/.985 | .980/.989/.983 |
| Acc nano/small/medium | 81.1/79.7/80.7 | 82.5/82.0/81.8 | 83.2/**84.4**/83.2 | 80.6/81.9/83.1 |
| Dice nano/small/medium | .771/.777/.778 | .778/**.780**/.774 | .777/.761/.761 | .767/.755/.765 |

Final selected model (small, 1024): **AUC 0.981 (0.965–0.997), binary accuracy 96.2%, Se 95.17%, Sp 98.46%, PPV 99.28%, NPV 90.14%, 6-class accuracy 86.19% (80.77–90.55), Dice 0.77 (0.56–0.98).**

Baselines on test: AUC 0.982–0.993 (DenseNet121 best, 0.993); 6-class accuracy VGG16 74.3, InceptionV3 77.6, EfficientNetB4 79.1, ResNet50V2 80.0, DenseNet121 83.8. Only VGG16 vs YOLO-small-1024 was significant. YOLO-nano-320: AUC 0.982, acc 83.3, Dice 0.78.

Confusion matrix: 15 errors are grade errors within the same location, 6 are location errors. Worst classes: Garden I–II and subtrochanteric.

### 1.10 Limitations and future work stated by the authors
1. Small number of annotated images; rare classes (undisplaced neck, subtrochanteric) perform worse. **Future: semi-automated labelling to enrich rare fractures.**
2. Single institution; **future: formal external generalisation study.**
3. No clinical reader study; **future: formal evaluation with radiologists** (second-reader benefit).
4. Open question: is segmentation more useful than a box or an arrow? **Future: test whether segmentation increases confidence and helps teaching.**

### 1.11 Critical observations (weak points we can exploit)
- **Dice decreases as resolution and model size increase**, while accuracy increases. The detector-style YOLO mask head (prototype masks at 1/4 resolution) is not built for boundary-accurate segmentation; there is room for a dedicated segmentation decoder.
- **Dice CI 0.56–0.98 is very wide**: per-case Dice varies a lot, and no per-class Dice or boundary metric (HD95, NSD) is reported.
- **Classification adds no significant gain over DenseNet121** (83.8 vs 86.2, not significant), and DenseNet has a better AUC. The segmentation branch is not shown to help classification; there is no ablation.
- Grading is learned as flat 6-way classification; the **hierarchy** (location → grade) and the **ordinal** nature of Garden and Evans grades are ignored, yet most errors are within-location grade errors.
- **No calibration, uncertainty, or failure detection** analysis.
- Model selection uses many test-reported grids; the test set is small (210 femurs, 16 Garden I–II, 12 subtrochanteric), so a few cases change accuracy by several points.
- The contralateral femur, available for most x-rays, is never used (see 2.7).
- No external validation, no reproducible code or hyper-parameters.

---

## Part 2. Dataset EDA (local copy)

All numbers below are produced by `EDA/eda_femoral.py` and stored in `EDA/summary.json` and `EDA/per_image_stats.csv`.

### 2.1 Folder structure and format
```
Femoral Fractures/
  train/ images (1043)  labels (680)  masks (1043)
  val/   images (0)     labels (0)    masks (0)      <- EMPTY
  test/  images (210)   labels (145)  masks (210)
```
- **Images:** 8-bit grayscale PNG, all exactly **1024 × 1536 (W × H, portrait 2:3)**. Already windowed (2–98%) and padded as in the paper.
- **Labels:** YOLO-seg text, one line per instance: `class x1 y1 x2 y2 …` (normalised). Only fractured femurs have a label file (680 = 1043 − 363; 145 = 210 − 65). Every label file has **exactly one polygon**, one class. No out-of-range coordinates.
- **Masks:** same size PNG, 0 = background, **value = YOLO class id + 1** (verified on all 825 fractured femurs). No-fracture masks are all zero; no fractured femur has an empty mask.
- **Class id mapping** (inferred, matches Table 1 counts exactly in both splits):
  `0 Neck Garden I–II · 1 Neck Garden III–IV · 2 Pertroch Evans I–II · 3 Pertroch Evans III–V · 4 Subtrochanteric` (no-fracture = no label).
- **Polygon vs mask consistency:** rasterised polygon vs mask IoU mean 0.997 (min 0.982), so masks and labels are the same annotation.
- **File names:** `<femurID>_<xrayID>.png`. 702 unique x-ray IDs in train, 146 in test, none shared across splits. **Patient IDs are not released.**

### 2.2 Validation split is missing
- The local `val/` folder is empty, and a comment on the IEEE DataPort page (found by the literature thread) says the official release has an empty val folder too. So the 201 validation femurs from the paper are **not available**; only train (1043) and test (210) are. Please glance at the Kaggle copy to be sure.
- Consequence: we must carve our own validation set from train. Since patient IDs are missing, split **grouped by x-ray ID** (keeps both femurs of an image together) and stratified by class. A small risk remains that one patient's two different x-rays land in train and val, which is acceptable for model selection but should be stated. The test set stays untouched and matches the paper, so our numbers are directly comparable.
- We train on 1043 femurs versus the paper's 1043 + 201 for selection, so we should report this honestly.

### 2.3 Class balance (Figure 01)
| Class | Train | Test | Train % |
|---|---|---|---|
| No fracture | 363 | 65 | 34.8 |
| Neck Garden I–II | 57 | 16 | 5.5 |
| Neck Garden III–IV | 254 | 41 | 24.4 |
| Pertroch Evans I–II | 134 | 34 | 12.8 |
| Pertroch Evans III–V | 199 | 42 | 19.1 |
| Subtrochanteric | 36 | 12 | 3.5 |
- Imbalance ratio of 7:1 between Garden III–IV and subtrochanteric. The two rare classes have only 16 and 12 test cases, so one case equals 6–8 points of per-class recall.

### 2.4 Fracture region size and position (Figures 02, 03, 04, 07)
Median mask area as % of ROI:
| Class | Train | Test |
|---|---|---|
| Neck G I–II | 2.9 | **1.9** |
| Neck G III–IV | 4.8 | **2.9** |
| Pertro E I–II | 5.4 | 5.7 |
| Pertro E III–V | 8.6 | 7.6 |
| Subtroch | 8.0 | 9.5 |
- Regions are **small** (2–10% of the ROI; 5th–95th percentile 34k–179k pixels at full resolution). Undisplaced neck fractures are the smallest targets.
- Size grows with severity (G I–II < G III–IV, E I–II < E III–V), so region size and shape carry grading information.
- Centroids are tightly clustered (neck around y≈0.36, pertrochanteric y≈0.42, subtrochanteric y≈0.51); location is strongly encoded by vertical position.
- Pertrochanteric centroids form **two clusters at x≈0.4 and x≈0.6**, because left and right femurs are **not mirrored to a canonical side** (Figure 06).
- Every mask is a **single connected region** (1 of 825 has 2 parts). The annotation is a **coarse region polygon around the fracture zone**, not a fracture-line trace (Figure 06), so Dice here means "region agreement", and boundaries are inherently fuzzy.

### 2.5 Image quality and acquisition artefacts (Figures 05, 06)
- About **33% of each ROI is black padding** (mean; up to 59%) from the 100% margin crop, and about 8.5% saturated white from 2–98% windowing; soft-tissue and bone-overlap detail is lost in the saturated parts.
- Visible artefacts: burned-in "R" laterality markers, detector edges and table lines at the bottom, CR scan-line noise, collimation borders, a different contrast between DX and CR. The paper says CR/DX and device info exist, but they are **not in the release**.
- No low-contrast outliers (std < 30) were found.

### 2.6 Annotation quality findings
1. **Exact duplicate image with two different masks:** `train/00397_00261.png` and `train/00399_00262.png` are pixel-identical (different femur and x-ray IDs, same class Evans III–V), but their masks differ: 157,150 vs 109,703 pixels, **IoU ≈ 0.70**. This is a natural sample of annotation variability and suggests that the ceiling for region Dice is well below 1. Keep one of them, or treat it as a soft label.
2. **Train–test annotation-style shift for neck fractures:** test neck masks are about 35–40% smaller than train neck masks of the same grade (table in 2.4). The test set was committee-reviewed and train was not (paper §Reference standard), which plausibly made the test polygons tighter. A model trained on train-style polygons will over-segment test neck fractures and lose Dice. This is a measurable label-noise / protocol-shift effect worth addressing.
3. Near-duplicate screening (perceptual hash + normalised cross-correlation) flags many train–test pairs at correlation about 0.97, but these are anatomically similar different patients (same canonical crop), not leaks. No x-ray ID appears in both splits.

### 2.7 Hidden structure: paired contralateral femurs
| | Train | Test |
|---|---|---|
| X-rays with 2 femurs (1 fractured + 1 healthy) | **334** | **64** |
| X-rays with 1 femur, fractured | 346 | 81 |
| X-rays with only healthy femur(s) | 22 | 1 |
- **No x-ray has two fractured femurs**, and almost every healthy femur (334 + 64 of 428) is the **contralateral side of a fractured hip in the same image**. So a healthy "reference twin" exists for about half the fractured cases, and the "no fracture" class is mainly made of contralateral femurs rather than of healthy patients. This matters for clinical realism (the AUC is mostly "fractured vs other side of the same patient") and is an unused signal: radiologists compare both sides.

---

## Part 3. Gaps and implications (inputs for idea selection, not a final idea)

| # | Gap observed | Evidence |
|---|---|---|
| G1 | Segmentation quality is not the focus: YOLO mask head, Dice falls with resolution, no per-class Dice or boundary metrics | Table 2; §1.11 |
| G2 | Rare classes (Garden I–II: 57 train, subtroch: 36 train) perform worst; the authors' own future work | Table 1, Fig 3, §2.3 |
| G3 | Flat 6-way classification ignores location → grade hierarchy and ordinal grades, while most errors are within-location | Fig 3 text |
| G4 | No evidence that segmentation helps classification (no ablation; DenseNet ≈ YOLO) | Table 3 |
| G5 | Annotation noise and train/test protocol shift (duplicate IoU 0.70; smaller test neck masks) | §2.6 |
| G6 | Contralateral femur never used, although available in about half of fractured cases | §2.7 |
| G7 | No uncertainty, calibration or failure analysis, important for a "trustworthy" segmentation claim | §1.11 |
| G8 | Laterality, padding and saturation are not normalised; mixed DX/CR without metadata | §2.4–2.5 |
| G9 | Official val split is missing in the release | §2.2 |
| G10 | Single site, no external test; public hip datasets exist only with image-level labels | paper limitations; see literature review |

Note: the paper reports only overall Dice. Per-class Dice needs model predictions, so we will report it (together with HD95 and NSD) once baselines are trained; this will be the first per-class segmentation breakdown on this dataset.

## Part 4. Recommended experimental protocol (for all later phases)
1. Keep the official **test set (210)** frozen; report AUC, binary accuracy, Se/Sp, 6-class accuracy, macro-F1, balanced accuracy, **per-class Dice, HD95, NSD**, plus calibration (ECE).
2. Create `splits/val_from_train.csv`: about 15% of train grouped by x-ray ID, stratified by class (or 5-fold grouped CV for robustness on the small rare classes).
3. Re-run the paper's YOLOv8-seg (small @1024 and nano @640) on Kaggle with 3 or more seeds as our own reproduction baseline, since their code and hyper-parameters are not public.
4. Use DeLong for AUC, McNemar / bootstrap for accuracy and Dice, with 95% CIs.
5. Optional pre-processing: mirror all femurs to one side, mask the black padding out of the loss and the normalisation.

## Figures
- `EDA/figures/01_class_distribution.png`: class counts per split
- `EDA/figures/02_mask_area_by_class.png`: fracture region size by class
- `EDA/figures/03_centroids_by_class.png`: centroid scatter (shows left/right clusters)
- `EDA/figures/04_mask_frequency_heatmaps.png`: pixel-wise fracture frequency per class
- `EDA/figures/05_intensity_stats.png`: intensity, contrast, sharpness per split
- `EDA/figures/06_samples_per_class.png`: 3 random examples per class with annotation contour
- `EDA/figures/07_area_vs_solidity.png`: mask shape vs size
