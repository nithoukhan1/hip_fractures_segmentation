# C04: Which segmenter outlines fractures best? Does a dedicated segmentation decoder beat YOLOv8-seg (RQ1)?

*Written 2026-11-27 from the filed runs:*
- `results/S01_yolov8s_seg_1024/` (register line 3);
- `results/S07_unet_r34/` and `results/S08_deeplabv3p_r34/` (lines 11 and 12);
- `results/S09_segformer_b2/` (line 18);
- `results/S10_m0_convnext_unet/` (line 8).

*Every number below comes from these files:*
- `stats.md` / `stats.json` in this folder: 95% intervals, paired tests and calibration (`python scripts/run_stats.py S01_yolov8s_seg_1024 S07_unet_r34 S08_deeplabv3p_r34 S09_segformer_b2 S10_m0_convnext_unet --compare --name C04_segmenters_vs_yolo`).
- `breakdown.md` / `breakdown.json` in this folder: per class, per fold and fold model, errors, mask areas, comparison with the article (`python scripts/run_breakdown.py` with the same five runs and `--name C04_segmenters_vs_yolo`).
- The boundary metrics (NSD at 10 and 20 px, HD95) are each run's own `metrics.json` (keys `nsd10_fractured_mean`, `nsd20_fractured_mean`, `hd95_detected_median`), written by the same evaluation code (`src/femurseg/evaluation.py`).

**What differs.**
- **S01** is the article's final model, YOLOv8s-seg at 1024 px, trained as in the article. Its mask is the mask of the top detected instance.
- **S07, S08 and S09** are standard semantic segmenters at 768 × 512 with the shared Step 2 recipe (`Analysis/05_Step2_Baselines_Protocol.md`): U-Net and DeepLabV3+ with a ResNet34 encoder, and SegFormer with a MiT-B2 encoder. They label every pixel with one of 6 classes; the mask is the pixels with fracture probability ≥ 0.5.
- **S10 = M0** is the ConvNeXt-T classifier S06 plus a U-Net decoder that learns a binary fracture mask (the core model of both papers).
- All five use the same folds, the same decision rule and the same threshold rule (Youden on the pooled out-of-fold femurs). Masks are scored at full resolution (1024 × 1536) against the released masks.
- **Primary endpoint here:** Dice on fractured femurs, with a missed fracture counted as 0 (`Analysis/04`; 680 fractured femurs out of fold, 145 on the test set). Classification is reported as a secondary result.

## 1. Verdict: M0 outlines fractures best; out of fold it beats YOLO and every standard segmenter, on the test set it beats U-Net and DeepLabV3+ only

| Endpoint | S01 YOLOv8s-seg | S07 U-Net | S08 DeepLabV3+ | S09 SegFormer | S10 M0 |
|---|---|---|---|---|---|
| **OOF, 1,043 femurs:** Dice, fractured femurs, missed = 0 (primary) | .740 (.722–.755) | .748 (.734–.762) | .749 (.734–.763) | .766 (.753–.778) | **.777** (.764–.789) |
| Fractures missed (of 680) | 51 | 21 | 28 | **16** | 19 |
| NSD at 10 px, fractured femurs | not computed | .221 | .222 | .238 | **.248** |
| HD95 (px), median, detected fractures | 63.0 | 64.4 | 61.7 | 61.6 | **60.8** |
| Macro-F1 | .744 (.707–.779) | .800 (.766–.832) | .800 (.765–.830) | .817 (.784–.847) | **.825** (.793–.855) |
| **Test, 5 fold models, mean ± SD:** Dice, fractured femurs | .693 ± .016 | .698 ± .009 | .704 ± .005 | .725 ± .008 | **.734 ± .005** |
| Macro-F1 | .791 ± .018 | .770 ± .027 | .784 ± .022 | **.797 ± .020** | .776 ± .017 |
| **Test, 5-model ensemble, 210 femurs:** Dice, fractured femurs | .725 (.694–.754) | .711 (.678–.740) | .698 (.661–.734) | .737 (.711–.760) | **.744** (.718–.766) |
| Fractures missed (of 145) | 5 | 5 | 10 | 2 | **1** |
| NSD at 10 px | not computed | .179 | .185 | **.202** | **.202** |
| HD95 (px), median | 68.3 | 68.7 | **65.4** | 68.7 | 65.8 |
| Macro-F1 | **.845** (.780–.898) | .799 (.714–.863) | .778 (.696–.847) | .804 (.721–.870) | .779 (.695–.849) |

Intervals: 95% cluster bootstrap over x-rays. NSD and HD95 are descriptive (no test was planned for them).

**Paired Dice tests** (Wilcoxon signed-rank, Pratt, on the fractured femurs, missed = 0; first run − second run; Holm within each pair over its five endpoints):

| Pair | OOF difference | OOF p (Holm) | Test-ensemble difference | Test-ensemble p (Holm) |
|---|---|---|---|---|
| M0 − YOLO | **+.037** | **< .0001** | +.018 | .36 |
| M0 − SegFormer | **+.011** | **.0003** | +.006 | .44 |
| M0 − U-Net | **+.029** | **< .0001** | **+.033** | **< .0001** |
| M0 − DeepLabV3+ | **+.028** | **< .0001** | **+.045** | **.0005** |
| SegFormer − YOLO | **+.026** | **.014** | +.012 | 1.00 |
| SegFormer − U-Net | **+.018** | **.002** | **+.027** | **.012** |
| SegFormer − DeepLabV3+ | **+.017** | **.013** | +.039 | .30 |
| U-Net − YOLO | +.008 | .61 | −.015 | .77 |
| DeepLabV3+ − YOLO | +.009 | .64 | −.027 | .68 |
| U-Net − DeepLabV3+ | −.001 | 1.00 | +.012 | .94 |

1. **RQ1 is answered "yes" out of fold, for M0 and SegFormer.**
   - M0's U-Net decoder on ConvNeXt-T has the highest Dice in every view: .777 out of fold, .734 ± .005 per test model, .744 for the test ensemble.
   - Out of fold, M0 is significantly better than all four other models (+.011 to +.037). SegFormer is the best standard segmenter and also beats YOLO (+.026).
2. **On the 210 test femurs, M0's lead over YOLO (+.018) and SegFormer (+.006) is not significant.** It stays significant over U-Net and DeepLabV3+ (+.033 and +.045). The test set has only 145 fractured femurs, and its neck masks are drawn differently (§4).
3. **The two ResNet34 segmenters (U-Net, DeepLabV3+) are not better than YOLO** in either view, and they are the same as each other.
4. **M0 and SegFormer lead in all three views.** The order of the other three changes between views (test models: DeepLabV3+ .704, U-Net .698, YOLO .693; test ensemble: YOLO .725, U-Net .711, DeepLabV3+ .698), and none of their differences is significant.

## 2. Where the Dice differences come from

**Mostly from fractures that are missed, which count as Dice 0.**

| Dice on the femurs each model detects (its own set) | S01 | S07 | S08 | S09 | S10 |
|---|---|---|---|---|---|
| OOF | **.800** | .772 | .781 | .784 | .799 |
| Test, 5 fold models | .741 ± .009 | .718 ± .009 | .733 ± .006 | .735 ± .005 | **.746 ± .002** |
| Test ensemble | **.751** | .736 | .750 | .747 | .749 |

- YOLO misses 51 of the 680 fractured femurs out of fold (7.5%), the segmenters 16–28. Most of YOLO's Dice deficit (.740 against .777) is these misses.
- On the fractures it does find, YOLO's masks are as good as M0's (.800 against .799) and better than those of the three standard segmenters (.772–.784). This comparison favours YOLO, because its own set leaves out the 51 femurs it misses, which are likely the harder ones.
- On the test set YOLO's ensemble misses only 5 of 145 fractures (M0: 1), which is why the test difference is small.

**Per class** (OOF pooled; test, 5 fold models mean ± SD; a missed fracture counts as 0; `breakdown.md`):

| Class (OOF / test femurs) | S01 | S07 | S08 | S09 | S10 |
|---|---|---|---|---|---|
| 2AB Neck Garden I-II (57 / 16), OOF | .534 | .588 | .556 | .592 | **.624** |
| test | .524 ± .036 | .544 ± .041 | .557 ± .058 | **.612 ± .013** | .578 ± .032 |
| 2CD Neck Garden III-IV (254 / 41), OOF | .751 | .748 | .744 | .755 | **.773** |
| test | .677 ± .013 | .679 ± .017 | .693 ± .005 | .686 ± .011 | **.699 ± .009** |
| 3AB Pertroch Evans I-II (134 / 34), OOF | .721 | .725 | .734 | .757 | **.763** |
| test | .713 ± .039 | .698 ± .029 | .700 ± .019 | .740 ± .009 | **.750 ± .008** |
| 3CDE Pertroch Evans III-V (199 / 42), OOF | .822 | .825 | .832 | .841 | **.852** |
| test | .781 ± .019 | .816 ± .005 | .812 ± .017 | .822 ± .016 | **.836 ± .004** |
| 4 Subtrochanteric (36 / 12), OOF | .601 | .662 | .682 | **.728** | .682 |
| test | .608 ± .024 | .558 ± .038 | .577 ± .024 | .631 ± .021 | **.655 ± .039** |

- **M0 is best or second best in every class.** Its gain over YOLO is largest where YOLO misses the most fractures: Garden I-II (OOF +.090; YOLO detects 74% of them, the segmenters 82–93%) and subtrochanteric (+.081; YOLO detects 81%).
- **SegFormer is the best model for subtrochanteric fractures out of fold** (.728) and for Garden I-II on the test set (.612). Both classes are small (36 and 57 femurs out of fold, 12 and 16 on the test set).
- **Garden I-II and subtrochanteric fractures stay the hardest to outline** for every model (Dice .52–.73), as they are the hardest to classify (C03).

## 3. Boundary metrics (descriptive)

| Metric | S01 | S07 | S08 | S09 | S10 |
|---|---|---|---|---|---|
| NSD at 2 px, OOF | .053 | .050 | .049 | .054 | **.057** |
| NSD at 10 px, OOF | not computed | .221 | .222 | .238 | **.248** |
| NSD at 20 px, OOF | not computed | .415 | .413 | .438 | **.452** |
| NSD at 10 px, test ensemble | not computed | .179 | .185 | **.202** | **.202** |
| HD95 (px), median, detected, OOF | 63.0 | 64.4 | 61.7 | 61.6 | **60.8** |
| HD95 (px), test, 5 fold models | 72.3 ± 1.4 | 72.4 ± 1.4 | 70.6 ± 0.8 | 71.8 ± 1.5 | **68.4 ± 1.0** |

- **The order is the same as for Dice:** M0 ≥ SegFormer > the ResNet34 segmenters. The HD95 medians of all five models lie within 4 px of each other (at 1024 × 1536 px), so no model draws clearly sharper borders.
- **All models outline the fracture region, not a sharp fracture line.** Only about 5% of the boundary lies within 2 px of the released boundary, and 22–25% within 10 px. The released masks are large regions (median 30,000–150,000 px), so a boundary tolerance of a few pixels is very strict for them.
- **S01 has NSD at 2 px only.** It was scored before NSD at 10 and 20 px were added to the evaluation, and its masks are not in the repository. At 2 px it is in the same range as the others (.053 OOF, .043 test ensemble against .049–.057 and .039–.047). A re-score of S01's saved masks would be needed only if a reviewer asks for NSD at 10 px for YOLO.

## 4. The test set: neck masks are drawn smaller, so every model's test Dice drops for neck fractures

| Class | True mask, training femurs (median px) | True mask, test femurs | Test / training | Predicted / true area on the test set, 5 models (S01, S07, S08, S09, S10) | Same out of fold |
|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 45,963 | 29,797 | **0.65** | 2.04, 1.74, 1.71, 1.73, 1.89 | 0.76–1.06 |
| 2CD Neck Garden III-IV | 75,612 | 45,844 | **0.61** | 1.78, 1.80, 1.76, 1.70, 1.71 | 0.87–0.99 |
| 3AB Pertroch Evans I-II | 85,540 | 88,794 | 1.04 | 1.00–1.16 | 0.93–1.06 |
| 3CDE Pertroch Evans III-V | 135,306 | 120,095 | 0.89 | 1.00–1.06 | 0.93–0.98 |
| 4 Subtrochanteric | 126,204 | 149,960 | 1.19 | 0.69–0.95 | 0.95–1.04 |

- All five models learned the training masks and draw neck fractures 1.7–2.0 times larger than the test masks. Their neck Dice drops from .744–.773 out of fold to .677–.699 on the test set (2CD), while pertrochanteric Dice changes little.
- This is a property of the test labels (the article says a committee reviewed them), not of one model, and it was already seen for YOLO in C01. It limits every test Dice the papers report. As decided in C01, Dice is reported per location, and the paper states the shift.
- The test set must not be used to correct it (Report 04 §3.4).

## 5. Classification (secondary here)

| | S01 | S07 | S08 | S09 | S10 |
|---|---|---|---|---|---|
| OOF fracture AUC | .975 | .990 | .990 | .994 | **.995** |
| OOF 6-class accuracy | 83.2% | 86.3% | 86.2% | 87.9% | **88.4%** |
| OOF errors: detection / location / grade | 62 / 37 / 76 | 32 / 38 / 73 | 41 / 36 / 67 | 27 / 34 / 65 | **22 / 34 / 65** |
| Test, 5 fold models: 6-class accuracy | 84.6 ± 2.1% | 84.6 ± 1.6% | 85.5 ± 1.9% | **86.8 ± 0.8%** | 86.0 ± 0.5% |
| Test ensemble: 6-class accuracy | **88.1%** | 86.7% | 86.2% | **88.1%** | 86.7% |
| Test ensemble errors: detection / location / grade | 6 / **2** / 17 | 5 / 5 / 18 | 10 / 5 / **14** | 2 / 7 / 16 | **1** / 7 / 20 |

- **Out of fold, every segmenter classifies better than YOLO:** 6-class accuracy +3.0 to +5.2 points, macro-F1 +.055 to +.081, fracture AUC +.014 to +.019, and fewer detection errors (all Holm p ≤ .017).
- **Among the segmenters there is little difference.** M0 is more accurate than U-Net and DeepLabV3+ out of fold (+2.1 and +2.2 points, Holm p = .046 and .035) and detects fractures better than DeepLabV3+ (Holm p = .002). M0 and SegFormer do not differ on any classification endpoint (smallest Holm p = 1.00).
- **On the test set no segmenter differs from YOLO in classification** (smallest Holm p = .31). DeepLabV3+ misses more fractures than SegFormer and M0 (detection −.038 and −.043, Holm p = .039 and .016).
- **The pattern of C03 holds:** YOLO makes the most detection errors out of fold and the fewest location errors on the test set (2 in the ensemble against 5–7; test location accuracy per model .966 against .948–.956). Grade errors are similar for all models (65–76 OOF, 14–20 test). Per test model, SegFormer has the highest accuracy (86.8 ± 0.8%) and M0 the smallest spread (86.0 ± 0.5%).

## 6. Calibration (reported only; `stats.md`)

| Run | Temperature | OOF ECE before → after | Test-ensemble ECE before → after | Test fold models, ECE before → after |
|---|---|---|---|---|
| S01 YOLOv8s-seg | not applicable | .079 | .083 | .082 |
| S07 U-Net | 2.22 | .115 → .035 | .083 → .099 | .140 → .062 |
| S08 DeepLabV3+ | 2.36 | .115 → .031 | .075 → .098 | .124 → .055 |
| S09 SegFormer | 2.39 | .096 → .050 | .074 → .084 | .115 → .064 |
| S10 M0 | 3.14 | .108 → .040 | .103 → .064 | .131 → .062 |

- All four networks are over-confident as single models, and one temperature fitted on the OOF femurs removes most of it, also for each fold model on the test set.
- **For the test ensembles of S07–S09 the same temperature makes the ECE worse** (.074–.083 → .084–.099). Averaging five models already softens their probabilities, so the OOF temperature over-corrects. M0's ensemble stays over-confident before scaling (.103) and improves (.064).
- The temperature never changes a decision. Paper A reports calibration for the fold models and the ensemble separately (Analysis/06 §6).

## 7. Decision

- **RQ1:** a dedicated segmentation decoder outlines fractures better than YOLOv8-seg out of fold (M0 +.037 Dice, SegFormer +.026). On the 210 test femurs the difference is smaller and not significant (M0 +.018, Holm p = .36). The papers state both results, and that the gain comes mainly from fewer missed fractures.
- **M0 stays the base model of both papers.** It has the best Dice in every view and is significantly better than all three standard segmenters out of fold. Its classification equals SegFormer's (no significant difference) and is above YOLO's out of fold.
- **SegFormer (S09) is the strongest standard segmenter.** It enters Paper A's main table (C09) as planned (A6 against S01 and S04–S10), and it is the segmentation baseline to beat in the Dice columns.
- **U-Net and DeepLabV3+ (S07, S08) stay in the baseline table only.** They are no better than YOLO on Dice.
- **For both papers:** Dice is reported per location because of the test-set neck shift (§4). Garden I-II and subtrochanteric fractures are the hardest to outline as well as to classify.
- **With C04, Step 2 is complete:** S01–S10 are filed and analysed (C01–C05). This is milestone **M1, baseline table v1** (planned for 13 Dec 2026): `results/analysis/M1_baseline_table/`.
