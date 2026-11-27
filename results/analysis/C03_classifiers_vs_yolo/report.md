# C03: Do image classifiers grade better than YOLO? Does our retraining reproduce the article's classifiers?

*Written 2026-11-27 from the filed runs:*
- `results/S01_yolov8s_seg_1024/` (register line 3);
- `results/S04_densenet121/` and `results/S05_efficientnet_b4/` (lines 9 and 10);
- `results/S06_convnext_tiny/` (line 7).

*Every number below comes from the files in this folder:*
- `stats.md` / `stats.json`: 95% intervals, paired tests and calibration (`python scripts/run_stats.py S01_yolov8s_seg_1024 S04_densenet121 S05_efficientnet_b4 S06_convnext_tiny --compare --name C03_classifiers_vs_yolo`).
- `breakdown.md` / `breakdown.json`: per class, per fold and fold model, errors, comparison with the article (`python scripts/run_breakdown.py` with the same four runs and `--name C03_classifiers_vs_yolo`).

**What differs.**
- **S01** is the article's final model, YOLOv8s-seg at 1024 px, trained as in the article. It decides from detected instances.
- **S04, S05 and S06** are plain image classifiers at 768 × 512 with the shared Step 2 recipe (`Analysis/05_Step2_Baselines_Protocol.md`). They are DenseNet121 (the article's best classifier), EfficientNet-B4 (an article baseline) and ConvNeXt-T (= A0).
- All four use the same folds, the same decision rule and the same threshold rule (Youden on the pooled out-of-fold femurs).
- The classifiers have no mask, so this analysis is about classification only.

## 1. Verdict: on the 1,043 out-of-fold femurs, DenseNet121 and ConvNeXt-T classify better than YOLO; on the 210 test femurs, YOLO's ensemble is ahead, not significantly

| Endpoint | S01 YOLOv8s-seg | S04 DenseNet121 | S05 EfficientNet-B4 | S06 ConvNeXt-T |
|---|---|---|---|---|
| **OOF, 1,043 femurs:** fracture AUC | .975 (.967–.983) | **.992** (.987–.996) | .990 (.985–.995) | .990 (.984–.995) |
| 6-class accuracy | 83.2% (81.0–85.4) | **86.8%** (84.6–88.6) | 84.8% (82.6–86.8) | 86.4% (84.3–88.4) |
| Macro-F1 (primary) | .744 (.707–.779) | **.802** (.766–.833) | .755 (.717–.789) | .784 (.746–.816) |
| Garden I-II recall (57 femurs) | .509 | .579 | .386 | .474 |
| **Test, 5 fold models, mean ± SD:** 6-class accuracy | 84.6 ± 2.1% | 84.8 ± 2.0% | 80.6 ± 1.5% | 83.5 ± 1.6% |
| Macro-F1 | **.791 ± .018** | .749 ± .049 | .736 ± .021 | .732 ± .024 |
| **Test, 5-model ensemble, 210 femurs:** fracture AUC | .994 | .999 | .994 | 1.000 |
| 6-class accuracy | **88.1%** (83.6–92.3) | 85.7% (80.4–90.6) | 81.0% (75.5–86.4) | 84.8% (79.6–89.6) |
| Macro-F1 | **.845** (.780–.898) | .760 (.670–.832) | .744 (.658–.814) | .742 (.651–.816) |

Intervals: 95% cluster bootstrap over x-rays.

**Paired tests against S01** (classifier − YOLO, Holm within each pair):

| | OOF accuracy | OOF macro-F1 | OOF AUC | Test-ensemble accuracy | Test-ensemble macro-F1 |
|---|---|---|---|---|---|
| S04 DenseNet121 | **+3.5 points (p = .002)** | **+.058 (p = .002)** | **+.016 (p < .001)** | −2.4 points (p = .95) | −.086 (p = .24) |
| S05 EfficientNet-B4 | +1.5 points (p = .38) | +.011 (p = .59) | **+.015 (p = .001)** | −7.1 points (p = .067) | −.101 (p = .070) |
| S06 ConvNeXt-T | **+3.2 points (p = .009)** | +.039 (p = .050) | **+.015 (p = .003)** | −3.3 points (p = .53) | −.103 (p = .12) |

1. **Out of fold, the classifiers find fractures better than YOLO.**
   - Fracture AUC is .990–.992 against .975, significant for all three.
   - The fracture vs no-fracture decision is better by 2.1–3.0 points (Holm p ≤ .012).
   - DenseNet121 and ConvNeXt-T are also better on 6-class accuracy (+3.5 and +3.2 points). On macro-F1, DenseNet121 is significantly better (+.058) and ConvNeXt-T borderline (+.039, p = .050).
2. **On the test set the order reverses for grading, but not significantly.**
   - YOLO's 5-model ensemble is the best test classifier (88.1%, macro-F1 .845).
   - Model by model, YOLO and DenseNet121 have the same accuracy (84.6% vs 84.8%), but YOLO's macro-F1 is higher (.791 vs .749).
   - Ensembling helps YOLO much more than the classifiers: +3.5 accuracy points against +0.9 for DenseNet121.
   - The 210 test femurs cannot resolve these differences (smallest Holm p = .067).
3. **The classifiers rank DenseNet121 ≥ ConvNeXt-T > EfficientNet-B4.**
   - DenseNet121 vs EfficientNet-B4: OOF macro-F1 +.047 (Holm p = .022).
   - DenseNet121 vs ConvNeXt-T: no difference on any endpoint (smallest p = .23).

## 2. Where the differences come from

| Errors (as in the article) | S01 OOF | S04 OOF | S05 OOF | S06 OOF | S01 test ens. | S04 test ens. | S05 test ens. | S06 test ens. |
|---|---|---|---|---|---|---|---|---|
| Detection (fracture vs none) | **62** | 32 | 40 | 31 | 6 | 4 | 12 | 3 |
| Location | 37 | 39 | 44 | 43 | **2** | 9 | 8 | 9 |
| Grade (right location) | 76 | 67 | 75 | 68 | 17 | 17 | 20 | 20 |
| **Total** | **175** | **138** | **159** | **142** | **25** | **30** | **40** | **32** |

- **YOLO's weakness is detection:** it misses or invents twice as many fractures out of fold (62 against 31–40). A femur with no detected instance gets "no fracture" (OOF fracture sensitivity .925 against .954–.966).
- **YOLO's strength is location on the test set:**
  - 2 location errors in the ensemble against 8–9;
  - test location accuracy .966 ± .012 per model against .931–.934;
  - subtrochanteric recall per test model .733 against .417–.567 (12 femurs).
  - Out of fold, location is the same (.941 against .932–.940). The test advantage may come from YOLO's 1024-px input and instance boxes, or from the small test set.
- **Grade errors are similar for all four** (67–76 OOF, 17–20 test). No model solves grading within a location: Garden I-II stays the weakest class (OOF recall .386–.579; test .275–.438 per model).

## 3. Reproduction of the article's classifiers (test set, Table 3)

| Model | Article (one model) | Ours, 5 fold models | Ours, ensemble |
|---|---|---|---|
| DenseNet121: fracture AUC | .993 (.985–1.000) | .997 ± .001 | .999 |
| DenseNet121: 6-class accuracy | 83.8% (78.1–88.5) | 84.8 ± 2.0% | 85.7% |
| EfficientNet-B4: fracture AUC | .990 (.981–.998) | .990 ± .002 | .994 |
| EfficientNet-B4: 6-class accuracy | 79.0% (72.9–84.3) | 80.6 ± 1.5% | 81.0% |

**Both are reproduced:** our values lie inside the article's 95% intervals and slightly above their point estimates. This is despite training each model on about 834 femurs rather than the article's 1,043 plus a 201-femur validation set. The article's ranking (DenseNet121 above EfficientNet-B4, YOLO at the top on the test set) also holds.

## 4. Calibration (reported only; `stats.md`)

| Run | Temperature | OOF ECE before → after | Test-ensemble ECE before → after |
|---|---|---|---|
| S01 YOLOv8s-seg | not applicable | .079 | .083 |
| S04 DenseNet121 | 2.44 | .116 → .026 | .083 → .055 |
| S05 EfficientNet-B4 | 3.27 | .122 → .025 | .104 → .064 |
| S06 ConvNeXt-T | 3.43 | .127 → .046 | .102 → .066 |

- All three classifiers are over-confident, and a single temperature fitted on the OOF femurs removes most of it.
- YOLO's confidences are better calibrated before any scaling. A temperature cannot be fitted for YOLO: 99 of its 1,043 OOF femurs get probability 0 for their true class.
- S05 has one OOF femur whose true class is written as probability 0, below the 8 decimals of the per-case file. `scripts/run_stats.py` now reads such a written 0 as 10⁻¹² for network runs (2026-11-27), as it already did for every other class.
  - S05's temperature lies between 3.19 and 3.27, depending on how that one femur is treated, and its OOF ECE after scaling between .025 and .026.
  - Every other value in this table is unchanged by the fix.

## 5. Decision

- **Image classifiers are kept as baselines in both papers.** DenseNet121 (S04) is the strongest plain classifier and joins Paper A's main table (C09) next to S01, S06 and M0.
- **ConvNeXt-T stays the backbone of M0, SLG-Net and RefFrac.**
  - It is as good as DenseNet121 here (no significant difference).
  - It is the backbone for which M0's mask task was shown to help (C05: M0 OOF macro-F1 .825, above every model in this table).
- **What the next models must fix is the same for all baselines:** grade errors within a location (Garden I-II above all), and calibration. YOLO's test-set location advantage is noted for C09.
- **C04** (the segmenters S07–S10 against S01, with the formal Dice test of M0) follows when S09 is filed.
