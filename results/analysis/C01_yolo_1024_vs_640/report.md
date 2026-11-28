# YOLO resolution analysis: What the YOLO baselines YOLOv8s-seg (1024 px) and YOLOv8s-seg at 640 px show

*Written 2026-11-26 from the filed runs `results/S01_yolov8s_seg_1024/` and `results/S02_yolov8s_seg_640/` (register lines 3 and 4). Every number below comes from the two files in this folder:*
- `stats.md` / `stats.json`: 95% intervals and paired YOLOv8s-seg vs YOLOv8s-seg at 640 px tests (`python scripts/run_stats.py S01_yolov8s_seg_1024 S02_yolov8s_seg_640 --compare --name C01_yolo_1024_vs_640`).
- `breakdown.md` / `breakdown.json`: per class, mean ± SD over folds and fold models, errors, areas, comparison with the article (`python scripts/run_breakdown.py S01_yolov8s_seg_1024 S02_yolov8s_seg_640 --name C01_yolo_1024_vs_640`).

Article = González et al., Insights Imaging 2025;16:86. Its numbers are in `src/femurseg/reference.py` (Table 2, Table 3 and the Fig. 3 confusion matrix, checked by `tests/test_reference.py`).

## 1. Verdict: the article's classification results are reproduced; its Dice is not quite

| Test set (210 femurs) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, 5-model ensemble |
|---|---|---|---|---|
| **YOLOv8s-seg vs article small @1024:** fracture AUC | .986 ± .008 | .985 ± .005 | .981 | .994 (DeLong 95% CI .988–1.000) |
| 6-class accuracy | 84.4 ± 2.5% | 84.6 ± 2.1% | 86.2% (181/210) | **88.1% (185/210; exact CI 82.9–92.1)** |
| Dice (article: definition not stated; ours: detected fractures) | .761 ± .008 | .741 ± .009 | .77 | .751 |
| **YOLOv8s-seg at 640 px vs article small @640:** fracture AUC | .985 ± .006 | .979 ± .007 | | .994 |
| 6-class accuracy | 82.0 ± 3.0% | 82.2 ± 2.1% | | 83.8% |
| Dice | .780 ± .011 | .741 ± .002 | | .752 |

1. **Classification matches the article** at both sizes, although each of our fold models trains on about 834 femurs (4 of 5 folds) and the article's on all 1,043 plus a 201-femur validation set.
2. **The article's final model (86.2%) is one selected run.** Applying its selection rule to our folds (keep the model with the best accuracy on its own held-out fold, no test data) picks fold 1, which scores 83.8% on the test set. A single selected run is within the normal spread (84.6 ± 2.1%), so we report all 5 models and the ensemble rather than one.
3. **The 5-model ensemble is our strongest baseline:** 88.1% accuracy and 25 errors, against 29 for the article's final model (detection 6 vs 8, location 2 vs 6, grade 17 vs 15). New methods must beat this.
4. **Dice is about .02 (1024) and .04 (640) lower than the article's**, and the article's finding that 640 px gives better masks than 1024 px does not hold for us (YOLOv8s-seg vs YOLOv8s-seg at 640 px Dice: p = .50 OOF, p = .56 test). We cannot pin the cause without their code. Candidates: less training data per model, final weights instead of a best-validation checkpoint, and the unstated Dice definition. The same predictions give .69 to .78 under common definitions (`breakdown.md`, "Dice definitions"). The spread of our per-femur Dice (.52–.99, mean ± 1.96 SD) is close to the article's ".56–.98", so its Dice probably averages detected fractures, as ours does.

## 2. Where the classification errors are

Test recall per class (YOLOv8s-seg; ensemble in brackets; article final model from its Fig. 3):

| Class | n test | Article final | Ours, 5 models mean ± SD | OOF pooled (n = 1,043) |
|---|---|---|---|---|
| No fracture | 65 | .985 | .978 ± .008 (.985) | .970 |
| Neck Garden I–II | 16 | .500 | **.438 ± .117** (.500) | **.509** (95% CI .379–.641) |
| Neck Garden III–IV | 41 | .951 | .961 ± .028 (.976) | .831 |
| Pertroch Evans I–II | 34 | .882 | .888 ± .060 (.971) | .716 |
| Pertroch Evans III–V | 42 | .738 | **.681 ± .087** (.690) | .794 |
| Subtrochanteric | 12 | .750 | .733 ± .124 (.917) | .611 |

1. **Grade, not location, is the problem.** When a fracture is found, its location is right in 96.6 ± 1.2% of cases, but the grade is right in only 86.1 ± 2.9% of those (test, fold models). 17 of the ensemble's 25 errors are grade errors.
2. The two weak spots are **Garden I–II called Garden III–IV** (6 of 16 in the ensemble; 10 of 57 OOF) and **Evans III–V called Evans I–II** (10 of 42 in the ensemble; 24 of 199 OOF).
3. Rare classes are unstable across folds: Garden I–II recall varies ± .215 and subtrochanteric ± .240 between held-out folds (OOF, SD over 5 folds). This is why the papers use pooled OOF predictions (n = 1,043) with cluster-bootstrap intervals, not single folds.
4. Some classes score much higher on the test set than OOF (Garden III–IV .96 vs .83; Evans I–II .89 vs .72). The article says its test labels were reviewed by a committee; the training labels were not. Cleaner test labels would explain this, but it cannot be proven from the data.

→ Paper A (SLG-Net: region, then location, then grade) targets exactly points 1 and 2. Paper B (RefFrac) targets Garden I–II, the class with the lowest recall.

## 3. Segmentation per class

Dice on fractured femurs (missed = 0), YOLOv8s-seg, test fold models mean ± SD: Garden I–II **.52 ± .04**, Garden III–IV .68 ± .01, Evans I–II .71 ± .04, Evans III–V .78 ± .02, subtrochanteric .61 ± .02 (OOF: .53, .75, .72, .82, .60). Undisplaced neck fractures are the hardest to find and to outline.

Boundaries are far off: the median HD95 of detected fractures is 63 px OOF and 72 px on test (image height 1,536 px). NSD at 2 px tolerance is about .05 and says nothing at this scale.

## 4. Finding: the test masks of neck fractures are drawn about 35–40% smaller

| Class | True mask, train (median px) | True mask, test | Test / train | YOLOv8s-seg predicted, OOF | YOLOv8s-seg predicted, test |
|---|---|---|---|---|---|
| Neck Garden I–II | 45,963 | 29,797 | **0.65** | 54,558 | 57,046 |
| Neck Garden III–IV | 75,612 | 45,844 | **0.61** | 75,926 | 76,820 |
| Pertroch Evans I–II | 85,540 | 88,794 | 1.04 | 93,944 | 96,513 |
| Pertroch Evans III–V | 135,306 | 120,095 | 0.89 | 131,154 | 130,396 |
| Subtrochanteric | 126,204 | 149,960 | 1.19 | 144,031 | 125,186 |

1. The models draw the same size of neck mask on test femurs as on held-out training femurs (test / OOF predicted area 1.01–1.05). Measured against the smaller test masks, they over-segment neck fractures by a factor of about 2 (predicted / true area: 2.04 for Garden I–II, 1.78 for Garden III–IV; about 1 for the other classes, and 0.99 overall OOF).
2. This explains most of the drop in Dice from OOF to test (.800 → .741, detected fractures): Garden III–IV drops from .79 to .68 and Garden I–II from .73 to .64, but Evans III–V only from .84 to .82 and Evans I–II from .79 to .76.
3. So the drop is an **annotation difference between the training and the (committee-reviewed) test masks**, not overfitting. It caps the test Dice of neck fractures for any model trained on these labels, the article's included.
4. Consequences for both papers:
   - Report Dice per location, OOF and test side by side, and state this difference in the data section (measured, no clinical claim).
   - Never correct for it with anything learned from the test set (for example rescaling masks to the test size).
   - Methods are compared on the same test set, so comparisons stay fair; only the absolute test Dice of neck fractures is lower.

## 5. 1024 px versus 640 px (YOLOv8s-seg vs YOLOv8s-seg at 640 px, paired tests on the same femurs)

| Scope | Accuracy | Macro-F1 | Dice (fractured, missed = 0) | AUC |
|---|---|---|---|---|
| OOF, 1,043 femurs | 83.2 vs 82.6% (McNemar p = .64) | .744 vs .751 (p = .68) | .740 vs .746 (p = .50) | .975 vs .975 (p = .94) |
| Test ensemble, 210 femurs | 88.1 vs 83.8% (p = .049; Holm .20) | .845 vs .787 (p = .040; Holm .20) | .725 vs .731 (p = .56) | .994 vs .994 (p = .96) |

There is no significant difference after the Holm correction. YOLOv8s-seg at 640 px takes half the time (81 vs 176 min). **Decision:** YOLOv8s-seg (1024 px) stays the reference baseline because it is the article's final setting and scores best on the test set. Resolution is not a lever for YOLO on this data, so YOLO11s-seg runs at 1024 px for a like-for-like comparison with YOLOv8s-seg.

## 6. Runtime on Kaggle (2 × T4, two folds at a time)

YOLOv8s-seg took 176 min (folds 54–59 min of training each), YOLOv8s-seg at 640 px 81 min (23–27 min). The plan's estimates were 5–6 h and 2–3 h. Steps 0–1 used about 4.4 h of session time in total.

## 7. Decisions

1. Baseline values for the papers: YOLOv8s-seg's fold models (mean ± SD) and its 5-model ensemble on the test set, and YOLOv8s-seg's pooled OOF for development comparisons. The article's own numbers are quoted next to them.
2. Primary-endpoint baselines (YOLOv8s-seg, OOF pooled / test fold models / test ensemble):
   - 6-class macro-F1 (Paper A): .744 / .791 ± .018 / .845
   - Dice on fractured femurs, missed = 0 (Papers A and B): .740 / .693 ± .016 / .725
   - Garden I–II recall (Paper B): .509 / .438 ± .117 / .500
3. Dice is reported per location as well, OOF and test side by side.
4. NSD at 2 px is uninformative. The Step 2 training code records NSD at wider tolerances too (10 and 20 px, about 0.7% and 1.3% of the image height). YOLOv8s-seg and YOLOv8s-seg at 640 px can be re-scored from their saved masks on a CPU if needed.
5. YOLO11s-seg at 1024 px, same protocol as YOLOv8s-seg.
