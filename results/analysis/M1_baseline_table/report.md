# Milestone 1 (baseline table): baseline table v1 (ten baselines)

*Written 2026-11-27, when the last baseline (SegFormer-B2) was filed. Planned date in `Analysis/04` section 7: 13 Dec 2026.*

This table puts the ten baseline runs of Steps 1 and 2 side by side. All of them were trained on the same 5 folds (`splits/folds_v1.csv`) and scored by the same code with the same decision rule and threshold rule (Youden on the pooled out-of-fold femurs, applied unchanged to the test set).

*Every number below comes from the files in this folder:*
- `stats.md` / `stats.json`: 95% intervals (cluster bootstrap over x-rays) and calibration (`python scripts/run_stats.py S01_yolov8s_seg_1024 S02_yolov8s_seg_640 S03_yolo11s_seg_1024 S04_densenet121 S05_efficientnet_b4 S06_convnext_tiny S07_unet_r34 S08_deeplabv3p_r34 S09_segformer_b2 S10_m0_convnext_unet --name M1_baseline_table`).
- `breakdown.md` / `breakdown.json`: per class, per fold and fold model, errors, comparison with the article (`python scripts/run_breakdown.py` with the same ten runs and `--name M1_baseline_table`).
- The paired tests are not repeated here: they are in the planned the baseline analyses (`results/analysis/`), each with its own Holm correction.

| Run | Model | Input | Measured run time | Register line | Analyses |
|---|---|---|---|---|---|
| YOLOv8s-seg | YOLOv8s-seg (the article's final model) | 1024 px | 2.9 h | 3 | First four baseline analyses |
| YOLOv8s-seg at 640 px | YOLOv8s-seg | 640 px | 1.4 h | 4 | YOLO resolution analysis |
| YOLO11s-seg | YOLO11s-seg | 1024 px | 3.0 h | 6 | YOLO version analysis |
| DenseNet-121 | DenseNet121 classifier | 768 × 512 | 1.4 h | 9 | Classifier analysis |
| EfficientNet-B4 | EfficientNet-B4 classifier | 768 × 512 | 1.9 h | 10 | Classifier analysis |
| ConvNeXt-T classifier | ConvNeXt-T classifier | 768 × 512 | 1.5 h | 7 | Classifier analysis, the mask-task analysis |
| ResNet-34 U-Net | U-Net (ResNet34) segmenter | 768 × 512 | 1.5 h | 11 | Segmenter analysis |
| ResNet-34 DeepLabV3+ | DeepLabV3+ (ResNet34) segmenter | 768 × 512 | 1.2 h | 12 | Segmenter analysis |
| SegFormer-B2 | SegFormer (MiT-B2) segmenter | 768 × 512 | 2.8 h | 18 | Segmenter analysis |
| ConvNeXt U-Net | **ConvNeXt U-Net**: ConvNeXt-T + U-Net decoder | 768 × 512 | 2.3 h | 8 | Segmenter analysis, the mask-task analysis |

Run times are for 5 folds and the test set on Kaggle's T4 x2.

## 1. Out of fold: 1,043 femurs (701 x-rays), 95% intervals

| Run | Fracture AUC | 6-class accuracy | Macro-F1 (primary) | Garden I-II recall (57 femurs) | Dice, fractured femurs, missed = 0 (primary) | Temperature | ECE before → after |
|---|---|---|---|---|---|---|---|
| YOLOv8s-seg | .975 (.967–.983) | 83.2% (81.0–85.4) | .744 (.707–.779) | .509 (.379–.641) | .740 (.722–.755) | n/a | .079 |
| YOLOv8s-seg at 640 px | .975 (.967–.983) | 82.6% (80.2–84.8) | .751 (.715–.783) | .544 (.411–.673) | .746 (.730–.762) | n/a | .082 |
| YOLO11s-seg | .974 (.963–.983) | 82.2% (79.8–84.4) | .719 (.678–.754) | .386 (.267–.517) | .743 (.726–.759) | n/a | .073 |
| DenseNet-121 | .992 (.987–.996) | 86.8% (84.6–88.6) | .802 (.766–.833) | .579 (.446–.707) | no mask | 2.44 | .116 → .026 |
| EfficientNet-B4 | .990 (.985–.995) | 84.8% (82.6–86.8) | .755 (.717–.789) | .386 (.268–.517) | no mask | 3.27 | .122 → .025 |
| ConvNeXt-T classifier | .990 (.984–.995) | 86.4% (84.3–88.4) | .784 (.746–.816) | .474 (.339–.605) | no mask | 3.43 | .127 → .046 |
| ResNet-34 U-Net | .990 (.984–.994) | 86.3% (84.1–88.3) | .800 (.766–.832) | **.614** (.485–.739) | .748 (.734–.762) | 2.22 | .115 → .035 |
| ResNet-34 DeepLabV3+ | .990 (.985–.994) | 86.2% (84.0–88.2) | .800 (.765–.830) | .596 (.469–.721) | .749 (.734–.763) | 2.36 | .115 → .031 |
| SegFormer-B2 | .994 (.990–.997) | 87.9% (85.8–89.8) | .817 (.784–.847) | .579 (.450–.714) | .766 (.753–.778) | 2.39 | .096 → .050 |
| **ConvNeXt U-Net** | **.995** (.990–.998) | **88.4%** (86.4–90.2) | **.825** (.793–.855) | .544 (.412–.677) | **.777** (.764–.789) | 3.14 | .108 → .040 |

## 2. Test set: 5-model ensemble, 210 femurs (146 x-rays), 95% intervals

| Run | Fracture AUC | 6-class accuracy | Macro-F1 (primary) | Garden I-II recall (16 femurs) | Dice, fractured femurs, missed = 0 (primary) | ECE before → after |
|---|---|---|---|---|---|---|
| YOLOv8s-seg | .994 (.986–.999) | **88.1%** (83.6–92.3) | **.845** (.780–.898) | .500 (.235–.750) | .725 (.694–.754) | .083 |
| YOLOv8s-seg at 640 px | .994 (.986–.999) | 83.8% (78.3–88.5) | .787 (.707–.849) | **.562** (.308–.818) | .731 (.699–.758) | .067 |
| YOLO11s-seg | .994 (.985–1.000) | 85.2% (80.3–89.9) | .801 (.729–.860) | .438 (.182–.692) | .730 (.699–.756) | .100 |
| DenseNet-121 | .999 (.998–1.000) | 85.7% (80.4–90.6) | .760 (.670–.832) | .250 (.062–.500) | no mask | .083 → .055 |
| EfficientNet-B4 | .994 (.984–.999) | 81.0% (75.5–86.4) | .744 (.658–.814) | .312 (.083–.562) | no mask | .104 → .064 |
| ConvNeXt-T classifier | **1.000** (.999–1.000) | 84.8% (79.6–89.6) | .742 (.651–.816) | .250 (.050–.500) | no mask | .102 → .066 |
| ResNet-34 U-Net | .999 (.996–1.000) | 86.7% (81.6–90.8) | .799 (.714–.863) | .375 (.133–.625) | .711 (.678–.740) | .083 → .099 |
| ResNet-34 DeepLabV3+ | .999 (.998–1.000) | 86.2% (81.2–90.7) | .778 (.696–.847) | .250 (.056–.500) | .698 (.661–.734) | .075 → .098 |
| SegFormer-B2 | .999 (.996–1.000) | **88.1%** (83.4–92.3) | .804 (.721–.870) | .375 (.125–.625) | .737 (.711–.760) | .074 → .084 |
| **ConvNeXt U-Net** | .999 (.998–1.000) | 86.7% (82.0–91.2) | .779 (.695–.849) | .312 (.091–.556) | **.744** (.718–.766) | .103 → .064 |

## 3. Test set: the 5 fold models one by one, mean ± SD

| Run | Fracture AUC | 6-class accuracy | Macro-F1 | Dice, fractured femurs, missed = 0 |
|---|---|---|---|---|
| YOLOv8s-seg | .985 ± .005 | 84.6 ± 2.1% | .791 ± .018 | .693 ± .016 |
| YOLOv8s-seg at 640 px | .979 ± .007 | 82.2 ± 2.1% | .758 ± .036 | .696 ± .009 |
| YOLO11s-seg | .988 ± .006 | 81.1 ± 4.0% | .750 ± .062 | .693 ± .007 |
| DenseNet-121 | .997 ± .001 | 84.8 ± 2.0% | .749 ± .049 | no mask |
| EfficientNet-B4 | .990 ± .002 | 80.6 ± 1.5% | .736 ± .021 | no mask |
| ConvNeXt-T classifier | .998 ± .001 | 83.5 ± 1.6% | .732 ± .024 | no mask |
| ResNet-34 U-Net | .993 ± .005 | 84.6 ± 1.6% | .770 ± .027 | .698 ± .009 |
| ResNet-34 DeepLabV3+ | .993 ± .004 | 85.5 ± 1.9% | .784 ± .022 | .704 ± .005 |
| SegFormer-B2 | .997 ± .002 | **86.8 ± 0.8%** | **.797 ± .020** | .725 ± .008 |
| **ConvNeXt U-Net** | **.999 ± .001** | 86.0 ± 0.5% | .776 ± .017 | **.734 ± .005** |

## 4. What the baselines show (from the baseline analyses)

1. **The article is reproduced.** YOLOv8s-seg at 1024 px matches the article's test accuracy and AUC (YOLO resolution analysis), and so do its DenseNet121 and EfficientNet-B4 classifiers (classifier analysis). The article's Dice is not quite reached (about .02 lower; without the article's code the cause cannot be pinned down, the YOLO resolution analysis). Neither 640 px (YOLO resolution analysis) nor YOLO11 (YOLO version analysis) changes anything.
2. **Out of fold, the networks at 768 × 512 classify better than YOLO.** All except EfficientNet-B4 have a significantly higher 6-class accuracy, mainly because YOLO misses more fractures (classifier analysis, the segmenter analysis). On the 210 test femurs, YOLO's ensemble has the highest macro-F1 (.845), but no network differs from it significantly.
3. **ConvNeXt U-Net is the strongest baseline:** best out-of-fold macro-F1 (.825) and best Dice in every view (segmenter analysis). Learning the mask helps its classification (mask-task analysis: ConvNeXt U-Net vs ConvNeXt-T classifier).
4. **SegFormer (SegFormer-B2) is the strongest standard segmenter** and equals ConvNeXt U-Net in classification (segmenter analysis).
5. **Open problems for both papers:**
   - Grade errors within a location, Garden I-II above all: recall .39–.61 out of fold and .25–.56 on the test ensemble for every model.
   - Subtrochanteric fractures, the smallest class (36 + 12 femurs).
   - Calibration: every network is over-confident (ECE .096–.127 before scaling).
   - The test neck masks are drawn 35–40% smaller than the training ones (YOLO resolution analysis, the segmenter analysis), which lowers every model's test Dice for neck fractures.

**Numbers for Papers A and B to beat** (ConvNeXt U-Net): OOF macro-F1 .825 and Dice .777; test ensemble macro-F1 .779 and Dice .744 (YOLO's ensemble: macro-F1 .845).
