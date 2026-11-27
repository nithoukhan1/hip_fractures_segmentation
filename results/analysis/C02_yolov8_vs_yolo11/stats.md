# Statistics: S01_yolov8s_seg_1024, S03_yolo11s_seg_1024

95% intervals: cluster bootstrap over x-rays (2000 resamples, seed 2026); AUC also DeLong; 6-class accuracy also exact (Clopper-Pearson). Decisions use each run's OOF threshold.

## S01_yolov8s_seg_1024: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.975 | 0.967–0.983 |
| fracture_sensitivity | 0.925 | 0.905–0.943 |
| fracture_specificity | 0.970 | 0.951–0.986 |
| accuracy_6class | 0.832 | 0.810–0.854 |
| macro_f1 | 0.744 | 0.707–0.779 |
| balanced_accuracy | 0.738 | 0.698–0.778 |
| kappa | 0.779 | 0.749–0.807 |
| dice_fractured_mean | 0.740 | 0.722–0.755 |
| dice_detected_mean | 0.800 | 0.792–0.807 |
| garden12_recall | 0.509 | 0.379–0.641 |
| auc_fracture (DeLong) | 0.975 | 0.967–0.984 |
| accuracy_6class (exact) | 868/1043 = 0.832 | 0.808–0.854 |

## S01_yolov8s_seg_1024: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.986–0.999 |
| fracture_sensitivity | 0.966 | 0.932–0.993 |
| fracture_specificity | 0.985 | 0.949–1.000 |
| accuracy_6class | 0.881 | 0.836–0.923 |
| macro_f1 | 0.845 | 0.780–0.898 |
| balanced_accuracy | 0.840 | 0.781–0.894 |
| kappa | 0.849 | 0.792–0.900 |
| dice_fractured_mean | 0.725 | 0.694–0.754 |
| dice_detected_mean | 0.751 | 0.731–0.770 |
| garden12_recall | 0.500 | 0.235–0.750 |
| auc_fracture (DeLong) | 0.994 | 0.988–1.000 |
| accuracy_6class (exact) | 185/210 = 0.881 | 0.829–0.921 |

## S03_yolo11s_seg_1024: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.974 | 0.963–0.983 |
| fracture_sensitivity | 0.928 | 0.909–0.947 |
| fracture_specificity | 0.956 | 0.932–0.975 |
| accuracy_6class | 0.822 | 0.798–0.844 |
| macro_f1 | 0.719 | 0.678–0.754 |
| balanced_accuracy | 0.717 | 0.679–0.754 |
| kappa | 0.765 | 0.735–0.793 |
| dice_fractured_mean | 0.743 | 0.726–0.759 |
| dice_detected_mean | 0.800 | 0.793–0.808 |
| garden12_recall | 0.386 | 0.267–0.517 |
| auc_fracture (DeLong) | 0.974 | 0.964–0.983 |
| accuracy_6class (exact) | 857/1043 = 0.822 | 0.797–0.844 |

## S03_yolo11s_seg_1024: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.985–1.000 |
| fracture_sensitivity | 0.972 | 0.945–0.993 |
| fracture_specificity | 0.969 | 0.921–1.000 |
| accuracy_6class | 0.852 | 0.803–0.899 |
| macro_f1 | 0.801 | 0.729–0.860 |
| balanced_accuracy | 0.801 | 0.734–0.863 |
| kappa | 0.813 | 0.752–0.871 |
| dice_fractured_mean | 0.730 | 0.699–0.756 |
| dice_detected_mean | 0.750 | 0.730–0.770 |
| garden12_recall | 0.438 | 0.182–0.692 |
| auc_fracture (DeLong) | 0.994 | 0.987–1.000 |
| accuracy_6class (exact) | 179/210 = 0.852 | 0.797–0.897 |

## S01_yolov8s_seg_1024 vs S03_yolo11s_seg_1024: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.7915 | 1.0000 |
| accuracy_6class | McNemar | +0.011 | 0.3426 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.003 | 0.7660 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.025 | 0.1910 | 0.9548 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.003 | 0.9158 | 1.0000 |

## S01_yolov8s_seg_1024 vs S03_yolo11s_seg_1024: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 0.7972 | 1.0000 |
| accuracy_6class | McNemar | +0.029 | 0.1460 | 0.7199 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.044 | 0.1440 | 0.7199 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.004 | 0.8598 | 1.0000 |
