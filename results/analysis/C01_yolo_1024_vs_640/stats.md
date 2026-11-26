# Statistics: S01_yolov8s_seg_1024, S02_yolov8s_seg_640

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

## S02_yolov8s_seg_640: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.975 | 0.967–0.983 |
| fracture_sensitivity | 0.935 | 0.917–0.953 |
| fracture_specificity | 0.970 | 0.950–0.986 |
| accuracy_6class | 0.826 | 0.802–0.848 |
| macro_f1 | 0.751 | 0.715–0.783 |
| balanced_accuracy | 0.765 | 0.728–0.799 |
| kappa | 0.772 | 0.742–0.799 |
| dice_fractured_mean | 0.746 | 0.730–0.762 |
| dice_detected_mean | 0.798 | 0.791–0.805 |
| garden12_recall | 0.544 | 0.411–0.673 |
| auc_fracture (DeLong) | 0.975 | 0.967–0.984 |
| accuracy_6class (exact) | 862/1043 = 0.826 | 0.802–0.849 |

## S02_yolov8s_seg_640: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.986–0.999 |
| fracture_sensitivity | 0.972 | 0.944–0.993 |
| fracture_specificity | 0.938 | 0.873–0.986 |
| accuracy_6class | 0.838 | 0.783–0.885 |
| macro_f1 | 0.787 | 0.707–0.849 |
| balanced_accuracy | 0.774 | 0.701–0.844 |
| kappa | 0.795 | 0.728–0.852 |
| dice_fractured_mean | 0.731 | 0.699–0.758 |
| dice_detected_mean | 0.752 | 0.731–0.771 |
| garden12_recall | 0.562 | 0.308–0.818 |
| auc_fracture (DeLong) | 0.994 | 0.988–1.000 |
| accuracy_6class (exact) | 176/210 = 0.838 | 0.781–0.885 |

## S01_yolov8s_seg_1024 vs S02_yolov8s_seg_640: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.9405 | 1.0000 |
| accuracy_6class | McNemar | +0.006 | 0.6368 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.007 | 0.4011 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.007 | 0.6841 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.006 | 0.4985 | 1.0000 |

## S01_yolov8s_seg_1024 vs S02_yolov8s_seg_640: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 0.9611 | 1.0000 |
| accuracy_6class | McNemar | +0.043 | 0.0490 | 0.2020 |
| fracture_detection | McNemar (binary fracture decision) | +0.010 | 0.7266 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.058 | 0.0404 | 0.2020 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.005 | 0.5577 | 1.0000 |
