# Statistics: S01_yolov8s_seg_1024, S04_densenet121, S05_efficientnet_b4, S06_convnext_tiny

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

## S04_densenet121: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.992 | 0.987–0.996 |
| fracture_sensitivity | 0.962 | 0.947–0.975 |
| fracture_specificity | 0.983 | 0.969–0.995 |
| accuracy_6class | 0.868 | 0.846–0.886 |
| macro_f1 | 0.802 | 0.766–0.833 |
| balanced_accuracy | 0.787 | 0.748–0.822 |
| kappa | 0.825 | 0.798–0.850 |
| dice_fractured_mean | n/a | n/a–n/a |
| dice_detected_mean | n/a | n/a–n/a |
| garden12_recall | 0.579 | 0.446–0.707 |
| auc_fracture (DeLong) | 0.992 | 0.987–0.996 |
| accuracy_6class (exact) | 905/1043 = 0.868 | 0.846–0.888 |

## S04_densenet121: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.998–1.000 |
| fracture_sensitivity | 0.986 | 0.966–1.000 |
| fracture_specificity | 0.969 | 0.924–1.000 |
| accuracy_6class | 0.857 | 0.804–0.906 |
| macro_f1 | 0.760 | 0.670–0.832 |
| balanced_accuracy | 0.744 | 0.677–0.813 |
| kappa | 0.817 | 0.751–0.878 |
| dice_fractured_mean | n/a | n/a–n/a |
| dice_detected_mean | n/a | n/a–n/a |
| garden12_recall | 0.250 | 0.062–0.500 |
| auc_fracture (DeLong) | 0.999 | 0.998–1.000 |
| accuracy_6class (exact) | 180/210 = 0.857 | 0.802–0.901 |

## S05_efficientnet_b4: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.990 | 0.985–0.995 |
| fracture_sensitivity | 0.954 | 0.938–0.969 |
| fracture_specificity | 0.975 | 0.958–0.989 |
| accuracy_6class | 0.848 | 0.826–0.868 |
| macro_f1 | 0.755 | 0.717–0.789 |
| balanced_accuracy | 0.740 | 0.701–0.777 |
| kappa | 0.798 | 0.769–0.825 |
| dice_fractured_mean | n/a | n/a–n/a |
| dice_detected_mean | n/a | n/a–n/a |
| garden12_recall | 0.386 | 0.268–0.517 |
| auc_fracture (DeLong) | 0.990 | 0.985–0.995 |
| accuracy_6class (exact) | 884/1043 = 0.848 | 0.824–0.869 |

## S05_efficientnet_b4: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.984–0.999 |
| fracture_sensitivity | 0.924 | 0.877–0.966 |
| fracture_specificity | 0.985 | 0.951–1.000 |
| accuracy_6class | 0.810 | 0.755–0.864 |
| macro_f1 | 0.744 | 0.658–0.814 |
| balanced_accuracy | 0.720 | 0.648–0.791 |
| kappa | 0.755 | 0.684–0.823 |
| dice_fractured_mean | n/a | n/a–n/a |
| dice_detected_mean | n/a | n/a–n/a |
| garden12_recall | 0.312 | 0.083–0.562 |
| auc_fracture (DeLong) | 0.994 | 0.987–1.000 |
| accuracy_6class (exact) | 170/210 = 0.810 | 0.750–0.860 |

## S06_convnext_tiny: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.990 | 0.984–0.995 |
| fracture_sensitivity | 0.966 | 0.951–0.978 |
| fracture_specificity | 0.978 | 0.962–0.992 |
| accuracy_6class | 0.864 | 0.843–0.884 |
| macro_f1 | 0.784 | 0.746–0.816 |
| balanced_accuracy | 0.763 | 0.725–0.800 |
| kappa | 0.820 | 0.793–0.845 |
| dice_fractured_mean | n/a | n/a–n/a |
| dice_detected_mean | n/a | n/a–n/a |
| garden12_recall | 0.474 | 0.339–0.605 |
| auc_fracture (DeLong) | 0.990 | 0.985–0.996 |
| accuracy_6class (exact) | 901/1043 = 0.864 | 0.842–0.884 |

## S06_convnext_tiny: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 1.000 | 0.999–1.000 |
| fracture_sensitivity | 0.993 | 0.979–1.000 |
| fracture_specificity | 0.969 | 0.919–1.000 |
| accuracy_6class | 0.848 | 0.796–0.896 |
| macro_f1 | 0.742 | 0.651–0.816 |
| balanced_accuracy | 0.727 | 0.661–0.796 |
| kappa | 0.805 | 0.741–0.865 |
| dice_fractured_mean | n/a | n/a–n/a |
| dice_detected_mean | n/a | n/a–n/a |
| garden12_recall | 0.250 | 0.050–0.500 |
| auc_fracture (DeLong) | 1.000 | 0.999–1.000 |
| accuracy_6class (exact) | 178/210 = 0.848 | 0.792–0.893 |

## Calibration: temperature scaling fitted on each run's OOF femurs

| Run | Temperature | Femurs | ECE before → after | Brier before → after | NLL before → after |
|---|---|---|---|---|---|
| S01_yolov8s_seg_1024 | n/a | OOF (fitted here): 1043 | 0.079 → n/a | 0.323 → n/a | n/a → n/a |
| S01_yolov8s_seg_1024 | n/a | test ensemble: 210 | 0.083 → n/a | 0.275 → n/a | n/a → n/a |
| S01_yolov8s_seg_1024 | n/a | test fold models: 5 models | 0.082 → n/a (SD 0.007 → n/a) | 0.309 → n/a (SD 0.035 → n/a) | n/a → n/a (SD n/a → n/a) |
| S04_densenet121 | 2.44 | OOF (fitted here): 1043 | 0.116 → 0.026 | 0.244 → 0.214 | 0.809 → 0.466 |
| S04_densenet121 | 2.44 | test ensemble: 210 | 0.083 → 0.055 | 0.214 → 0.199 | 0.504 → 0.405 |
| S04_densenet121 | 2.44 | test fold models: 5 models | 0.137 → 0.065 (SD 0.016 → 0.012) | 0.271 → 0.231 (SD 0.035 → 0.025) | 0.852 → 0.484 (SD 0.132 → 0.045) |
| S05_efficientnet_b4 | 3.27 | OOF (fitted here): 1043 | 0.122 → 0.025 | 0.266 → 0.226 | 0.976 → 0.472 |
| S05_efficientnet_b4 | 3.27 | test ensemble: 210 | 0.104 → 0.064 | 0.276 → 0.254 | 0.674 → 0.493 |
| S05_efficientnet_b4 | 3.27 | test fold models: 5 models | 0.166 → 0.066 (SD 0.014 → 0.005) | 0.343 → 0.277 (SD 0.024 → 0.014) | 1.171 → 0.544 (SD 0.209 → 0.048) |
| S06_convnext_tiny | 3.43 | OOF (fitted here): 1043 | 0.127 → 0.046 | 0.259 → 0.215 | 1.130 → 0.459 |
| S06_convnext_tiny | 3.43 | test ensemble: 210 | 0.102 → 0.066 | 0.239 → 0.214 | 0.730 → 0.436 |
| S06_convnext_tiny | 3.43 | test fold models: 5 models | 0.156 → 0.066 (SD 0.014 → 0.013) | 0.311 → 0.252 (SD 0.031 → 0.019) | 1.323 → 0.522 (SD 0.089 → 0.024) |

ECE: 15 equal-width bins of the top-class probability. The temperature keeps each femur's most probable class, and no threshold or decision uses it: it is reported only.
S01_yolov8s_seg_1024: not applicable, 99 of 1043 OOF femurs give their true class probability 0, so the NLL is undefined (YOLO-style confidences: no probability for a class without an instance).
S05_efficientnet_b4: 1 of 1043 OOF femurs have their true class probability written as 0 (the per-case files keep 8 decimals, so it is below 5e-9); it is read as 1e-12, like every written 0.

## S01_yolov8s_seg_1024 vs S04_densenet121: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.016 | 0.0001 | 0.0002 |
| accuracy_6class | McNemar | -0.035 | 0.0009 | 0.0018 |
| fracture_detection | McNemar (binary fracture decision) | -0.029 | 0.0000 | 0.0001 |
| macro_f1 | randomisation (swap per x-ray) | -0.058 | 0.0014 | 0.0018 |

## S01_yolov8s_seg_1024 vs S04_densenet121: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.005 | 0.0824 | 0.2472 |
| accuracy_6class | McNemar | +0.024 | 0.4731 | 0.9463 |
| fracture_detection | McNemar (binary fracture decision) | -0.010 | 0.7266 | 0.9463 |
| macro_f1 | randomisation (swap per x-ray) | +0.086 | 0.0610 | 0.2440 |

## S01_yolov8s_seg_1024 vs S05_efficientnet_b4: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.015 | 0.0002 | 0.0008 |
| accuracy_6class | McNemar | -0.015 | 0.1915 | 0.3830 |
| fracture_detection | McNemar (binary fracture decision) | -0.021 | 0.0038 | 0.0115 |
| macro_f1 | randomisation (swap per x-ray) | -0.011 | 0.5941 | 0.5941 |

## S01_yolov8s_seg_1024 vs S05_efficientnet_b4: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.9035 | 0.9035 |
| accuracy_6class | McNemar | +0.071 | 0.0167 | 0.0667 |
| fracture_detection | McNemar (binary fracture decision) | +0.029 | 0.2101 | 0.4202 |
| macro_f1 | randomisation (swap per x-ray) | +0.101 | 0.0234 | 0.0702 |

## S01_yolov8s_seg_1024 vs S06_convnext_tiny: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.015 | 0.0009 | 0.0027 |
| accuracy_6class | McNemar | -0.032 | 0.0046 | 0.0093 |
| fracture_detection | McNemar (binary fracture decision) | -0.030 | 0.0001 | 0.0004 |
| macro_f1 | randomisation (swap per x-ray) | -0.039 | 0.0496 | 0.0496 |

## S01_yolov8s_seg_1024 vs S06_convnext_tiny: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.006 | 0.0726 | 0.2177 |
| accuracy_6class | McNemar | +0.033 | 0.2649 | 0.5299 |
| fracture_detection | McNemar (binary fracture decision) | -0.014 | 0.5078 | 0.5299 |
| macro_f1 | randomisation (swap per x-ray) | +0.103 | 0.0300 | 0.1200 |

## S04_densenet121 vs S05_efficientnet_b4: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.5202 | 0.5202 |
| accuracy_6class | McNemar | +0.020 | 0.0396 | 0.1188 |
| fracture_detection | McNemar (binary fracture decision) | +0.008 | 0.2153 | 0.4307 |
| macro_f1 | randomisation (swap per x-ray) | +0.047 | 0.0054 | 0.0216 |

## S04_densenet121 vs S05_efficientnet_b4: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.006 | 0.0686 | 0.2057 |
| accuracy_6class | McNemar | +0.048 | 0.0755 | 0.2057 |
| fracture_detection | McNemar (binary fracture decision) | +0.038 | 0.0215 | 0.0859 |
| macro_f1 | randomisation (swap per x-ray) | +0.016 | 0.6715 | 0.6715 |

## S04_densenet121 vs S06_convnext_tiny: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.5202 | 1.0000 |
| accuracy_6class | McNemar | +0.004 | 0.7436 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.001 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.019 | 0.2272 | 0.9086 |

## S04_densenet121 vs S06_convnext_tiny: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.3695 | 1.0000 |
| accuracy_6class | McNemar | +0.010 | 0.8238 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.018 | 0.6467 | 1.0000 |

## S05_efficientnet_b4 vs S06_convnext_tiny: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 0.9446 | 0.9446 |
| accuracy_6class | McNemar | -0.016 | 0.0893 | 0.3255 |
| fracture_detection | McNemar (binary fracture decision) | -0.009 | 0.1877 | 0.3755 |
| macro_f1 | randomisation (swap per x-ray) | -0.028 | 0.0814 | 0.3255 |

## S05_efficientnet_b4 vs S06_convnext_tiny: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.006 | 0.0545 | 0.1634 |
| accuracy_6class | McNemar | -0.038 | 0.2295 | 0.4590 |
| fracture_detection | McNemar (binary fracture decision) | -0.043 | 0.0225 | 0.0898 |
| macro_f1 | randomisation (swap per x-ray) | +0.002 | 0.9636 | 0.9636 |
