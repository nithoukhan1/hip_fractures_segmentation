# Statistics: S06_convnext_tiny, S10_m0_convnext_unet

95% intervals: cluster bootstrap over x-rays (2000 resamples, seed 2026); AUC also DeLong; 6-class accuracy also exact (Clopper-Pearson). Decisions use each run's OOF threshold.

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

## S10_m0_convnext_unet: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.995 | 0.990–0.998 |
| fracture_sensitivity | 0.972 | 0.960–0.984 |
| fracture_specificity | 0.992 | 0.981–1.000 |
| accuracy_6class | 0.884 | 0.864–0.902 |
| macro_f1 | 0.825 | 0.793–0.855 |
| balanced_accuracy | 0.804 | 0.768–0.839 |
| kappa | 0.847 | 0.821–0.870 |
| dice_fractured_mean | 0.777 | 0.764–0.789 |
| dice_detected_mean | 0.799 | 0.790–0.807 |
| garden12_recall | 0.544 | 0.412–0.677 |
| auc_fracture (DeLong) | 0.995 | 0.991–0.998 |
| accuracy_6class (exact) | 922/1043 = 0.884 | 0.863–0.903 |

## S10_m0_convnext_unet: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.998–1.000 |
| fracture_sensitivity | 0.993 | 0.979–1.000 |
| fracture_specificity | 1.000 | 1.000–1.000 |
| accuracy_6class | 0.867 | 0.820–0.912 |
| macro_f1 | 0.779 | 0.695–0.849 |
| balanced_accuracy | 0.757 | 0.692–0.825 |
| kappa | 0.829 | 0.770–0.886 |
| dice_fractured_mean | 0.744 | 0.718–0.766 |
| dice_detected_mean | 0.749 | 0.725–0.770 |
| garden12_recall | 0.312 | 0.091–0.556 |
| auc_fracture (DeLong) | 0.999 | 0.998–1.000 |
| accuracy_6class (exact) | 182/210 = 0.867 | 0.813–0.910 |

## S06_convnext_tiny vs S10_m0_convnext_unet: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.004 | 0.0290 | 0.0581 |
| accuracy_6class | McNemar | -0.020 | 0.0086 | 0.0259 |
| fracture_detection | McNemar (binary fracture decision) | -0.009 | 0.0784 | 0.0784 |
| macro_f1 | randomisation (swap per x-ray) | -0.042 | 0.0032 | 0.0128 |

## S06_convnext_tiny vs S10_m0_convnext_unet: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 0.5961 | 1.0000 |
| accuracy_6class | McNemar | -0.019 | 0.3438 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.010 | 0.6250 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.037 | 0.2196 | 0.8782 |
