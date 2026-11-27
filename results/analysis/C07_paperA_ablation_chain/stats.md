# Statistics: S06_convnext_tiny, S10_m0_convnext_unet, A2_region, A3_geometry, A4_hierarchy, A5_shared_grade, A6_slgnet

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

## A2_region: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.996 | 0.993–0.998 |
| fracture_sensitivity | 0.978 | 0.966–0.988 |
| fracture_specificity | 0.986 | 0.972–0.997 |
| accuracy_6class | 0.883 | 0.863–0.901 |
| macro_f1 | 0.820 | 0.784–0.850 |
| balanced_accuracy | 0.801 | 0.764–0.837 |
| kappa | 0.846 | 0.820–0.870 |
| dice_fractured_mean | 0.772 | 0.760–0.785 |
| dice_detected_mean | 0.790 | 0.780–0.799 |
| garden12_recall | 0.544 | 0.407–0.678 |
| auc_fracture (DeLong) | 0.996 | 0.993–0.998 |
| accuracy_6class (exact) | 921/1043 = 0.883 | 0.862–0.902 |

## A2_region: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.996–1.000 |
| fracture_sensitivity | 0.993 | 0.979–1.000 |
| fracture_specificity | 0.954 | 0.897–1.000 |
| accuracy_6class | 0.862 | 0.815–0.906 |
| macro_f1 | 0.785 | 0.705–0.847 |
| balanced_accuracy | 0.769 | 0.703–0.833 |
| kappa | 0.824 | 0.765–0.877 |
| dice_fractured_mean | 0.738 | 0.712–0.762 |
| dice_detected_mean | 0.743 | 0.718–0.765 |
| garden12_recall | 0.250 | 0.050–0.500 |
| auc_fracture (DeLong) | 0.999 | 0.997–1.000 |
| accuracy_6class (exact) | 181/210 = 0.862 | 0.808–0.906 |

## A3_geometry: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.996 | 0.992–0.998 |
| fracture_sensitivity | 0.981 | 0.970–0.991 |
| fracture_specificity | 0.972 | 0.953–0.989 |
| accuracy_6class | 0.889 | 0.869–0.907 |
| macro_f1 | 0.838 | 0.805–0.866 |
| balanced_accuracy | 0.825 | 0.789–0.860 |
| kappa | 0.854 | 0.829–0.878 |
| dice_fractured_mean | 0.777 | 0.765–0.788 |
| dice_detected_mean | 0.792 | 0.783–0.801 |
| garden12_recall | 0.684 | 0.561–0.803 |
| auc_fracture (DeLong) | 0.996 | 0.992–0.999 |
| accuracy_6class (exact) | 927/1043 = 0.889 | 0.868–0.907 |

## A3_geometry: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.996–1.000 |
| fracture_sensitivity | 0.993 | 0.979–1.000 |
| fracture_specificity | 0.908 | 0.828–0.970 |
| accuracy_6class | 0.857 | 0.803–0.904 |
| macro_f1 | 0.786 | 0.703–0.854 |
| balanced_accuracy | 0.767 | 0.698–0.833 |
| kappa | 0.818 | 0.750–0.876 |
| dice_fractured_mean | 0.734 | 0.706–0.760 |
| dice_detected_mean | 0.739 | 0.713–0.763 |
| garden12_recall | 0.312 | 0.077–0.562 |
| auc_fracture (DeLong) | 0.999 | 0.997–1.000 |
| accuracy_6class (exact) | 180/210 = 0.857 | 0.802–0.901 |

## A4_hierarchy: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.989–0.998 |
| fracture_sensitivity | 0.979 | 0.969–0.990 |
| fracture_specificity | 0.986 | 0.972–0.997 |
| accuracy_6class | 0.878 | 0.858–0.897 |
| macro_f1 | 0.823 | 0.790–0.852 |
| balanced_accuracy | 0.813 | 0.776–0.847 |
| kappa | 0.840 | 0.815–0.864 |
| dice_fractured_mean | 0.770 | 0.756–0.783 |
| dice_detected_mean | 0.786 | 0.776–0.797 |
| garden12_recall | 0.649 | 0.526–0.771 |
| auc_fracture (DeLong) | 0.994 | 0.989–0.998 |
| accuracy_6class (exact) | 916/1043 = 0.878 | 0.857–0.897 |

## A4_hierarchy: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.997–1.000 |
| fracture_sensitivity | 0.993 | 0.979–1.000 |
| fracture_specificity | 0.969 | 0.919–1.000 |
| accuracy_6class | 0.871 | 0.824–0.914 |
| macro_f1 | 0.800 | 0.719–0.866 |
| balanced_accuracy | 0.781 | 0.714–0.850 |
| kappa | 0.836 | 0.776–0.890 |
| dice_fractured_mean | 0.732 | 0.703–0.759 |
| dice_detected_mean | 0.737 | 0.710–0.762 |
| garden12_recall | 0.375 | 0.133–0.636 |
| auc_fracture (DeLong) | 0.999 | 0.998–1.000 |
| accuracy_6class (exact) | 183/210 = 0.871 | 0.818–0.914 |

## A5_shared_grade: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.989–0.998 |
| fracture_sensitivity | 0.982 | 0.972–0.991 |
| fracture_specificity | 0.986 | 0.973–0.997 |
| accuracy_6class | 0.880 | 0.861–0.898 |
| macro_f1 | 0.820 | 0.786–0.850 |
| balanced_accuracy | 0.805 | 0.768–0.839 |
| kappa | 0.842 | 0.817–0.866 |
| dice_fractured_mean | 0.769 | 0.755–0.782 |
| dice_detected_mean | 0.783 | 0.772–0.794 |
| garden12_recall | 0.614 | 0.491–0.745 |
| auc_fracture (DeLong) | 0.994 | 0.990–0.998 |
| accuracy_6class (exact) | 918/1043 = 0.880 | 0.859–0.899 |

## A5_shared_grade: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 1.000 | 0.998–1.000 |
| fracture_sensitivity | 0.993 | 0.979–1.000 |
| fracture_specificity | 0.985 | 0.945–1.000 |
| accuracy_6class | 0.871 | 0.827–0.914 |
| macro_f1 | 0.792 | 0.712–0.857 |
| balanced_accuracy | 0.772 | 0.705–0.839 |
| kappa | 0.836 | 0.780–0.889 |
| dice_fractured_mean | 0.734 | 0.705–0.760 |
| dice_detected_mean | 0.739 | 0.711–0.763 |
| garden12_recall | 0.312 | 0.083–0.556 |
| auc_fracture (DeLong) | 1.000 | 0.999–1.000 |
| accuracy_6class (exact) | 183/210 = 0.871 | 0.818–0.914 |

## A6_slgnet: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.996 | 0.993–0.999 |
| fracture_sensitivity | 0.972 | 0.959–0.983 |
| fracture_specificity | 0.983 | 0.969–0.995 |
| accuracy_6class | 0.870 | 0.849–0.889 |
| macro_f1 | 0.806 | 0.771–0.836 |
| balanced_accuracy | 0.802 | 0.766–0.836 |
| kappa | 0.829 | 0.802–0.854 |
| dice_fractured_mean | 0.774 | 0.761–0.786 |
| dice_detected_mean | 0.796 | 0.788–0.805 |
| garden12_recall | 0.614 | 0.491–0.741 |
| auc_fracture (DeLong) | 0.996 | 0.994–0.999 |
| accuracy_6class (exact) | 907/1043 = 0.870 | 0.848–0.889 |

## A6_slgnet: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.996–1.000 |
| fracture_sensitivity | 0.986 | 0.966–1.000 |
| fracture_specificity | 0.985 | 0.945–1.000 |
| accuracy_6class | 0.886 | 0.842–0.925 |
| macro_f1 | 0.820 | 0.747–0.879 |
| balanced_accuracy | 0.805 | 0.743–0.864 |
| kappa | 0.854 | 0.799–0.904 |
| dice_fractured_mean | 0.737 | 0.711–0.761 |
| dice_detected_mean | 0.747 | 0.725–0.768 |
| garden12_recall | 0.312 | 0.083–0.556 |
| auc_fracture (DeLong) | 0.999 | 0.997–1.000 |
| accuracy_6class (exact) | 186/210 = 0.886 | 0.835–0.925 |

## Calibration: temperature scaling fitted on each run's OOF femurs

| Run | Temperature | Femurs | ECE before → after | Brier before → after | NLL before → after |
|---|---|---|---|---|---|
| S06_convnext_tiny | 3.43 | OOF (fitted here): 1043 | 0.127 → 0.046 | 0.259 → 0.215 | 1.130 → 0.459 |
| S06_convnext_tiny | 3.43 | test ensemble: 210 | 0.102 → 0.066 | 0.239 → 0.214 | 0.730 → 0.436 |
| S06_convnext_tiny | 3.43 | test fold models: 5 models | 0.156 → 0.066 (SD 0.014 → 0.013) | 0.311 → 0.252 (SD 0.031 → 0.019) | 1.323 → 0.522 (SD 0.089 → 0.024) |
| S10_m0_convnext_unet | 3.14 | OOF (fitted here): 1043 | 0.108 → 0.040 | 0.224 → 0.194 | 0.941 → 0.424 |
| S10_m0_convnext_unet | 3.14 | test ensemble: 210 | 0.103 → 0.064 | 0.223 → 0.198 | 0.644 → 0.397 |
| S10_m0_convnext_unet | 3.14 | test fold models: 5 models | 0.131 → 0.062 (SD 0.010 → 0.019) | 0.266 → 0.225 (SD 0.011 → 0.011) | 1.047 → 0.465 (SD 0.085 → 0.021) |
| A2_region | 2.74 | OOF (fitted here): 1043 | 0.104 → 0.030 | 0.219 → 0.192 | 0.851 → 0.433 |
| A2_region | 2.74 | test ensemble: 210 | 0.089 → 0.059 | 0.203 → 0.187 | 0.543 → 0.385 |
| A2_region | 2.74 | test fold models: 5 models | 0.124 → 0.059 (SD 0.009 → 0.005) | 0.247 → 0.205 (SD 0.021 → 0.020) | 0.827 → 0.433 (SD 0.115 → 0.043) |
| A3_geometry | 2.64 | OOF (fitted here): 1043 | 0.097 → 0.045 | 0.201 → 0.178 | 0.762 → 0.402 |
| A3_geometry | 2.64 | test ensemble: 210 | 0.087 → 0.074 | 0.193 → 0.191 | 0.573 → 0.403 |
| A3_geometry | 2.64 | test fold models: 5 models | 0.120 → 0.064 (SD 0.010 → 0.008) | 0.244 → 0.210 (SD 0.017 → 0.015) | 0.855 → 0.447 (SD 0.114 → 0.035) |
| A4_hierarchy | 3.21 | OOF (fitted here): 1043 | 0.115 → 0.029 | 0.238 → 0.200 | 1.032 → 0.449 |
| A4_hierarchy | 3.21 | test ensemble: 210 | 0.078 → 0.067 | 0.200 → 0.185 | 0.530 → 0.368 |
| A4_hierarchy | 3.21 | test fold models: 5 models | 0.128 → 0.054 (SD 0.013 → 0.025) | 0.258 → 0.220 (SD 0.027 → 0.019) | 1.059 → 0.459 (SD 0.116 → 0.031) |
| A5_shared_grade | 3.38 | OOF (fitted here): 1043 | 0.114 → 0.039 | 0.239 → 0.202 | 1.106 → 0.464 |
| A5_shared_grade | 3.38 | test ensemble: 210 | 0.097 → 0.069 | 0.225 → 0.191 | 0.576 → 0.377 |
| A5_shared_grade | 3.38 | test fold models: 5 models | 0.138 → 0.064 (SD 0.012 → 0.011) | 0.277 → 0.228 (SD 0.024 → 0.023) | 1.108 → 0.464 (SD 0.181 → 0.052) |
| A6_slgnet | 3.41 | OOF (fitted here): 1043 | 0.119 → 0.037 | 0.248 → 0.215 | 1.179 → 0.489 |
| A6_slgnet | 3.41 | test ensemble: 210 | 0.081 → 0.083 | 0.203 → 0.195 | 0.578 → 0.389 |
| A6_slgnet | 3.41 | test fold models: 5 models | 0.131 → 0.066 (SD 0.019 → 0.015) | 0.260 → 0.215 (SD 0.032 → 0.021) | 0.985 → 0.438 (SD 0.141 → 0.038) |

ECE: 15 equal-width bins of the top-class probability. The temperature keeps each femur's most probable class, and no threshold or decision uses it: it is reported only.
A5_shared_grade: 3 of 1043 OOF femurs have their true class probability written as 0 (the per-case files keep 8 decimals, so it is below 5e-9); it is read as 1e-12, like every written 0.
A6_slgnet: 4 of 1043 OOF femurs have their true class probability written as 0 (the per-case files keep 8 decimals, so it is below 5e-9); it is read as 1e-12, like every written 0.

p (Holm, family): p_holm_family: Holm across the 7 compared pairs, each endpoint and scope separately (an endpoint counts only where it was tested).

## S10_m0_convnext_unet vs S06_convnext_tiny: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.004 | 0.0290 | 0.0581 | 0.2033 |
| accuracy_6class | McNemar | +0.020 | 0.0086 | 0.0259 | 0.0605 |
| fracture_detection | McNemar (binary fracture decision) | +0.009 | 0.0784 | 0.0784 | 0.4701 |
| macro_f1 | randomisation (swap per x-ray) | +0.042 | 0.0032 | 0.0128 | 0.0224 |

## S10_m0_convnext_unet vs S06_convnext_tiny: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.5961 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.019 | 0.3438 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.010 | 0.6250 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.037 | 0.2196 | 0.8782 | 1.0000 |

## A2_region vs S10_m0_convnext_unet: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.3878 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | -0.001 | 1.0000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.002 | 0.7905 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.006 | 0.6353 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.004 | 0.0007 | 0.0033 | 0.0039 |

## A2_region vs S10_m0_convnext_unet: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.4032 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.014 | 0.2500 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.006 | 0.8700 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.006 | 0.9827 | 1.0000 | 1.0000 |

## A3_geometry vs A2_region: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.8085 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.006 | 0.5118 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.003 | 0.6291 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.018 | 0.2066 | 0.8262 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.005 | 0.1375 | 0.6873 | 0.4124 |

## A3_geometry vs A2_region: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.8238 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.014 | 0.3750 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.001 | 0.9610 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.004 | 0.8235 | 1.0000 | 1.0000 |

## A4_hierarchy vs A3_geometry: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.002 | 0.3754 | 0.7960 | 1.0000 |
| accuracy_6class | McNemar | -0.011 | 0.1925 | 0.7701 | 0.9243 |
| fracture_detection | McNemar (binary fracture decision) | +0.004 | 0.5034 | 0.7960 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.015 | 0.2653 | 0.7960 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.007 | 0.0469 | 0.2347 | 0.2347 |

## A4_hierarchy vs A3_geometry: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 0.6274 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.014 | 0.5811 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.019 | 0.2188 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.014 | 0.5995 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.001 | 0.9057 | 1.0000 | 1.0000 |

## A5_shared_grade vs A4_hierarchy: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.6504 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.002 | 0.8939 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.002 | 0.7539 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.003 | 0.8152 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.001 | 0.3546 | 1.0000 | 0.7092 |

## A5_shared_grade vs A4_hierarchy: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.3040 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.005 | 1.0000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.008 | 0.8846 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.001 | 0.5650 | 1.0000 | 1.0000 |

## A6_slgnet vs A5_shared_grade: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.002 | 0.0692 | 0.2869 | 0.4150 |
| accuracy_6class | McNemar | -0.011 | 0.1849 | 0.5546 | 0.9243 |
| fracture_detection | McNemar (binary fracture decision) | -0.008 | 0.0574 | 0.2869 | 0.4016 |
| macro_f1 | randomisation (swap per x-ray) | -0.014 | 0.2511 | 0.5546 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.005 | 0.7853 | 0.7853 | 0.7853 |

## A6_slgnet vs A5_shared_grade: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.2849 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.014 | 0.2500 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.028 | 0.1210 | 0.6049 | 0.8468 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.003 | 0.5791 | 1.0000 | 1.0000 |

## A6_slgnet vs S10_m0_convnext_unet: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.002 | 0.2465 | 0.5201 | 1.0000 |
| accuracy_6class | McNemar | -0.014 | 0.0769 | 0.3803 | 0.4616 |
| fracture_detection | McNemar (binary fracture decision) | -0.003 | 0.6072 | 0.6072 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.020 | 0.1734 | 0.5201 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.002 | 0.0761 | 0.3803 | 0.3042 |

## A6_slgnet vs S10_m0_convnext_unet: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.2876 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.019 | 0.3438 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.010 | 0.5000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.041 | 0.2138 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.006 | 0.6598 | 1.0000 | 1.0000 |
