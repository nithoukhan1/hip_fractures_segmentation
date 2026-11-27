# Statistics: S10_m0_convnext_unet, A6_dev_base, A6_dev_scale32, A6_dev_base_s2, A6_dev_scale32_s2

95% intervals: cluster bootstrap over x-rays (2000 resamples, seed 2026); AUC also DeLong; 6-class accuracy also exact (Clopper-Pearson). Decisions use each run's OOF threshold, fitted again on the femurs of fold(s) 0 only (test set not used).

## S10_m0_convnext_unet: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.996 | 0.989–1.000 |
| fracture_sensitivity | 0.985 | 0.963–1.000 |
| fracture_specificity | 0.972 | 0.930–1.000 |
| accuracy_6class | 0.885 | 0.837–0.927 |
| macro_f1 | 0.822 | 0.735–0.889 |
| balanced_accuracy | 0.809 | 0.722–0.890 |
| kappa | 0.849 | 0.788–0.903 |
| dice_fractured_mean | 0.749 | 0.710–0.783 |
| dice_detected_mean | 0.760 | 0.723–0.791 |
| garden12_recall | 0.545 | 0.250–0.857 |
| auc_fracture (DeLong) | 0.996 | 0.990–1.000 |
| accuracy_6class (exact) | 184/208 = 0.885 | 0.833–0.925 |

## A6_dev_base: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.998 | 0.995–1.000 |
| fracture_sensitivity | 0.978 | 0.949–1.000 |
| fracture_specificity | 0.986 | 0.956–1.000 |
| accuracy_6class | 0.885 | 0.840–0.926 |
| macro_f1 | 0.775 | 0.675–0.853 |
| balanced_accuracy | 0.762 | 0.673–0.854 |
| kappa | 0.848 | 0.791–0.901 |
| dice_fractured_mean | 0.754 | 0.718–0.785 |
| dice_detected_mean | 0.771 | 0.742–0.797 |
| garden12_recall | 0.545 | 0.235–0.846 |
| auc_fracture (DeLong) | 0.998 | 0.995–1.000 |
| accuracy_6class (exact) | 184/208 = 0.885 | 0.833–0.925 |

## A6_dev_scale32: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.996–1.000 |
| fracture_sensitivity | 0.985 | 0.963–1.000 |
| fracture_specificity | 1.000 | 1.000–1.000 |
| accuracy_6class | 0.889 | 0.847–0.931 |
| macro_f1 | 0.827 | 0.727–0.897 |
| balanced_accuracy | 0.820 | 0.735–0.903 |
| kappa | 0.855 | 0.799–0.909 |
| dice_fractured_mean | 0.760 | 0.729–0.786 |
| dice_detected_mean | 0.771 | 0.745–0.794 |
| garden12_recall | 0.818 | 0.556–1.000 |
| auc_fracture (DeLong) | 0.999 | 0.996–1.000 |
| accuracy_6class (exact) | 185/208 = 0.889 | 0.839–0.929 |

## A6_dev_base_s2: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.998 | 0.995–1.000 |
| fracture_sensitivity | 0.985 | 0.963–1.000 |
| fracture_specificity | 0.986 | 0.956–1.000 |
| accuracy_6class | 0.865 | 0.816–0.911 |
| macro_f1 | 0.771 | 0.668–0.847 |
| balanced_accuracy | 0.766 | 0.671–0.855 |
| kappa | 0.824 | 0.760–0.882 |
| dice_fractured_mean | 0.744 | 0.703–0.780 |
| dice_detected_mean | 0.755 | 0.717–0.787 |
| garden12_recall | 0.545 | 0.235–0.846 |
| auc_fracture (DeLong) | 0.998 | 0.996–1.000 |
| accuracy_6class (exact) | 180/208 = 0.865 | 0.811–0.909 |

## A6_dev_scale32_s2: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.996–1.000 |
| fracture_sensitivity | 0.978 | 0.949–1.000 |
| fracture_specificity | 1.000 | 1.000–1.000 |
| accuracy_6class | 0.889 | 0.846–0.931 |
| macro_f1 | 0.827 | 0.728–0.896 |
| balanced_accuracy | 0.806 | 0.716–0.889 |
| kappa | 0.854 | 0.797–0.908 |
| dice_fractured_mean | 0.770 | 0.740–0.795 |
| dice_detected_mean | 0.787 | 0.766–0.807 |
| garden12_recall | 0.727 | 0.429–1.000 |
| auc_fracture (DeLong) | 0.999 | 0.997–1.000 |
| accuracy_6class (exact) | 185/208 = 0.889 | 0.839–0.929 |

## Calibration: temperature scaling fitted on each run's OOF femurs

| Run | Temperature | Femurs | ECE before → after | Brier before → after | NLL before → after |
|---|---|---|---|---|---|
| S10_m0_convnext_unet | 3.22 | OOF (fitted here): 208 | 0.125 → 0.051 | 0.242 → 0.205 | 1.054 → 0.461 |
| A6_dev_base | 3.14 | OOF (fitted here): 208 | 0.104 → 0.056 | 0.213 → 0.179 | 0.936 → 0.416 |
| A6_dev_scale32 | 3.25 | OOF (fitted here): 208 | 0.120 → 0.050 | 0.236 → 0.197 | 1.022 → 0.437 |
| A6_dev_base_s2 | 3.30 | OOF (fitted here): 208 | 0.128 → 0.079 | 0.255 → 0.210 | 1.128 → 0.474 |
| A6_dev_scale32_s2 | 3.22 | OOF (fitted here): 208 | 0.109 → 0.058 | 0.222 → 0.193 | 0.982 → 0.431 |

ECE: 15 equal-width bins of the top-class probability. The temperature keeps each femur's most probable class, and no threshold or decision uses it: it is reported only.

## S10_m0_convnext_unet vs A6_dev_base: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.002 | 0.3986 | 1.0000 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.047 | 0.2533 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.005 | 0.6545 | 1.0000 |

## S10_m0_convnext_unet vs A6_dev_scale32: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.002 | 0.2478 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.010 | 0.5000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.005 | 0.8746 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.011 | 0.2077 | 1.0000 |

## S10_m0_convnext_unet vs A6_dev_base_s2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.002 | 0.3478 | 1.0000 |
| accuracy_6class | McNemar | +0.019 | 0.4240 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.051 | 0.2482 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.005 | 0.4542 | 1.0000 |

## S10_m0_convnext_unet vs A6_dev_scale32_s2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.003 | 0.2204 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.005 | 0.8868 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.021 | 0.4628 | 1.0000 |

## A6_dev_base vs A6_dev_scale32: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.6346 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.010 | 0.6250 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.052 | 0.1756 | 0.8778 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.006 | 0.9489 | 1.0000 |

## A6_dev_base vs A6_dev_base_s2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.8880 | 1.0000 |
| accuracy_6class | McNemar | +0.019 | 0.3438 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.004 | 0.9052 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.010 | 0.8095 | 1.0000 |

## A6_dev_base vs A6_dev_scale32_s2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.4554 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.051 | 0.1238 | 0.6189 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.016 | 0.9178 | 1.0000 |

## A6_dev_scale32 vs A6_dev_base_s2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 0.6128 | 1.0000 |
| accuracy_6class | McNemar | +0.024 | 0.1797 | 0.8984 |
| fracture_detection | McNemar (binary fracture decision) | +0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.056 | 0.1866 | 0.8984 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.016 | 0.9861 | 1.0000 |

## A6_dev_scale32 vs A6_dev_scale32_s2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.7400 | 1.0000 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.001 | 0.9562 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.010 | 0.5999 | 1.0000 |

## A6_dev_base_s2 vs A6_dev_scale32_s2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.3955 | 1.0000 |
| accuracy_6class | McNemar | -0.024 | 0.2668 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.056 | 0.1816 | 0.9078 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.026 | 0.7544 | 1.0000 |
