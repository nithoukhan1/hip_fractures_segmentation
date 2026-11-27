# Statistics: S10_m0_convnext_unet, R4_dev_base, R4_dev_base_s2, R4_dev_global, R4_dev_margin

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

## R4_dev_base: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.997 | 0.993–1.000 |
| fracture_sensitivity | 0.978 | 0.949–1.000 |
| fracture_specificity | 1.000 | 1.000–1.000 |
| accuracy_6class | 0.880 | 0.835–0.922 |
| macro_f1 | 0.817 | 0.715–0.889 |
| balanced_accuracy | 0.786 | 0.694–0.875 |
| kappa | 0.841 | 0.782–0.896 |
| dice_fractured_mean | 0.765 | 0.735–0.792 |
| dice_detected_mean | 0.783 | 0.760–0.803 |
| garden12_recall | 0.636 | 0.333–0.909 |
| auc_fracture (DeLong) | 0.997 | 0.994–1.000 |
| accuracy_6class (exact) | 183/208 = 0.880 | 0.828–0.921 |

## R4_dev_base_s2: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.997–1.000 |
| fracture_sensitivity | 0.993 | 0.977–1.000 |
| fracture_specificity | 0.986 | 0.957–1.000 |
| accuracy_6class | 0.894 | 0.850–0.933 |
| macro_f1 | 0.819 | 0.717–0.890 |
| balanced_accuracy | 0.804 | 0.713–0.890 |
| kappa | 0.861 | 0.804–0.913 |
| dice_fractured_mean | 0.774 | 0.747–0.798 |
| dice_detected_mean | 0.780 | 0.754–0.802 |
| garden12_recall | 0.636 | 0.333–0.909 |
| auc_fracture (DeLong) | 0.999 | 0.998–1.000 |
| accuracy_6class (exact) | 186/208 = 0.894 | 0.844–0.933 |

## R4_dev_global: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.996–1.000 |
| fracture_sensitivity | 0.985 | 0.963–1.000 |
| fracture_specificity | 0.986 | 0.953–1.000 |
| accuracy_6class | 0.870 | 0.821–0.917 |
| macro_f1 | 0.796 | 0.688–0.872 |
| balanced_accuracy | 0.776 | 0.685–0.868 |
| kappa | 0.829 | 0.768–0.890 |
| dice_fractured_mean | 0.778 | 0.753–0.800 |
| dice_detected_mean | 0.790 | 0.772–0.807 |
| garden12_recall | 0.545 | 0.235–0.846 |
| auc_fracture (DeLong) | 0.999 | 0.997–1.000 |
| accuracy_6class (exact) | 181/208 = 0.870 | 0.817–0.913 |

## R4_dev_margin: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.997 | 0.993–1.000 |
| fracture_sensitivity | 0.956 | 0.918–0.986 |
| fracture_specificity | 1.000 | 1.000–1.000 |
| accuracy_6class | 0.889 | 0.846–0.932 |
| macro_f1 | 0.829 | 0.724–0.897 |
| balanced_accuracy | 0.808 | 0.712–0.894 |
| kappa | 0.854 | 0.796–0.910 |
| dice_fractured_mean | 0.755 | 0.721–0.786 |
| dice_detected_mean | 0.790 | 0.769–0.809 |
| garden12_recall | 0.727 | 0.417–1.000 |
| auc_fracture (DeLong) | 0.997 | 0.994–1.000 |
| accuracy_6class (exact) | 185/208 = 0.889 | 0.839–0.929 |

## Calibration: temperature scaling fitted on each run's OOF femurs

| Run | Temperature | Femurs | ECE before → after | Brier before → after | NLL before → after |
|---|---|---|---|---|---|
| S10_m0_convnext_unet | 3.22 | OOF (fitted here): 208 | 0.125 → 0.051 | 0.242 → 0.205 | 1.054 → 0.461 |
| R4_dev_base | 3.77 | OOF (fitted here): 208 | 0.128 → 0.037 | 0.254 → 0.213 | 1.255 → 0.462 |
| R4_dev_base_s2 | 3.36 | OOF (fitted here): 208 | 0.098 → 0.021 | 0.205 → 0.185 | 1.026 → 0.431 |
| R4_dev_global | 3.35 | OOF (fitted here): 208 | 0.122 → 0.050 | 0.253 → 0.211 | 1.067 → 0.444 |
| R4_dev_margin | 2.96 | OOF (fitted here): 208 | 0.098 → 0.059 | 0.197 → 0.174 | 0.808 → 0.388 |

ECE: 15 equal-width bins of the top-class probability. The temperature keeps each femur's most probable class, and no threshold or decision uses it: it is reported only.

## S10_m0_convnext_unet vs R4_dev_base: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.5678 | 1.0000 |
| accuracy_6class | McNemar | +0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.005 | 0.8596 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.017 | 0.7282 | 1.0000 |

## S10_m0_convnext_unet vs R4_dev_base_s2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.003 | 0.2083 | 1.0000 |
| accuracy_6class | McNemar | -0.010 | 0.7744 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.010 | 0.5000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.003 | 0.9208 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.025 | 0.3225 | 1.0000 |

## S10_m0_convnext_unet vs R4_dev_global: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.003 | 0.2522 | 1.0000 |
| accuracy_6class | McNemar | +0.014 | 0.5811 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.026 | 0.4781 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.030 | 0.3430 | 1.0000 |

## S10_m0_convnext_unet vs R4_dev_margin: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.6814 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.010 | 0.6875 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.007 | 0.8092 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.007 | 0.6413 | 1.0000 |

## R4_dev_base vs R4_dev_base_s2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.002 | 0.2337 | 0.9349 |
| accuracy_6class | McNemar | -0.014 | 0.4531 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.002 | 0.8910 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.009 | 0.1483 | 0.7415 |

## R4_dev_base vs R4_dev_global: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.2988 | 1.0000 |
| accuracy_6class | McNemar | +0.010 | 0.7744 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.021 | 0.3477 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.013 | 0.0844 | 0.4219 |

## R4_dev_base vs R4_dev_margin: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 0.8897 | 1.0000 |
| accuracy_6class | McNemar | -0.010 | 0.7744 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.014 | 0.3750 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.013 | 0.7285 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.010 | 0.8757 | 1.0000 |

## R4_dev_base_s2 vs R4_dev_global: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 0.6366 | 1.0000 |
| accuracy_6class | McNemar | +0.024 | 0.2668 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.023 | 0.3135 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.005 | 0.8357 | 1.0000 |

## R4_dev_base_s2 vs R4_dev_margin: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.002 | 0.1618 | 0.6471 |
| accuracy_6class | McNemar | +0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.019 | 0.2188 | 0.6562 |
| macro_f1 | randomisation (swap per x-ray) | -0.011 | 0.7768 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.018 | 0.1155 | 0.5777 |

## R4_dev_global vs R4_dev_margin: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.002 | 0.1763 | 0.7054 |
| accuracy_6class | McNemar | -0.019 | 0.4240 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.014 | 0.3750 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.034 | 0.4203 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.023 | 0.1101 | 0.5506 |
