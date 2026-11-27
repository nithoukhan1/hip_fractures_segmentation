# Statistics: S10_m0_convnext_unet, A6_dev_base, A6_dev_no_tf, A6_dev_detach, A6_dev_scale32

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

## A6_dev_no_tf: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.998 | 0.994–1.000 |
| fracture_sensitivity | 0.985 | 0.963–1.000 |
| fracture_specificity | 0.986 | 0.955–1.000 |
| accuracy_6class | 0.885 | 0.841–0.926 |
| macro_f1 | 0.783 | 0.689–0.861 |
| balanced_accuracy | 0.788 | 0.700–0.880 |
| kappa | 0.849 | 0.792–0.901 |
| dice_fractured_mean | 0.754 | 0.718–0.785 |
| dice_detected_mean | 0.765 | 0.734–0.793 |
| garden12_recall | 0.727 | 0.417–1.000 |
| auc_fracture (DeLong) | 0.998 | 0.995–1.000 |
| accuracy_6class (exact) | 184/208 = 0.885 | 0.833–0.925 |

## A6_dev_detach: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.998 | 0.995–1.000 |
| fracture_sensitivity | 0.985 | 0.962–1.000 |
| fracture_specificity | 1.000 | 1.000–1.000 |
| accuracy_6class | 0.885 | 0.842–0.926 |
| macro_f1 | 0.794 | 0.689–0.876 |
| balanced_accuracy | 0.783 | 0.696–0.875 |
| kappa | 0.848 | 0.792–0.901 |
| dice_fractured_mean | 0.754 | 0.718–0.785 |
| dice_detected_mean | 0.765 | 0.734–0.792 |
| garden12_recall | 0.727 | 0.444–1.000 |
| auc_fracture (DeLong) | 0.998 | 0.996–1.000 |
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

## Calibration: temperature scaling fitted on each run's OOF femurs

| Run | Temperature | Femurs | ECE before → after | Brier before → after | NLL before → after |
|---|---|---|---|---|---|
| S10_m0_convnext_unet | 3.22 | OOF (fitted here): 208 | 0.125 → 0.051 | 0.242 → 0.205 | 1.054 → 0.461 |
| A6_dev_base | 3.14 | OOF (fitted here): 208 | 0.104 → 0.056 | 0.213 → 0.179 | 0.936 → 0.416 |
| A6_dev_no_tf | 3.30 | OOF (fitted here): 208 | 0.110 → 0.038 | 0.217 → 0.186 | 1.024 → 0.429 |
| A6_dev_detach | 3.38 | OOF (fitted here): 208 | 0.111 → 0.074 | 0.220 → 0.185 | 1.113 → 0.461 |
| A6_dev_scale32 | 3.25 | OOF (fitted here): 208 | 0.120 → 0.050 | 0.236 → 0.197 | 1.022 → 0.437 |

ECE: 15 equal-width bins of the top-class probability. The temperature keeps each femur's most probable class, and no threshold or decision uses it: it is reported only.
A6_dev_detach: 1 of 208 OOF femurs have their true class probability written as 0 (the per-case files keep 8 decimals, so it is below 5e-9); it is read as 1e-12, like every written 0.

## S10_m0_convnext_unet vs A6_dev_base: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.002 | 0.3986 | 1.0000 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.047 | 0.2533 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.005 | 0.6545 | 1.0000 |

## S10_m0_convnext_unet vs A6_dev_no_tf: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.002 | 0.3706 | 1.0000 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.039 | 0.3415 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.005 | 0.0666 | 0.3329 |

## S10_m0_convnext_unet vs A6_dev_detach: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.002 | 0.1965 | 0.9826 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.010 | 0.6250 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.028 | 0.5333 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.005 | 0.3885 | 1.0000 |

## S10_m0_convnext_unet vs A6_dev_scale32: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.002 | 0.2478 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.010 | 0.5000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.005 | 0.8746 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.011 | 0.2077 | 1.0000 |

## A6_dev_base vs A6_dev_no_tf: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 0.8709 | 1.0000 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.008 | 0.7083 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.000 | 0.1881 | 0.9403 |

## A6_dev_base vs A6_dev_detach: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.6744 | 1.0000 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.010 | 0.5000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.019 | 0.4989 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.000 | 0.4966 | 1.0000 |

## A6_dev_base vs A6_dev_scale32: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.6346 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.010 | 0.6250 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.052 | 0.1756 | 0.8778 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.006 | 0.9489 | 1.0000 |

## A6_dev_no_tf vs A6_dev_detach: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.5022 | 1.0000 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.011 | 0.7051 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.000 | 0.1837 | 0.9186 |

## A6_dev_no_tf vs A6_dev_scale32: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.4964 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.044 | 0.2681 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.006 | 0.4856 | 1.0000 |

## A6_dev_detach vs A6_dev_scale32: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.8902 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.033 | 0.3585 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.006 | 0.2501 | 1.0000 |
