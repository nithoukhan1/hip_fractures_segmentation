# Statistics: S10_m0_convnext_unet, A1_m0_deletion, N_segzoom_fold0, N_segzoom_fold0_seed2, N_res1152_fold0, N_res1152_fold0_seed2, N_convnext_s_fold0, N_convnext_s_fold0_seed2

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

## A1_m0_deletion: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.985–1.000 |
| fracture_sensitivity | 0.956 | 0.919–0.986 |
| fracture_specificity | 1.000 | 1.000–1.000 |
| accuracy_6class | 0.885 | 0.839–0.925 |
| macro_f1 | 0.811 | 0.706–0.891 |
| balanced_accuracy | 0.770 | 0.683–0.866 |
| kappa | 0.847 | 0.785–0.900 |
| dice_fractured_mean | 0.761 | 0.725–0.791 |
| dice_detected_mean | 0.796 | 0.776–0.813 |
| garden12_recall | 0.636 | 0.333–0.917 |
| auc_fracture (DeLong) | 0.994 | 0.987–1.000 |
| accuracy_6class (exact) | 184/208 = 0.885 | 0.833–0.925 |

## N_segzoom_fold0: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.991 | 0.976–1.000 |
| fracture_sensitivity | 0.985 | 0.963–1.000 |
| fracture_specificity | 0.972 | 0.931–1.000 |
| accuracy_6class | 0.885 | 0.839–0.928 |
| macro_f1 | 0.816 | 0.717–0.889 |
| balanced_accuracy | 0.797 | 0.706–0.884 |
| kappa | 0.848 | 0.790–0.904 |
| dice_fractured_mean | 0.763 | 0.729–0.791 |
| dice_detected_mean | 0.774 | 0.747–0.799 |
| garden12_recall | 0.636 | 0.333–0.909 |
| auc_fracture (DeLong) | 0.991 | 0.979–1.000 |
| accuracy_6class (exact) | 184/208 = 0.885 | 0.833–0.925 |

## N_segzoom_fold0_seed2: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.997 | 0.992–1.000 |
| fracture_sensitivity | 0.985 | 0.963–1.000 |
| fracture_specificity | 0.986 | 0.957–1.000 |
| accuracy_6class | 0.894 | 0.850–0.935 |
| macro_f1 | 0.843 | 0.766–0.906 |
| balanced_accuracy | 0.832 | 0.755–0.906 |
| kappa | 0.861 | 0.803–0.914 |
| dice_fractured_mean | 0.763 | 0.731–0.793 |
| dice_detected_mean | 0.774 | 0.744–0.800 |
| garden12_recall | 0.545 | 0.231–0.857 |
| auc_fracture (DeLong) | 0.997 | 0.993–1.000 |
| accuracy_6class (exact) | 186/208 = 0.894 | 0.844–0.933 |

## N_res1152_fold0: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.997 | 0.990–1.000 |
| fracture_sensitivity | 0.978 | 0.949–1.000 |
| fracture_specificity | 0.986 | 0.955–1.000 |
| accuracy_6class | 0.880 | 0.834–0.922 |
| macro_f1 | 0.783 | 0.673–0.860 |
| balanced_accuracy | 0.764 | 0.676–0.856 |
| kappa | 0.842 | 0.782–0.897 |
| dice_fractured_mean | 0.768 | 0.739–0.794 |
| dice_detected_mean | 0.785 | 0.763–0.804 |
| garden12_recall | 0.545 | 0.235–0.846 |
| auc_fracture (DeLong) | 0.997 | 0.991–1.000 |
| accuracy_6class (exact) | 183/208 = 0.880 | 0.828–0.921 |

## N_res1152_fold0_seed2: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.995–1.000 |
| fracture_sensitivity | 0.993 | 0.977–1.000 |
| fracture_specificity | 1.000 | 1.000–1.000 |
| accuracy_6class | 0.870 | 0.822–0.915 |
| macro_f1 | 0.790 | 0.687–0.864 |
| balanced_accuracy | 0.772 | 0.682–0.861 |
| kappa | 0.830 | 0.768–0.887 |
| dice_fractured_mean | 0.779 | 0.755–0.800 |
| dice_detected_mean | 0.785 | 0.765–0.803 |
| garden12_recall | 0.545 | 0.235–0.846 |
| auc_fracture (DeLong) | 0.999 | 0.996–1.000 |
| accuracy_6class (exact) | 181/208 = 0.870 | 0.817–0.913 |

## N_convnext_s_fold0: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.996 | 0.987–1.000 |
| fracture_sensitivity | 0.978 | 0.949–1.000 |
| fracture_specificity | 0.986 | 0.956–1.000 |
| accuracy_6class | 0.894 | 0.850–0.936 |
| macro_f1 | 0.814 | 0.706–0.888 |
| balanced_accuracy | 0.781 | 0.690–0.872 |
| kappa | 0.860 | 0.801–0.914 |
| dice_fractured_mean | 0.759 | 0.725–0.790 |
| dice_detected_mean | 0.777 | 0.748–0.801 |
| garden12_recall | 0.455 | 0.143–0.778 |
| auc_fracture (DeLong) | 0.996 | 0.989–1.000 |
| accuracy_6class (exact) | 186/208 = 0.894 | 0.844–0.933 |

## N_convnext_s_fold0_seed2: oof (208 femurs, 139 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.998 | 0.995–1.000 |
| fracture_sensitivity | 0.993 | 0.977–1.000 |
| fracture_specificity | 0.986 | 0.957–1.000 |
| accuracy_6class | 0.865 | 0.821–0.910 |
| macro_f1 | 0.788 | 0.686–0.863 |
| balanced_accuracy | 0.769 | 0.677–0.861 |
| kappa | 0.823 | 0.765–0.880 |
| dice_fractured_mean | 0.761 | 0.728–0.790 |
| dice_detected_mean | 0.766 | 0.733–0.793 |
| garden12_recall | 0.636 | 0.333–0.909 |
| auc_fracture (DeLong) | 0.998 | 0.996–1.000 |
| accuracy_6class (exact) | 180/208 = 0.865 | 0.811–0.909 |

## Calibration: temperature scaling fitted on each run's OOF femurs

| Run | Temperature | Femurs | ECE before → after | Brier before → after | NLL before → after |
|---|---|---|---|---|---|
| S10_m0_convnext_unet | 3.22 | OOF (fitted here): 208 | 0.125 → 0.051 | 0.242 → 0.205 | 1.054 → 0.461 |
| A1_m0_deletion | 3.15 | OOF (fitted here): 208 | 0.116 → 0.031 | 0.222 → 0.189 | 0.957 → 0.428 |
| N_segzoom_fold0 | 3.22 | OOF (fitted here): 208 | 0.115 → 0.033 | 0.223 → 0.197 | 0.968 → 0.436 |
| N_segzoom_fold0_seed2 | 3.22 | OOF (fitted here): 208 | 0.119 → 0.057 | 0.220 → 0.188 | 0.956 → 0.424 |
| N_res1152_fold0 | 3.10 | OOF (fitted here): 208 | 0.106 → 0.036 | 0.207 → 0.177 | 0.848 → 0.396 |
| N_res1152_fold0_seed2 | 3.33 | OOF (fitted here): 208 | 0.114 → 0.076 | 0.240 → 0.206 | 1.061 → 0.462 |
| N_convnext_s_fold0 | 3.18 | OOF (fitted here): 208 | 0.106 → 0.055 | 0.216 → 0.191 | 1.013 → 0.442 |
| N_convnext_s_fold0_seed2 | 3.14 | OOF (fitted here): 208 | 0.137 → 0.074 | 0.261 → 0.201 | 0.954 → 0.430 |

ECE: 15 equal-width bins of the top-class probability. The temperature keeps each femur's most probable class, and no threshold or decision uses it: it is reported only.

## N_segzoom_fold0 vs S10_m0_convnext_unet: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.005 | 0.2001 | 1.0000 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.006 | 0.8506 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.014 | 0.8128 | 1.0000 |

## N_segzoom_fold0_seed2 vs A1_m0_deletion: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.003 | 0.3077 | 1.0000 |
| accuracy_6class | McNemar | +0.010 | 0.7905 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.014 | 0.4531 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.032 | 0.4593 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.002 | 0.9075 | 1.0000 |

## N_res1152_fold0 vs S10_m0_convnext_unet: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.6218 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.039 | 0.3353 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.019 | 0.8963 | 1.0000 |

## N_res1152_fold0_seed2 vs A1_m0_deletion: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.005 | 0.0937 | 0.3748 |
| accuracy_6class | McNemar | -0.014 | 0.6291 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.024 | 0.0625 | 0.3125 |
| macro_f1 | randomisation (swap per x-ray) | -0.022 | 0.5775 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.018 | 0.9230 | 1.0000 |

## N_convnext_s_fold0 vs S10_m0_convnext_unet: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.5027 | 1.0000 |
| accuracy_6class | McNemar | +0.010 | 0.7266 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.008 | 0.7848 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.011 | 0.6926 | 1.0000 |

## N_convnext_s_fold0_seed2 vs A1_m0_deletion: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.004 | 0.1639 | 0.8196 |
| accuracy_6class | McNemar | -0.019 | 0.4545 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.019 | 0.2188 | 0.8750 |
| macro_f1 | randomisation (swap per x-ray) | -0.024 | 0.5201 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.000 | 0.4432 | 1.0000 |
