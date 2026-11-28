# Statistics: R4_refrac, R1_aasn, R5_concat, R6a_no_align, R6b_affine

95% intervals: cluster bootstrap over x-rays (2000 resamples, seed 2026); AUC also DeLong; 6-class accuracy also exact (Clopper-Pearson). Decisions use each run's OOF threshold.

## R4_refrac: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.990–0.998 |
| fracture_sensitivity | 0.975 | 0.963–0.987 |
| fracture_specificity | 0.981 | 0.965–0.994 |
| accuracy_6class | 0.882 | 0.862–0.901 |
| macro_f1 | 0.826 | 0.792–0.855 |
| balanced_accuracy | 0.820 | 0.784–0.853 |
| kappa | 0.845 | 0.819–0.869 |
| dice_fractured_mean | 0.777 | 0.764–0.789 |
| dice_detected_mean | 0.797 | 0.788–0.805 |
| garden12_recall | 0.702 | 0.579–0.814 |
| auc_fracture (DeLong) | 0.994 | 0.991–0.998 |
| accuracy_6class (exact) | 920/1043 = 0.882 | 0.861–0.901 |

## R4_refrac: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 1.000 | 0.999–1.000 |
| fracture_sensitivity | 0.986 | 0.966–1.000 |
| fracture_specificity | 0.985 | 0.948–1.000 |
| accuracy_6class | 0.857 | 0.807–0.902 |
| macro_f1 | 0.779 | 0.696–0.844 |
| balanced_accuracy | 0.760 | 0.692–0.826 |
| kappa | 0.818 | 0.754–0.873 |
| dice_fractured_mean | 0.739 | 0.713–0.763 |
| dice_detected_mean | 0.749 | 0.726–0.771 |
| garden12_recall | 0.312 | 0.077–0.562 |
| auc_fracture (DeLong) | 1.000 | 0.999–1.000 |
| accuracy_6class (exact) | 180/210 = 0.857 | 0.802–0.901 |

## R1_aasn: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.988–0.997 |
| fracture_sensitivity | 0.982 | 0.972–0.991 |
| fracture_specificity | 0.981 | 0.965–0.994 |
| accuracy_6class | 0.877 | 0.856–0.896 |
| macro_f1 | 0.820 | 0.784–0.849 |
| balanced_accuracy | 0.808 | 0.771–0.842 |
| kappa | 0.839 | 0.812–0.863 |
| dice_fractured_mean | 0.781 | 0.769–0.792 |
| dice_detected_mean | 0.795 | 0.787–0.804 |
| garden12_recall | 0.579 | 0.447–0.707 |
| auc_fracture (DeLong) | 0.994 | 0.989–0.998 |
| accuracy_6class (exact) | 915/1043 = 0.877 | 0.856–0.897 |

## R1_aasn: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.998–1.000 |
| fracture_sensitivity | 0.986 | 0.966–1.000 |
| fracture_specificity | 0.954 | 0.896–1.000 |
| accuracy_6class | 0.862 | 0.813–0.905 |
| macro_f1 | 0.798 | 0.716–0.860 |
| balanced_accuracy | 0.781 | 0.709–0.846 |
| kappa | 0.824 | 0.763–0.878 |
| dice_fractured_mean | 0.738 | 0.712–0.760 |
| dice_detected_mean | 0.748 | 0.727–0.768 |
| garden12_recall | 0.438 | 0.182–0.688 |
| auc_fracture (DeLong) | 0.999 | 0.998–1.000 |
| accuracy_6class (exact) | 181/210 = 0.862 | 0.808–0.906 |

## R5_concat: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.996 | 0.992–0.998 |
| fracture_sensitivity | 0.974 | 0.961–0.985 |
| fracture_specificity | 0.992 | 0.981–1.000 |
| accuracy_6class | 0.887 | 0.868–0.905 |
| macro_f1 | 0.827 | 0.792–0.859 |
| balanced_accuracy | 0.808 | 0.772–0.844 |
| kappa | 0.851 | 0.826–0.875 |
| dice_fractured_mean | 0.780 | 0.768–0.792 |
| dice_detected_mean | 0.802 | 0.794–0.809 |
| garden12_recall | 0.561 | 0.431–0.689 |
| auc_fracture (DeLong) | 0.996 | 0.993–0.999 |
| accuracy_6class (exact) | 925/1043 = 0.887 | 0.866–0.905 |

## R5_concat: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.996–1.000 |
| fracture_sensitivity | 0.986 | 0.966–1.000 |
| fracture_specificity | 0.985 | 0.949–1.000 |
| accuracy_6class | 0.848 | 0.798–0.894 |
| macro_f1 | 0.741 | 0.657–0.811 |
| balanced_accuracy | 0.729 | 0.664–0.796 |
| kappa | 0.805 | 0.742–0.863 |
| dice_fractured_mean | 0.740 | 0.715–0.763 |
| dice_detected_mean | 0.751 | 0.730–0.771 |
| garden12_recall | 0.188 | 0.000–0.412 |
| auc_fracture (DeLong) | 0.999 | 0.997–1.000 |
| accuracy_6class (exact) | 178/210 = 0.848 | 0.792–0.893 |

## R6a_no_align: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.995 | 0.990–0.998 |
| fracture_sensitivity | 0.974 | 0.961–0.985 |
| fracture_specificity | 0.983 | 0.969–0.995 |
| accuracy_6class | 0.871 | 0.850–0.890 |
| macro_f1 | 0.807 | 0.774–0.836 |
| balanced_accuracy | 0.797 | 0.761–0.831 |
| kappa | 0.830 | 0.803–0.854 |
| dice_fractured_mean | 0.777 | 0.764–0.789 |
| dice_detected_mean | 0.798 | 0.789–0.806 |
| garden12_recall | 0.561 | 0.434–0.689 |
| auc_fracture (DeLong) | 0.995 | 0.991–0.998 |
| accuracy_6class (exact) | 908/1043 = 0.871 | 0.849–0.890 |

## R6a_no_align: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.996–1.000 |
| fracture_sensitivity | 0.986 | 0.966–1.000 |
| fracture_specificity | 0.969 | 0.921–1.000 |
| accuracy_6class | 0.857 | 0.807–0.902 |
| macro_f1 | 0.788 | 0.707–0.851 |
| balanced_accuracy | 0.771 | 0.703–0.833 |
| kappa | 0.818 | 0.754–0.873 |
| dice_fractured_mean | 0.741 | 0.716–0.764 |
| dice_detected_mean | 0.751 | 0.730–0.772 |
| garden12_recall | 0.312 | 0.077–0.562 |
| auc_fracture (DeLong) | 0.999 | 0.997–1.000 |
| accuracy_6class (exact) | 180/210 = 0.857 | 0.802–0.901 |

## R6b_affine: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.995 | 0.990–0.998 |
| fracture_sensitivity | 0.971 | 0.957–0.982 |
| fracture_specificity | 0.989 | 0.977–0.997 |
| accuracy_6class | 0.877 | 0.857–0.896 |
| macro_f1 | 0.814 | 0.780–0.845 |
| balanced_accuracy | 0.798 | 0.761–0.833 |
| kappa | 0.838 | 0.812–0.862 |
| dice_fractured_mean | 0.776 | 0.763–0.788 |
| dice_detected_mean | 0.799 | 0.791–0.807 |
| garden12_recall | 0.561 | 0.431–0.692 |
| auc_fracture (DeLong) | 0.995 | 0.991–0.999 |
| accuracy_6class (exact) | 915/1043 = 0.877 | 0.856–0.897 |

## R6b_affine: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.998 | 0.995–1.000 |
| fracture_sensitivity | 0.986 | 0.966–1.000 |
| fracture_specificity | 0.954 | 0.897–1.000 |
| accuracy_6class | 0.862 | 0.811–0.909 |
| macro_f1 | 0.792 | 0.718–0.858 |
| balanced_accuracy | 0.778 | 0.715–0.845 |
| kappa | 0.824 | 0.759–0.883 |
| dice_fractured_mean | 0.742 | 0.717–0.765 |
| dice_detected_mean | 0.753 | 0.733–0.772 |
| garden12_recall | 0.312 | 0.083–0.556 |
| auc_fracture (DeLong) | 0.998 | 0.996–1.000 |
| accuracy_6class (exact) | 181/210 = 0.862 | 0.808–0.906 |

## Calibration: temperature scaling fitted on each run's OOF femurs

| Run | Temperature | Femurs | ECE before → after | Brier before → after | NLL before → after |
|---|---|---|---|---|---|
| R4_refrac | 3.56 | OOF (fitted here): 1043 | 0.108 → 0.029 | 0.225 → 0.193 | 1.058 → 0.418 |
| R4_refrac | 3.56 | test ensemble: 210 | 0.106 → 0.089 | 0.221 → 0.191 | 0.634 → 0.383 |
| R4_refrac | 3.56 | test fold models: 5 models | 0.139 → 0.070 (SD 0.014 → 0.013) | 0.275 → 0.225 (SD 0.026 → 0.014) | 1.151 → 0.453 (SD 0.120 → 0.030) |
| R1_aasn | 3.45 | OOF (fitted here): 1043 | 0.112 → 0.030 | 0.233 → 0.197 | 1.057 → 0.426 |
| R1_aasn | 3.45 | test ensemble: 210 | 0.084 → 0.111 | 0.219 → 0.201 | 0.682 → 0.398 |
| R1_aasn | 3.45 | test fold models: 5 models | 0.138 → 0.056 (SD 0.010 → 0.012) | 0.274 → 0.227 (SD 0.023 → 0.016) | 1.150 → 0.460 (SD 0.147 → 0.033) |
| R5_concat | 3.42 | OOF (fitted here): 1043 | 0.107 → 0.032 | 0.219 → 0.191 | 1.016 → 0.418 |
| R5_concat | 3.42 | test ensemble: 210 | 0.111 → 0.075 | 0.219 → 0.195 | 0.655 → 0.390 |
| R5_concat | 3.42 | test fold models: 5 models | 0.138 → 0.065 (SD 0.009 → 0.019) | 0.270 → 0.228 (SD 0.017 → 0.013) | 1.165 → 0.466 (SD 0.115 → 0.028) |
| R6a_no_align | 3.84 | OOF (fitted here): 1043 | 0.116 → 0.032 | 0.240 → 0.201 | 1.196 → 0.444 |
| R6a_no_align | 3.84 | test ensemble: 210 | 0.088 → 0.080 | 0.208 → 0.194 | 0.592 → 0.383 |
| R6a_no_align | 3.84 | test fold models: 5 models | 0.137 → 0.059 (SD 0.013 → 0.014) | 0.271 → 0.226 (SD 0.021 → 0.015) | 1.197 → 0.450 (SD 0.144 → 0.035) |
| R6b_affine | 3.81 | OOF (fitted here): 1043 | 0.115 → 0.032 | 0.234 → 0.197 | 1.166 → 0.435 |
| R6b_affine | 3.81 | test ensemble: 210 | 0.087 → 0.088 | 0.217 → 0.200 | 0.616 → 0.389 |
| R6b_affine | 3.81 | test fold models: 5 models | 0.138 → 0.065 (SD 0.013 → 0.009) | 0.278 → 0.220 (SD 0.024 → 0.023) | 1.093 → 0.426 (SD 0.202 → 0.047) |

ECE: 15 equal-width bins of the top-class probability. The temperature keeps each femur's most probable class, and no threshold or decision uses it: it is reported only.
R6a_no_align: 3 of 1043 OOF femurs have their true class probability written as 0 (the per-case files keep 8 decimals, so it is below 5e-9); it is read as 1e-12, like every written 0.
R6b_affine: 2 of 1043 OOF femurs have their true class probability written as 0 (the per-case files keep 8 decimals, so it is below 5e-9); it is read as 1e-12, like every written 0.

p (Holm, family): p_holm_family: Holm across the 4 compared pairs, each endpoint and scope separately (an endpoint counts only where it was tested).

## R4_refrac vs R1_aasn: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.5981 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.005 | 0.6089 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 0.3323 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.006 | 0.6491 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.004 | 0.9820 | 1.0000 | 1.0000 |

## R4_refrac vs R1_aasn: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 0.4137 | 1.0000 | 0.7583 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.010 | 0.6250 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.019 | 0.4401 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.001 | 0.0911 | 0.4556 | 0.3644 |

## R4_refrac vs R5_concat: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.1548 | 0.6190 | 0.6190 |
| accuracy_6class | McNemar | -0.005 | 0.5831 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.003 | 0.5811 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.002 | 0.9090 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.004 | 0.0827 | 0.4133 | 0.3306 |

## R4_refrac vs R5_concat: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.2633 | 1.0000 | 0.7583 |
| accuracy_6class | McNemar | +0.010 | 0.7744 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.038 | 0.1718 | 0.8588 | 0.6871 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.001 | 0.9591 | 1.0000 | 1.0000 |

## R4_refrac vs R6a_no_align: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.9437 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.012 | 0.1619 | 0.8094 | 0.6475 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.019 | 0.1726 | 0.8094 | 0.6903 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.000 | 0.5837 | 1.0000 | 1.0000 |

## R4_refrac vs R6a_no_align: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.2528 | 1.0000 | 0.7583 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.005 | 1.0000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.009 | 0.6485 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.002 | 0.9308 | 1.0000 | 1.0000 |

## R4_refrac vs R6b_affine: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.7444 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.005 | 0.5901 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.012 | 0.3219 | 1.0000 | 0.9658 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.001 | 0.8415 | 1.0000 | 1.0000 |

## R4_refrac vs R6b_affine: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.1731 | 0.8654 | 0.6923 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.010 | 0.5000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.013 | 0.6375 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.004 | 0.3744 | 1.0000 | 1.0000 |
