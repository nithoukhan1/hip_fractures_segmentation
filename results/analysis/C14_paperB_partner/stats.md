# Statistics: S10_m0_convnext_unet, R3_nearest_ref, R4_refrac, R7_partner

95% intervals: cluster bootstrap over x-rays (2000 resamples, seed 2026); AUC also DeLong; 6-class accuracy also exact (Clopper-Pearson). Decisions use each run's OOF threshold.

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

## R3_nearest_ref: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.993 | 0.988–0.997 |
| fracture_sensitivity | 0.984 | 0.974–0.993 |
| fracture_specificity | 0.972 | 0.954–0.989 |
| accuracy_6class | 0.880 | 0.860–0.899 |
| macro_f1 | 0.826 | 0.791–0.856 |
| balanced_accuracy | 0.818 | 0.781–0.853 |
| kappa | 0.843 | 0.817–0.867 |
| dice_fractured_mean | 0.775 | 0.763–0.788 |
| dice_detected_mean | 0.788 | 0.777–0.799 |
| garden12_recall | 0.684 | 0.558–0.800 |
| auc_fracture (DeLong) | 0.993 | 0.988–0.998 |
| accuracy_6class (exact) | 918/1043 = 0.880 | 0.859–0.899 |

## R3_nearest_ref: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.997–1.000 |
| fracture_sensitivity | 1.000 | 1.000–1.000 |
| fracture_specificity | 0.938 | 0.877–0.986 |
| accuracy_6class | 0.843 | 0.793–0.891 |
| macro_f1 | 0.765 | 0.679–0.835 |
| balanced_accuracy | 0.751 | 0.682–0.818 |
| kappa | 0.800 | 0.738–0.860 |
| dice_fractured_mean | 0.742 | 0.717–0.764 |
| dice_detected_mean | 0.742 | 0.717–0.764 |
| garden12_recall | 0.375 | 0.118–0.625 |
| auc_fracture (DeLong) | 0.999 | 0.998–1.000 |
| accuracy_6class (exact) | 177/210 = 0.843 | 0.786–0.889 |

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

## R7_partner: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.995 | 0.991–0.998 |
| fracture_sensitivity | 0.975 | 0.962–0.985 |
| fracture_specificity | 0.992 | 0.981–1.000 |
| accuracy_6class | 0.878 | 0.857–0.897 |
| macro_f1 | 0.819 | 0.784–0.847 |
| balanced_accuracy | 0.811 | 0.775–0.845 |
| kappa | 0.840 | 0.813–0.864 |
| dice_fractured_mean | 0.780 | 0.767–0.792 |
| dice_detected_mean | 0.800 | 0.791–0.808 |
| garden12_recall | 0.632 | 0.500–0.754 |
| auc_fracture (DeLong) | 0.995 | 0.991–0.998 |
| accuracy_6class (exact) | 916/1043 = 0.878 | 0.857–0.897 |

## R7_partner: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.997–1.000 |
| fracture_sensitivity | 0.986 | 0.966–1.000 |
| fracture_specificity | 0.985 | 0.949–1.000 |
| accuracy_6class | 0.867 | 0.818–0.910 |
| macro_f1 | 0.796 | 0.715–0.862 |
| balanced_accuracy | 0.776 | 0.710–0.847 |
| kappa | 0.830 | 0.769–0.884 |
| dice_fractured_mean | 0.741 | 0.716–0.764 |
| dice_detected_mean | 0.752 | 0.731–0.772 |
| garden12_recall | 0.375 | 0.133–0.615 |
| auc_fracture (DeLong) | 0.999 | 0.997–1.000 |
| accuracy_6class (exact) | 182/210 = 0.867 | 0.813–0.910 |

## Calibration: temperature scaling fitted on each run's OOF femurs

| Run | Temperature | Femurs | ECE before → after | Brier before → after | NLL before → after |
|---|---|---|---|---|---|
| S10_m0_convnext_unet | 3.14 | OOF (fitted here): 1043 | 0.108 → 0.040 | 0.224 → 0.194 | 0.941 → 0.424 |
| S10_m0_convnext_unet | 3.14 | test ensemble: 210 | 0.103 → 0.064 | 0.223 → 0.198 | 0.644 → 0.397 |
| S10_m0_convnext_unet | 3.14 | test fold models: 5 models | 0.131 → 0.062 (SD 0.010 → 0.019) | 0.266 → 0.225 (SD 0.011 → 0.011) | 1.047 → 0.465 (SD 0.085 → 0.021) |
| R3_nearest_ref | 3.44 | OOF (fitted here): 1043 | 0.113 → 0.030 | 0.236 → 0.201 | 1.065 → 0.432 |
| R3_nearest_ref | 3.44 | test ensemble: 210 | 0.095 → 0.075 | 0.223 → 0.195 | 0.601 → 0.385 |
| R3_nearest_ref | 3.44 | test fold models: 5 models | 0.141 → 0.058 (SD 0.007 → 0.012) | 0.282 → 0.227 (SD 0.016 → 0.019) | 1.163 → 0.465 (SD 0.163 → 0.043) |
| R4_refrac | 3.56 | OOF (fitted here): 1043 | 0.108 → 0.029 | 0.225 → 0.193 | 1.058 → 0.418 |
| R4_refrac | 3.56 | test ensemble: 210 | 0.106 → 0.089 | 0.221 → 0.191 | 0.634 → 0.383 |
| R4_refrac | 3.56 | test fold models: 5 models | 0.139 → 0.070 (SD 0.014 → 0.013) | 0.275 → 0.225 (SD 0.026 → 0.014) | 1.151 → 0.453 (SD 0.120 → 0.030) |
| R7_partner | 3.58 | OOF (fitted here): 1043 | 0.115 → 0.035 | 0.237 → 0.201 | 1.111 → 0.432 |
| R7_partner | 3.58 | test ensemble: 210 | 0.081 → 0.065 | 0.205 → 0.186 | 0.569 → 0.376 |
| R7_partner | 3.58 | test fold models: 5 models | 0.133 → 0.047 (SD 0.016 → 0.016) | 0.267 → 0.220 (SD 0.034 → 0.026) | 1.154 → 0.455 (SD 0.196 → 0.054) |

ECE: 15 equal-width bins of the top-class probability. The temperature keeps each femur's most probable class, and no threshold or decision uses it: it is reported only.
