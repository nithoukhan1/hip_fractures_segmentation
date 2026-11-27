# Statistics: A6_slgnet, S01_yolov8s_seg_1024, S04_densenet121, S05_efficientnet_b4, S06_convnext_tiny, S07_unet_r34, S08_deeplabv3p_r34, S09_segformer_b2, S10_m0_convnext_unet

95% intervals: cluster bootstrap over x-rays (2000 resamples, seed 2026); AUC also DeLong; 6-class accuracy also exact (Clopper-Pearson). Decisions use each run's OOF threshold.

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

## S07_unet_r34: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.990 | 0.984–0.994 |
| fracture_sensitivity | 0.969 | 0.956–0.982 |
| fracture_specificity | 0.970 | 0.951–0.986 |
| accuracy_6class | 0.863 | 0.841–0.883 |
| macro_f1 | 0.800 | 0.766–0.832 |
| balanced_accuracy | 0.802 | 0.765–0.836 |
| kappa | 0.820 | 0.792–0.846 |
| dice_fractured_mean | 0.748 | 0.734–0.762 |
| dice_detected_mean | 0.772 | 0.761–0.782 |
| garden12_recall | 0.614 | 0.485–0.739 |
| auc_fracture (DeLong) | 0.990 | 0.984–0.995 |
| accuracy_6class (exact) | 900/1043 = 0.863 | 0.841–0.883 |

## S07_unet_r34: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.996–1.000 |
| fracture_sensitivity | 0.966 | 0.932–0.993 |
| fracture_specificity | 1.000 | 1.000–1.000 |
| accuracy_6class | 0.867 | 0.816–0.908 |
| macro_f1 | 0.799 | 0.714–0.863 |
| balanced_accuracy | 0.771 | 0.699–0.838 |
| kappa | 0.829 | 0.765–0.882 |
| dice_fractured_mean | 0.711 | 0.678–0.740 |
| dice_detected_mean | 0.736 | 0.712–0.757 |
| garden12_recall | 0.375 | 0.133–0.625 |
| auc_fracture (DeLong) | 0.999 | 0.997–1.000 |
| accuracy_6class (exact) | 182/210 = 0.867 | 0.813–0.910 |

## S08_deeplabv3p_r34: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.990 | 0.985–0.994 |
| fracture_sensitivity | 0.959 | 0.943–0.972 |
| fracture_specificity | 0.964 | 0.942–0.983 |
| accuracy_6class | 0.862 | 0.840–0.882 |
| macro_f1 | 0.800 | 0.765–0.830 |
| balanced_accuracy | 0.798 | 0.764–0.831 |
| kappa | 0.818 | 0.790–0.845 |
| dice_fractured_mean | 0.749 | 0.734–0.763 |
| dice_detected_mean | 0.781 | 0.772–0.791 |
| garden12_recall | 0.596 | 0.469–0.721 |
| auc_fracture (DeLong) | 0.990 | 0.985–0.995 |
| accuracy_6class (exact) | 899/1043 = 0.862 | 0.840–0.882 |

## S08_deeplabv3p_r34: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.998–1.000 |
| fracture_sensitivity | 0.931 | 0.889–0.972 |
| fracture_specificity | 1.000 | 1.000–1.000 |
| accuracy_6class | 0.862 | 0.812–0.907 |
| macro_f1 | 0.778 | 0.696–0.847 |
| balanced_accuracy | 0.753 | 0.688–0.820 |
| kappa | 0.822 | 0.760–0.878 |
| dice_fractured_mean | 0.698 | 0.661–0.734 |
| dice_detected_mean | 0.750 | 0.729–0.771 |
| garden12_recall | 0.250 | 0.056–0.500 |
| auc_fracture (DeLong) | 0.999 | 0.998–1.000 |
| accuracy_6class (exact) | 181/210 = 0.862 | 0.808–0.906 |

## S09_segformer_b2: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.990–0.997 |
| fracture_sensitivity | 0.976 | 0.965–0.987 |
| fracture_specificity | 0.970 | 0.951–0.986 |
| accuracy_6class | 0.879 | 0.858–0.898 |
| macro_f1 | 0.817 | 0.784–0.847 |
| balanced_accuracy | 0.810 | 0.775–0.844 |
| kappa | 0.841 | 0.814–0.865 |
| dice_fractured_mean | 0.766 | 0.753–0.778 |
| dice_detected_mean | 0.784 | 0.774–0.794 |
| garden12_recall | 0.579 | 0.450–0.714 |
| auc_fracture (DeLong) | 0.994 | 0.990–0.998 |
| accuracy_6class (exact) | 917/1043 = 0.879 | 0.858–0.898 |

## S09_segformer_b2: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.999 | 0.996–1.000 |
| fracture_sensitivity | 0.986 | 0.966–1.000 |
| fracture_specificity | 1.000 | 1.000–1.000 |
| accuracy_6class | 0.881 | 0.834–0.923 |
| macro_f1 | 0.804 | 0.721–0.870 |
| balanced_accuracy | 0.786 | 0.718–0.854 |
| kappa | 0.848 | 0.790–0.900 |
| dice_fractured_mean | 0.737 | 0.711–0.760 |
| dice_detected_mean | 0.747 | 0.725–0.767 |
| garden12_recall | 0.375 | 0.125–0.625 |
| auc_fracture (DeLong) | 0.999 | 0.997–1.000 |
| accuracy_6class (exact) | 185/210 = 0.881 | 0.829–0.921 |

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

## Calibration: temperature scaling fitted on each run's OOF femurs

| Run | Temperature | Femurs | ECE before → after | Brier before → after | NLL before → after |
|---|---|---|---|---|---|
| A6_slgnet | 3.41 | OOF (fitted here): 1043 | 0.119 → 0.037 | 0.248 → 0.215 | 1.179 → 0.489 |
| A6_slgnet | 3.41 | test ensemble: 210 | 0.081 → 0.083 | 0.203 → 0.195 | 0.578 → 0.389 |
| A6_slgnet | 3.41 | test fold models: 5 models | 0.131 → 0.066 (SD 0.019 → 0.015) | 0.260 → 0.215 (SD 0.032 → 0.021) | 0.985 → 0.438 (SD 0.141 → 0.038) |
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
| S07_unet_r34 | 2.22 | OOF (fitted here): 1043 | 0.115 → 0.035 | 0.258 → 0.240 | 0.927 → 0.573 |
| S07_unet_r34 | 2.22 | test ensemble: 210 | 0.083 → 0.099 | 0.206 → 0.220 | 0.558 → 0.485 |
| S07_unet_r34 | 2.22 | test fold models: 5 models | 0.140 → 0.062 (SD 0.017 → 0.025) | 0.280 → 0.252 (SD 0.028 → 0.021) | 0.931 → 0.576 (SD 0.083 → 0.037) |
| S08_deeplabv3p_r34 | 2.36 | OOF (fitted here): 1043 | 0.115 → 0.031 | 0.250 → 0.223 | 0.863 → 0.509 |
| S08_deeplabv3p_r34 | 2.36 | test ensemble: 210 | 0.075 → 0.098 | 0.194 → 0.209 | 0.525 → 0.444 |
| S08_deeplabv3p_r34 | 2.36 | test fold models: 5 models | 0.124 → 0.055 (SD 0.016 → 0.024) | 0.267 → 0.239 (SD 0.031 → 0.022) | 0.909 → 0.532 (SD 0.072 → 0.037) |
| S09_segformer_b2 | 2.39 | OOF (fitted here): 1043 | 0.096 → 0.050 | 0.221 → 0.209 | 0.804 → 0.475 |
| S09_segformer_b2 | 2.39 | test ensemble: 210 | 0.074 → 0.084 | 0.191 → 0.198 | 0.530 → 0.420 |
| S09_segformer_b2 | 2.39 | test fold models: 5 models | 0.115 → 0.064 (SD 0.011 → 0.013) | 0.243 → 0.223 (SD 0.023 → 0.020) | 0.852 → 0.489 (SD 0.086 → 0.035) |
| S10_m0_convnext_unet | 3.14 | OOF (fitted here): 1043 | 0.108 → 0.040 | 0.224 → 0.194 | 0.941 → 0.424 |
| S10_m0_convnext_unet | 3.14 | test ensemble: 210 | 0.103 → 0.064 | 0.223 → 0.198 | 0.644 → 0.397 |
| S10_m0_convnext_unet | 3.14 | test fold models: 5 models | 0.131 → 0.062 (SD 0.010 → 0.019) | 0.266 → 0.225 (SD 0.011 → 0.011) | 1.047 → 0.465 (SD 0.085 → 0.021) |

ECE: 15 equal-width bins of the top-class probability. The temperature keeps each femur's most probable class, and no threshold or decision uses it: it is reported only.
A6_slgnet: 4 of 1043 OOF femurs have their true class probability written as 0 (the per-case files keep 8 decimals, so it is below 5e-9); it is read as 1e-12, like every written 0.
S01_yolov8s_seg_1024: not applicable, 99 of 1043 OOF femurs give their true class probability 0, so the NLL is undefined (YOLO-style confidences: no probability for a class without an instance).
S05_efficientnet_b4: 1 of 1043 OOF femurs have their true class probability written as 0 (the per-case files keep 8 decimals, so it is below 5e-9); it is read as 1e-12, like every written 0.

p (Holm, family): p_holm_family: Holm across the 8 compared pairs, each endpoint and scope separately (an endpoint counts only where it was tested).

## A6_slgnet vs S01_yolov8s_seg_1024: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.021 | 0.0000 | 0.0000 | 0.0000 |
| accuracy_6class | McNemar | +0.037 | 0.0004 | 0.0008 | 0.0032 |
| fracture_detection | McNemar (binary fracture decision) | +0.035 | 0.0000 | 0.0000 | 0.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.061 | 0.0008 | 0.0008 | 0.0064 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.034 | 0.0000 | 0.0000 | 0.0000 |

## A6_slgnet vs S01_yolov8s_seg_1024: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.005 | 0.1124 | 0.5620 | 0.8909 |
| accuracy_6class | McNemar | +0.005 | 1.0000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.014 | 0.4531 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.025 | 0.5111 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.012 | 0.4116 | 1.0000 | 0.8232 |

## A6_slgnet vs S04_densenet121: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.005 | 0.0023 | 0.0090 | 0.0135 |
| accuracy_6class | McNemar | +0.002 | 0.9132 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.007 | 0.2100 | 0.6301 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.004 | 0.8192 | 1.0000 | 1.0000 |

## A6_slgnet vs S04_densenet121: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.3526 | 0.7895 | 1.0000 |
| accuracy_6class | McNemar | +0.029 | 0.2632 | 0.7895 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.005 | 1.0000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.061 | 0.1574 | 0.6295 | 0.9442 |

## A6_slgnet vs S05_efficientnet_b4: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.006 | 0.0018 | 0.0074 | 0.0129 |
| accuracy_6class | McNemar | +0.022 | 0.0235 | 0.0401 | 0.1644 |
| fracture_detection | McNemar (binary fracture decision) | +0.014 | 0.0201 | 0.0401 | 0.1204 |
| macro_f1 | randomisation (swap per x-ray) | +0.050 | 0.0028 | 0.0084 | 0.0196 |

## A6_slgnet vs S05_efficientnet_b4: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.005 | 0.1114 | 0.1800 | 0.8909 |
| accuracy_6class | McNemar | +0.076 | 0.0070 | 0.0280 | 0.0560 |
| fracture_detection | McNemar (binary fracture decision) | +0.043 | 0.0117 | 0.0352 | 0.0938 |
| macro_f1 | randomisation (swap per x-ray) | +0.076 | 0.0900 | 0.1800 | 0.6299 |

## A6_slgnet vs S06_convnext_tiny: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.006 | 0.0031 | 0.0126 | 0.0157 |
| accuracy_6class | McNemar | +0.006 | 0.5504 | 0.5726 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.006 | 0.2863 | 0.5726 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.022 | 0.1400 | 0.4199 | 0.8398 |

## A6_slgnet vs S06_convnext_tiny: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.2399 | 0.4798 | 1.0000 |
| accuracy_6class | McNemar | +0.038 | 0.0215 | 0.0645 | 0.1504 |
| fracture_detection | McNemar (binary fracture decision) | +0.000 | 1.0000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.079 | 0.0064 | 0.0256 | 0.0512 |

## A6_slgnet vs S07_unet_r34: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.007 | 0.0048 | 0.0193 | 0.0157 |
| accuracy_6class | McNemar | +0.007 | 0.4887 | 0.9774 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.007 | 0.2649 | 0.7948 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.006 | 0.6729 | 0.9774 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.026 | 0.0000 | 0.0000 | 0.0000 |

## A6_slgnet vs S07_unet_r34: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 1.0000 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.019 | 0.4240 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.010 | 0.6250 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.021 | 0.4995 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.026 | 0.0000 | 0.0002 | 0.0002 |

## A6_slgnet vs S08_deeplabv3p_r34: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.006 | 0.0034 | 0.0134 | 0.0157 |
| accuracy_6class | McNemar | +0.008 | 0.4340 | 0.8681 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.015 | 0.0090 | 0.0271 | 0.0633 |
| macro_f1 | randomisation (swap per x-ray) | +0.006 | 0.6853 | 0.8681 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.025 | 0.0000 | 0.0000 | 0.0000 |

## A6_slgnet vs S08_deeplabv3p_r34: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.5191 | 0.6035 | 1.0000 |
| accuracy_6class | McNemar | +0.024 | 0.3018 | 0.6035 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.033 | 0.0391 | 0.1562 | 0.2734 |
| macro_f1 | randomisation (swap per x-ray) | +0.042 | 0.1920 | 0.5759 | 0.9598 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.039 | 0.0024 | 0.0122 | 0.0098 |

## A6_slgnet vs S09_segformer_b2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.002 | 0.2497 | 0.9986 | 0.4931 |
| accuracy_6class | McNemar | -0.010 | 0.3143 | 0.9986 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.002 | 0.8450 | 0.9986 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.012 | 0.4197 | 0.9986 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.008 | 0.0016 | 0.0079 | 0.0032 |

## A6_slgnet vs S09_segformer_b2: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.8881 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.005 | 1.0000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.016 | 0.5661 | 1.0000 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.000 | 0.2379 | 1.0000 | 0.7136 |

## A6_slgnet vs S10_m0_convnext_unet: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | +0.002 | 0.2465 | 0.5201 | 0.4931 |
| accuracy_6class | McNemar | -0.014 | 0.0769 | 0.3803 | 0.4616 |
| fracture_detection | McNemar (binary fracture decision) | -0.003 | 0.6072 | 0.6072 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.020 | 0.1734 | 0.5201 | 0.8668 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.002 | 0.0761 | 0.3803 | 0.0761 |

## A6_slgnet vs S10_m0_convnext_unet: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) | p (Holm, family) |
|---|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.2876 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | +0.019 | 0.3438 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.010 | 0.5000 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.041 | 0.2138 | 1.0000 | 0.9598 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.006 | 0.6598 | 1.0000 | 0.8232 |
