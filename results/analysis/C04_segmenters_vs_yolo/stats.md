# Statistics: S01_yolov8s_seg_1024, S07_unet_r34, S08_deeplabv3p_r34, S09_segformer_b2, S10_m0_convnext_unet

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
| S01_yolov8s_seg_1024 | n/a | OOF (fitted here): 1043 | 0.079 → n/a | 0.323 → n/a | n/a → n/a |
| S01_yolov8s_seg_1024 | n/a | test ensemble: 210 | 0.083 → n/a | 0.275 → n/a | n/a → n/a |
| S01_yolov8s_seg_1024 | n/a | test fold models: 5 models | 0.082 → n/a (SD 0.007 → n/a) | 0.309 → n/a (SD 0.035 → n/a) | n/a → n/a (SD n/a → n/a) |
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
S01_yolov8s_seg_1024: not applicable, 99 of 1043 OOF femurs give their true class probability 0, so the NLL is undefined (YOLO-style confidences: no probability for a class without an instance).

## S01_yolov8s_seg_1024 vs S07_unet_r34: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.014 | 0.0008 | 0.0033 |
| accuracy_6class | McNemar | -0.031 | 0.0045 | 0.0089 |
| fracture_detection | McNemar (binary fracture decision) | -0.029 | 0.0001 | 0.0007 |
| macro_f1 | randomisation (swap per x-ray) | -0.056 | 0.0022 | 0.0066 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.008 | 0.6064 | 0.6064 |

## S01_yolov8s_seg_1024 vs S07_unet_r34: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.005 | 0.0783 | 0.3916 |
| accuracy_6class | McNemar | +0.014 | 0.7011 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.046 | 0.2378 | 0.7721 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.015 | 0.1930 | 0.7721 |

## S01_yolov8s_seg_1024 vs S08_deeplabv3p_r34: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.015 | 0.0002 | 0.0011 |
| accuracy_6class | McNemar | -0.030 | 0.0066 | 0.0165 |
| fracture_detection | McNemar (binary fracture decision) | -0.020 | 0.0055 | 0.0165 |
| macro_f1 | randomisation (swap per x-ray) | -0.055 | 0.0020 | 0.0080 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.009 | 0.6364 | 0.6364 |

## S01_yolov8s_seg_1024 vs S08_deeplabv3p_r34: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.005 | 0.0800 | 0.3999 |
| accuracy_6class | McNemar | +0.019 | 0.5716 | 0.6823 |
| fracture_detection | McNemar (binary fracture decision) | +0.019 | 0.2891 | 0.6823 |
| macro_f1 | randomisation (swap per x-ray) | +0.067 | 0.1218 | 0.4871 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.027 | 0.2274 | 0.6823 |

## S01_yolov8s_seg_1024 vs S09_segformer_b2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.019 | 0.0000 | 0.0001 |
| accuracy_6class | McNemar | -0.047 | 0.0000 | 0.0001 |
| fracture_detection | McNemar (binary fracture decision) | -0.034 | 0.0000 | 0.0001 |
| macro_f1 | randomisation (swap per x-ray) | -0.073 | 0.0006 | 0.0012 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.026 | 0.0139 | 0.0139 |

## S01_yolov8s_seg_1024 vs S09_segformer_b2: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.005 | 0.0798 | 0.3988 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.019 | 0.2188 | 0.8750 |
| macro_f1 | randomisation (swap per x-ray) | +0.041 | 0.2687 | 0.8750 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.012 | 0.7762 | 1.0000 |

## S01_yolov8s_seg_1024 vs S10_m0_convnext_unet: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.019 | 0.0000 | 0.0000 |
| accuracy_6class | McNemar | -0.052 | 0.0000 | 0.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.038 | 0.0000 | 0.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.081 | 0.0002 | 0.0002 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.037 | 0.0000 | 0.0000 |

## S01_yolov8s_seg_1024 vs S10_m0_convnext_unet: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.005 | 0.0681 | 0.3125 |
| accuracy_6class | McNemar | +0.014 | 0.6900 | 0.6900 |
| fracture_detection | McNemar (binary fracture decision) | -0.024 | 0.0625 | 0.3125 |
| macro_f1 | randomisation (swap per x-ray) | +0.066 | 0.1188 | 0.3563 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.018 | 0.1795 | 0.3591 |

## S07_unet_r34 vs S08_deeplabv3p_r34: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.8195 | 1.0000 |
| accuracy_6class | McNemar | +0.001 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.009 | 0.1996 | 0.9980 |
| macro_f1 | randomisation (swap per x-ray) | +0.000 | 0.9840 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.001 | 0.2232 | 0.9980 |

## S07_unet_r34 vs S08_deeplabv3p_r34: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.5294 | 1.0000 |
| accuracy_6class | McNemar | +0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | +0.024 | 0.0625 | 0.3125 |
| macro_f1 | randomisation (swap per x-ray) | +0.020 | 0.4437 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.012 | 0.2347 | 0.9389 |

## S07_unet_r34 vs S09_segformer_b2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.005 | 0.0631 | 0.2525 |
| accuracy_6class | McNemar | -0.016 | 0.0712 | 0.2525 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 0.4996 | 0.4996 |
| macro_f1 | randomisation (swap per x-ray) | -0.017 | 0.2048 | 0.4095 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.018 | 0.0004 | 0.0022 |

## S07_unet_r34 vs S09_segformer_b2: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.8241 | 1.0000 |
| accuracy_6class | McNemar | -0.014 | 0.6072 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.014 | 0.2500 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.006 | 0.8756 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.027 | 0.0024 | 0.0118 |

## S07_unet_r34 vs S10_m0_convnext_unet: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.005 | 0.0749 | 0.1917 |
| accuracy_6class | McNemar | -0.021 | 0.0115 | 0.0461 |
| fracture_detection | McNemar (binary fracture decision) | -0.010 | 0.0639 | 0.1917 |
| macro_f1 | randomisation (swap per x-ray) | -0.025 | 0.0720 | 0.1917 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.029 | 0.0000 | 0.0000 |

## S07_unet_r34 vs S10_m0_convnext_unet: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.2518 | 0.7553 |
| accuracy_6class | McNemar | +0.000 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.019 | 0.1250 | 0.5000 |
| macro_f1 | randomisation (swap per x-ray) | +0.020 | 0.4789 | 0.9578 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.033 | 0.0000 | 0.0000 |

## S08_deeplabv3p_r34 vs S09_segformer_b2: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.004 | 0.1031 | 0.2176 |
| accuracy_6class | McNemar | -0.017 | 0.0725 | 0.2176 |
| fracture_detection | McNemar (binary fracture decision) | -0.013 | 0.0488 | 0.1951 |
| macro_f1 | randomisation (swap per x-ray) | -0.018 | 0.2631 | 0.2631 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.017 | 0.0026 | 0.0128 |

## S08_deeplabv3p_r34 vs S09_segformer_b2: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.001 | 0.6072 | 1.0000 |
| accuracy_6class | McNemar | -0.019 | 0.4545 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.038 | 0.0078 | 0.0391 |
| macro_f1 | randomisation (swap per x-ray) | -0.026 | 0.4667 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.039 | 0.0747 | 0.2988 |

## S08_deeplabv3p_r34 vs S10_m0_convnext_unet: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.004 | 0.0502 | 0.1003 |
| accuracy_6class | McNemar | -0.022 | 0.0117 | 0.0350 |
| fracture_detection | McNemar (binary fracture decision) | -0.018 | 0.0005 | 0.0022 |
| macro_f1 | randomisation (swap per x-ray) | -0.026 | 0.0902 | 0.1003 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.028 | 0.0000 | 0.0000 |

## S08_deeplabv3p_r34 vs S10_m0_convnext_unet: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 1.0000 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.043 | 0.0039 | 0.0156 |
| macro_f1 | randomisation (swap per x-ray) | -0.001 | 0.9966 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.045 | 0.0001 | 0.0005 |

## S09_segformer_b2 vs S10_m0_convnext_unet: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.8106 | 1.0000 |
| accuracy_6class | McNemar | -0.005 | 0.6305 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 0.4421 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.008 | 0.5897 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.011 | 0.0001 | 0.0003 |

## S09_segformer_b2 vs S10_m0_convnext_unet: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.001 | 0.3403 | 1.0000 |
| accuracy_6class | McNemar | +0.014 | 0.6291 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.005 | 1.0000 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.025 | 0.5113 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.006 | 0.0874 | 0.4370 |
