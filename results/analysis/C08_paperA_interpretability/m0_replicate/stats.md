# Statistics: A1_m0_deletion, S10_m0_convnext_unet

95% intervals: cluster bootstrap over x-rays (2000 resamples, seed 2026); AUC also DeLong; 6-class accuracy also exact (Clopper-Pearson). Decisions use each run's OOF threshold.

## A1_m0_deletion: oof (1043 femurs, 701 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.990–0.997 |
| fracture_sensitivity | 0.971 | 0.957–0.982 |
| fracture_specificity | 0.989 | 0.976–0.997 |
| accuracy_6class | 0.883 | 0.862–0.902 |
| macro_f1 | 0.825 | 0.791–0.855 |
| balanced_accuracy | 0.811 | 0.774–0.844 |
| kappa | 0.846 | 0.819–0.870 |
| dice_fractured_mean | 0.778 | 0.765–0.790 |
| dice_detected_mean | 0.802 | 0.794–0.809 |
| garden12_recall | 0.614 | 0.480–0.738 |
| auc_fracture (DeLong) | 0.994 | 0.991–0.998 |
| accuracy_6class (exact) | 921/1043 = 0.883 | 0.862–0.902 |

## A1_m0_deletion: test_ensemble (210 femurs, 146 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 1.000 | 0.999–1.000 |
| fracture_sensitivity | 1.000 | 1.000–1.000 |
| fracture_specificity | 0.938 | 0.875–0.986 |
| accuracy_6class | 0.852 | 0.797–0.899 |
| macro_f1 | 0.783 | 0.695–0.852 |
| balanced_accuracy | 0.763 | 0.690–0.832 |
| kappa | 0.812 | 0.744–0.870 |
| dice_fractured_mean | 0.738 | 0.712–0.762 |
| dice_detected_mean | 0.738 | 0.712–0.762 |
| garden12_recall | 0.438 | 0.182–0.692 |
| auc_fracture (DeLong) | 1.000 | 0.999–1.000 |
| accuracy_6class (exact) | 179/210 = 0.852 | 0.797–0.897 |

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
| A1_m0_deletion | 3.15 | OOF (fitted here): 1043 | 0.108 → 0.032 | 0.227 → 0.194 | 0.945 → 0.422 |
| A1_m0_deletion | 3.15 | test ensemble: 210 | 0.089 → 0.067 | 0.224 → 0.198 | 0.638 → 0.403 |
| A1_m0_deletion | 3.15 | test fold models: 5 models | 0.139 → 0.072 (SD 0.014 → 0.018) | 0.275 → 0.228 (SD 0.025 → 0.017) | 1.070 → 0.472 (SD 0.128 → 0.035) |
| S10_m0_convnext_unet | 3.14 | OOF (fitted here): 1043 | 0.108 → 0.040 | 0.224 → 0.194 | 0.941 → 0.424 |
| S10_m0_convnext_unet | 3.14 | test ensemble: 210 | 0.103 → 0.064 | 0.223 → 0.198 | 0.644 → 0.397 |
| S10_m0_convnext_unet | 3.14 | test fold models: 5 models | 0.131 → 0.062 (SD 0.010 → 0.019) | 0.266 → 0.225 (SD 0.011 → 0.011) | 1.047 → 0.465 (SD 0.085 → 0.021) |

ECE: 15 equal-width bins of the top-class probability. The temperature keeps each femur's most probable class, and no threshold or decision uses it: it is reported only.

## A1_m0_deletion vs S10_m0_convnext_unet: oof

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | -0.000 | 0.5823 | 1.0000 |
| accuracy_6class | McNemar | -0.001 | 1.0000 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.002 | 0.7905 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | -0.000 | 0.9970 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | +0.001 | 0.7858 | 1.0000 |

## A1_m0_deletion vs S10_m0_convnext_unet: test_ensemble

| Endpoint | Test | Difference (A − B) | p | p (Holm) |
|---|---|---|---|---|
| auc_fracture | DeLong | +0.000 | 0.4216 | 1.0000 |
| accuracy_6class | McNemar | -0.014 | 0.5078 | 1.0000 |
| fracture_detection | McNemar (binary fracture decision) | -0.014 | 0.3750 | 1.0000 |
| macro_f1 | randomisation (swap per x-ray) | +0.004 | 0.8736 | 1.0000 |
| dice_fractured | Wilcoxon signed-rank (Pratt), fractured femurs, missed = 0 | -0.006 | 0.0673 | 0.3365 |
