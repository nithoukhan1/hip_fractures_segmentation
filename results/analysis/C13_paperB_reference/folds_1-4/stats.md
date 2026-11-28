# Statistics: S10_m0_convnext_unet, R2_random_ref, R3_nearest_ref, R4_refrac

95% intervals: cluster bootstrap over x-rays (2000 resamples, seed 2026); AUC also DeLong; 6-class accuracy also exact (Clopper-Pearson). Decisions use each run's OOF threshold, fitted again on the femurs of fold(s) 1, 2, 3, 4 only (test set not used).

## S10_m0_convnext_unet: oof (835 femurs, 562 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.990–0.998 |
| fracture_sensitivity | 0.978 | 0.965–0.989 |
| fracture_specificity | 0.990 | 0.976–1.000 |
| accuracy_6class | 0.885 | 0.864–0.906 |
| macro_f1 | 0.828 | 0.793–0.860 |
| balanced_accuracy | 0.811 | 0.771–0.850 |
| kappa | 0.848 | 0.821–0.876 |
| dice_fractured_mean | 0.784 | 0.771–0.797 |
| dice_detected_mean | 0.802 | 0.793–0.810 |
| garden12_recall | 0.543 | 0.407–0.694 |
| auc_fracture (DeLong) | 0.994 | 0.990–0.999 |
| accuracy_6class (exact) | 739/835 = 0.885 | 0.861–0.906 |

## R2_random_ref: oof (835 femurs, 562 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.995 | 0.990–0.998 |
| fracture_sensitivity | 0.976 | 0.963–0.989 |
| fracture_specificity | 0.983 | 0.966–0.997 |
| accuracy_6class | 0.878 | 0.856–0.901 |
| macro_f1 | 0.814 | 0.776–0.848 |
| balanced_accuracy | 0.806 | 0.766–0.844 |
| kappa | 0.839 | 0.810–0.869 |
| dice_fractured_mean | 0.783 | 0.770–0.796 |
| dice_detected_mean | 0.802 | 0.794–0.811 |
| garden12_recall | 0.587 | 0.444–0.727 |
| auc_fracture (DeLong) | 0.995 | 0.991–0.998 |
| accuracy_6class (exact) | 733/835 = 0.878 | 0.854–0.899 |

## R3_nearest_ref: oof (835 femurs, 562 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.993 | 0.987–0.998 |
| fracture_sensitivity | 0.983 | 0.972–0.993 |
| fracture_specificity | 0.969 | 0.947–0.987 |
| accuracy_6class | 0.880 | 0.858–0.900 |
| macro_f1 | 0.829 | 0.790–0.862 |
| balanced_accuracy | 0.823 | 0.785–0.863 |
| kappa | 0.843 | 0.815–0.869 |
| dice_fractured_mean | 0.780 | 0.767–0.794 |
| dice_detected_mean | 0.794 | 0.782–0.804 |
| garden12_recall | 0.717 | 0.579–0.848 |
| auc_fracture (DeLong) | 0.993 | 0.987–0.998 |
| accuracy_6class (exact) | 735/835 = 0.880 | 0.856–0.901 |

## R4_refrac: oof (835 femurs, 562 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.993 | 0.988–0.998 |
| fracture_sensitivity | 0.967 | 0.951–0.982 |
| fracture_specificity | 0.986 | 0.971–0.997 |
| accuracy_6class | 0.877 | 0.855–0.897 |
| macro_f1 | 0.821 | 0.784–0.854 |
| balanced_accuracy | 0.818 | 0.780–0.856 |
| kappa | 0.838 | 0.810–0.865 |
| dice_fractured_mean | 0.778 | 0.763–0.792 |
| dice_detected_mean | 0.805 | 0.796–0.813 |
| garden12_recall | 0.652 | 0.510–0.793 |
| auc_fracture (DeLong) | 0.993 | 0.989–0.998 |
| accuracy_6class (exact) | 732/835 = 0.877 | 0.852–0.898 |

## Calibration: temperature scaling fitted on each run's OOF femurs

| Run | Temperature | Femurs | ECE before → after | Brier before → after | NLL before → after |
|---|---|---|---|---|---|
| S10_m0_convnext_unet | 3.11 | OOF (fitted here): 835 | 0.105 → 0.040 | 0.219 → 0.191 | 0.913 → 0.415 |
| R2_random_ref | 3.55 | OOF (fitted here): 835 | 0.118 → 0.048 | 0.239 → 0.200 | 1.085 → 0.429 |
| R3_nearest_ref | 3.44 | OOF (fitted here): 835 | 0.113 → 0.030 | 0.232 → 0.199 | 1.057 → 0.428 |
| R4_refrac | 3.52 | OOF (fitted here): 835 | 0.110 → 0.031 | 0.228 → 0.196 | 1.047 → 0.419 |

ECE: 15 equal-width bins of the top-class probability. The temperature keeps each femur's most probable class, and no threshold or decision uses it: it is reported only.
