# Statistics: S06_convnext_tiny, S10_m0_convnext_unet, A2_region, A3_geometry, A4_hierarchy, A5_shared_grade, A6_slgnet

95% intervals: cluster bootstrap over x-rays (2000 resamples, seed 2026); AUC also DeLong; 6-class accuracy also exact (Clopper-Pearson). Decisions use each run's OOF threshold, fitted again on the femurs of fold(s) 1, 2, 3, 4 only (test set not used).

## S06_convnext_tiny: oof (835 femurs, 562 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.990 | 0.984–0.996 |
| fracture_sensitivity | 0.969 | 0.953–0.983 |
| fracture_specificity | 0.979 | 0.962–0.993 |
| accuracy_6class | 0.867 | 0.844–0.890 |
| macro_f1 | 0.786 | 0.746–0.824 |
| balanced_accuracy | 0.767 | 0.727–0.809 |
| kappa | 0.824 | 0.794–0.853 |
| dice_fractured_mean | n/a | n/a–n/a |
| dice_detected_mean | n/a | n/a–n/a |
| garden12_recall | 0.435 | 0.294–0.583 |
| auc_fracture (DeLong) | 0.990 | 0.985–0.996 |
| accuracy_6class (exact) | 724/835 = 0.867 | 0.842–0.889 |

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

## A2_region: oof (835 femurs, 562 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.995 | 0.992–0.998 |
| fracture_sensitivity | 0.976 | 0.963–0.989 |
| fracture_specificity | 0.983 | 0.965–0.997 |
| accuracy_6class | 0.883 | 0.861–0.904 |
| macro_f1 | 0.823 | 0.786–0.856 |
| balanced_accuracy | 0.809 | 0.771–0.847 |
| kappa | 0.845 | 0.817–0.874 |
| dice_fractured_mean | 0.776 | 0.762–0.790 |
| dice_detected_mean | 0.795 | 0.786–0.805 |
| garden12_recall | 0.522 | 0.383–0.667 |
| auc_fracture (DeLong) | 0.995 | 0.992–0.999 |
| accuracy_6class (exact) | 737/835 = 0.883 | 0.859–0.904 |

## A3_geometry: oof (835 femurs, 562 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.995 | 0.991–0.999 |
| fracture_sensitivity | 0.971 | 0.956–0.984 |
| fracture_specificity | 0.986 | 0.972–0.997 |
| accuracy_6class | 0.887 | 0.865–0.909 |
| macro_f1 | 0.836 | 0.800–0.869 |
| balanced_accuracy | 0.820 | 0.781–0.860 |
| kappa | 0.852 | 0.823–0.879 |
| dice_fractured_mean | 0.777 | 0.762–0.791 |
| dice_detected_mean | 0.801 | 0.792–0.809 |
| garden12_recall | 0.630 | 0.488–0.771 |
| auc_fracture (DeLong) | 0.995 | 0.991–0.999 |
| accuracy_6class (exact) | 741/835 = 0.887 | 0.864–0.908 |

## A4_hierarchy: oof (835 femurs, 562 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.993 | 0.987–0.998 |
| fracture_sensitivity | 0.982 | 0.969–0.993 |
| fracture_specificity | 0.986 | 0.971–0.997 |
| accuracy_6class | 0.881 | 0.859–0.902 |
| macro_f1 | 0.832 | 0.796–0.864 |
| balanced_accuracy | 0.827 | 0.788–0.864 |
| kappa | 0.844 | 0.815–0.871 |
| dice_fractured_mean | 0.777 | 0.762–0.790 |
| dice_detected_mean | 0.791 | 0.780–0.802 |
| garden12_recall | 0.652 | 0.511–0.792 |
| auc_fracture (DeLong) | 0.993 | 0.988–0.998 |
| accuracy_6class (exact) | 736/835 = 0.881 | 0.858–0.903 |

## A5_shared_grade: oof (835 femurs, 562 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.994 | 0.988–0.998 |
| fracture_sensitivity | 0.983 | 0.973–0.993 |
| fracture_specificity | 0.983 | 0.966–0.997 |
| accuracy_6class | 0.875 | 0.853–0.897 |
| macro_f1 | 0.815 | 0.777–0.848 |
| balanced_accuracy | 0.803 | 0.761–0.841 |
| kappa | 0.836 | 0.806–0.864 |
| dice_fractured_mean | 0.774 | 0.759–0.788 |
| dice_detected_mean | 0.787 | 0.775–0.799 |
| garden12_recall | 0.587 | 0.440–0.721 |
| auc_fracture (DeLong) | 0.994 | 0.989–0.998 |
| accuracy_6class (exact) | 731/835 = 0.875 | 0.851–0.897 |

## A6_slgnet: oof (835 femurs, 562 x-rays)

| Metric | Estimate | 95% CI |
|---|---|---|
| auc_fracture | 0.996 | 0.993–0.999 |
| fracture_sensitivity | 0.971 | 0.956–0.984 |
| fracture_specificity | 0.986 | 0.971–0.997 |
| accuracy_6class | 0.871 | 0.848–0.893 |
| macro_f1 | 0.808 | 0.770–0.843 |
| balanced_accuracy | 0.806 | 0.765–0.844 |
| kappa | 0.830 | 0.801–0.859 |
| dice_fractured_mean | 0.779 | 0.765–0.793 |
| dice_detected_mean | 0.802 | 0.795–0.810 |
| garden12_recall | 0.587 | 0.452–0.726 |
| auc_fracture (DeLong) | 0.996 | 0.993–0.999 |
| accuracy_6class (exact) | 727/835 = 0.871 | 0.846–0.893 |

## Calibration: temperature scaling fitted on each run's OOF femurs

| Run | Temperature | Femurs | ECE before → after | Brier before → after | NLL before → after |
|---|---|---|---|---|---|
| S06_convnext_tiny | 3.41 | OOF (fitted here): 835 | 0.124 → 0.043 | 0.255 → 0.215 | 1.120 → 0.456 |
| S10_m0_convnext_unet | 3.11 | OOF (fitted here): 835 | 0.105 → 0.040 | 0.219 → 0.191 | 0.913 → 0.415 |
| A2_region | 2.76 | OOF (fitted here): 835 | 0.106 → 0.034 | 0.223 → 0.196 | 0.876 → 0.438 |
| A3_geometry | 2.66 | OOF (fitted here): 835 | 0.098 → 0.046 | 0.205 → 0.181 | 0.776 → 0.406 |
| A4_hierarchy | 3.21 | OOF (fitted here): 835 | 0.116 → 0.027 | 0.235 → 0.198 | 1.026 → 0.445 |
| A5_shared_grade | 3.37 | OOF (fitted here): 835 | 0.115 → 0.041 | 0.244 → 0.205 | 1.107 → 0.467 |
| A6_slgnet | 3.43 | OOF (fitted here): 835 | 0.116 → 0.035 | 0.244 → 0.214 | 1.192 → 0.493 |

ECE: 15 equal-width bins of the top-class probability. The temperature keeps each femur's most probable class, and no threshold or decision uses it: it is reported only.
A5_shared_grade: 3 of 835 OOF femurs have their true class probability written as 0 (the per-case files keep 8 decimals, so it is below 5e-9); it is read as 1e-12, like every written 0.
A6_slgnet: 4 of 835 OOF femurs have their true class probability written as 0 (the per-case files keep 8 decimals, so it is below 5e-9); it is read as 1e-12, like every written 0.
