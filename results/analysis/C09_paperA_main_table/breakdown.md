# Breakdown: A6_slgnet, S01_yolov8s_seg_1024, S04_densenet121, S05_efficientnet_b4, S06_convnext_tiny, S07_unet_r34, S08_deeplabv3p_r34, S09_segformer_b2, S10_m0_convnext_unet

Made by `scripts/run_breakdown.py` from the filed run folders. Decisions use each run's OOF threshold. Mean ± SD is over the 5 held-out folds (OOF) or the 5 fold models (test); SD uses n − 1.

## A6_slgnet

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A6` at 768x512 px; fracture threshold 0.0156 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 88.5%) is fold 1.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.996 | 0.997 ± 0.003 | 0.999 ± 0.001 | 0.999 | 0.999 |
| Fracture sensitivity | 0.972 | 0.972 ± 0.016 | 0.972 ± 0.007 | 0.966 | 0.986 |
| Fracture specificity | 0.983 | 0.984 ± 0.011 | 0.991 ± 0.008 | 0.985 | 0.985 |
| 6-class accuracy (%) | 87.0 | 87.0 ± 1.1 | 86.3 ± 1.5 | 84.3 | 88.6 |
| Macro-F1 | 0.806 | 0.805 ± 0.024 | 0.789 ± 0.015 | 0.769 | 0.820 |
| Balanced accuracy | 0.802 | 0.803 ± 0.016 | 0.774 ± 0.018 | 0.744 | 0.805 |
| Cohen's kappa | 0.829 | 0.829 ± 0.014 | 0.825 ± 0.019 | 0.799 | 0.854 |
| Location accuracy (fractures called) | 0.946 | 0.945 ± 0.009 | 0.960 ± 0.009 | 0.964 | 0.972 |
| Grade accuracy, location right | 0.875 | 0.875 ± 0.016 | 0.854 ± 0.018 | 0.828 | 0.869 |
| Dice, fractured femurs (missed = 0) | 0.774 | 0.774 ± 0.015 | 0.725 ± 0.007 | 0.717 | 0.737 |
| Dice, detected fractures only | 0.796 | 0.796 ± 0.015 | 0.745 ± 0.003 | 0.742 | 0.747 |
| Predicted / true area, median | 0.96 | 0.96 ± 0.02 | 1.24 ± 0.05 | 1.29 | 1.23 |
| HD95 (px), median, detected | 62.0 | 62.9 ± 2.5 | 68.2 ± 1.5 | 67.6 | 66.6 |
| Errors: detection | 25.0 | 5.0 ± 1.4 | 4.6 ± 0.9 | 6.0 | 3.0 |
| Errors: location | 36.0 | 7.2 ± 1.1 | 5.6 ± 1.3 | 5.0 | 4.0 |
| Errors: grade | 75.0 | 15.0 ± 2.0 | 18.6 ± 2.3 | 22.0 | 17.0 |
| Errors: total | 136.0 | 27.2 ± 2.3 | 28.8 ± 3.1 | 33.0 | 24.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.983 | 0.984 ± 0.011 | 0.991 ± 0.008 | 0.985 | 0.985 |
| 2AB Neck Garden I-II | 57 / 16 | 0.614 | 0.617 ± 0.134 | 0.287 ± 0.034 | 0.312 | 0.312 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.878 | 0.878 ± 0.071 | 0.985 ± 0.013 | 0.976 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.784 | 0.784 ± 0.071 | 0.894 ± 0.049 | 0.824 | 0.971 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.804 | 0.805 ± 0.069 | 0.786 ± 0.051 | 0.786 | 0.810 |
| 4 Subtrochanteric | 36 / 12 | 0.750 | 0.750 ± 0.118 | 0.700 ± 0.075 | 0.583 | 0.750 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.612 | 0.613 ± 0.030 | 0.573 ± 0.022 | 0.603 | 0.895 / 0.838 ± 0.034 | 0.684 / 0.685 ± 0.018 | 0.90 / 1.69 ± 0.13 |
| 2CD Neck Garden III-IV | 0.769 | 0.769 ± 0.018 | 0.692 ± 0.007 | 0.699 | 0.976 / 1.000 ± 0.000 | 0.788 / 0.692 ± 0.007 | 0.94 / 1.73 ± 0.05 |
| 3AB Pertroch Evans I-II | 0.774 | 0.774 ± 0.049 | 0.727 ± 0.022 | 0.747 | 0.970 / 0.965 ± 0.025 | 0.797 / 0.754 ± 0.009 | 1.03 / 1.13 ± 0.05 |
| 3CDE Pertroch Evans III-V | 0.846 | 0.846 ± 0.014 | 0.836 ± 0.004 | 0.842 | 1.000 / 1.000 ± 0.000 | 0.846 / 0.836 ± 0.004 | 0.96 / 1.03 ± 0.02 |
| 4 Subtrochanteric | 0.674 | 0.673 ± 0.097 | 0.641 ± 0.034 | 0.651 | 0.917 / 0.983 ± 0.037 | 0.735 / 0.652 ± 0.029 | 1.06 / 0.91 ± 0.04 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 357 | 1 | 4 | 1 | 0 | 0 |
| 2AB | 6 | 35 | 11 | 5 | 0 | 0 |
| 2CD | 6 | 16 | 223 | 7 | 2 | 0 |
| 3AB | 4 | 0 | 1 | 105 | 22 | 2 |
| 3CDE | 0 | 1 | 6 | 26 | 160 | 6 |
| 4 | 3 | 0 | 0 | 1 | 5 | 27 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 64 | 1 | 0 | 0 | 0 | 0 |
| 2AB | 2 | 5 | 9 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 33 | 1 | 0 |
| 3CDE | 0 | 0 | 1 | 7 | 34 | 0 |
| 4 | 0 | 0 | 0 | 0 | 3 | 9 |

## S01_yolov8s_seg_1024

Model `yolov8s-seg.pt` at 1024 px; fracture threshold 0.1272 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 85.6%) is fold 1.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.975 | 0.976 ± 0.010 | 0.985 ± 0.005 | 0.981 | 0.994 |
| Fracture sensitivity | 0.925 | 0.925 ± 0.026 | 0.935 ± 0.014 | 0.917 | 0.966 |
| Fracture specificity | 0.970 | 0.970 ± 0.025 | 0.978 ± 0.008 | 0.969 | 0.985 |
| 6-class accuracy (%) | 83.2 | 83.2 ± 2.6 | 84.6 ± 2.1 | 83.8 | 88.1 |
| Macro-F1 | 0.744 | 0.740 ± 0.045 | 0.791 ± 0.018 | 0.793 | 0.845 |
| Balanced accuracy | 0.738 | 0.738 ± 0.048 | 0.780 ± 0.015 | 0.774 | 0.840 |
| Cohen's kappa | 0.779 | 0.779 ± 0.033 | 0.803 ± 0.027 | 0.793 | 0.849 |
| Location accuracy (fractures called) | 0.941 | 0.942 ± 0.019 | 0.966 ± 0.012 | 0.962 | 0.986 |
| Grade accuracy, location right | 0.867 | 0.867 ± 0.051 | 0.861 ± 0.029 | 0.876 | 0.866 |
| Dice, fractured femurs (missed = 0) | 0.740 | 0.740 ± 0.023 | 0.693 ± 0.016 | 0.683 | 0.725 |
| Dice, detected fractures only | 0.800 | 0.800 ± 0.014 | 0.741 ± 0.009 | 0.745 | 0.751 |
| Predicted / true area, median | 0.99 | 0.99 ± 0.01 | 1.28 ± 0.02 | 1.27 | 1.26 |
| HD95 (px), median, detected | 63.0 | 63.4 ± 4.5 | 72.3 ± 1.4 | 72.0 | 68.3 |
| Errors: detection | 62.0 | 12.4 ± 4.4 | 10.8 ± 2.2 | 14.0 | 6.0 |
| Errors: location | 37.0 | 7.4 ± 2.6 | 4.6 ± 1.5 | 5.0 | 2.0 |
| Errors: grade | 76.0 | 15.2 ± 5.8 | 17.0 ± 3.5 | 15.0 | 17.0 |
| Errors: total | 175.0 | 35.0 ± 5.4 | 32.4 ± 4.5 | 34.0 | 25.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.970 | 0.970 ± 0.025 | 0.978 ± 0.008 | 0.969 | 0.985 |
| 2AB Neck Garden I-II | 57 / 16 | 0.509 | 0.512 ± 0.215 | 0.438 ± 0.117 | 0.625 | 0.500 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.831 | 0.831 ± 0.076 | 0.961 ± 0.028 | 0.951 | 0.976 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.716 | 0.717 ± 0.113 | 0.888 ± 0.060 | 0.824 | 0.971 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.794 | 0.795 ± 0.100 | 0.681 ± 0.087 | 0.690 | 0.690 |
| 4 Subtrochanteric | 36 / 12 | 0.611 | 0.607 ± 0.240 | 0.733 ± 0.124 | 0.583 | 0.917 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.534 | 0.533 ± 0.070 | 0.524 ± 0.036 | 0.556 | 0.737 / 0.825 ± 0.052 | 0.725 / 0.635 ± 0.023 | 1.06 / 2.04 ± 0.09 |
| 2CD Neck Garden III-IV | 0.751 | 0.752 ± 0.028 | 0.677 ± 0.013 | 0.694 | 0.949 / 0.990 ± 0.013 | 0.792 / 0.684 ± 0.007 | 0.99 / 1.78 ± 0.06 |
| 3AB Pertroch Evans I-II | 0.721 | 0.720 ± 0.071 | 0.713 ± 0.039 | 0.754 | 0.918 / 0.935 ± 0.038 | 0.785 / 0.762 ± 0.013 | 1.06 / 1.16 ± 0.02 |
| 3CDE Pertroch Evans III-V | 0.822 | 0.822 ± 0.021 | 0.781 ± 0.019 | 0.814 | 0.975 / 0.952 ± 0.017 | 0.843 / 0.820 ± 0.012 | 0.97 / 1.05 ± 0.03 |
| 4 Subtrochanteric | 0.601 | 0.598 ± 0.163 | 0.608 ± 0.024 | 0.669 | 0.806 / 0.833 ± 0.059 | 0.746 / 0.731 ± 0.027 | 1.03 / 0.95 ± 0.06 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 352 | 7 | 2 | 0 | 1 | 1 |
| 2AB | 15 | 29 | 10 | 3 | 0 | 0 |
| 2CD | 13 | 21 | 211 | 8 | 1 | 0 |
| 3AB | 11 | 3 | 1 | 96 | 21 | 2 |
| 3CDE | 5 | 1 | 3 | 24 | 158 | 8 |
| 4 | 7 | 0 | 0 | 0 | 7 | 22 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 64 | 1 | 0 | 0 | 0 | 0 |
| 2AB | 2 | 8 | 6 | 0 | 0 | 0 |
| 2CD | 0 | 1 | 40 | 0 | 0 | 0 |
| 3AB | 1 | 0 | 0 | 33 | 0 | 0 |
| 3CDE | 1 | 0 | 1 | 10 | 29 | 1 |
| 4 | 1 | 0 | 0 | 0 | 0 | 11 |

## S04_densenet121

Network `cls:densenet121.tv_in1k` at 768x512 px; fracture threshold 0.1175 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 90.0%) is fold 1.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.992 | 0.992 ± 0.005 | 0.997 ± 0.001 | 0.999 | 0.999 |
| Fracture sensitivity | 0.962 | 0.962 ± 0.014 | 0.967 ± 0.009 | 0.972 | 0.986 |
| Fracture specificity | 0.983 | 0.983 ± 0.015 | 0.988 ± 0.013 | 1.000 | 0.969 |
| 6-class accuracy (%) | 86.8 | 86.8 ± 2.6 | 84.8 ± 2.0 | 85.7 | 85.7 |
| Macro-F1 | 0.802 | 0.796 ± 0.062 | 0.749 ± 0.049 | 0.771 | 0.760 |
| Balanced accuracy | 0.787 | 0.787 ± 0.069 | 0.730 ± 0.042 | 0.744 | 0.744 |
| Cohen's kappa | 0.825 | 0.825 ± 0.035 | 0.804 ± 0.026 | 0.817 | 0.817 |
| Location accuracy (fractures called) | 0.940 | 0.940 ± 0.014 | 0.931 ± 0.009 | 0.929 | 0.937 |
| Grade accuracy, location right | 0.886 | 0.887 ± 0.040 | 0.866 ± 0.023 | 0.873 | 0.867 |
| Dice, fractured femurs (missed = 0) | n/a | n/a | n/a | n/a | n/a |
| Dice, detected fractures only | n/a | n/a | n/a | n/a | n/a |
| Predicted / true area, median | n/a | n/a | n/a | n/a | n/a |
| HD95 (px), median, detected | n/a | n/a | n/a | n/a | n/a |
| Errors: detection | 32.0 | 6.4 ± 1.9 | 5.6 ± 1.1 | 4.0 | 4.0 |
| Errors: location | 39.0 | 7.8 ± 1.9 | 9.6 ± 1.1 | 10.0 | 9.0 |
| Errors: grade | 67.0 | 13.4 ± 4.8 | 16.8 ± 2.8 | 16.0 | 17.0 |
| Errors: total | 138.0 | 27.6 ± 5.4 | 32.0 ± 4.2 | 30.0 | 30.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.983 | 0.983 ± 0.015 | 0.988 ± 0.013 | 1.000 | 0.969 |
| 2AB Neck Garden I-II | 57 / 16 | 0.579 | 0.577 ± 0.190 | 0.312 ± 0.133 | 0.375 | 0.250 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.890 | 0.890 ± 0.060 | 0.971 ± 0.011 | 0.951 | 0.976 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.761 | 0.762 ± 0.075 | 0.876 ± 0.057 | 0.912 | 0.912 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.814 | 0.815 ± 0.099 | 0.814 ± 0.020 | 0.810 | 0.857 |
| 4 Subtrochanteric | 36 / 12 | 0.694 | 0.693 ± 0.257 | 0.417 ± 0.102 | 0.417 | 0.500 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | n/a | n/a | n/a | n/a | 0.895 / 0.812 ± 0.044 | n/a / n/a | n/a / n/a |
| 2CD Neck Garden III-IV | n/a | n/a | n/a | n/a | 0.961 / 0.990 ± 0.013 | n/a / n/a | n/a / n/a |
| 3AB Pertroch Evans I-II | n/a | n/a | n/a | n/a | 0.948 / 0.965 ± 0.025 | n/a / n/a | n/a / n/a |
| 3CDE Pertroch Evans III-V | n/a | n/a | n/a | n/a | 1.000 / 1.000 ± 0.000 | n/a / n/a | n/a / n/a |
| 4 Subtrochanteric | n/a | n/a | n/a | n/a | 0.917 / 0.983 ± 0.037 | n/a / n/a | n/a / n/a |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 357 | 2 | 3 | 1 | 0 | 0 |
| 2AB | 6 | 33 | 13 | 5 | 0 | 0 |
| 2CD | 10 | 9 | 226 | 6 | 3 | 0 |
| 3AB | 7 | 1 | 4 | 102 | 20 | 0 |
| 3CDE | 0 | 1 | 5 | 25 | 162 | 6 |
| 4 | 3 | 0 | 1 | 1 | 6 | 25 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 63 | 0 | 1 | 1 | 0 | 0 |
| 2AB | 2 | 4 | 10 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 40 | 1 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 31 | 2 | 1 |
| 3CDE | 0 | 0 | 1 | 5 | 36 | 0 |
| 4 | 0 | 0 | 0 | 1 | 5 | 6 |

## S05_efficientnet_b4

Network `cls:efficientnet_b4.ra2_in1k` at 768x512 px; fracture threshold 0.4486 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 86.5%) is fold 0.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.990 | 0.991 ± 0.006 | 0.990 ± 0.002 | 0.987 | 0.994 |
| Fracture sensitivity | 0.954 | 0.954 ± 0.018 | 0.932 ± 0.013 | 0.924 | 0.924 |
| Fracture specificity | 0.975 | 0.975 ± 0.006 | 0.972 ± 0.013 | 0.985 | 0.985 |
| 6-class accuracy (%) | 84.8 | 84.8 ± 1.4 | 80.6 ± 1.5 | 82.9 | 81.0 |
| Macro-F1 | 0.755 | 0.751 ± 0.038 | 0.736 ± 0.021 | 0.751 | 0.744 |
| Balanced accuracy | 0.740 | 0.740 ± 0.053 | 0.710 ± 0.023 | 0.720 | 0.720 |
| Cohen's kappa | 0.798 | 0.798 ± 0.020 | 0.750 ± 0.019 | 0.779 | 0.755 |
| Location accuracy (fractures called) | 0.932 | 0.932 ± 0.018 | 0.934 ± 0.015 | 0.955 | 0.940 |
| Grade accuracy, location right | 0.871 | 0.871 ± 0.025 | 0.831 ± 0.019 | 0.852 | 0.831 |
| Dice, fractured femurs (missed = 0) | n/a | n/a | n/a | n/a | n/a |
| Dice, detected fractures only | n/a | n/a | n/a | n/a | n/a |
| Predicted / true area, median | n/a | n/a | n/a | n/a | n/a |
| HD95 (px), median, detected | n/a | n/a | n/a | n/a | n/a |
| Errors: detection | 40.0 | 8.0 ± 2.6 | 11.6 ± 1.1 | 12.0 | 12.0 |
| Errors: location | 44.0 | 8.8 ± 2.3 | 9.0 ± 2.1 | 6.0 | 8.0 |
| Errors: grade | 75.0 | 15.0 ± 3.1 | 20.2 ± 2.2 | 18.0 | 20.0 |
| Errors: total | 159.0 | 31.8 ± 3.0 | 40.8 ± 3.1 | 36.0 | 40.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.975 | 0.975 ± 0.006 | 0.972 ± 0.013 | 0.985 | 0.985 |
| 2AB Neck Garden I-II | 57 / 16 | 0.386 | 0.383 ± 0.174 | 0.338 ± 0.114 | 0.312 | 0.312 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.909 | 0.909 ± 0.049 | 0.956 ± 0.011 | 0.976 | 0.951 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.709 | 0.709 ± 0.104 | 0.671 ± 0.038 | 0.735 | 0.618 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.794 | 0.795 ± 0.092 | 0.757 ± 0.035 | 0.810 | 0.786 |
| 4 Subtrochanteric | 36 / 12 | 0.667 | 0.668 ± 0.187 | 0.567 ± 0.109 | 0.500 | 0.667 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | n/a | n/a | n/a | n/a | 0.807 / 0.787 ± 0.071 | n/a / n/a | n/a / n/a |
| 2CD Neck Garden III-IV | n/a | n/a | n/a | n/a | 0.965 / 0.985 ± 0.013 | n/a / n/a | n/a / n/a |
| 3AB Pertroch Evans I-II | n/a | n/a | n/a | n/a | 0.940 / 0.882 ± 0.021 | n/a / n/a | n/a / n/a |
| 3CDE Pertroch Evans III-V | n/a | n/a | n/a | n/a | 0.995 / 0.967 ± 0.021 | n/a / n/a | n/a / n/a |
| 4 Subtrochanteric | n/a | n/a | n/a | n/a | 0.944 / 0.967 ± 0.075 | n/a / n/a | n/a / n/a |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 354 | 3 | 5 | 0 | 0 | 1 |
| 2AB | 11 | 22 | 21 | 3 | 0 | 0 |
| 2CD | 9 | 8 | 231 | 4 | 2 | 0 |
| 3AB | 8 | 1 | 9 | 95 | 20 | 1 |
| 3CDE | 1 | 2 | 5 | 26 | 158 | 7 |
| 4 | 2 | 0 | 0 | 1 | 9 | 24 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 64 | 0 | 0 | 0 | 1 | 0 |
| 2AB | 3 | 5 | 8 | 0 | 0 | 0 |
| 2CD | 1 | 0 | 39 | 1 | 0 | 0 |
| 3AB | 5 | 0 | 1 | 21 | 7 | 0 |
| 3CDE | 2 | 0 | 1 | 5 | 33 | 1 |
| 4 | 0 | 0 | 0 | 1 | 3 | 8 |

## S06_convnext_tiny

Network `cls:convnext_tiny.fb_in22k_ft_in1k` at 768x512 px; fracture threshold 0.1228 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 88.0%) is fold 4.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.990 | 0.992 ± 0.004 | 0.998 ± 0.001 | 0.998 | 1.000 |
| Fracture sensitivity | 0.966 | 0.966 ± 0.021 | 0.963 ± 0.017 | 0.959 | 0.993 |
| Fracture specificity | 0.978 | 0.978 ± 0.012 | 0.994 ± 0.014 | 1.000 | 0.969 |
| 6-class accuracy (%) | 86.4 | 86.4 ± 1.5 | 83.5 ± 1.6 | 82.4 | 84.8 |
| Macro-F1 | 0.784 | 0.780 ± 0.041 | 0.732 ± 0.024 | 0.727 | 0.742 |
| Balanced accuracy | 0.763 | 0.763 ± 0.052 | 0.714 ± 0.023 | 0.708 | 0.727 |
| Cohen's kappa | 0.820 | 0.820 ± 0.020 | 0.788 ± 0.021 | 0.773 | 0.805 |
| Location accuracy (fractures called) | 0.935 | 0.935 ± 0.020 | 0.931 ± 0.006 | 0.935 | 0.938 |
| Grade accuracy, location right | 0.885 | 0.885 ± 0.018 | 0.847 ± 0.026 | 0.823 | 0.846 |
| Dice, fractured femurs (missed = 0) | n/a | n/a | n/a | n/a | n/a |
| Dice, detected fractures only | n/a | n/a | n/a | n/a | n/a |
| Predicted / true area, median | n/a | n/a | n/a | n/a | n/a |
| HD95 (px), median, detected | n/a | n/a | n/a | n/a | n/a |
| Errors: detection | 31.0 | 6.2 ± 2.7 | 5.8 ± 2.2 | 6.0 | 3.0 |
| Errors: location | 43.0 | 8.6 ± 2.7 | 9.6 ± 0.9 | 9.0 | 9.0 |
| Errors: grade | 68.0 | 13.6 ± 2.4 | 19.2 ± 3.3 | 22.0 | 20.0 |
| Errors: total | 142.0 | 28.4 ± 3.0 | 34.6 ± 3.4 | 37.0 | 32.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.978 | 0.978 ± 0.012 | 0.994 ± 0.014 | 1.000 | 0.969 |
| 2AB Neck Garden I-II | 57 / 16 | 0.474 | 0.476 ± 0.149 | 0.275 ± 0.034 | 0.250 | 0.250 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.921 | 0.921 ± 0.050 | 0.980 ± 0.020 | 1.000 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.761 | 0.762 ± 0.076 | 0.853 ± 0.075 | 0.735 | 0.941 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.804 | 0.805 ± 0.085 | 0.767 ± 0.052 | 0.762 | 0.786 |
| 4 Subtrochanteric | 36 / 12 | 0.639 | 0.636 ± 0.243 | 0.417 ± 0.132 | 0.500 | 0.417 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | n/a | n/a | n/a | n/a | 0.895 / 0.825 ± 0.112 | n/a / n/a | n/a / n/a |
| 2CD Neck Garden III-IV | n/a | n/a | n/a | n/a | 0.976 / 0.985 ± 0.013 | n/a / n/a | n/a / n/a |
| 3AB Pertroch Evans I-II | n/a | n/a | n/a | n/a | 0.940 / 0.953 ± 0.034 | n/a / n/a | n/a / n/a |
| 3CDE Pertroch Evans III-V | n/a | n/a | n/a | n/a | 0.995 / 1.000 ± 0.000 | n/a / n/a | n/a / n/a |
| 4 Subtrochanteric | n/a | n/a | n/a | n/a | 0.944 / 0.967 ± 0.046 | n/a / n/a | n/a / n/a |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 355 | 2 | 5 | 1 | 0 | 0 |
| 2AB | 6 | 27 | 18 | 6 | 0 | 0 |
| 2CD | 6 | 7 | 234 | 5 | 2 | 0 |
| 3AB | 8 | 1 | 4 | 102 | 18 | 1 |
| 3CDE | 1 | 0 | 8 | 25 | 160 | 5 |
| 4 | 2 | 0 | 0 | 3 | 8 | 23 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 63 | 2 | 0 | 0 | 0 | 0 |
| 2AB | 1 | 4 | 11 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 1 | 32 | 1 | 0 |
| 3CDE | 0 | 0 | 1 | 8 | 33 | 0 |
| 4 | 0 | 0 | 0 | 1 | 6 | 5 |

## S07_unet_r34

Network `seg:Unet:resnet34` at 768x512 px; fracture threshold 0.6997 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 88.0%) is fold 3.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.990 | 0.990 ± 0.006 | 0.993 ± 0.005 | 0.994 | 0.999 |
| Fracture sensitivity | 0.969 | 0.969 ± 0.012 | 0.972 ± 0.005 | 0.972 | 0.966 |
| Fracture specificity | 0.970 | 0.970 ± 0.024 | 0.972 ± 0.013 | 0.985 | 1.000 |
| 6-class accuracy (%) | 86.3 | 86.3 ± 1.1 | 84.6 ± 1.6 | 82.9 | 86.7 |
| Macro-F1 | 0.800 | 0.798 ± 0.029 | 0.770 ± 0.027 | 0.757 | 0.799 |
| Balanced accuracy | 0.802 | 0.803 ± 0.039 | 0.752 ± 0.026 | 0.745 | 0.771 |
| Cohen's kappa | 0.820 | 0.820 ± 0.015 | 0.803 ± 0.021 | 0.782 | 0.829 |
| Location accuracy (fractures called) | 0.942 | 0.942 ± 0.023 | 0.953 ± 0.011 | 0.943 | 0.964 |
| Grade accuracy, location right | 0.877 | 0.877 ± 0.021 | 0.844 ± 0.023 | 0.817 | 0.859 |
| Dice, fractured femurs (missed = 0) | 0.748 | 0.748 ± 0.011 | 0.698 ± 0.009 | 0.707 | 0.711 |
| Dice, detected fractures only | 0.772 | 0.772 ± 0.007 | 0.718 ± 0.009 | 0.727 | 0.736 |
| Predicted / true area, median | 0.93 | 0.93 ± 0.03 | 1.16 ± 0.03 | 1.14 | 1.14 |
| HD95 (px), median, detected | 64.4 | 64.1 ± 1.2 | 72.4 ± 1.4 | 73.0 | 68.7 |
| Errors: detection | 32.0 | 6.4 ± 2.3 | 5.8 ± 0.8 | 5.0 | 5.0 |
| Errors: location | 38.0 | 7.6 ± 3.0 | 6.6 ± 1.5 | 8.0 | 5.0 |
| Errors: grade | 73.0 | 14.6 ± 3.0 | 20.0 ± 2.9 | 23.0 | 18.0 |
| Errors: total | 143.0 | 28.6 ± 2.3 | 32.4 ± 3.4 | 36.0 | 28.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.970 | 0.970 ± 0.024 | 0.972 ± 0.013 | 0.985 | 1.000 |
| 2AB Neck Garden I-II | 57 / 16 | 0.614 | 0.615 ± 0.137 | 0.375 ± 0.088 | 0.438 | 0.375 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.874 | 0.874 ± 0.059 | 0.985 ± 0.013 | 0.976 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.784 | 0.783 ± 0.110 | 0.882 ± 0.047 | 0.824 | 0.882 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.794 | 0.795 ± 0.095 | 0.748 ± 0.052 | 0.667 | 0.786 |
| 4 Subtrochanteric | 36 / 12 | 0.778 | 0.779 ± 0.161 | 0.550 ± 0.075 | 0.583 | 0.583 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.588 | 0.587 ± 0.059 | 0.544 ± 0.041 | 0.576 | 0.930 / 0.850 ± 0.034 | 0.632 / 0.639 ± 0.034 | 0.76 / 1.74 ± 0.08 |
| 2CD Neck Garden III-IV | 0.748 | 0.748 ± 0.019 | 0.679 ± 0.017 | 0.689 | 0.972 / 1.000 ± 0.000 | 0.769 / 0.679 ± 0.017 | 0.93 / 1.80 ± 0.11 |
| 3AB Pertroch Evans I-II | 0.725 | 0.725 ± 0.047 | 0.698 ± 0.029 | 0.700 | 0.963 / 0.982 ± 0.016 | 0.753 / 0.711 ± 0.019 | 0.94 / 1.00 ± 0.08 |
| 3CDE Pertroch Evans III-V | 0.825 | 0.825 ± 0.014 | 0.816 ± 0.005 | 0.829 | 0.990 / 1.000 ± 0.000 | 0.833 / 0.816 ± 0.005 | 0.93 / 1.00 ± 0.02 |
| 4 Subtrochanteric | 0.662 | 0.661 ± 0.100 | 0.558 ± 0.038 | 0.580 | 0.917 / 0.917 ± 0.059 | 0.723 / 0.610 ± 0.046 | 0.95 / 0.69 ± 0.08 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 352 | 6 | 3 | 1 | 0 | 1 |
| 2AB | 4 | 35 | 14 | 4 | 0 | 0 |
| 2CD | 7 | 15 | 222 | 9 | 1 | 0 |
| 3AB | 5 | 3 | 0 | 105 | 18 | 3 |
| 3CDE | 2 | 1 | 8 | 26 | 158 | 4 |
| 4 | 3 | 0 | 0 | 1 | 4 | 28 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 65 | 0 | 0 | 0 | 0 | 0 |
| 2AB | 3 | 6 | 7 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 1 | 0 | 0 | 30 | 3 | 0 |
| 3CDE | 0 | 0 | 1 | 8 | 33 | 0 |
| 4 | 1 | 0 | 0 | 1 | 3 | 7 |

## S08_deeplabv3p_r34

Network `seg:DeepLabV3Plus:resnet34` at 768x512 px; fracture threshold 0.9539 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 88.9%) is fold 0.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.990 | 0.990 ± 0.003 | 0.993 ± 0.004 | 0.995 | 0.999 |
| Fracture sensitivity | 0.959 | 0.959 ± 0.018 | 0.961 ± 0.010 | 0.952 | 0.931 |
| Fracture specificity | 0.964 | 0.964 ± 0.012 | 0.978 ± 0.014 | 1.000 | 1.000 |
| 6-class accuracy (%) | 86.2 | 86.2 ± 2.3 | 85.5 ± 1.9 | 86.7 | 86.2 |
| Macro-F1 | 0.800 | 0.798 ± 0.027 | 0.784 ± 0.022 | 0.793 | 0.778 |
| Balanced accuracy | 0.798 | 0.799 ± 0.035 | 0.764 ± 0.021 | 0.776 | 0.753 |
| Cohen's kappa | 0.818 | 0.819 ± 0.029 | 0.815 ± 0.024 | 0.829 | 0.822 |
| Location accuracy (fractures called) | 0.945 | 0.945 ± 0.020 | 0.956 ± 0.006 | 0.964 | 0.963 |
| Grade accuracy, location right | 0.886 | 0.886 ± 0.041 | 0.864 ± 0.030 | 0.872 | 0.886 |
| Dice, fractured femurs (missed = 0) | 0.749 | 0.749 ± 0.017 | 0.704 ± 0.005 | 0.702 | 0.698 |
| Dice, detected fractures only | 0.781 | 0.781 ± 0.004 | 0.733 ± 0.006 | 0.737 | 0.750 |
| Predicted / true area, median | 0.93 | 0.93 ± 0.03 | 1.17 ± 0.03 | 1.17 | 1.16 |
| HD95 (px), median, detected | 61.7 | 62.3 ± 2.3 | 70.6 ± 0.8 | 71.6 | 65.4 |
| Errors: detection | 41.0 | 8.2 ± 1.6 | 7.0 ± 1.2 | 7.0 | 10.0 |
| Errors: location | 36.0 | 7.2 ± 2.6 | 6.2 ± 0.8 | 5.0 | 5.0 |
| Errors: grade | 67.0 | 13.4 ± 4.8 | 17.2 ± 3.9 | 16.0 | 14.0 |
| Errors: total | 144.0 | 28.8 ± 4.8 | 30.4 ± 3.9 | 28.0 | 29.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.964 | 0.964 ± 0.012 | 0.978 ± 0.014 | 1.000 | 1.000 |
| 2AB Neck Garden I-II | 57 / 16 | 0.596 | 0.597 ± 0.134 | 0.375 ± 0.125 | 0.312 | 0.250 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.894 | 0.894 ± 0.061 | 0.990 ± 0.013 | 0.976 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.769 | 0.768 ± 0.081 | 0.888 ± 0.038 | 0.941 | 0.853 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.789 | 0.790 ± 0.106 | 0.767 ± 0.085 | 0.762 | 0.833 |
| 4 Subtrochanteric | 36 / 12 | 0.778 | 0.779 ± 0.161 | 0.583 ± 0.059 | 0.667 | 0.583 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.556 | 0.556 ± 0.015 | 0.557 ± 0.058 | 0.441 | 0.825 / 0.825 ± 0.112 | 0.675 / 0.678 ± 0.035 | 0.82 / 1.71 ± 0.14 |
| 2CD Neck Garden III-IV | 0.744 | 0.744 ± 0.030 | 0.693 ± 0.005 | 0.706 | 0.972 / 1.000 ± 0.000 | 0.765 / 0.693 ± 0.005 | 0.87 / 1.76 ± 0.06 |
| 3AB Pertroch Evans I-II | 0.734 | 0.734 ± 0.052 | 0.700 ± 0.019 | 0.686 | 0.940 / 0.959 ± 0.026 | 0.780 / 0.730 ± 0.007 | 0.99 / 1.06 ± 0.04 |
| 3CDE Pertroch Evans III-V | 0.832 | 0.833 ± 0.015 | 0.812 ± 0.017 | 0.823 | 0.995 / 0.986 ± 0.013 | 0.837 / 0.823 ± 0.007 | 0.96 / 1.03 ± 0.03 |
| 4 Subtrochanteric | 0.682 | 0.681 ± 0.052 | 0.577 ± 0.024 | 0.618 | 0.944 / 0.933 ± 0.037 | 0.722 / 0.619 ± 0.027 | 0.96 / 0.74 ± 0.04 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 350 | 5 | 5 | 2 | 1 | 0 |
| 2AB | 10 | 34 | 10 | 3 | 0 | 0 |
| 2CD | 7 | 12 | 227 | 7 | 1 | 0 |
| 3AB | 8 | 2 | 1 | 103 | 19 | 1 |
| 3CDE | 1 | 0 | 7 | 26 | 157 | 8 |
| 4 | 2 | 0 | 0 | 2 | 4 | 28 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 65 | 0 | 0 | 0 | 0 | 0 |
| 2AB | 6 | 4 | 6 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 3 | 0 | 0 | 29 | 2 | 0 |
| 3CDE | 0 | 0 | 1 | 6 | 35 | 0 |
| 4 | 1 | 0 | 0 | 0 | 4 | 7 |

## S09_segformer_b2

Network `seg:Segformer:mit_b2` at 768x512 px; fracture threshold 0.8732 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 90.0%) is fold 1.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.993 ± 0.005 | 0.997 ± 0.002 | 0.999 | 0.999 |
| Fracture sensitivity | 0.976 | 0.976 ± 0.012 | 0.986 ± 0.005 | 0.986 | 0.986 |
| Fracture specificity | 0.970 | 0.970 ± 0.027 | 0.982 ± 0.020 | 0.969 | 1.000 |
| 6-class accuracy (%) | 87.9 | 87.9 ± 1.4 | 86.8 ± 0.8 | 85.7 | 88.1 |
| Macro-F1 | 0.817 | 0.817 ± 0.035 | 0.797 ± 0.020 | 0.781 | 0.804 |
| Balanced accuracy | 0.810 | 0.810 ± 0.025 | 0.777 ± 0.019 | 0.761 | 0.786 |
| Cohen's kappa | 0.841 | 0.841 ± 0.019 | 0.831 ± 0.010 | 0.818 | 0.848 |
| Location accuracy (fractures called) | 0.949 | 0.949 ± 0.006 | 0.948 ± 0.010 | 0.965 | 0.951 |
| Grade accuracy, location right | 0.892 | 0.892 ± 0.034 | 0.867 ± 0.015 | 0.840 | 0.876 |
| Dice, fractured femurs (missed = 0) | 0.766 | 0.766 ± 0.008 | 0.725 ± 0.008 | 0.723 | 0.737 |
| Dice, detected fractures only | 0.784 | 0.784 ± 0.011 | 0.735 ± 0.005 | 0.733 | 0.747 |
| Predicted / true area, median | 0.93 | 0.94 ± 0.03 | 1.20 ± 0.03 | 1.22 | 1.20 |
| HD95 (px), median, detected | 61.6 | 62.2 ± 2.1 | 71.8 ± 1.5 | 72.1 | 68.7 |
| Errors: detection | 27.0 | 5.4 ± 1.3 | 3.2 ± 1.8 | 4.0 | 2.0 |
| Errors: location | 34.0 | 6.8 ± 0.8 | 7.4 ± 1.5 | 5.0 | 7.0 |
| Errors: grade | 65.0 | 13.0 ± 4.2 | 17.2 ± 2.2 | 21.0 | 16.0 |
| Errors: total | 126.0 | 25.2 ± 3.0 | 27.8 ± 1.6 | 30.0 | 25.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.970 | 0.970 ± 0.027 | 0.982 ± 0.020 | 0.969 | 1.000 |
| 2AB Neck Garden I-II | 57 / 16 | 0.579 | 0.582 ± 0.154 | 0.412 ± 0.095 | 0.312 | 0.375 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.906 | 0.906 ± 0.074 | 0.985 ± 0.013 | 0.976 | 0.976 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.769 | 0.769 ± 0.060 | 0.935 ± 0.053 | 0.941 | 0.971 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.859 | 0.860 ± 0.069 | 0.781 ± 0.011 | 0.786 | 0.810 |
| 4 Subtrochanteric | 36 / 12 | 0.778 | 0.775 ± 0.083 | 0.567 ± 0.109 | 0.583 | 0.583 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.592 | 0.593 ± 0.057 | 0.612 ± 0.013 | 0.620 | 0.912 / 0.900 ± 0.034 | 0.649 / 0.681 ± 0.015 | 0.78 / 1.73 ± 0.14 |
| 2CD Neck Garden III-IV | 0.755 | 0.756 ± 0.022 | 0.686 ± 0.011 | 0.699 | 0.980 / 1.000 ± 0.000 | 0.771 / 0.686 ± 0.011 | 0.93 / 1.70 ± 0.11 |
| 3AB Pertroch Evans I-II | 0.757 | 0.757 ± 0.038 | 0.740 ± 0.009 | 0.751 | 0.963 / 1.000 ± 0.000 | 0.786 / 0.740 ± 0.009 | 0.93 / 1.03 ± 0.08 |
| 3CDE Pertroch Evans III-V | 0.841 | 0.841 ± 0.015 | 0.822 ± 0.016 | 0.835 | 1.000 / 0.995 ± 0.011 | 0.841 / 0.826 ± 0.009 | 0.98 / 1.00 ± 0.04 |
| 4 Subtrochanteric | 0.728 | 0.729 ± 0.037 | 0.631 ± 0.021 | 0.641 | 0.972 / 0.983 ± 0.037 | 0.749 / 0.642 ± 0.018 | 1.04 / 0.93 ± 0.13 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 352 | 4 | 3 | 1 | 1 | 2 |
| 2AB | 5 | 33 | 14 | 4 | 1 | 0 |
| 2CD | 5 | 9 | 230 | 10 | 0 | 0 |
| 3AB | 5 | 2 | 1 | 103 | 22 | 1 |
| 3CDE | 0 | 1 | 4 | 20 | 171 | 3 |
| 4 | 1 | 0 | 0 | 1 | 6 | 28 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 65 | 0 | 0 | 0 | 0 | 0 |
| 2AB | 2 | 6 | 8 | 0 | 0 | 0 |
| 2CD | 0 | 1 | 40 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 33 | 1 | 0 |
| 3CDE | 0 | 0 | 1 | 6 | 34 | 1 |
| 4 | 0 | 0 | 0 | 1 | 4 | 7 |

## S10_m0_convnext_unet

Network `m0:convnext_tiny.fb_in22k_ft_in1k` at 768x512 px; fracture threshold 0.0182 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 89.5%) is fold 1.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.995 | 0.995 ± 0.003 | 0.999 ± 0.001 | 1.000 | 0.999 |
| Fracture sensitivity | 0.972 | 0.972 ± 0.016 | 0.983 ± 0.006 | 0.979 | 0.993 |
| Fracture specificity | 0.992 | 0.992 ± 0.008 | 1.000 ± 0.000 | 1.000 | 1.000 |
| 6-class accuracy (%) | 88.4 | 88.4 ± 1.2 | 86.0 ± 0.5 | 86.2 | 86.7 |
| Macro-F1 | 0.825 | 0.825 ± 0.033 | 0.776 ± 0.017 | 0.783 | 0.779 |
| Balanced accuracy | 0.804 | 0.804 ± 0.041 | 0.757 ± 0.015 | 0.762 | 0.757 |
| Cohen's kappa | 0.847 | 0.847 ± 0.016 | 0.821 ± 0.007 | 0.823 | 0.829 |
| Location accuracy (fractures called) | 0.949 | 0.948 ± 0.022 | 0.951 ± 0.005 | 0.958 | 0.951 |
| Grade accuracy, location right | 0.892 | 0.892 ± 0.026 | 0.845 ± 0.013 | 0.845 | 0.847 |
| Dice, fractured femurs (missed = 0) | 0.777 | 0.777 ± 0.019 | 0.734 ± 0.005 | 0.733 | 0.744 |
| Dice, detected fractures only | 0.799 | 0.799 ± 0.007 | 0.746 ± 0.002 | 0.748 | 0.749 |
| Predicted / true area, median | 0.98 | 0.97 ± 0.02 | 1.26 ± 0.03 | 1.26 | 1.24 |
| HD95 (px), median, detected | 60.8 | 60.7 ± 2.6 | 68.4 ± 1.0 | 67.7 | 65.8 |
| Errors: detection | 22.0 | 4.4 ± 1.8 | 2.4 ± 0.9 | 3.0 | 1.0 |
| Errors: location | 34.0 | 6.8 ± 2.9 | 7.0 ± 0.7 | 6.0 | 7.0 |
| Errors: grade | 65.0 | 13.0 ± 3.4 | 20.0 ± 1.9 | 20.0 | 20.0 |
| Errors: total | 121.0 | 24.2 ± 2.5 | 29.4 ± 1.1 | 29.0 | 28.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.992 | 0.992 ± 0.008 | 1.000 ± 0.000 | 1.000 | 1.000 |
| 2AB Neck Garden I-II | 57 / 16 | 0.544 | 0.544 ± 0.109 | 0.312 ± 0.077 | 0.312 | 0.312 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.909 | 0.910 ± 0.053 | 0.990 ± 0.013 | 0.976 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.791 | 0.791 ± 0.052 | 0.953 ± 0.016 | 0.941 | 0.971 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.839 | 0.840 ± 0.067 | 0.738 ± 0.034 | 0.762 | 0.762 |
| 4 Subtrochanteric | 36 / 12 | 0.750 | 0.750 ± 0.156 | 0.550 ± 0.046 | 0.583 | 0.500 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.624 | 0.625 ± 0.056 | 0.578 ± 0.032 | 0.599 | 0.895 / 0.863 ± 0.052 | 0.697 / 0.671 ± 0.010 | 0.96 / 1.89 ± 0.07 |
| 2CD Neck Garden III-IV | 0.773 | 0.773 ± 0.023 | 0.699 ± 0.009 | 0.704 | 0.980 / 1.000 ± 0.000 | 0.788 / 0.699 ± 0.009 | 0.93 / 1.71 ± 0.06 |
| 3AB Pertroch Evans I-II | 0.763 | 0.763 ± 0.056 | 0.750 ± 0.008 | 0.766 | 0.955 / 0.994 ± 0.013 | 0.799 / 0.755 ± 0.008 | 1.01 / 1.09 ± 0.05 |
| 3CDE Pertroch Evans III-V | 0.852 | 0.852 ± 0.009 | 0.836 ± 0.004 | 0.842 | 1.000 / 1.000 ± 0.000 | 0.852 / 0.836 ± 0.004 | 0.98 / 1.06 ± 0.02 |
| 4 Subtrochanteric | 0.682 | 0.682 ± 0.098 | 0.655 ± 0.039 | 0.665 | 0.944 / 1.000 ± 0.000 | 0.722 / 0.655 ± 0.039 | 0.99 / 0.95 ± 0.04 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 360 | 0 | 3 | 0 | 0 | 0 |
| 2AB | 6 | 31 | 13 | 7 | 0 | 0 |
| 2CD | 5 | 10 | 231 | 8 | 0 | 0 |
| 3AB | 6 | 1 | 4 | 106 | 17 | 0 |
| 3CDE | 0 | 1 | 5 | 25 | 167 | 1 |
| 4 | 2 | 0 | 0 | 2 | 5 | 27 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 65 | 0 | 0 | 0 | 0 | 0 |
| 2AB | 1 | 5 | 10 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 33 | 1 | 0 |
| 3CDE | 0 | 0 | 1 | 9 | 32 | 0 |
| 4 | 0 | 0 | 0 | 1 | 5 | 6 |

## Runs side by side (test, 5 fold models mean ± SD; OOF pooled in brackets)

| Metric | A6_slgnet | S01_yolov8s_seg_1024 | S04_densenet121 | S05_efficientnet_b4 | S06_convnext_tiny | S07_unet_r34 | S08_deeplabv3p_r34 | S09_segformer_b2 | S10_m0_convnext_unet |
|---|---|---|---|---|---|---|---|---|---|
| 6-class accuracy (%) | 86.3 ± 1.5 (87.0) | 84.6 ± 2.1 (83.2) | 84.8 ± 2.0 (86.8) | 80.6 ± 1.5 (84.8) | 83.5 ± 1.6 (86.4) | 84.6 ± 1.6 (86.3) | 85.5 ± 1.9 (86.2) | 86.8 ± 0.8 (87.9) | 86.0 ± 0.5 (88.4) |
| Macro-F1 | 0.789 ± 0.015 (0.806) | 0.791 ± 0.018 (0.744) | 0.749 ± 0.049 (0.802) | 0.736 ± 0.021 (0.755) | 0.732 ± 0.024 (0.784) | 0.770 ± 0.027 (0.800) | 0.784 ± 0.022 (0.800) | 0.797 ± 0.020 (0.817) | 0.776 ± 0.017 (0.825) |
| Dice, fractured (missed = 0) | 0.725 ± 0.007 (0.774) | 0.693 ± 0.016 (0.740) | n/a (n/a) | n/a (n/a) | n/a (n/a) | 0.698 ± 0.009 (0.748) | 0.704 ± 0.005 (0.749) | 0.725 ± 0.008 (0.766) | 0.734 ± 0.005 (0.777) |
| Recall NF | 0.991 ± 0.008 (0.983) | 0.978 ± 0.008 (0.970) | 0.988 ± 0.013 (0.983) | 0.972 ± 0.013 (0.975) | 0.994 ± 0.014 (0.978) | 0.972 ± 0.013 (0.970) | 0.978 ± 0.014 (0.964) | 0.982 ± 0.020 (0.970) | 1.000 ± 0.000 (0.992) |
| Recall 2AB | 0.287 ± 0.034 (0.614) | 0.438 ± 0.117 (0.509) | 0.312 ± 0.133 (0.579) | 0.338 ± 0.114 (0.386) | 0.275 ± 0.034 (0.474) | 0.375 ± 0.088 (0.614) | 0.375 ± 0.125 (0.596) | 0.412 ± 0.095 (0.579) | 0.312 ± 0.077 (0.544) |
| Recall 2CD | 0.985 ± 0.013 (0.878) | 0.961 ± 0.028 (0.831) | 0.971 ± 0.011 (0.890) | 0.956 ± 0.011 (0.909) | 0.980 ± 0.020 (0.921) | 0.985 ± 0.013 (0.874) | 0.990 ± 0.013 (0.894) | 0.985 ± 0.013 (0.906) | 0.990 ± 0.013 (0.909) |
| Recall 3AB | 0.894 ± 0.049 (0.784) | 0.888 ± 0.060 (0.716) | 0.876 ± 0.057 (0.761) | 0.671 ± 0.038 (0.709) | 0.853 ± 0.075 (0.761) | 0.882 ± 0.047 (0.784) | 0.888 ± 0.038 (0.769) | 0.935 ± 0.053 (0.769) | 0.953 ± 0.016 (0.791) |
| Recall 3CDE | 0.786 ± 0.051 (0.804) | 0.681 ± 0.087 (0.794) | 0.814 ± 0.020 (0.814) | 0.757 ± 0.035 (0.794) | 0.767 ± 0.052 (0.804) | 0.748 ± 0.052 (0.794) | 0.767 ± 0.085 (0.789) | 0.781 ± 0.011 (0.859) | 0.738 ± 0.034 (0.839) |
| Recall 4 | 0.700 ± 0.075 (0.750) | 0.733 ± 0.124 (0.611) | 0.417 ± 0.102 (0.694) | 0.567 ± 0.109 (0.667) | 0.417 ± 0.132 (0.639) | 0.550 ± 0.075 (0.778) | 0.583 ± 0.059 (0.778) | 0.567 ± 0.109 (0.778) | 0.550 ± 0.046 (0.750) |
| Dice 2AB | 0.573 ± 0.022 (0.612) | 0.524 ± 0.036 (0.534) | n/a (n/a) | n/a (n/a) | n/a (n/a) | 0.544 ± 0.041 (0.588) | 0.557 ± 0.058 (0.556) | 0.612 ± 0.013 (0.592) | 0.578 ± 0.032 (0.624) |
| Dice 2CD | 0.692 ± 0.007 (0.769) | 0.677 ± 0.013 (0.751) | n/a (n/a) | n/a (n/a) | n/a (n/a) | 0.679 ± 0.017 (0.748) | 0.693 ± 0.005 (0.744) | 0.686 ± 0.011 (0.755) | 0.699 ± 0.009 (0.773) |
| Dice 3AB | 0.727 ± 0.022 (0.774) | 0.713 ± 0.039 (0.721) | n/a (n/a) | n/a (n/a) | n/a (n/a) | 0.698 ± 0.029 (0.725) | 0.700 ± 0.019 (0.734) | 0.740 ± 0.009 (0.757) | 0.750 ± 0.008 (0.763) |
| Dice 3CDE | 0.836 ± 0.004 (0.846) | 0.781 ± 0.019 (0.822) | n/a (n/a) | n/a (n/a) | n/a (n/a) | 0.816 ± 0.005 (0.825) | 0.812 ± 0.017 (0.832) | 0.822 ± 0.016 (0.841) | 0.836 ± 0.004 (0.852) |
| Dice 4 | 0.641 ± 0.034 (0.674) | 0.608 ± 0.024 (0.601) | n/a (n/a) | n/a (n/a) | n/a (n/a) | 0.558 ± 0.038 (0.662) | 0.577 ± 0.024 (0.682) | 0.631 ± 0.021 (0.728) | 0.655 ± 0.039 (0.682) |

Paired tests between the runs (same femurs) are in `stats.md` of the same folder, if it was made with `scripts/run_stats.py ... --compare`.

## Comparison with the base article

Source: González et al., Insights Imaging 2025;16:86 (doi 10.1186/s13244-025-01958-y). Table 2 gives mean ± SD of six training runs per setting on the same 210 test femurs; the final model is the run with the best validation accuracy (Table 3, confusion matrix in Fig. 3). Our fold models are trained on 4/5 of the 1,043 training femurs (about 834), the article's on all 1,043 plus a 201-femur validation set; our counterpart of its selected model is the fold model with the best accuracy on its own held-out fold.

### A6_slgnet vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.999 ± 0.001 | 0.981 | 0.999 | 0.999 |
| 6-class accuracy (%) | n/a | 86.3 ± 1.5 | 86.2 | 84.3 | 88.6 |
| Fracture sensitivity | n/a | 0.972 ± 0.007 | 0.952 | 0.966 | 0.986 |
| Fracture specificity | n/a | 0.991 ± 0.008 | 0.985 | 0.985 | 0.985 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.789 ± 0.015 | 0.814 | 0.769 | 0.820 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.745 ± 0.003 | 0.77 | 0.742 | 0.747 |
| Recall NF No fracture | n/a | 0.991 ± 0.008 | 0.985 | 0.985 | 0.985 |
| Recall 2AB Neck Garden I-II | n/a | 0.287 ± 0.034 | 0.500 | 0.312 | 0.312 |
| Recall 2CD Neck Garden III-IV | n/a | 0.985 ± 0.013 | 0.951 | 0.976 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.894 ± 0.049 | 0.882 | 0.824 | 0.971 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.786 ± 0.051 | 0.738 | 0.786 | 0.810 |
| Recall 4 Subtrochanteric | n/a | 0.700 ± 0.075 | 0.750 | 0.583 | 0.750 |
| Errors: detection | n/a | 4.6 ± 0.9 | 8 | 6 | 3 |
| Errors: location | n/a | 5.6 ± 1.3 | 6 | 5 | 4 |
| Errors: grade | n/a | 18.6 ± 2.3 | 15 | 22 | 17 |
| Errors: total | n/a | 28.8 ± 3.1 | 29 | 33 | 24 |

### S01_yolov8s_seg_1024 vs the article's YOLOv8-seg small @ 1024

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | 0.986 ± 0.008 | 0.985 ± 0.005 | 0.981 | 0.981 | 0.994 |
| 6-class accuracy (%) | 84.4 ± 2.5 | 84.6 ± 2.1 | 86.2 | 83.8 | 88.1 |
| Fracture sensitivity | n/a | 0.935 ± 0.014 | 0.952 | 0.917 | 0.966 |
| Fracture specificity | n/a | 0.978 ± 0.008 | 0.985 | 0.969 | 0.985 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.791 ± 0.018 | 0.814 | 0.793 | 0.845 |
| Dice (article: definition not stated; ours: detected fractures only) | 0.761 ± 0.008 | 0.741 ± 0.009 | 0.77 | 0.745 | 0.751 |
| Recall NF No fracture | n/a | 0.978 ± 0.008 | 0.985 | 0.969 | 0.985 |
| Recall 2AB Neck Garden I-II | n/a | 0.438 ± 0.117 | 0.500 | 0.625 | 0.500 |
| Recall 2CD Neck Garden III-IV | n/a | 0.961 ± 0.028 | 0.951 | 0.951 | 0.976 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.888 ± 0.060 | 0.882 | 0.824 | 0.971 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.681 ± 0.087 | 0.738 | 0.690 | 0.690 |
| Recall 4 Subtrochanteric | n/a | 0.733 ± 0.124 | 0.750 | 0.583 | 0.917 |
| Errors: detection | n/a | 10.8 ± 2.2 | 8 | 14 | 6 |
| Errors: location | n/a | 4.6 ± 1.5 | 6 | 5 | 2 |
| Errors: grade | n/a | 17.0 ± 3.5 | 15 | 15 | 17 |
| Errors: total | n/a | 32.4 ± 4.5 | 29 | 34 | 25 |

### S04_densenet121 vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.997 ± 0.001 | 0.981 | 0.999 | 0.999 |
| 6-class accuracy (%) | n/a | 84.8 ± 2.0 | 86.2 | 85.7 | 85.7 |
| Fracture sensitivity | n/a | 0.967 ± 0.009 | 0.952 | 0.972 | 0.986 |
| Fracture specificity | n/a | 0.988 ± 0.013 | 0.985 | 1.000 | 0.969 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.749 ± 0.049 | 0.814 | 0.771 | 0.760 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | n/a | 0.77 | n/a | n/a |
| Recall NF No fracture | n/a | 0.988 ± 0.013 | 0.985 | 1.000 | 0.969 |
| Recall 2AB Neck Garden I-II | n/a | 0.312 ± 0.133 | 0.500 | 0.375 | 0.250 |
| Recall 2CD Neck Garden III-IV | n/a | 0.971 ± 0.011 | 0.951 | 0.951 | 0.976 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.876 ± 0.057 | 0.882 | 0.912 | 0.912 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.814 ± 0.020 | 0.738 | 0.810 | 0.857 |
| Recall 4 Subtrochanteric | n/a | 0.417 ± 0.102 | 0.750 | 0.417 | 0.500 |
| Errors: detection | n/a | 5.6 ± 1.1 | 8 | 4 | 4 |
| Errors: location | n/a | 9.6 ± 1.1 | 6 | 10 | 9 |
| Errors: grade | n/a | 16.8 ± 2.8 | 15 | 16 | 17 |
| Errors: total | n/a | 32.0 ± 4.2 | 29 | 30 | 30 |

The article's own DenseNet121 (Table 3, one run on the same test set): fracture AUC 0.993 (0.985–1.000), 6-class accuracy 83.8% (78.1–88.5).

### S05_efficientnet_b4 vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.990 ± 0.002 | 0.981 | 0.987 | 0.994 |
| 6-class accuracy (%) | n/a | 80.6 ± 1.5 | 86.2 | 82.9 | 81.0 |
| Fracture sensitivity | n/a | 0.932 ± 0.013 | 0.952 | 0.924 | 0.924 |
| Fracture specificity | n/a | 0.972 ± 0.013 | 0.985 | 0.985 | 0.985 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.736 ± 0.021 | 0.814 | 0.751 | 0.744 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | n/a | 0.77 | n/a | n/a |
| Recall NF No fracture | n/a | 0.972 ± 0.013 | 0.985 | 0.985 | 0.985 |
| Recall 2AB Neck Garden I-II | n/a | 0.338 ± 0.114 | 0.500 | 0.312 | 0.312 |
| Recall 2CD Neck Garden III-IV | n/a | 0.956 ± 0.011 | 0.951 | 0.976 | 0.951 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.671 ± 0.038 | 0.882 | 0.735 | 0.618 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.757 ± 0.035 | 0.738 | 0.810 | 0.786 |
| Recall 4 Subtrochanteric | n/a | 0.567 ± 0.109 | 0.750 | 0.500 | 0.667 |
| Errors: detection | n/a | 11.6 ± 1.1 | 8 | 12 | 12 |
| Errors: location | n/a | 9.0 ± 2.1 | 6 | 6 | 8 |
| Errors: grade | n/a | 20.2 ± 2.2 | 15 | 18 | 20 |
| Errors: total | n/a | 40.8 ± 3.1 | 29 | 36 | 40 |

The article's own EfficientNetB4 (Table 3, one run on the same test set): fracture AUC 0.990 (0.981–0.998), 6-class accuracy 79.0% (72.9–84.3).

### S06_convnext_tiny vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.998 ± 0.001 | 0.981 | 0.998 | 1.000 |
| 6-class accuracy (%) | n/a | 83.5 ± 1.6 | 86.2 | 82.4 | 84.8 |
| Fracture sensitivity | n/a | 0.963 ± 0.017 | 0.952 | 0.959 | 0.993 |
| Fracture specificity | n/a | 0.994 ± 0.014 | 0.985 | 1.000 | 0.969 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.732 ± 0.024 | 0.814 | 0.727 | 0.742 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | n/a | 0.77 | n/a | n/a |
| Recall NF No fracture | n/a | 0.994 ± 0.014 | 0.985 | 1.000 | 0.969 |
| Recall 2AB Neck Garden I-II | n/a | 0.275 ± 0.034 | 0.500 | 0.250 | 0.250 |
| Recall 2CD Neck Garden III-IV | n/a | 0.980 ± 0.020 | 0.951 | 1.000 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.853 ± 0.075 | 0.882 | 0.735 | 0.941 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.767 ± 0.052 | 0.738 | 0.762 | 0.786 |
| Recall 4 Subtrochanteric | n/a | 0.417 ± 0.132 | 0.750 | 0.500 | 0.417 |
| Errors: detection | n/a | 5.8 ± 2.2 | 8 | 6 | 3 |
| Errors: location | n/a | 9.6 ± 0.9 | 6 | 9 | 9 |
| Errors: grade | n/a | 19.2 ± 3.3 | 15 | 22 | 20 |
| Errors: total | n/a | 34.6 ± 3.4 | 29 | 37 | 32 |

### S07_unet_r34 vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.993 ± 0.005 | 0.981 | 0.994 | 0.999 |
| 6-class accuracy (%) | n/a | 84.6 ± 1.6 | 86.2 | 82.9 | 86.7 |
| Fracture sensitivity | n/a | 0.972 ± 0.005 | 0.952 | 0.972 | 0.966 |
| Fracture specificity | n/a | 0.972 ± 0.013 | 0.985 | 0.985 | 1.000 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.770 ± 0.027 | 0.814 | 0.757 | 0.799 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.718 ± 0.009 | 0.77 | 0.727 | 0.736 |
| Recall NF No fracture | n/a | 0.972 ± 0.013 | 0.985 | 0.985 | 1.000 |
| Recall 2AB Neck Garden I-II | n/a | 0.375 ± 0.088 | 0.500 | 0.438 | 0.375 |
| Recall 2CD Neck Garden III-IV | n/a | 0.985 ± 0.013 | 0.951 | 0.976 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.882 ± 0.047 | 0.882 | 0.824 | 0.882 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.748 ± 0.052 | 0.738 | 0.667 | 0.786 |
| Recall 4 Subtrochanteric | n/a | 0.550 ± 0.075 | 0.750 | 0.583 | 0.583 |
| Errors: detection | n/a | 5.8 ± 0.8 | 8 | 5 | 5 |
| Errors: location | n/a | 6.6 ± 1.5 | 6 | 8 | 5 |
| Errors: grade | n/a | 20.0 ± 2.9 | 15 | 23 | 18 |
| Errors: total | n/a | 32.4 ± 3.4 | 29 | 36 | 28 |

### S08_deeplabv3p_r34 vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.993 ± 0.004 | 0.981 | 0.995 | 0.999 |
| 6-class accuracy (%) | n/a | 85.5 ± 1.9 | 86.2 | 86.7 | 86.2 |
| Fracture sensitivity | n/a | 0.961 ± 0.010 | 0.952 | 0.952 | 0.931 |
| Fracture specificity | n/a | 0.978 ± 0.014 | 0.985 | 1.000 | 1.000 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.784 ± 0.022 | 0.814 | 0.793 | 0.778 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.733 ± 0.006 | 0.77 | 0.737 | 0.750 |
| Recall NF No fracture | n/a | 0.978 ± 0.014 | 0.985 | 1.000 | 1.000 |
| Recall 2AB Neck Garden I-II | n/a | 0.375 ± 0.125 | 0.500 | 0.312 | 0.250 |
| Recall 2CD Neck Garden III-IV | n/a | 0.990 ± 0.013 | 0.951 | 0.976 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.888 ± 0.038 | 0.882 | 0.941 | 0.853 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.767 ± 0.085 | 0.738 | 0.762 | 0.833 |
| Recall 4 Subtrochanteric | n/a | 0.583 ± 0.059 | 0.750 | 0.667 | 0.583 |
| Errors: detection | n/a | 7.0 ± 1.2 | 8 | 7 | 10 |
| Errors: location | n/a | 6.2 ± 0.8 | 6 | 5 | 5 |
| Errors: grade | n/a | 17.2 ± 3.9 | 15 | 16 | 14 |
| Errors: total | n/a | 30.4 ± 3.9 | 29 | 28 | 29 |

### S09_segformer_b2 vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.997 ± 0.002 | 0.981 | 0.999 | 0.999 |
| 6-class accuracy (%) | n/a | 86.8 ± 0.8 | 86.2 | 85.7 | 88.1 |
| Fracture sensitivity | n/a | 0.986 ± 0.005 | 0.952 | 0.986 | 0.986 |
| Fracture specificity | n/a | 0.982 ± 0.020 | 0.985 | 0.969 | 1.000 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.797 ± 0.020 | 0.814 | 0.781 | 0.804 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.735 ± 0.005 | 0.77 | 0.733 | 0.747 |
| Recall NF No fracture | n/a | 0.982 ± 0.020 | 0.985 | 0.969 | 1.000 |
| Recall 2AB Neck Garden I-II | n/a | 0.412 ± 0.095 | 0.500 | 0.312 | 0.375 |
| Recall 2CD Neck Garden III-IV | n/a | 0.985 ± 0.013 | 0.951 | 0.976 | 0.976 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.935 ± 0.053 | 0.882 | 0.941 | 0.971 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.781 ± 0.011 | 0.738 | 0.786 | 0.810 |
| Recall 4 Subtrochanteric | n/a | 0.567 ± 0.109 | 0.750 | 0.583 | 0.583 |
| Errors: detection | n/a | 3.2 ± 1.8 | 8 | 4 | 2 |
| Errors: location | n/a | 7.4 ± 1.5 | 6 | 5 | 7 |
| Errors: grade | n/a | 17.2 ± 2.2 | 15 | 21 | 16 |
| Errors: total | n/a | 27.8 ± 1.6 | 29 | 30 | 25 |

### S10_m0_convnext_unet vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.999 ± 0.001 | 0.981 | 1.000 | 0.999 |
| 6-class accuracy (%) | n/a | 86.0 ± 0.5 | 86.2 | 86.2 | 86.7 |
| Fracture sensitivity | n/a | 0.983 ± 0.006 | 0.952 | 0.979 | 0.993 |
| Fracture specificity | n/a | 1.000 ± 0.000 | 0.985 | 1.000 | 1.000 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.776 ± 0.017 | 0.814 | 0.783 | 0.779 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.746 ± 0.002 | 0.77 | 0.748 | 0.749 |
| Recall NF No fracture | n/a | 1.000 ± 0.000 | 0.985 | 1.000 | 1.000 |
| Recall 2AB Neck Garden I-II | n/a | 0.312 ± 0.077 | 0.500 | 0.312 | 0.312 |
| Recall 2CD Neck Garden III-IV | n/a | 0.990 ± 0.013 | 0.951 | 0.976 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.953 ± 0.016 | 0.882 | 0.941 | 0.971 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.738 ± 0.034 | 0.738 | 0.762 | 0.762 |
| Recall 4 Subtrochanteric | n/a | 0.550 ± 0.046 | 0.750 | 0.583 | 0.500 |
| Errors: detection | n/a | 2.4 ± 0.9 | 8 | 3 | 1 |
| Errors: location | n/a | 7.0 ± 0.7 | 6 | 6 | 7 |
| Errors: grade | n/a | 20.0 ± 1.9 | 15 | 20 | 20 |
| Errors: total | n/a | 29.4 ± 1.1 | 29 | 29 | 28 |

Article's final model, confusion matrix (Fig. 3):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 64 | 0 | 1 | 0 | 0 | 0 |
| 2AB | 3 | 8 | 3 | 2 | 0 | 0 |
| 2CD | 0 | 2 | 39 | 0 | 0 | 0 |
| 3AB | 1 | 0 | 0 | 30 | 3 | 0 |
| 3CDE | 1 | 0 | 1 | 7 | 31 | 2 |
| 4 | 2 | 0 | 0 | 0 | 1 | 9 |

## Dice definitions

The article reports "the average Dice between annotated fracture regions and the regions automatically detected": 0.77 (0.56–0.98) for its final model. That interval is about mean ± 1.96 SD of single femurs, so it probably averages detected fractures only. The same predictions give quite different numbers under other common definitions (test set, mean ± SD of the 5 fold models; the ensemble has no union masks):

| Definition | A6_slgnet | S01_yolov8s_seg_1024 | S04_densenet121 | S05_efficientnet_b4 | S06_convnext_tiny | S07_unet_r34 | S08_deeplabv3p_r34 | S09_segformer_b2 | S10_m0_convnext_unet |
|---|---|---|---|---|---|---|---|---|---|
| Detected fractures only, top instance (our main definition) | 0.745 ± 0.003 | 0.741 ± 0.009 | n/a | n/a | n/a | 0.718 ± 0.009 | 0.733 ± 0.006 | 0.735 ± 0.005 | 0.746 ± 0.002 |
| Same at Ultralytics' default threshold 0.25 | n/a | 0.744 ± 0.007 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Same at 0.25, union of all instances ≥ 0.25 | n/a | 0.745 ± 0.007 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| All fractured femurs, missed = 0 | 0.725 ± 0.007 | 0.693 ± 0.016 | n/a | n/a | n/a | 0.698 ± 0.009 | 0.704 ± 0.005 | 0.725 ± 0.008 | 0.734 ± 0.005 |
| All 210 femurs, no-fracture femur = 1 if nothing called else 0 | 0.807 ± 0.005 | 0.781 ± 0.011 | n/a | n/a | n/a | 0.783 ± 0.006 | 0.789 ± 0.005 | 0.805 ± 0.011 | 0.816 ± 0.004 |
| Global (pooled pixels over fractured femurs) | 0.762 ± 0.005 | 0.733 ± 0.013 | n/a | n/a | n/a | 0.737 ± 0.007 | 0.744 ± 0.005 | 0.754 ± 0.008 | 0.767 ± 0.004 |
| Per-femur spread, detected only: mean ± 1.96 SD (compare the article's 0.56–0.98) | 0.49–1.00 (ensemble) | 0.52–0.99 (ensemble) | n/a | n/a | n/a | 0.47–1.00 (ensemble) | 0.51–0.99 (ensemble) | 0.50–1.00 (ensemble) | 0.48–1.00 (ensemble) |

## OOF versus test segmentation

| Run | Dice detected, OOF pooled | Dice detected, test models | Predicted / true area, OOF (median, IQR) | Predicted / true area, test models (median, IQR, pooled over models) |
|---|---|---|---|---|
| A6_slgnet | 0.796 | 0.745 ± 0.003 | 0.96 (0.78–1.17) | 1.23 (0.93–1.69) |
| S01_yolov8s_seg_1024 | 0.800 | 0.741 ± 0.009 | 0.99 (0.84–1.22) | 1.27 (0.94–1.80) |
| S04_densenet121 | n/a | n/a | n/a | n/a |
| S05_efficientnet_b4 | n/a | n/a | n/a | n/a |
| S06_convnext_tiny | n/a | n/a | n/a | n/a |
| S07_unet_r34 | 0.772 | 0.718 ± 0.009 | 0.93 (0.74–1.15) | 1.17 (0.85–1.76) |
| S08_deeplabv3p_r34 | 0.781 | 0.733 ± 0.006 | 0.93 (0.75–1.15) | 1.17 (0.88–1.75) |
| S09_segformer_b2 | 0.784 | 0.735 ± 0.005 | 0.93 (0.74–1.15) | 1.19 (0.87–1.72) |
| S10_m0_convnext_unet | 0.799 | 0.746 ± 0.002 | 0.98 (0.79–1.18) | 1.27 (0.94–1.74) |

Median mask area in pixels by class: true masks of the training femurs and of the test femurs (the article says its test labels were reviewed by a committee), and the masks the models predict for called fractures (OOF; test pooled over the 5 fold models).

| Class | True, train | True, test | Test / train | Predicted OOF, A6_slgnet | Predicted test, A6_slgnet | Predicted OOF, S01_yolov8s_seg_1024 | Predicted test, S01_yolov8s_seg_1024 | Predicted OOF, S04_densenet121 | Predicted test, S04_densenet121 | Predicted OOF, S05_efficientnet_b4 | Predicted test, S05_efficientnet_b4 | Predicted OOF, S06_convnext_tiny | Predicted test, S06_convnext_tiny | Predicted OOF, S07_unet_r34 | Predicted test, S07_unet_r34 | Predicted OOF, S08_deeplabv3p_r34 | Predicted test, S08_deeplabv3p_r34 | Predicted OOF, S09_segformer_b2 | Predicted test, S09_segformer_b2 | Predicted OOF, S10_m0_convnext_unet | Predicted test, S10_m0_convnext_unet |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 45,963 | 29,797 | 0.65 | 44,552 | 50,646 | 54,558 | 57,046 | n/a | n/a | n/a | n/a | n/a | n/a | 40,114 | 46,235 | 38,032 | 45,420 | 37,198 | 46,262 | 42,584 | 54,829 |
| 2CD Neck Garden III-IV | 75,612 | 45,844 | 0.61 | 74,452 | 78,553 | 75,926 | 76,820 | n/a | n/a | n/a | n/a | n/a | n/a | 71,564 | 82,886 | 71,951 | 81,019 | 72,908 | 78,273 | 74,660 | 77,694 |
| 3AB Pertroch Evans I-II | 85,540 | 88,794 | 1.04 | 90,712 | 88,074 | 93,944 | 96,513 | n/a | n/a | n/a | n/a | n/a | n/a | 86,042 | 85,245 | 85,475 | 86,284 | 85,871 | 83,822 | 91,848 | 89,882 |
| 3CDE Pertroch Evans III-V | 135,306 | 120,095 | 0.89 | 129,562 | 134,645 | 131,154 | 130,396 | n/a | n/a | n/a | n/a | n/a | n/a | 125,625 | 133,345 | 130,645 | 139,181 | 130,116 | 130,197 | 133,085 | 138,130 |
| 4 Subtrochanteric | 126,204 | 149,960 | 1.19 | 124,134 | 119,801 | 144,031 | 125,186 | n/a | n/a | n/a | n/a | n/a | n/a | 125,503 | 100,960 | 119,581 | 103,458 | 130,040 | 118,053 | 136,438 | 124,222 |
