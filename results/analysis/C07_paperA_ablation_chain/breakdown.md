# Breakdown: S06_convnext_tiny, S10_m0_convnext_unet, A2_region, A3_geometry, A4_hierarchy, A5_shared_grade, A6_slgnet

Made by `scripts/run_breakdown.py` from the filed run folders. Decisions use each run's OOF threshold. Mean ± SD is over the 5 held-out folds (OOF) or the 5 fold models (test); SD uses n − 1.

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

## A2_region

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A2` at 768x512 px; fracture threshold 0.0581 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 88.9%) is fold 4.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.996 | 0.996 ± 0.003 | 0.999 ± 0.001 | 0.999 | 0.999 |
| Fracture sensitivity | 0.978 | 0.978 ± 0.010 | 0.985 ± 0.008 | 0.993 | 0.993 |
| Fracture specificity | 0.986 | 0.986 ± 0.010 | 0.988 ± 0.013 | 0.985 | 0.954 |
| 6-class accuracy (%) | 88.3 | 88.3 ± 0.5 | 86.3 ± 1.0 | 84.8 | 86.2 |
| Macro-F1 | 0.820 | 0.817 ± 0.016 | 0.779 ± 0.010 | 0.776 | 0.785 |
| Balanced accuracy | 0.801 | 0.801 ± 0.034 | 0.762 ± 0.006 | 0.758 | 0.769 |
| Cohen's kappa | 0.846 | 0.846 ± 0.007 | 0.825 ± 0.012 | 0.805 | 0.824 |
| Location accuracy (fractures called) | 0.944 | 0.944 ± 0.022 | 0.955 ± 0.008 | 0.951 | 0.965 |
| Grade accuracy, location right | 0.892 | 0.892 ± 0.019 | 0.850 ± 0.016 | 0.822 | 0.847 |
| Dice, fractured femurs (missed = 0) | 0.772 | 0.772 ± 0.012 | 0.726 ± 0.007 | 0.718 | 0.738 |
| Dice, detected fractures only | 0.790 | 0.790 ± 0.012 | 0.737 ± 0.009 | 0.723 | 0.743 |
| Predicted / true area, median | 0.95 | 0.95 ± 0.03 | 1.23 ± 0.02 | 1.21 | 1.22 |
| HD95 (px), median, detected | 60.8 | 61.5 ± 2.6 | 68.4 ± 1.8 | 68.7 | 68.1 |
| Errors: detection | 20.0 | 4.0 ± 1.6 | 3.0 ± 1.6 | 2.0 | 4.0 |
| Errors: location | 37.0 | 7.4 ± 3.0 | 6.4 ± 1.1 | 7.0 | 5.0 |
| Errors: grade | 65.0 | 13.0 ± 2.5 | 19.4 ± 2.1 | 23.0 | 20.0 |
| Errors: total | 122.0 | 24.4 ± 1.1 | 28.8 ± 2.0 | 32.0 | 29.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.986 | 0.986 ± 0.010 | 0.988 ± 0.013 | 0.985 | 0.954 |
| 2AB Neck Garden I-II | 57 / 16 | 0.544 | 0.544 ± 0.109 | 0.263 ± 0.052 | 0.312 | 0.250 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.902 | 0.902 ± 0.072 | 0.995 ± 0.011 | 1.000 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.799 | 0.799 ± 0.019 | 0.906 ± 0.064 | 0.824 | 0.912 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.854 | 0.855 ± 0.059 | 0.805 ± 0.031 | 0.762 | 0.833 |
| 4 Subtrochanteric | 36 / 12 | 0.722 | 0.721 ± 0.226 | 0.617 ± 0.046 | 0.667 | 0.667 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.618 | 0.618 ± 0.030 | 0.615 ± 0.025 | 0.633 | 0.912 / 0.900 ± 0.056 | 0.678 / 0.684 ± 0.022 | 0.89 / 1.72 ± 0.13 |
| 2CD Neck Garden III-IV | 0.757 | 0.757 ± 0.022 | 0.688 ± 0.009 | 0.696 | 0.972 / 0.995 ± 0.011 | 0.778 / 0.692 ± 0.008 | 0.93 / 1.68 ± 0.07 |
| 3AB Pertroch Evans I-II | 0.778 | 0.778 ± 0.034 | 0.723 ± 0.020 | 0.739 | 0.985 / 0.988 ± 0.016 | 0.789 / 0.731 ± 0.019 | 0.96 / 1.12 ± 0.06 |
| 3CDE Pertroch Evans III-V | 0.848 | 0.848 ± 0.013 | 0.837 ± 0.006 | 0.843 | 1.000 / 1.000 ± 0.000 | 0.848 / 0.837 ± 0.006 | 0.96 / 1.03 ± 0.01 |
| 4 Subtrochanteric | 0.688 | 0.688 ± 0.080 | 0.624 ± 0.044 | 0.647 | 0.972 / 1.000 ± 0.000 | 0.707 / 0.624 ± 0.044 | 0.96 / 0.90 ± 0.08 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 358 | 1 | 2 | 2 | 0 | 0 |
| 2AB | 5 | 31 | 17 | 4 | 0 | 0 |
| 2CD | 7 | 8 | 229 | 9 | 1 | 0 |
| 3AB | 2 | 2 | 2 | 107 | 21 | 0 |
| 3CDE | 0 | 1 | 6 | 19 | 170 | 3 |
| 4 | 1 | 0 | 0 | 2 | 7 | 26 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 62 | 1 | 2 | 0 | 0 | 0 |
| 2AB | 1 | 4 | 11 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 31 | 3 | 0 |
| 3CDE | 0 | 0 | 1 | 6 | 35 | 0 |
| 4 | 0 | 0 | 0 | 1 | 3 | 8 |

## A3_geometry

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A3` at 768x512 px; fracture threshold 0.0577 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 89.9%) is fold 0.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.996 | 0.997 ± 0.002 | 0.998 ± 0.001 | 0.999 | 0.999 |
| Fracture sensitivity | 0.981 | 0.981 ± 0.013 | 0.982 ± 0.010 | 0.979 | 0.993 |
| Fracture specificity | 0.972 | 0.973 ± 0.026 | 0.969 ± 0.011 | 0.985 | 0.908 |
| 6-class accuracy (%) | 88.9 | 88.9 ± 0.9 | 86.5 ± 1.1 | 87.6 | 85.7 |
| Macro-F1 | 0.838 | 0.838 ± 0.009 | 0.782 ± 0.012 | 0.770 | 0.786 |
| Balanced accuracy | 0.825 | 0.825 ± 0.015 | 0.761 ± 0.012 | 0.750 | 0.767 |
| Cohen's kappa | 0.854 | 0.854 ± 0.010 | 0.827 ± 0.014 | 0.841 | 0.818 |
| Location accuracy (fractures called) | 0.951 | 0.951 ± 0.013 | 0.951 ± 0.009 | 0.937 | 0.958 |
| Grade accuracy, location right | 0.901 | 0.902 ± 0.024 | 0.870 ± 0.026 | 0.898 | 0.870 |
| Dice, fractured femurs (missed = 0) | 0.777 | 0.777 ± 0.018 | 0.729 ± 0.007 | 0.731 | 0.734 |
| Dice, detected fractures only | 0.792 | 0.792 ± 0.009 | 0.742 ± 0.007 | 0.746 | 0.739 |
| Predicted / true area, median | 0.95 | 0.94 ± 0.02 | 1.21 ± 0.04 | 1.19 | 1.20 |
| HD95 (px), median, detected | 61.2 | 60.9 ± 3.4 | 69.0 ± 1.1 | 68.3 | 66.0 |
| Errors: detection | 23.0 | 4.6 ± 1.1 | 4.6 ± 1.3 | 4.0 | 7.0 |
| Errors: location | 33.0 | 6.6 ± 1.7 | 7.0 ± 1.2 | 9.0 | 6.0 |
| Errors: grade | 60.0 | 12.0 ± 3.1 | 16.8 ± 3.5 | 13.0 | 17.0 |
| Errors: total | 116.0 | 23.2 ± 1.8 | 28.4 ± 2.3 | 26.0 | 30.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.972 | 0.973 ± 0.026 | 0.969 ± 0.011 | 0.985 | 0.908 |
| 2AB Neck Garden I-II | 57 / 16 | 0.684 | 0.686 ± 0.143 | 0.312 ± 0.077 | 0.250 | 0.312 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.913 | 0.913 ± 0.058 | 1.000 ± 0.000 | 1.000 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.806 | 0.806 ± 0.040 | 0.912 ± 0.036 | 0.941 | 0.941 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.849 | 0.850 ± 0.052 | 0.838 ± 0.062 | 0.905 | 0.857 |
| 4 Subtrochanteric | 36 / 12 | 0.722 | 0.721 ± 0.102 | 0.533 ± 0.095 | 0.417 | 0.583 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.629 | 0.630 ± 0.036 | 0.593 ± 0.027 | 0.586 | 0.912 / 0.875 ± 0.062 | 0.690 / 0.680 ± 0.040 | 0.86 / 1.72 ± 0.06 |
| 2CD Neck Garden III-IV | 0.763 | 0.763 ± 0.022 | 0.695 ± 0.009 | 0.700 | 0.984 / 1.000 ± 0.000 | 0.775 / 0.695 ± 0.009 | 0.93 / 1.69 ± 0.05 |
| 3AB Pertroch Evans I-II | 0.776 | 0.777 ± 0.047 | 0.734 ± 0.012 | 0.740 | 0.978 / 0.994 ± 0.013 | 0.794 / 0.738 ± 0.008 | 0.96 / 1.11 ± 0.05 |
| 3CDE Pertroch Evans III-V | 0.848 | 0.848 ± 0.017 | 0.841 ± 0.005 | 0.846 | 1.000 / 1.000 ± 0.000 | 0.848 / 0.841 ± 0.005 | 0.96 / 1.04 ± 0.02 |
| 4 Subtrochanteric | 0.716 | 0.716 ± 0.069 | 0.620 ± 0.023 | 0.635 | 0.972 / 0.967 ± 0.046 | 0.737 / 0.642 ± 0.028 | 0.98 / 0.89 ± 0.05 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 353 | 5 | 4 | 1 | 0 | 0 |
| 2AB | 5 | 39 | 9 | 4 | 0 | 0 |
| 2CD | 4 | 10 | 232 | 7 | 1 | 0 |
| 3AB | 3 | 0 | 2 | 108 | 21 | 0 |
| 3CDE | 0 | 1 | 7 | 20 | 169 | 2 |
| 4 | 1 | 0 | 0 | 1 | 8 | 26 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 59 | 1 | 4 | 1 | 0 | 0 |
| 2AB | 1 | 5 | 10 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 32 | 2 | 0 |
| 3CDE | 0 | 0 | 1 | 5 | 36 | 0 |
| 4 | 0 | 0 | 0 | 1 | 4 | 7 |

## A4_hierarchy

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A4` at 768x512 px; fracture threshold 0.0149 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 89.5%) is fold 1.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.995 ± 0.005 | 0.998 ± 0.001 | 0.998 | 0.999 |
| Fracture sensitivity | 0.979 | 0.979 ± 0.012 | 0.975 ± 0.004 | 0.979 | 0.993 |
| Fracture specificity | 0.986 | 0.986 ± 0.010 | 0.985 ± 0.015 | 1.000 | 0.969 |
| 6-class accuracy (%) | 87.8 | 87.8 ± 1.4 | 86.3 ± 1.3 | 87.1 | 87.1 |
| Macro-F1 | 0.823 | 0.821 ± 0.035 | 0.775 ± 0.016 | 0.783 | 0.800 |
| Balanced accuracy | 0.813 | 0.813 ± 0.047 | 0.755 ± 0.015 | 0.762 | 0.781 |
| Cohen's kappa | 0.840 | 0.840 ± 0.018 | 0.824 ± 0.017 | 0.835 | 0.836 |
| Location accuracy (fractures called) | 0.946 | 0.946 ± 0.017 | 0.948 ± 0.007 | 0.958 | 0.958 |
| Grade accuracy, location right | 0.881 | 0.881 ± 0.025 | 0.868 ± 0.023 | 0.860 | 0.863 |
| Dice, fractured femurs (missed = 0) | 0.770 | 0.770 ± 0.018 | 0.727 ± 0.005 | 0.734 | 0.732 |
| Dice, detected fractures only | 0.786 | 0.787 ± 0.018 | 0.745 ± 0.004 | 0.749 | 0.737 |
| Predicted / true area, median | 0.95 | 0.95 ± 0.03 | 1.26 ± 0.03 | 1.29 | 1.23 |
| HD95 (px), median, detected | 61.0 | 61.4 ± 2.4 | 67.1 ± 1.1 | 67.5 | 64.6 |
| Errors: detection | 19.0 | 3.8 ± 2.2 | 4.6 ± 1.1 | 3.0 | 3.0 |
| Errors: location | 36.0 | 7.2 ± 2.3 | 7.4 ± 0.9 | 6.0 | 6.0 |
| Errors: grade | 72.0 | 14.4 ± 3.2 | 16.8 ± 2.8 | 18.0 | 18.0 |
| Errors: total | 127.0 | 25.4 ± 2.8 | 28.8 ± 2.8 | 27.0 | 27.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.986 | 0.986 ± 0.010 | 0.985 ± 0.015 | 1.000 | 0.969 |
| 2AB Neck Garden I-II | 57 / 16 | 0.649 | 0.647 ± 0.111 | 0.287 ± 0.084 | 0.250 | 0.375 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.882 | 0.882 ± 0.072 | 0.985 ± 0.013 | 1.000 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.784 | 0.783 ± 0.032 | 0.906 ± 0.044 | 0.882 | 0.971 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.829 | 0.830 ± 0.081 | 0.833 ± 0.063 | 0.857 | 0.786 |
| 4 Subtrochanteric | 36 / 12 | 0.750 | 0.750 ± 0.211 | 0.533 ± 0.075 | 0.583 | 0.583 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.595 | 0.595 ± 0.023 | 0.579 ± 0.018 | 0.576 | 0.947 / 0.863 ± 0.052 | 0.628 / 0.673 ± 0.026 | 0.84 / 1.73 ± 0.02 |
| 2CD Neck Garden III-IV | 0.762 | 0.762 ± 0.019 | 0.687 ± 0.013 | 0.693 | 0.980 / 0.995 ± 0.011 | 0.777 / 0.691 ± 0.012 | 0.91 / 1.74 ± 0.03 |
| 3AB Pertroch Evans I-II | 0.765 | 0.765 ± 0.060 | 0.737 ± 0.026 | 0.745 | 0.970 / 0.971 ± 0.029 | 0.789 / 0.759 ± 0.011 | 0.98 / 1.14 ± 0.06 |
| 3CDE Pertroch Evans III-V | 0.846 | 0.846 ± 0.013 | 0.837 ± 0.002 | 0.843 | 1.000 / 1.000 ± 0.000 | 0.846 / 0.837 ± 0.002 | 0.95 / 1.06 ± 0.03 |
| 4 Subtrochanteric | 0.703 | 0.703 ± 0.089 | 0.646 ± 0.035 | 0.651 | 0.944 / 0.983 ± 0.037 | 0.744 / 0.657 ± 0.015 | 1.00 / 0.92 ± 0.06 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 358 | 2 | 3 | 0 | 0 | 0 |
| 2AB | 3 | 37 | 13 | 4 | 0 | 0 |
| 2CD | 5 | 14 | 224 | 9 | 2 | 0 |
| 3AB | 4 | 2 | 2 | 105 | 21 | 0 |
| 3CDE | 0 | 1 | 6 | 24 | 165 | 3 |
| 4 | 2 | 0 | 0 | 1 | 6 | 27 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 63 | 2 | 0 | 0 | 0 | 0 |
| 2AB | 1 | 6 | 9 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 33 | 1 | 0 |
| 3CDE | 0 | 0 | 1 | 8 | 33 | 0 |
| 4 | 0 | 0 | 0 | 1 | 4 | 7 |

## A5_shared_grade

Network `slg:convnext_tiny.fb_in22k_ft_in1k:A5` at 768x512 px; fracture threshold 0.0077 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 89.9%) is fold 0.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.996 ± 0.004 | 0.998 ± 0.003 | 1.000 | 1.000 |
| Fracture sensitivity | 0.982 | 0.982 ± 0.012 | 0.978 ± 0.009 | 0.966 | 0.993 |
| Fracture specificity | 0.986 | 0.986 ± 0.014 | 0.994 ± 0.008 | 1.000 | 0.985 |
| 6-class accuracy (%) | 88.0 | 88.0 ± 1.4 | 85.5 ± 1.0 | 86.7 | 87.1 |
| Macro-F1 | 0.820 | 0.821 ± 0.032 | 0.774 ± 0.015 | 0.798 | 0.792 |
| Balanced accuracy | 0.805 | 0.806 ± 0.025 | 0.752 ± 0.014 | 0.774 | 0.772 |
| Cohen's kappa | 0.842 | 0.842 ± 0.019 | 0.815 ± 0.013 | 0.829 | 0.836 |
| Location accuracy (fractures called) | 0.948 | 0.948 ± 0.017 | 0.952 ± 0.006 | 0.957 | 0.958 |
| Grade accuracy, location right | 0.880 | 0.880 ± 0.040 | 0.844 ± 0.016 | 0.865 | 0.855 |
| Dice, fractured femurs (missed = 0) | 0.769 | 0.769 ± 0.015 | 0.727 ± 0.005 | 0.720 | 0.734 |
| Dice, detected fractures only | 0.783 | 0.783 ± 0.010 | 0.743 ± 0.002 | 0.745 | 0.739 |
| Predicted / true area, median | 0.96 | 0.96 ± 0.02 | 1.23 ± 0.01 | 1.25 | 1.24 |
| HD95 (px), median, detected | 61.2 | 60.9 ± 1.7 | 68.7 ± 1.8 | 69.2 | 64.0 |
| Errors: detection | 17.0 | 3.4 ± 2.1 | 3.6 ± 1.5 | 5.0 | 2.0 |
| Errors: location | 35.0 | 7.0 ± 2.2 | 6.8 ± 0.8 | 6.0 | 6.0 |
| Errors: grade | 73.0 | 14.6 ± 5.2 | 20.0 ± 2.1 | 17.0 | 19.0 |
| Errors: total | 125.0 | 25.0 ± 3.0 | 30.4 ± 2.2 | 28.0 | 27.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.986 | 0.986 ± 0.014 | 0.994 ± 0.008 | 1.000 | 0.985 |
| 2AB Neck Garden I-II | 57 / 16 | 0.614 | 0.618 ± 0.128 | 0.300 ± 0.052 | 0.312 | 0.312 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.902 | 0.902 ± 0.042 | 0.985 ± 0.013 | 0.976 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.761 | 0.761 ± 0.042 | 0.865 ± 0.057 | 0.882 | 0.941 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.844 | 0.845 ± 0.064 | 0.800 ± 0.013 | 0.810 | 0.810 |
| 4 Subtrochanteric | 36 / 12 | 0.722 | 0.721 ± 0.102 | 0.567 ± 0.070 | 0.667 | 0.583 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.615 | 0.615 ± 0.044 | 0.579 ± 0.026 | 0.576 | 0.965 / 0.863 ± 0.052 | 0.637 / 0.672 ± 0.020 | 0.87 / 1.74 ± 0.07 |
| 2CD Neck Garden III-IV | 0.759 | 0.759 ± 0.029 | 0.691 ± 0.004 | 0.698 | 0.984 / 0.995 ± 0.011 | 0.771 / 0.694 ± 0.006 | 0.95 / 1.74 ± 0.04 |
| 3AB Pertroch Evans I-II | 0.760 | 0.760 ± 0.053 | 0.731 ± 0.015 | 0.743 | 0.970 / 0.976 ± 0.025 | 0.784 / 0.749 ± 0.005 | 0.97 / 1.13 ± 0.08 |
| 3CDE Pertroch Evans III-V | 0.849 | 0.849 ± 0.008 | 0.837 ± 0.001 | 0.843 | 1.000 / 1.000 ± 0.000 | 0.849 / 0.837 ± 0.001 | 0.96 / 1.03 ± 0.03 |
| 4 Subtrochanteric | 0.677 | 0.677 ± 0.064 | 0.647 ± 0.028 | 0.654 | 0.944 / 1.000 ± 0.000 | 0.717 / 0.647 ± 0.028 | 0.97 / 0.94 ± 0.03 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 358 | 1 | 3 | 1 | 0 | 0 |
| 2AB | 2 | 35 | 15 | 5 | 0 | 0 |
| 2CD | 4 | 11 | 229 | 7 | 3 | 0 |
| 3AB | 4 | 2 | 1 | 102 | 25 | 0 |
| 3CDE | 0 | 1 | 5 | 22 | 168 | 3 |
| 4 | 2 | 0 | 0 | 2 | 6 | 26 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 64 | 1 | 0 | 0 | 0 | 0 |
| 2AB | 1 | 5 | 10 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 32 | 2 | 0 |
| 3CDE | 0 | 0 | 1 | 7 | 34 | 0 |
| 4 | 0 | 0 | 0 | 1 | 4 | 7 |

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

## Runs side by side (test, 5 fold models mean ± SD; OOF pooled in brackets)

| Metric | S06_convnext_tiny | S10_m0_convnext_unet | A2_region | A3_geometry | A4_hierarchy | A5_shared_grade | A6_slgnet |
|---|---|---|---|---|---|---|---|
| 6-class accuracy (%) | 83.5 ± 1.6 (86.4) | 86.0 ± 0.5 (88.4) | 86.3 ± 1.0 (88.3) | 86.5 ± 1.1 (88.9) | 86.3 ± 1.3 (87.8) | 85.5 ± 1.0 (88.0) | 86.3 ± 1.5 (87.0) |
| Macro-F1 | 0.732 ± 0.024 (0.784) | 0.776 ± 0.017 (0.825) | 0.779 ± 0.010 (0.820) | 0.782 ± 0.012 (0.838) | 0.775 ± 0.016 (0.823) | 0.774 ± 0.015 (0.820) | 0.789 ± 0.015 (0.806) |
| Dice, fractured (missed = 0) | n/a (n/a) | 0.734 ± 0.005 (0.777) | 0.726 ± 0.007 (0.772) | 0.729 ± 0.007 (0.777) | 0.727 ± 0.005 (0.770) | 0.727 ± 0.005 (0.769) | 0.725 ± 0.007 (0.774) |
| Recall NF | 0.994 ± 0.014 (0.978) | 1.000 ± 0.000 (0.992) | 0.988 ± 0.013 (0.986) | 0.969 ± 0.011 (0.972) | 0.985 ± 0.015 (0.986) | 0.994 ± 0.008 (0.986) | 0.991 ± 0.008 (0.983) |
| Recall 2AB | 0.275 ± 0.034 (0.474) | 0.312 ± 0.077 (0.544) | 0.263 ± 0.052 (0.544) | 0.312 ± 0.077 (0.684) | 0.287 ± 0.084 (0.649) | 0.300 ± 0.052 (0.614) | 0.287 ± 0.034 (0.614) |
| Recall 2CD | 0.980 ± 0.020 (0.921) | 0.990 ± 0.013 (0.909) | 0.995 ± 0.011 (0.902) | 1.000 ± 0.000 (0.913) | 0.985 ± 0.013 (0.882) | 0.985 ± 0.013 (0.902) | 0.985 ± 0.013 (0.878) |
| Recall 3AB | 0.853 ± 0.075 (0.761) | 0.953 ± 0.016 (0.791) | 0.906 ± 0.064 (0.799) | 0.912 ± 0.036 (0.806) | 0.906 ± 0.044 (0.784) | 0.865 ± 0.057 (0.761) | 0.894 ± 0.049 (0.784) |
| Recall 3CDE | 0.767 ± 0.052 (0.804) | 0.738 ± 0.034 (0.839) | 0.805 ± 0.031 (0.854) | 0.838 ± 0.062 (0.849) | 0.833 ± 0.063 (0.829) | 0.800 ± 0.013 (0.844) | 0.786 ± 0.051 (0.804) |
| Recall 4 | 0.417 ± 0.132 (0.639) | 0.550 ± 0.046 (0.750) | 0.617 ± 0.046 (0.722) | 0.533 ± 0.095 (0.722) | 0.533 ± 0.075 (0.750) | 0.567 ± 0.070 (0.722) | 0.700 ± 0.075 (0.750) |
| Dice 2AB | n/a (n/a) | 0.578 ± 0.032 (0.624) | 0.615 ± 0.025 (0.618) | 0.593 ± 0.027 (0.629) | 0.579 ± 0.018 (0.595) | 0.579 ± 0.026 (0.615) | 0.573 ± 0.022 (0.612) |
| Dice 2CD | n/a (n/a) | 0.699 ± 0.009 (0.773) | 0.688 ± 0.009 (0.757) | 0.695 ± 0.009 (0.763) | 0.687 ± 0.013 (0.762) | 0.691 ± 0.004 (0.759) | 0.692 ± 0.007 (0.769) |
| Dice 3AB | n/a (n/a) | 0.750 ± 0.008 (0.763) | 0.723 ± 0.020 (0.778) | 0.734 ± 0.012 (0.776) | 0.737 ± 0.026 (0.765) | 0.731 ± 0.015 (0.760) | 0.727 ± 0.022 (0.774) |
| Dice 3CDE | n/a (n/a) | 0.836 ± 0.004 (0.852) | 0.837 ± 0.006 (0.848) | 0.841 ± 0.005 (0.848) | 0.837 ± 0.002 (0.846) | 0.837 ± 0.001 (0.849) | 0.836 ± 0.004 (0.846) |
| Dice 4 | n/a (n/a) | 0.655 ± 0.039 (0.682) | 0.624 ± 0.044 (0.688) | 0.620 ± 0.023 (0.716) | 0.646 ± 0.035 (0.703) | 0.647 ± 0.028 (0.677) | 0.641 ± 0.034 (0.674) |

Paired tests between the runs (same femurs) are in `stats.md` of the same folder, if it was made with `scripts/run_stats.py ... --compare`.

## Comparison with the base article

Source: González et al., Insights Imaging 2025;16:86 (doi 10.1186/s13244-025-01958-y). Table 2 gives mean ± SD of six training runs per setting on the same 210 test femurs; the final model is the run with the best validation accuracy (Table 3, confusion matrix in Fig. 3). Our fold models are trained on 4/5 of the 1,043 training femurs (about 834), the article's on all 1,043 plus a 201-femur validation set; our counterpart of its selected model is the fold model with the best accuracy on its own held-out fold.

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

### A2_region vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.999 ± 0.001 | 0.981 | 0.999 | 0.999 |
| 6-class accuracy (%) | n/a | 86.3 ± 1.0 | 86.2 | 84.8 | 86.2 |
| Fracture sensitivity | n/a | 0.985 ± 0.008 | 0.952 | 0.993 | 0.993 |
| Fracture specificity | n/a | 0.988 ± 0.013 | 0.985 | 0.985 | 0.954 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.779 ± 0.010 | 0.814 | 0.776 | 0.785 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.737 ± 0.009 | 0.77 | 0.723 | 0.743 |
| Recall NF No fracture | n/a | 0.988 ± 0.013 | 0.985 | 0.985 | 0.954 |
| Recall 2AB Neck Garden I-II | n/a | 0.263 ± 0.052 | 0.500 | 0.312 | 0.250 |
| Recall 2CD Neck Garden III-IV | n/a | 0.995 ± 0.011 | 0.951 | 1.000 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.906 ± 0.064 | 0.882 | 0.824 | 0.912 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.805 ± 0.031 | 0.738 | 0.762 | 0.833 |
| Recall 4 Subtrochanteric | n/a | 0.617 ± 0.046 | 0.750 | 0.667 | 0.667 |
| Errors: detection | n/a | 3.0 ± 1.6 | 8 | 2 | 4 |
| Errors: location | n/a | 6.4 ± 1.1 | 6 | 7 | 5 |
| Errors: grade | n/a | 19.4 ± 2.1 | 15 | 23 | 20 |
| Errors: total | n/a | 28.8 ± 2.0 | 29 | 32 | 29 |

### A3_geometry vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.998 ± 0.001 | 0.981 | 0.999 | 0.999 |
| 6-class accuracy (%) | n/a | 86.5 ± 1.1 | 86.2 | 87.6 | 85.7 |
| Fracture sensitivity | n/a | 0.982 ± 0.010 | 0.952 | 0.979 | 0.993 |
| Fracture specificity | n/a | 0.969 ± 0.011 | 0.985 | 0.985 | 0.908 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.782 ± 0.012 | 0.814 | 0.770 | 0.786 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.742 ± 0.007 | 0.77 | 0.746 | 0.739 |
| Recall NF No fracture | n/a | 0.969 ± 0.011 | 0.985 | 0.985 | 0.908 |
| Recall 2AB Neck Garden I-II | n/a | 0.312 ± 0.077 | 0.500 | 0.250 | 0.312 |
| Recall 2CD Neck Garden III-IV | n/a | 1.000 ± 0.000 | 0.951 | 1.000 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.912 ± 0.036 | 0.882 | 0.941 | 0.941 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.838 ± 0.062 | 0.738 | 0.905 | 0.857 |
| Recall 4 Subtrochanteric | n/a | 0.533 ± 0.095 | 0.750 | 0.417 | 0.583 |
| Errors: detection | n/a | 4.6 ± 1.3 | 8 | 4 | 7 |
| Errors: location | n/a | 7.0 ± 1.2 | 6 | 9 | 6 |
| Errors: grade | n/a | 16.8 ± 3.5 | 15 | 13 | 17 |
| Errors: total | n/a | 28.4 ± 2.3 | 29 | 26 | 30 |

### A4_hierarchy vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.998 ± 0.001 | 0.981 | 0.998 | 0.999 |
| 6-class accuracy (%) | n/a | 86.3 ± 1.3 | 86.2 | 87.1 | 87.1 |
| Fracture sensitivity | n/a | 0.975 ± 0.004 | 0.952 | 0.979 | 0.993 |
| Fracture specificity | n/a | 0.985 ± 0.015 | 0.985 | 1.000 | 0.969 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.775 ± 0.016 | 0.814 | 0.783 | 0.800 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.745 ± 0.004 | 0.77 | 0.749 | 0.737 |
| Recall NF No fracture | n/a | 0.985 ± 0.015 | 0.985 | 1.000 | 0.969 |
| Recall 2AB Neck Garden I-II | n/a | 0.287 ± 0.084 | 0.500 | 0.250 | 0.375 |
| Recall 2CD Neck Garden III-IV | n/a | 0.985 ± 0.013 | 0.951 | 1.000 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.906 ± 0.044 | 0.882 | 0.882 | 0.971 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.833 ± 0.063 | 0.738 | 0.857 | 0.786 |
| Recall 4 Subtrochanteric | n/a | 0.533 ± 0.075 | 0.750 | 0.583 | 0.583 |
| Errors: detection | n/a | 4.6 ± 1.1 | 8 | 3 | 3 |
| Errors: location | n/a | 7.4 ± 0.9 | 6 | 6 | 6 |
| Errors: grade | n/a | 16.8 ± 2.8 | 15 | 18 | 18 |
| Errors: total | n/a | 28.8 ± 2.8 | 29 | 27 | 27 |

### A5_shared_grade vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.998 ± 0.003 | 0.981 | 1.000 | 1.000 |
| 6-class accuracy (%) | n/a | 85.5 ± 1.0 | 86.2 | 86.7 | 87.1 |
| Fracture sensitivity | n/a | 0.978 ± 0.009 | 0.952 | 0.966 | 0.993 |
| Fracture specificity | n/a | 0.994 ± 0.008 | 0.985 | 1.000 | 0.985 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.774 ± 0.015 | 0.814 | 0.798 | 0.792 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.743 ± 0.002 | 0.77 | 0.745 | 0.739 |
| Recall NF No fracture | n/a | 0.994 ± 0.008 | 0.985 | 1.000 | 0.985 |
| Recall 2AB Neck Garden I-II | n/a | 0.300 ± 0.052 | 0.500 | 0.312 | 0.312 |
| Recall 2CD Neck Garden III-IV | n/a | 0.985 ± 0.013 | 0.951 | 0.976 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.865 ± 0.057 | 0.882 | 0.882 | 0.941 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.800 ± 0.013 | 0.738 | 0.810 | 0.810 |
| Recall 4 Subtrochanteric | n/a | 0.567 ± 0.070 | 0.750 | 0.667 | 0.583 |
| Errors: detection | n/a | 3.6 ± 1.5 | 8 | 5 | 2 |
| Errors: location | n/a | 6.8 ± 0.8 | 6 | 6 | 6 |
| Errors: grade | n/a | 20.0 ± 2.1 | 15 | 17 | 19 |
| Errors: total | n/a | 30.4 ± 2.2 | 29 | 28 | 27 |

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

| Definition | S06_convnext_tiny | S10_m0_convnext_unet | A2_region | A3_geometry | A4_hierarchy | A5_shared_grade | A6_slgnet |
|---|---|---|---|---|---|---|---|
| Detected fractures only, top instance (our main definition) | n/a | 0.746 ± 0.002 | 0.737 ± 0.009 | 0.742 ± 0.007 | 0.745 ± 0.004 | 0.743 ± 0.002 | 0.745 ± 0.003 |
| Same at Ultralytics' default threshold 0.25 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Same at 0.25, union of all instances ≥ 0.25 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| All fractured femurs, missed = 0 | n/a | 0.734 ± 0.005 | 0.726 ± 0.007 | 0.729 ± 0.007 | 0.727 ± 0.005 | 0.727 ± 0.005 | 0.725 ± 0.007 |
| All 210 femurs, no-fracture femur = 1 if nothing called else 0 | n/a | 0.816 ± 0.004 | 0.807 ± 0.008 | 0.803 ± 0.006 | 0.807 ± 0.007 | 0.809 ± 0.004 | 0.807 ± 0.005 |
| Global (pooled pixels over fractured femurs) | n/a | 0.767 ± 0.004 | 0.761 ± 0.006 | 0.764 ± 0.003 | 0.765 ± 0.003 | 0.764 ± 0.004 | 0.762 ± 0.005 |
| Per-femur spread, detected only: mean ± 1.96 SD (compare the article's 0.56–0.98) | n/a | 0.48–1.00 (ensemble) | 0.46–1.00 (ensemble) | 0.43–1.00 (ensemble) | 0.41–1.00 (ensemble) | 0.42–1.00 (ensemble) | 0.49–1.00 (ensemble) |

## OOF versus test segmentation

| Run | Dice detected, OOF pooled | Dice detected, test models | Predicted / true area, OOF (median, IQR) | Predicted / true area, test models (median, IQR, pooled over models) |
|---|---|---|---|---|
| S06_convnext_tiny | n/a | n/a | n/a | n/a |
| S10_m0_convnext_unet | 0.799 | 0.746 ± 0.002 | 0.98 (0.79–1.18) | 1.27 (0.94–1.74) |
| A2_region | 0.790 | 0.737 ± 0.009 | 0.95 (0.76–1.14) | 1.22 (0.92–1.69) |
| A3_geometry | 0.792 | 0.742 ± 0.007 | 0.95 (0.76–1.14) | 1.21 (0.92–1.71) |
| A4_hierarchy | 0.786 | 0.745 ± 0.004 | 0.95 (0.76–1.15) | 1.25 (0.94–1.72) |
| A5_shared_grade | 0.783 | 0.743 ± 0.002 | 0.96 (0.78–1.15) | 1.23 (0.93–1.70) |
| A6_slgnet | 0.796 | 0.745 ± 0.003 | 0.96 (0.78–1.17) | 1.23 (0.93–1.69) |

Median mask area in pixels by class: true masks of the training femurs and of the test femurs (the article says its test labels were reviewed by a committee), and the masks the models predict for called fractures (OOF; test pooled over the 5 fold models).

| Class | True, train | True, test | Test / train | Predicted OOF, S06_convnext_tiny | Predicted test, S06_convnext_tiny | Predicted OOF, S10_m0_convnext_unet | Predicted test, S10_m0_convnext_unet | Predicted OOF, A2_region | Predicted test, A2_region | Predicted OOF, A3_geometry | Predicted test, A3_geometry | Predicted OOF, A4_hierarchy | Predicted test, A4_hierarchy | Predicted OOF, A5_shared_grade | Predicted test, A5_shared_grade | Predicted OOF, A6_slgnet | Predicted test, A6_slgnet |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 45,963 | 29,797 | 0.65 | n/a | n/a | 42,584 | 54,829 | 40,743 | 44,176 | 39,198 | 44,617 | 41,028 | 48,601 | 44,384 | 50,108 | 44,552 | 50,646 |
| 2CD Neck Garden III-IV | 75,612 | 45,844 | 0.61 | n/a | n/a | 74,660 | 77,694 | 74,142 | 78,244 | 72,739 | 76,267 | 73,803 | 78,717 | 74,287 | 79,594 | 74,452 | 78,553 |
| 3AB Pertroch Evans I-II | 85,540 | 88,794 | 1.04 | n/a | n/a | 91,848 | 89,882 | 88,552 | 86,988 | 87,960 | 88,621 | 89,758 | 89,907 | 88,230 | 87,746 | 90,712 | 88,074 |
| 3CDE Pertroch Evans III-V | 135,306 | 120,095 | 0.89 | n/a | n/a | 133,085 | 138,130 | 129,095 | 131,636 | 130,376 | 137,744 | 130,175 | 135,180 | 130,051 | 133,474 | 129,562 | 134,645 |
| 4 Subtrochanteric | 126,204 | 149,960 | 1.19 | n/a | n/a | 136,438 | 124,222 | 124,231 | 115,689 | 128,920 | 117,838 | 138,067 | 121,342 | 134,966 | 125,400 | 124,134 | 119,801 |
