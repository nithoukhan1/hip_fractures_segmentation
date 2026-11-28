# Breakdown: S10_m0_convnext_unet, R3_nearest_ref, R4_refrac, R7_partner

Made by `scripts/run_breakdown.py` from the filed run folders. Decisions use each run's OOF threshold. Mean ± SD is over the 5 held-out folds (OOF) or the 5 fold models (test); SD uses n − 1.

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

## R3_nearest_ref

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R3` at 768x512 px; fracture threshold 0.0008 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 88.5%) is fold 1.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.993 | 0.993 ± 0.005 | 0.996 ± 0.002 | 0.997 | 0.999 |
| Fracture sensitivity | 0.984 | 0.984 ± 0.011 | 0.988 ± 0.006 | 0.993 | 1.000 |
| Fracture specificity | 0.972 | 0.972 ± 0.014 | 0.972 ± 0.017 | 0.985 | 0.938 |
| 6-class accuracy (%) | 88.0 | 88.0 ± 0.6 | 85.1 ± 1.1 | 86.7 | 84.3 |
| Macro-F1 | 0.826 | 0.827 ± 0.014 | 0.773 ± 0.023 | 0.806 | 0.765 |
| Balanced accuracy | 0.818 | 0.819 ± 0.019 | 0.755 ± 0.019 | 0.782 | 0.751 |
| Cohen's kappa | 0.843 | 0.843 ± 0.007 | 0.810 ± 0.014 | 0.830 | 0.800 |
| Location accuracy (fractures called) | 0.943 | 0.943 ± 0.011 | 0.950 ± 0.006 | 0.951 | 0.952 |
| Grade accuracy, location right | 0.891 | 0.891 ± 0.020 | 0.843 ± 0.016 | 0.854 | 0.833 |
| Dice, fractured femurs (missed = 0) | 0.775 | 0.776 ± 0.014 | 0.731 ± 0.003 | 0.732 | 0.742 |
| Dice, detected fractures only | 0.788 | 0.788 ± 0.018 | 0.740 ± 0.003 | 0.737 | 0.742 |
| Predicted / true area, median | 0.96 | 0.96 ± 0.03 | 1.28 ± 0.02 | 1.24 | 1.26 |
| HD95 (px), median, detected | 61.3 | 60.6 ± 2.6 | 70.1 ± 2.1 | 69.8 | 66.0 |
| Errors: detection | 21.0 | 4.2 ± 1.3 | 3.6 ± 1.5 | 2.0 | 4.0 |
| Errors: location | 38.0 | 7.6 ± 1.5 | 7.2 ± 0.8 | 7.0 | 7.0 |
| Errors: grade | 66.0 | 13.2 ± 2.6 | 20.4 ± 2.1 | 19.0 | 22.0 |
| Errors: total | 125.0 | 25.0 ± 1.2 | 31.2 ± 2.4 | 28.0 | 33.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.972 | 0.972 ± 0.014 | 0.972 ± 0.017 | 0.985 | 0.938 |
| 2AB Neck Garden I-II | 57 / 16 | 0.684 | 0.682 ± 0.089 | 0.350 ± 0.056 | 0.438 | 0.375 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.902 | 0.902 ± 0.062 | 0.990 ± 0.013 | 1.000 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.821 | 0.821 ± 0.041 | 0.953 ± 0.026 | 0.971 | 1.000 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.809 | 0.810 ± 0.083 | 0.729 ± 0.046 | 0.714 | 0.690 |
| 4 Subtrochanteric | 36 / 12 | 0.722 | 0.725 ± 0.083 | 0.533 ± 0.046 | 0.583 | 0.500 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.621 | 0.621 ± 0.018 | 0.578 ± 0.029 | 0.597 | 0.965 / 0.912 ± 0.034 | 0.643 / 0.634 ± 0.021 | 0.87 / 1.90 ± 0.04 |
| 2CD Neck Garden III-IV | 0.765 | 0.765 ± 0.027 | 0.693 ± 0.009 | 0.700 | 0.988 / 0.995 ± 0.011 | 0.774 / 0.697 ± 0.005 | 0.94 / 1.74 ± 0.03 |
| 3AB Pertroch Evans I-II | 0.772 | 0.772 ± 0.036 | 0.746 ± 0.011 | 0.763 | 0.963 / 0.994 ± 0.013 | 0.802 / 0.750 ± 0.013 | 1.00 / 1.11 ± 0.04 |
| 3CDE Pertroch Evans III-V | 0.851 | 0.851 ± 0.014 | 0.834 ± 0.002 | 0.839 | 1.000 / 1.000 ± 0.000 | 0.851 / 0.834 ± 0.002 | 0.96 / 1.07 ± 0.02 |
| 4 Subtrochanteric | 0.697 | 0.697 ± 0.095 | 0.656 ± 0.043 | 0.673 | 0.972 / 1.000 ± 0.000 | 0.717 / 0.656 ± 0.043 | 0.98 / 0.94 ± 0.05 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 353 | 3 | 2 | 5 | 0 | 0 |
| 2AB | 2 | 39 | 11 | 5 | 0 | 0 |
| 2CD | 3 | 10 | 229 | 12 | 0 | 0 |
| 3AB | 5 | 1 | 1 | 110 | 16 | 1 |
| 3CDE | 0 | 1 | 5 | 29 | 161 | 3 |
| 4 | 1 | 0 | 0 | 2 | 7 | 26 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 61 | 3 | 1 | 0 | 0 | 0 |
| 2AB | 0 | 6 | 10 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 34 | 0 | 0 |
| 3CDE | 0 | 0 | 1 | 12 | 29 | 0 |
| 4 | 0 | 0 | 0 | 2 | 4 | 6 |

## R4_refrac

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R4` at 768x512 px; fracture threshold 0.0071 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 90.4%) is fold 0.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.995 ± 0.003 | 0.998 ± 0.002 | 0.996 | 1.000 |
| Fracture sensitivity | 0.975 | 0.975 ± 0.006 | 0.981 ± 0.006 | 0.972 | 0.986 |
| Fracture specificity | 0.981 | 0.981 ± 0.016 | 0.997 ± 0.007 | 1.000 | 0.985 |
| 6-class accuracy (%) | 88.2 | 88.2 ± 1.5 | 86.0 ± 0.9 | 85.7 | 85.7 |
| Macro-F1 | 0.826 | 0.823 ± 0.004 | 0.777 ± 0.021 | 0.757 | 0.779 |
| Balanced accuracy | 0.820 | 0.821 ± 0.020 | 0.756 ± 0.017 | 0.739 | 0.760 |
| Cohen's kappa | 0.845 | 0.845 ± 0.019 | 0.821 ± 0.011 | 0.817 | 0.818 |
| Location accuracy (fractures called) | 0.941 | 0.941 ± 0.016 | 0.949 ± 0.006 | 0.950 | 0.958 |
| Grade accuracy, location right | 0.900 | 0.900 ± 0.033 | 0.851 ± 0.010 | 0.853 | 0.838 |
| Dice, fractured femurs (missed = 0) | 0.777 | 0.777 ± 0.011 | 0.730 ± 0.004 | 0.723 | 0.739 |
| Dice, detected fractures only | 0.797 | 0.797 ± 0.009 | 0.744 ± 0.001 | 0.743 | 0.749 |
| Predicted / true area, median | 0.97 | 0.98 ± 0.01 | 1.27 ± 0.02 | 1.28 | 1.27 |
| HD95 (px), median, detected | 60.8 | 60.6 ± 2.0 | 69.4 ± 2.2 | 71.1 | 66.2 |
| Errors: detection | 24.0 | 4.8 ± 1.1 | 3.0 ± 0.7 | 4.0 | 3.0 |
| Errors: location | 39.0 | 7.8 ± 2.2 | 7.2 ± 0.8 | 7.0 | 6.0 |
| Errors: grade | 60.0 | 12.0 ± 4.0 | 19.2 ± 1.1 | 19.0 | 21.0 |
| Errors: total | 123.0 | 24.6 ± 3.2 | 29.4 ± 1.8 | 30.0 | 30.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.981 | 0.981 ± 0.016 | 0.997 ± 0.007 | 1.000 | 0.985 |
| 2AB Neck Garden I-II | 57 / 16 | 0.702 | 0.703 ± 0.072 | 0.325 ± 0.052 | 0.312 | 0.312 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.898 | 0.898 ± 0.062 | 0.985 ± 0.013 | 0.976 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.806 | 0.805 ± 0.087 | 0.935 ± 0.044 | 0.941 | 0.941 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.814 | 0.815 ± 0.073 | 0.762 ± 0.024 | 0.786 | 0.738 |
| 4 Subtrochanteric | 36 / 12 | 0.722 | 0.721 ± 0.226 | 0.533 ± 0.075 | 0.417 | 0.583 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.635 | 0.635 ± 0.027 | 0.572 ± 0.027 | 0.599 | 0.947 / 0.863 ± 0.052 | 0.670 / 0.663 ± 0.021 | 0.88 / 1.93 ± 0.05 |
| 2CD Neck Garden III-IV | 0.761 | 0.761 ± 0.021 | 0.682 ± 0.006 | 0.682 | 0.972 / 0.990 ± 0.013 | 0.783 / 0.689 ± 0.006 | 0.94 / 1.71 ± 0.05 |
| 3AB Pertroch Evans I-II | 0.783 | 0.783 ± 0.036 | 0.756 ± 0.008 | 0.769 | 0.970 / 1.000 ± 0.000 | 0.808 / 0.756 ± 0.008 | 1.03 / 1.12 ± 0.04 |
| 3CDE Pertroch Evans III-V | 0.850 | 0.850 ± 0.011 | 0.839 ± 0.001 | 0.844 | 1.000 / 1.000 ± 0.000 | 0.850 / 0.839 ± 0.001 | 0.97 / 1.05 ± 0.02 |
| 4 Subtrochanteric | 0.682 | 0.682 ± 0.092 | 0.648 ± 0.031 | 0.667 | 0.917 / 0.983 ± 0.037 | 0.744 / 0.659 ± 0.008 | 1.05 / 0.95 ± 0.07 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 356 | 3 | 4 | 0 | 0 | 0 |
| 2AB | 3 | 40 | 9 | 5 | 0 | 0 |
| 2CD | 7 | 10 | 228 | 9 | 0 | 0 |
| 3AB | 4 | 4 | 2 | 108 | 15 | 1 |
| 3CDE | 0 | 1 | 6 | 26 | 162 | 4 |
| 4 | 3 | 1 | 0 | 1 | 5 | 26 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 64 | 0 | 1 | 0 | 0 | 0 |
| 2AB | 2 | 5 | 9 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 32 | 2 | 0 |
| 3CDE | 0 | 0 | 1 | 10 | 31 | 0 |
| 4 | 0 | 1 | 0 | 1 | 3 | 7 |

## R7_partner

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R7` at 768x512 px; fracture threshold 0.0601 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 88.9%) is fold 4.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.995 | 0.995 ± 0.003 | 0.998 ± 0.001 | 0.999 | 0.999 |
| Fracture sensitivity | 0.975 | 0.975 ± 0.004 | 0.978 ± 0.006 | 0.979 | 0.986 |
| Fracture specificity | 0.992 | 0.992 ± 0.012 | 0.994 ± 0.008 | 1.000 | 0.985 |
| 6-class accuracy (%) | 87.8 | 87.8 ± 0.7 | 85.5 ± 1.6 | 82.9 | 86.7 |
| Macro-F1 | 0.819 | 0.817 ± 0.015 | 0.772 ± 0.020 | 0.748 | 0.796 |
| Balanced accuracy | 0.811 | 0.811 ± 0.035 | 0.755 ± 0.025 | 0.727 | 0.776 |
| Cohen's kappa | 0.840 | 0.840 ± 0.009 | 0.815 ± 0.021 | 0.781 | 0.830 |
| Location accuracy (fractures called) | 0.943 | 0.943 ± 0.011 | 0.939 ± 0.008 | 0.930 | 0.958 |
| Grade accuracy, location right | 0.885 | 0.885 ± 0.006 | 0.857 ± 0.022 | 0.817 | 0.854 |
| Dice, fractured femurs (missed = 0) | 0.780 | 0.780 ± 0.012 | 0.728 ± 0.003 | 0.729 | 0.741 |
| Dice, detected fractures only | 0.800 | 0.800 ± 0.012 | 0.745 ± 0.003 | 0.744 | 0.752 |
| Predicted / true area, median | 0.97 | 0.97 ± 0.02 | 1.26 ± 0.01 | 1.25 | 1.27 |
| HD95 (px), median, detected | 60.1 | 60.6 ± 2.3 | 68.7 ± 3.1 | 66.3 | 66.0 |
| Errors: detection | 20.0 | 4.0 ± 1.2 | 3.6 ± 1.1 | 3.0 | 3.0 |
| Errors: location | 38.0 | 7.6 ± 1.5 | 8.6 ± 1.1 | 10.0 | 6.0 |
| Errors: grade | 69.0 | 13.8 ± 0.8 | 18.2 ± 2.8 | 23.0 | 19.0 |
| Errors: total | 127.0 | 25.4 ± 1.5 | 30.4 ± 3.4 | 36.0 | 28.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.992 | 0.992 ± 0.012 | 0.994 ± 0.008 | 1.000 | 0.985 |
| 2AB Neck Garden I-II | 57 / 16 | 0.632 | 0.630 ± 0.104 | 0.375 ± 0.088 | 0.375 | 0.375 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.882 | 0.882 ± 0.076 | 0.985 ± 0.022 | 1.000 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.799 | 0.798 ± 0.079 | 0.924 ± 0.074 | 0.794 | 1.000 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.814 | 0.815 ± 0.071 | 0.738 ± 0.038 | 0.690 | 0.714 |
| 4 Subtrochanteric | 36 / 12 | 0.750 | 0.750 ± 0.186 | 0.517 ± 0.149 | 0.500 | 0.583 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.651 | 0.651 ± 0.050 | 0.562 ± 0.025 | 0.596 | 0.930 / 0.838 ± 0.034 | 0.700 / 0.671 ± 0.012 | 0.96 / 1.90 ± 0.04 |
| 2CD Neck Garden III-IV | 0.770 | 0.770 ± 0.018 | 0.690 ± 0.012 | 0.696 | 0.972 / 0.995 ± 0.011 | 0.792 / 0.694 ± 0.009 | 0.94 / 1.69 ± 0.07 |
| 3AB Pertroch Evans I-II | 0.778 | 0.777 ± 0.042 | 0.746 ± 0.015 | 0.765 | 0.970 / 0.988 ± 0.016 | 0.801 / 0.755 ± 0.011 | 1.00 / 1.11 ± 0.04 |
| 3CDE Pertroch Evans III-V | 0.849 | 0.849 ± 0.018 | 0.837 ± 0.004 | 0.844 | 1.000 / 1.000 ± 0.000 | 0.849 / 0.837 ± 0.004 | 0.96 / 1.04 ± 0.03 |
| 4 Subtrochanteric | 0.681 | 0.681 ± 0.097 | 0.649 ± 0.026 | 0.664 | 0.944 / 1.000 ± 0.000 | 0.721 / 0.649 ± 0.026 | 1.05 / 0.97 ± 0.07 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 360 | 1 | 1 | 1 | 0 | 0 |
| 2AB | 4 | 36 | 12 | 5 | 0 | 0 |
| 2CD | 7 | 14 | 224 | 9 | 0 | 0 |
| 3AB | 4 | 2 | 2 | 107 | 18 | 1 |
| 3CDE | 0 | 0 | 8 | 25 | 162 | 4 |
| 4 | 2 | 0 | 0 | 2 | 5 | 27 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 64 | 1 | 0 | 0 | 0 | 0 |
| 2AB | 2 | 6 | 8 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 34 | 0 | 0 |
| 3CDE | 0 | 0 | 1 | 11 | 30 | 0 |
| 4 | 0 | 0 | 0 | 2 | 3 | 7 |

## Runs side by side (test, 5 fold models mean ± SD; OOF pooled in brackets)

| Metric | S10_m0_convnext_unet | R3_nearest_ref | R4_refrac | R7_partner |
|---|---|---|---|---|
| 6-class accuracy (%) | 86.0 ± 0.5 (88.4) | 85.1 ± 1.1 (88.0) | 86.0 ± 0.9 (88.2) | 85.5 ± 1.6 (87.8) |
| Macro-F1 | 0.776 ± 0.017 (0.825) | 0.773 ± 0.023 (0.826) | 0.777 ± 0.021 (0.826) | 0.772 ± 0.020 (0.819) |
| Dice, fractured (missed = 0) | 0.734 ± 0.005 (0.777) | 0.731 ± 0.003 (0.775) | 0.730 ± 0.004 (0.777) | 0.728 ± 0.003 (0.780) |
| Recall NF | 1.000 ± 0.000 (0.992) | 0.972 ± 0.017 (0.972) | 0.997 ± 0.007 (0.981) | 0.994 ± 0.008 (0.992) |
| Recall 2AB | 0.312 ± 0.077 (0.544) | 0.350 ± 0.056 (0.684) | 0.325 ± 0.052 (0.702) | 0.375 ± 0.088 (0.632) |
| Recall 2CD | 0.990 ± 0.013 (0.909) | 0.990 ± 0.013 (0.902) | 0.985 ± 0.013 (0.898) | 0.985 ± 0.022 (0.882) |
| Recall 3AB | 0.953 ± 0.016 (0.791) | 0.953 ± 0.026 (0.821) | 0.935 ± 0.044 (0.806) | 0.924 ± 0.074 (0.799) |
| Recall 3CDE | 0.738 ± 0.034 (0.839) | 0.729 ± 0.046 (0.809) | 0.762 ± 0.024 (0.814) | 0.738 ± 0.038 (0.814) |
| Recall 4 | 0.550 ± 0.046 (0.750) | 0.533 ± 0.046 (0.722) | 0.533 ± 0.075 (0.722) | 0.517 ± 0.149 (0.750) |
| Dice 2AB | 0.578 ± 0.032 (0.624) | 0.578 ± 0.029 (0.621) | 0.572 ± 0.027 (0.635) | 0.562 ± 0.025 (0.651) |
| Dice 2CD | 0.699 ± 0.009 (0.773) | 0.693 ± 0.009 (0.765) | 0.682 ± 0.006 (0.761) | 0.690 ± 0.012 (0.770) |
| Dice 3AB | 0.750 ± 0.008 (0.763) | 0.746 ± 0.011 (0.772) | 0.756 ± 0.008 (0.783) | 0.746 ± 0.015 (0.778) |
| Dice 3CDE | 0.836 ± 0.004 (0.852) | 0.834 ± 0.002 (0.851) | 0.839 ± 0.001 (0.850) | 0.837 ± 0.004 (0.849) |
| Dice 4 | 0.655 ± 0.039 (0.682) | 0.656 ± 0.043 (0.697) | 0.648 ± 0.031 (0.682) | 0.649 ± 0.026 (0.681) |

Paired tests between the runs (same femurs) are in `stats.md` of the same folder, if it was made with `scripts/run_stats.py ... --compare`.

## Comparison with the base article

Source: González et al., Insights Imaging 2025;16:86 (doi 10.1186/s13244-025-01958-y). Table 2 gives mean ± SD of six training runs per setting on the same 210 test femurs; the final model is the run with the best validation accuracy (Table 3, confusion matrix in Fig. 3). Our fold models are trained on 4/5 of the 1,043 training femurs (about 834), the article's on all 1,043 plus a 201-femur validation set; our counterpart of its selected model is the fold model with the best accuracy on its own held-out fold.

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

### R3_nearest_ref vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.996 ± 0.002 | 0.981 | 0.997 | 0.999 |
| 6-class accuracy (%) | n/a | 85.1 ± 1.1 | 86.2 | 86.7 | 84.3 |
| Fracture sensitivity | n/a | 0.988 ± 0.006 | 0.952 | 0.993 | 1.000 |
| Fracture specificity | n/a | 0.972 ± 0.017 | 0.985 | 0.985 | 0.938 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.773 ± 0.023 | 0.814 | 0.806 | 0.765 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.740 ± 0.003 | 0.77 | 0.737 | 0.742 |
| Recall NF No fracture | n/a | 0.972 ± 0.017 | 0.985 | 0.985 | 0.938 |
| Recall 2AB Neck Garden I-II | n/a | 0.350 ± 0.056 | 0.500 | 0.438 | 0.375 |
| Recall 2CD Neck Garden III-IV | n/a | 0.990 ± 0.013 | 0.951 | 1.000 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.953 ± 0.026 | 0.882 | 0.971 | 1.000 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.729 ± 0.046 | 0.738 | 0.714 | 0.690 |
| Recall 4 Subtrochanteric | n/a | 0.533 ± 0.046 | 0.750 | 0.583 | 0.500 |
| Errors: detection | n/a | 3.6 ± 1.5 | 8 | 2 | 4 |
| Errors: location | n/a | 7.2 ± 0.8 | 6 | 7 | 7 |
| Errors: grade | n/a | 20.4 ± 2.1 | 15 | 19 | 22 |
| Errors: total | n/a | 31.2 ± 2.4 | 29 | 28 | 33 |

### R4_refrac vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.998 ± 0.002 | 0.981 | 0.996 | 1.000 |
| 6-class accuracy (%) | n/a | 86.0 ± 0.9 | 86.2 | 85.7 | 85.7 |
| Fracture sensitivity | n/a | 0.981 ± 0.006 | 0.952 | 0.972 | 0.986 |
| Fracture specificity | n/a | 0.997 ± 0.007 | 0.985 | 1.000 | 0.985 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.777 ± 0.021 | 0.814 | 0.757 | 0.779 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.744 ± 0.001 | 0.77 | 0.743 | 0.749 |
| Recall NF No fracture | n/a | 0.997 ± 0.007 | 0.985 | 1.000 | 0.985 |
| Recall 2AB Neck Garden I-II | n/a | 0.325 ± 0.052 | 0.500 | 0.312 | 0.312 |
| Recall 2CD Neck Garden III-IV | n/a | 0.985 ± 0.013 | 0.951 | 0.976 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.935 ± 0.044 | 0.882 | 0.941 | 0.941 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.762 ± 0.024 | 0.738 | 0.786 | 0.738 |
| Recall 4 Subtrochanteric | n/a | 0.533 ± 0.075 | 0.750 | 0.417 | 0.583 |
| Errors: detection | n/a | 3.0 ± 0.7 | 8 | 4 | 3 |
| Errors: location | n/a | 7.2 ± 0.8 | 6 | 7 | 6 |
| Errors: grade | n/a | 19.2 ± 1.1 | 15 | 19 | 21 |
| Errors: total | n/a | 29.4 ± 1.8 | 29 | 30 | 30 |

### R7_partner vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.998 ± 0.001 | 0.981 | 0.999 | 0.999 |
| 6-class accuracy (%) | n/a | 85.5 ± 1.6 | 86.2 | 82.9 | 86.7 |
| Fracture sensitivity | n/a | 0.978 ± 0.006 | 0.952 | 0.979 | 0.986 |
| Fracture specificity | n/a | 0.994 ± 0.008 | 0.985 | 1.000 | 0.985 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.772 ± 0.020 | 0.814 | 0.748 | 0.796 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.745 ± 0.003 | 0.77 | 0.744 | 0.752 |
| Recall NF No fracture | n/a | 0.994 ± 0.008 | 0.985 | 1.000 | 0.985 |
| Recall 2AB Neck Garden I-II | n/a | 0.375 ± 0.088 | 0.500 | 0.375 | 0.375 |
| Recall 2CD Neck Garden III-IV | n/a | 0.985 ± 0.022 | 0.951 | 1.000 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.924 ± 0.074 | 0.882 | 0.794 | 1.000 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.738 ± 0.038 | 0.738 | 0.690 | 0.714 |
| Recall 4 Subtrochanteric | n/a | 0.517 ± 0.149 | 0.750 | 0.500 | 0.583 |
| Errors: detection | n/a | 3.6 ± 1.1 | 8 | 3 | 3 |
| Errors: location | n/a | 8.6 ± 1.1 | 6 | 10 | 6 |
| Errors: grade | n/a | 18.2 ± 2.8 | 15 | 23 | 19 |
| Errors: total | n/a | 30.4 ± 3.4 | 29 | 36 | 28 |

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

| Definition | S10_m0_convnext_unet | R3_nearest_ref | R4_refrac | R7_partner |
|---|---|---|---|---|
| Detected fractures only, top instance (our main definition) | 0.746 ± 0.002 | 0.740 ± 0.003 | 0.744 ± 0.001 | 0.745 ± 0.003 |
| Same at Ultralytics' default threshold 0.25 | n/a | n/a | n/a | n/a |
| Same at 0.25, union of all instances ≥ 0.25 | n/a | n/a | n/a | n/a |
| All fractured femurs, missed = 0 | 0.734 ± 0.005 | 0.731 ± 0.003 | 0.730 ± 0.004 | 0.728 ± 0.003 |
| All 210 femurs, no-fracture femur = 1 if nothing called else 0 | 0.816 ± 0.004 | 0.806 ± 0.005 | 0.813 ± 0.003 | 0.810 ± 0.004 |
| Global (pooled pixels over fractured femurs) | 0.767 ± 0.004 | 0.766 ± 0.003 | 0.766 ± 0.004 | 0.765 ± 0.002 |
| Per-femur spread, detected only: mean ± 1.96 SD (compare the article's 0.56–0.98) | 0.48–1.00 (ensemble) | 0.44–1.00 (ensemble) | 0.49–1.00 (ensemble) | 0.51–0.99 (ensemble) |

## OOF versus test segmentation

| Run | Dice detected, OOF pooled | Dice detected, test models | Predicted / true area, OOF (median, IQR) | Predicted / true area, test models (median, IQR, pooled over models) |
|---|---|---|---|---|
| S10_m0_convnext_unet | 0.799 | 0.746 ± 0.002 | 0.98 (0.79–1.18) | 1.27 (0.94–1.74) |
| R3_nearest_ref | 0.788 | 0.740 ± 0.003 | 0.96 (0.78–1.17) | 1.28 (0.93–1.78) |
| R4_refrac | 0.797 | 0.744 ± 0.001 | 0.97 (0.80–1.18) | 1.28 (0.94–1.76) |
| R7_partner | 0.800 | 0.745 ± 0.003 | 0.97 (0.80–1.17) | 1.26 (0.95–1.75) |

Median mask area in pixels by class: true masks of the training femurs and of the test femurs (the article says its test labels were reviewed by a committee), and the masks the models predict for called fractures (OOF; test pooled over the 5 fold models).

| Class | True, train | True, test | Test / train | Predicted OOF, S10_m0_convnext_unet | Predicted test, S10_m0_convnext_unet | Predicted OOF, R3_nearest_ref | Predicted test, R3_nearest_ref | Predicted OOF, R4_refrac | Predicted test, R4_refrac | Predicted OOF, R7_partner | Predicted test, R7_partner |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 45,963 | 29,797 | 0.65 | 42,584 | 54,829 | 41,673 | 51,720 | 43,444 | 52,675 | 46,792 | 51,612 |
| 2CD Neck Garden III-IV | 75,612 | 45,844 | 0.61 | 74,660 | 77,694 | 73,989 | 77,902 | 75,635 | 79,142 | 74,327 | 78,202 |
| 3AB Pertroch Evans I-II | 85,540 | 88,794 | 1.04 | 91,848 | 89,882 | 94,169 | 92,610 | 96,462 | 93,149 | 92,142 | 92,594 |
| 3CDE Pertroch Evans III-V | 135,306 | 120,095 | 0.89 | 133,085 | 138,130 | 128,926 | 138,587 | 132,074 | 137,379 | 129,294 | 134,986 |
| 4 Subtrochanteric | 126,204 | 149,960 | 1.19 | 136,438 | 124,222 | 136,596 | 124,055 | 134,393 | 121,561 | 128,038 | 122,670 |
