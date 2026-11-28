# Breakdown: R4_refrac, R1_aasn, R5_concat, R6a_no_align, R6b_affine

Made by `scripts/run_breakdown.py` from the filed run folders. Decisions use each run's OOF threshold. Mean ± SD is over the 5 held-out folds (OOF) or the 5 fold models (test); SD uses n − 1.

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

## R1_aasn

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R1` at 768x512 px; fracture threshold 0.0011 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 89.5%) is fold 1.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.994 | 0.995 ± 0.004 | 0.999 ± 0.000 | 0.999 | 0.999 |
| Fracture sensitivity | 0.982 | 0.982 ± 0.008 | 0.986 ± 0.005 | 0.986 | 0.986 |
| Fracture specificity | 0.981 | 0.981 ± 0.012 | 0.982 ± 0.017 | 0.985 | 0.954 |
| 6-class accuracy (%) | 87.7 | 87.7 ± 1.8 | 85.4 ± 1.1 | 84.3 | 86.2 |
| Macro-F1 | 0.820 | 0.819 ± 0.027 | 0.777 ± 0.016 | 0.763 | 0.798 |
| Balanced accuracy | 0.808 | 0.809 ± 0.033 | 0.758 ± 0.016 | 0.746 | 0.781 |
| Cohen's kappa | 0.839 | 0.839 ± 0.024 | 0.814 ± 0.014 | 0.800 | 0.824 |
| Location accuracy (fractures called) | 0.943 | 0.943 ± 0.019 | 0.947 ± 0.008 | 0.944 | 0.958 |
| Grade accuracy, location right | 0.882 | 0.882 ± 0.035 | 0.847 ± 0.026 | 0.828 | 0.862 |
| Dice, fractured femurs (missed = 0) | 0.781 | 0.781 ± 0.009 | 0.728 ± 0.003 | 0.731 | 0.738 |
| Dice, detected fractures only | 0.795 | 0.795 ± 0.009 | 0.738 ± 0.003 | 0.741 | 0.748 |
| Predicted / true area, median | 0.97 | 0.97 ± 0.01 | 1.26 ± 0.02 | 1.26 | 1.26 |
| HD95 (px), median, detected | 60.1 | 60.2 ± 3.0 | 70.8 ± 2.2 | 68.4 | 68.3 |
| Errors: detection | 19.0 | 3.8 ± 1.3 | 3.2 ± 1.5 | 3.0 | 5.0 |
| Errors: location | 38.0 | 7.6 ± 2.5 | 7.6 ± 1.1 | 8.0 | 6.0 |
| Errors: grade | 71.0 | 14.2 ± 4.3 | 19.8 ± 3.5 | 22.0 | 18.0 |
| Errors: total | 128.0 | 25.6 ± 3.8 | 30.6 ± 2.3 | 33.0 | 29.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.981 | 0.981 ± 0.012 | 0.982 ± 0.017 | 0.985 | 0.954 |
| 2AB Neck Garden I-II | 57 / 16 | 0.579 | 0.582 ± 0.154 | 0.375 ± 0.062 | 0.312 | 0.438 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.917 | 0.918 ± 0.061 | 0.995 ± 0.011 | 0.976 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.806 | 0.805 ± 0.045 | 0.941 ± 0.047 | 0.882 | 0.971 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.789 | 0.790 ± 0.081 | 0.724 ± 0.043 | 0.738 | 0.738 |
| 4 Subtrochanteric | 36 / 12 | 0.778 | 0.779 ± 0.161 | 0.533 ± 0.046 | 0.583 | 0.583 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.627 | 0.628 ± 0.059 | 0.555 ± 0.014 | 0.585 | 0.930 / 0.875 ± 0.044 | 0.674 / 0.635 ± 0.023 | 0.95 / 1.89 ± 0.11 |
| 2CD Neck Garden III-IV | 0.771 | 0.771 ± 0.022 | 0.691 ± 0.005 | 0.695 | 0.984 / 1.000 ± 0.000 | 0.783 / 0.691 ± 0.005 | 0.96 / 1.70 ± 0.05 |
| 3AB Pertroch Evans I-II | 0.786 | 0.785 ± 0.037 | 0.753 ± 0.008 | 0.763 | 0.978 / 1.000 ± 0.000 | 0.804 / 0.753 ± 0.008 | 1.03 / 1.14 ± 0.03 |
| 3CDE Pertroch Evans III-V | 0.847 | 0.847 ± 0.012 | 0.833 ± 0.006 | 0.839 | 1.000 / 1.000 ± 0.000 | 0.847 / 0.833 ± 0.006 | 0.95 / 1.04 ± 0.02 |
| 4 Subtrochanteric | 0.719 | 0.720 ± 0.057 | 0.643 ± 0.017 | 0.658 | 0.972 / 1.000 ± 0.000 | 0.740 / 0.643 ± 0.017 | 1.04 / 0.94 ± 0.05 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 356 | 4 | 2 | 1 | 0 | 0 |
| 2AB | 4 | 33 | 15 | 5 | 0 | 0 |
| 2CD | 4 | 8 | 233 | 8 | 1 | 0 |
| 3AB | 3 | 0 | 3 | 108 | 20 | 0 |
| 3CDE | 0 | 0 | 10 | 28 | 157 | 4 |
| 4 | 1 | 1 | 0 | 2 | 4 | 28 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 62 | 2 | 0 | 1 | 0 | 0 |
| 2AB | 2 | 7 | 7 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 33 | 1 | 0 |
| 3CDE | 0 | 0 | 1 | 10 | 31 | 0 |
| 4 | 0 | 1 | 0 | 1 | 3 | 7 |

## R5_concat

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R5` at 768x512 px; fracture threshold 0.0608 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 89.9%) is fold 0.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.996 | 0.997 ± 0.002 | 0.998 ± 0.002 | 0.997 | 0.999 |
| Fracture sensitivity | 0.974 | 0.974 ± 0.008 | 0.975 ± 0.009 | 0.966 | 0.986 |
| Fracture specificity | 0.992 | 0.992 ± 0.012 | 0.994 ± 0.008 | 0.985 | 0.985 |
| 6-class accuracy (%) | 88.7 | 88.7 ± 1.2 | 85.8 ± 0.9 | 85.2 | 84.8 |
| Macro-F1 | 0.827 | 0.825 ± 0.024 | 0.765 ± 0.021 | 0.756 | 0.741 |
| Balanced accuracy | 0.808 | 0.808 ± 0.028 | 0.748 ± 0.018 | 0.736 | 0.729 |
| Cohen's kappa | 0.851 | 0.851 ± 0.016 | 0.818 ± 0.011 | 0.811 | 0.805 |
| Location accuracy (fractures called) | 0.944 | 0.944 ± 0.012 | 0.948 ± 0.008 | 0.943 | 0.951 |
| Grade accuracy, location right | 0.900 | 0.900 ± 0.025 | 0.856 ± 0.017 | 0.866 | 0.831 |
| Dice, fractured femurs (missed = 0) | 0.780 | 0.780 ± 0.011 | 0.729 ± 0.006 | 0.720 | 0.740 |
| Dice, detected fractures only | 0.802 | 0.802 ± 0.009 | 0.747 ± 0.003 | 0.746 | 0.751 |
| Predicted / true area, median | 0.97 | 0.97 ± 0.02 | 1.27 ± 0.02 | 1.30 | 1.26 |
| HD95 (px), median, detected | 60.9 | 60.8 ± 3.5 | 68.1 ± 1.8 | 70.5 | 66.3 |
| Errors: detection | 21.0 | 4.2 ± 1.9 | 4.0 ± 1.6 | 6.0 | 3.0 |
| Errors: location | 37.0 | 7.4 ± 1.5 | 7.4 ± 1.1 | 8.0 | 7.0 |
| Errors: grade | 60.0 | 12.0 ± 3.1 | 18.4 ± 2.2 | 17.0 | 22.0 |
| Errors: total | 118.0 | 23.6 ± 2.6 | 29.8 ± 1.8 | 31.0 | 32.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.992 | 0.992 ± 0.012 | 0.994 ± 0.008 | 0.985 | 0.985 |
| 2AB Neck Garden I-II | 57 / 16 | 0.561 | 0.564 ± 0.091 | 0.275 ± 0.084 | 0.312 | 0.188 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.898 | 0.898 ± 0.080 | 0.995 ± 0.011 | 0.976 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.828 | 0.828 ± 0.069 | 0.929 ± 0.061 | 0.941 | 0.941 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.844 | 0.845 ± 0.064 | 0.776 ± 0.043 | 0.786 | 0.762 |
| 4 Subtrochanteric | 36 / 12 | 0.722 | 0.721 ± 0.226 | 0.517 ± 0.070 | 0.417 | 0.500 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.651 | 0.652 ± 0.043 | 0.546 ± 0.031 | 0.587 | 0.930 / 0.812 ± 0.044 | 0.700 / 0.672 ± 0.012 | 0.94 / 1.91 ± 0.09 |
| 2CD Neck Garden III-IV | 0.769 | 0.769 ± 0.022 | 0.692 ± 0.005 | 0.698 | 0.976 / 0.995 ± 0.011 | 0.788 / 0.696 ± 0.005 | 0.95 / 1.67 ± 0.03 |
| 3AB Pertroch Evans I-II | 0.772 | 0.772 ± 0.051 | 0.752 ± 0.008 | 0.764 | 0.955 / 0.994 ± 0.013 | 0.808 / 0.757 ± 0.009 | 1.02 / 1.13 ± 0.02 |
| 3CDE Pertroch Evans III-V | 0.852 | 0.852 ± 0.011 | 0.836 ± 0.001 | 0.841 | 1.000 / 1.000 ± 0.000 | 0.852 / 0.836 ± 0.001 | 0.97 / 1.04 ± 0.03 |
| 4 Subtrochanteric | 0.698 | 0.698 ± 0.084 | 0.653 ± 0.019 | 0.669 | 0.944 / 0.983 ± 0.037 | 0.739 / 0.665 ± 0.014 | 1.04 / 0.94 ± 0.08 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 360 | 1 | 1 | 1 | 0 | 0 |
| 2AB | 4 | 32 | 15 | 6 | 0 | 0 |
| 2CD | 6 | 8 | 228 | 12 | 0 | 0 |
| 3AB | 6 | 1 | 2 | 111 | 13 | 1 |
| 3CDE | 0 | 1 | 5 | 24 | 168 | 1 |
| 4 | 2 | 0 | 0 | 2 | 6 | 26 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 64 | 1 | 0 | 0 | 0 | 0 |
| 2AB | 2 | 3 | 11 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 32 | 2 | 0 |
| 3CDE | 0 | 0 | 1 | 9 | 32 | 0 |
| 4 | 0 | 0 | 0 | 1 | 5 | 6 |

## R6a_no_align

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R6a` at 768x512 px; fracture threshold 0.0013 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 89.4%) is fold 0.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.995 | 0.995 ± 0.002 | 0.997 ± 0.002 | 0.996 | 0.999 |
| Fracture sensitivity | 0.974 | 0.974 ± 0.008 | 0.975 ± 0.006 | 0.972 | 0.986 |
| Fracture specificity | 0.983 | 0.983 ± 0.006 | 0.985 ± 0.015 | 0.985 | 0.969 |
| 6-class accuracy (%) | 87.1 | 87.1 ± 1.6 | 85.9 ± 1.5 | 85.2 | 85.7 |
| Macro-F1 | 0.807 | 0.805 ± 0.018 | 0.786 ± 0.039 | 0.735 | 0.788 |
| Balanced accuracy | 0.797 | 0.796 ± 0.022 | 0.771 ± 0.036 | 0.722 | 0.771 |
| Cohen's kappa | 0.830 | 0.830 ± 0.021 | 0.820 ± 0.019 | 0.811 | 0.818 |
| Location accuracy (fractures called) | 0.935 | 0.935 ± 0.011 | 0.953 ± 0.016 | 0.929 | 0.965 |
| Grade accuracy, location right | 0.885 | 0.885 ± 0.033 | 0.855 ± 0.018 | 0.873 | 0.838 |
| Dice, fractured femurs (missed = 0) | 0.777 | 0.777 ± 0.012 | 0.727 ± 0.004 | 0.724 | 0.741 |
| Dice, detected fractures only | 0.798 | 0.798 ± 0.009 | 0.746 ± 0.004 | 0.744 | 0.751 |
| Predicted / true area, median | 0.98 | 0.98 ± 0.01 | 1.27 ± 0.03 | 1.32 | 1.28 |
| HD95 (px), median, detected | 59.9 | 60.2 ± 3.2 | 67.5 ± 1.4 | 66.7 | 63.8 |
| Errors: detection | 24.0 | 4.8 ± 1.5 | 4.6 ± 0.9 | 5.0 | 4.0 |
| Errors: location | 43.0 | 8.6 ± 1.5 | 6.6 ± 2.2 | 10.0 | 5.0 |
| Errors: grade | 68.0 | 13.6 ± 3.9 | 18.4 ± 2.3 | 16.0 | 21.0 |
| Errors: total | 135.0 | 27.0 ± 3.4 | 29.6 ± 3.0 | 31.0 | 30.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.983 | 0.983 ± 0.006 | 0.985 ± 0.015 | 0.985 | 0.969 |
| 2AB Neck Garden I-II | 57 / 16 | 0.561 | 0.562 ± 0.098 | 0.338 ± 0.114 | 0.188 | 0.312 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.874 | 0.874 ± 0.066 | 0.980 ± 0.020 | 0.976 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.784 | 0.783 ± 0.063 | 0.900 ± 0.026 | 0.912 | 0.941 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.829 | 0.830 ± 0.064 | 0.771 ± 0.062 | 0.857 | 0.738 |
| 4 Subtrochanteric | 36 / 12 | 0.750 | 0.746 ± 0.189 | 0.650 ± 0.149 | 0.417 | 0.667 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.635 | 0.636 ± 0.044 | 0.571 ± 0.020 | 0.594 | 0.912 / 0.863 ± 0.028 | 0.696 / 0.662 ± 0.014 | 0.96 / 1.93 ± 0.05 |
| 2CD Neck Garden III-IV | 0.761 | 0.761 ± 0.017 | 0.691 ± 0.010 | 0.700 | 0.965 / 0.990 ± 0.013 | 0.789 / 0.698 ± 0.003 | 0.96 / 1.68 ± 0.07 |
| 3AB Pertroch Evans I-II | 0.781 | 0.780 ± 0.042 | 0.737 ± 0.016 | 0.763 | 0.978 / 0.971 ± 0.029 | 0.798 / 0.760 ± 0.012 | 1.03 / 1.13 ± 0.03 |
| 3CDE Pertroch Evans III-V | 0.846 | 0.846 ± 0.018 | 0.837 ± 0.004 | 0.842 | 1.000 / 1.000 ± 0.000 | 0.846 / 0.837 ± 0.004 | 0.97 / 1.05 ± 0.03 |
| 4 Subtrochanteric | 0.715 | 0.716 ± 0.051 | 0.647 ± 0.038 | 0.660 | 0.972 / 1.000 ± 0.000 | 0.735 / 0.647 ± 0.038 | 1.03 / 0.95 ± 0.02 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 357 | 2 | 2 | 2 | 0 | 0 |
| 2AB | 5 | 32 | 15 | 5 | 0 | 0 |
| 2CD | 9 | 13 | 222 | 10 | 0 | 0 |
| 3AB | 3 | 3 | 5 | 105 | 18 | 0 |
| 3CDE | 0 | 2 | 6 | 22 | 165 | 4 |
| 4 | 1 | 0 | 0 | 2 | 6 | 27 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 63 | 1 | 0 | 1 | 0 | 0 |
| 2AB | 2 | 5 | 9 | 0 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 32 | 2 | 0 |
| 3CDE | 0 | 0 | 1 | 10 | 31 | 0 |
| 4 | 0 | 0 | 0 | 1 | 3 | 8 |

## R6b_affine

Network `ref:convnext_tiny.fb_in22k_ft_in1k:R6b` at 768x512 px; fracture threshold 0.0167 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 88.5%) is fold 1.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.995 | 0.995 ± 0.006 | 0.998 ± 0.001 | 0.997 | 0.998 |
| Fracture sensitivity | 0.971 | 0.971 ± 0.007 | 0.974 ± 0.003 | 0.972 | 0.986 |
| Fracture specificity | 0.989 | 0.989 ± 0.006 | 0.988 ± 0.013 | 0.985 | 0.954 |
| 6-class accuracy (%) | 87.7 | 87.7 ± 0.6 | 85.3 ± 0.7 | 86.2 | 86.2 |
| Macro-F1 | 0.814 | 0.810 ± 0.032 | 0.763 ± 0.024 | 0.774 | 0.792 |
| Balanced accuracy | 0.798 | 0.798 ± 0.042 | 0.747 ± 0.022 | 0.761 | 0.778 |
| Cohen's kappa | 0.838 | 0.838 ± 0.007 | 0.812 ± 0.009 | 0.823 | 0.824 |
| Location accuracy (fractures called) | 0.942 | 0.942 ± 0.020 | 0.939 ± 0.011 | 0.957 | 0.958 |
| Grade accuracy, location right | 0.889 | 0.890 ± 0.022 | 0.861 ± 0.017 | 0.858 | 0.860 |
| Dice, fractured femurs (missed = 0) | 0.776 | 0.776 ± 0.013 | 0.729 ± 0.004 | 0.728 | 0.742 |
| Dice, detected fractures only | 0.799 | 0.799 ± 0.009 | 0.749 ± 0.004 | 0.748 | 0.753 |
| Predicted / true area, median | 0.97 | 0.98 ± 0.01 | 1.28 ± 0.03 | 1.23 | 1.26 |
| HD95 (px), median, detected | 60.4 | 61.4 ± 3.5 | 69.2 ± 0.9 | 69.1 | 66.0 |
| Errors: detection | 24.0 | 4.8 ± 1.1 | 4.6 ± 1.1 | 5.0 | 5.0 |
| Errors: location | 38.0 | 7.6 ± 2.6 | 8.6 ± 1.5 | 6.0 | 6.0 |
| Errors: grade | 66.0 | 13.2 ± 2.8 | 17.6 ± 2.2 | 18.0 | 18.0 |
| Errors: total | 128.0 | 25.6 ± 1.1 | 30.8 ± 1.5 | 29.0 | 29.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.989 | 0.989 ± 0.006 | 0.988 ± 0.013 | 0.985 | 0.954 |
| 2AB Neck Garden I-II | 57 / 16 | 0.561 | 0.561 ± 0.136 | 0.287 ± 0.105 | 0.188 | 0.312 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.906 | 0.906 ± 0.058 | 0.990 ± 0.013 | 1.000 | 1.000 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.813 | 0.813 ± 0.077 | 0.918 ± 0.048 | 0.941 | 0.971 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.799 | 0.799 ± 0.072 | 0.767 ± 0.052 | 0.786 | 0.762 |
| 4 Subtrochanteric | 36 / 12 | 0.722 | 0.721 ± 0.226 | 0.533 ± 0.126 | 0.667 | 0.667 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.618 | 0.619 ± 0.039 | 0.566 ± 0.033 | 0.607 | 0.860 / 0.825 ± 0.052 | 0.719 / 0.686 ± 0.007 | 0.97 / 1.85 ± 0.08 |
| 2CD Neck Garden III-IV | 0.765 | 0.765 ± 0.028 | 0.695 ± 0.006 | 0.698 | 0.976 / 1.000 ± 0.000 | 0.784 / 0.695 ± 0.006 | 0.95 / 1.72 ± 0.03 |
| 3AB Pertroch Evans I-II | 0.773 | 0.773 ± 0.043 | 0.740 ± 0.010 | 0.758 | 0.970 / 0.976 ± 0.013 | 0.797 / 0.758 ± 0.004 | 1.02 / 1.16 ± 0.04 |
| 3CDE Pertroch Evans III-V | 0.850 | 0.850 ± 0.012 | 0.836 ± 0.003 | 0.843 | 1.000 / 1.000 ± 0.000 | 0.850 / 0.836 ± 0.003 | 0.97 / 1.06 ± 0.02 |
| 4 Subtrochanteric | 0.694 | 0.694 ± 0.081 | 0.658 ± 0.028 | 0.678 | 0.944 / 0.983 ± 0.037 | 0.735 / 0.669 ± 0.032 | 0.98 / 0.96 ± 0.07 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 359 | 1 | 2 | 1 | 0 | 0 |
| 2AB | 8 | 32 | 12 | 5 | 0 | 0 |
| 2CD | 6 | 9 | 230 | 9 | 0 | 0 |
| 3AB | 4 | 2 | 3 | 109 | 16 | 0 |
| 3CDE | 0 | 2 | 6 | 29 | 159 | 3 |
| 4 | 2 | 1 | 0 | 1 | 6 | 26 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 62 | 1 | 1 | 1 | 0 | 0 |
| 2AB | 2 | 5 | 8 | 1 | 0 | 0 |
| 2CD | 0 | 0 | 41 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 33 | 1 | 0 |
| 3CDE | 0 | 0 | 1 | 9 | 32 | 0 |
| 4 | 0 | 1 | 0 | 0 | 3 | 8 |

## Runs side by side (test, 5 fold models mean ± SD; OOF pooled in brackets)

| Metric | R4_refrac | R1_aasn | R5_concat | R6a_no_align | R6b_affine |
|---|---|---|---|---|---|
| 6-class accuracy (%) | 86.0 ± 0.9 (88.2) | 85.4 ± 1.1 (87.7) | 85.8 ± 0.9 (88.7) | 85.9 ± 1.5 (87.1) | 85.3 ± 0.7 (87.7) |
| Macro-F1 | 0.777 ± 0.021 (0.826) | 0.777 ± 0.016 (0.820) | 0.765 ± 0.021 (0.827) | 0.786 ± 0.039 (0.807) | 0.763 ± 0.024 (0.814) |
| Dice, fractured (missed = 0) | 0.730 ± 0.004 (0.777) | 0.728 ± 0.003 (0.781) | 0.729 ± 0.006 (0.780) | 0.727 ± 0.004 (0.777) | 0.729 ± 0.004 (0.776) |
| Recall NF | 0.997 ± 0.007 (0.981) | 0.982 ± 0.017 (0.981) | 0.994 ± 0.008 (0.992) | 0.985 ± 0.015 (0.983) | 0.988 ± 0.013 (0.989) |
| Recall 2AB | 0.325 ± 0.052 (0.702) | 0.375 ± 0.062 (0.579) | 0.275 ± 0.084 (0.561) | 0.338 ± 0.114 (0.561) | 0.287 ± 0.105 (0.561) |
| Recall 2CD | 0.985 ± 0.013 (0.898) | 0.995 ± 0.011 (0.917) | 0.995 ± 0.011 (0.898) | 0.980 ± 0.020 (0.874) | 0.990 ± 0.013 (0.906) |
| Recall 3AB | 0.935 ± 0.044 (0.806) | 0.941 ± 0.047 (0.806) | 0.929 ± 0.061 (0.828) | 0.900 ± 0.026 (0.784) | 0.918 ± 0.048 (0.813) |
| Recall 3CDE | 0.762 ± 0.024 (0.814) | 0.724 ± 0.043 (0.789) | 0.776 ± 0.043 (0.844) | 0.771 ± 0.062 (0.829) | 0.767 ± 0.052 (0.799) |
| Recall 4 | 0.533 ± 0.075 (0.722) | 0.533 ± 0.046 (0.778) | 0.517 ± 0.070 (0.722) | 0.650 ± 0.149 (0.750) | 0.533 ± 0.126 (0.722) |
| Dice 2AB | 0.572 ± 0.027 (0.635) | 0.555 ± 0.014 (0.627) | 0.546 ± 0.031 (0.651) | 0.571 ± 0.020 (0.635) | 0.566 ± 0.033 (0.618) |
| Dice 2CD | 0.682 ± 0.006 (0.761) | 0.691 ± 0.005 (0.771) | 0.692 ± 0.005 (0.769) | 0.691 ± 0.010 (0.761) | 0.695 ± 0.006 (0.765) |
| Dice 3AB | 0.756 ± 0.008 (0.783) | 0.753 ± 0.008 (0.786) | 0.752 ± 0.008 (0.772) | 0.737 ± 0.016 (0.781) | 0.740 ± 0.010 (0.773) |
| Dice 3CDE | 0.839 ± 0.001 (0.850) | 0.833 ± 0.006 (0.847) | 0.836 ± 0.001 (0.852) | 0.837 ± 0.004 (0.846) | 0.836 ± 0.003 (0.850) |
| Dice 4 | 0.648 ± 0.031 (0.682) | 0.643 ± 0.017 (0.719) | 0.653 ± 0.019 (0.698) | 0.647 ± 0.038 (0.715) | 0.658 ± 0.028 (0.694) |

Paired tests between the runs (same femurs) are in `stats.md` of the same folder, if it was made with `scripts/run_stats.py ... --compare`.

## Comparison with the base article

Source: González et al., Insights Imaging 2025;16:86 (doi 10.1186/s13244-025-01958-y). Table 2 gives mean ± SD of six training runs per setting on the same 210 test femurs; the final model is the run with the best validation accuracy (Table 3, confusion matrix in Fig. 3). Our fold models are trained on 4/5 of the 1,043 training femurs (about 834), the article's on all 1,043 plus a 201-femur validation set; our counterpart of its selected model is the fold model with the best accuracy on its own held-out fold.

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

### R1_aasn vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.999 ± 0.000 | 0.981 | 0.999 | 0.999 |
| 6-class accuracy (%) | n/a | 85.4 ± 1.1 | 86.2 | 84.3 | 86.2 |
| Fracture sensitivity | n/a | 0.986 ± 0.005 | 0.952 | 0.986 | 0.986 |
| Fracture specificity | n/a | 0.982 ± 0.017 | 0.985 | 0.985 | 0.954 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.777 ± 0.016 | 0.814 | 0.763 | 0.798 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.738 ± 0.003 | 0.77 | 0.741 | 0.748 |
| Recall NF No fracture | n/a | 0.982 ± 0.017 | 0.985 | 0.985 | 0.954 |
| Recall 2AB Neck Garden I-II | n/a | 0.375 ± 0.062 | 0.500 | 0.312 | 0.438 |
| Recall 2CD Neck Garden III-IV | n/a | 0.995 ± 0.011 | 0.951 | 0.976 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.941 ± 0.047 | 0.882 | 0.882 | 0.971 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.724 ± 0.043 | 0.738 | 0.738 | 0.738 |
| Recall 4 Subtrochanteric | n/a | 0.533 ± 0.046 | 0.750 | 0.583 | 0.583 |
| Errors: detection | n/a | 3.2 ± 1.5 | 8 | 3 | 5 |
| Errors: location | n/a | 7.6 ± 1.1 | 6 | 8 | 6 |
| Errors: grade | n/a | 19.8 ± 3.5 | 15 | 22 | 18 |
| Errors: total | n/a | 30.6 ± 2.3 | 29 | 33 | 29 |

### R5_concat vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.998 ± 0.002 | 0.981 | 0.997 | 0.999 |
| 6-class accuracy (%) | n/a | 85.8 ± 0.9 | 86.2 | 85.2 | 84.8 |
| Fracture sensitivity | n/a | 0.975 ± 0.009 | 0.952 | 0.966 | 0.986 |
| Fracture specificity | n/a | 0.994 ± 0.008 | 0.985 | 0.985 | 0.985 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.765 ± 0.021 | 0.814 | 0.756 | 0.741 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.747 ± 0.003 | 0.77 | 0.746 | 0.751 |
| Recall NF No fracture | n/a | 0.994 ± 0.008 | 0.985 | 0.985 | 0.985 |
| Recall 2AB Neck Garden I-II | n/a | 0.275 ± 0.084 | 0.500 | 0.312 | 0.188 |
| Recall 2CD Neck Garden III-IV | n/a | 0.995 ± 0.011 | 0.951 | 0.976 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.929 ± 0.061 | 0.882 | 0.941 | 0.941 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.776 ± 0.043 | 0.738 | 0.786 | 0.762 |
| Recall 4 Subtrochanteric | n/a | 0.517 ± 0.070 | 0.750 | 0.417 | 0.500 |
| Errors: detection | n/a | 4.0 ± 1.6 | 8 | 6 | 3 |
| Errors: location | n/a | 7.4 ± 1.1 | 6 | 8 | 7 |
| Errors: grade | n/a | 18.4 ± 2.2 | 15 | 17 | 22 |
| Errors: total | n/a | 29.8 ± 1.8 | 29 | 31 | 32 |

### R6a_no_align vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.997 ± 0.002 | 0.981 | 0.996 | 0.999 |
| 6-class accuracy (%) | n/a | 85.9 ± 1.5 | 86.2 | 85.2 | 85.7 |
| Fracture sensitivity | n/a | 0.975 ± 0.006 | 0.952 | 0.972 | 0.986 |
| Fracture specificity | n/a | 0.985 ± 0.015 | 0.985 | 0.985 | 0.969 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.786 ± 0.039 | 0.814 | 0.735 | 0.788 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.746 ± 0.004 | 0.77 | 0.744 | 0.751 |
| Recall NF No fracture | n/a | 0.985 ± 0.015 | 0.985 | 0.985 | 0.969 |
| Recall 2AB Neck Garden I-II | n/a | 0.338 ± 0.114 | 0.500 | 0.188 | 0.312 |
| Recall 2CD Neck Garden III-IV | n/a | 0.980 ± 0.020 | 0.951 | 0.976 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.900 ± 0.026 | 0.882 | 0.912 | 0.941 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.771 ± 0.062 | 0.738 | 0.857 | 0.738 |
| Recall 4 Subtrochanteric | n/a | 0.650 ± 0.149 | 0.750 | 0.417 | 0.667 |
| Errors: detection | n/a | 4.6 ± 0.9 | 8 | 5 | 4 |
| Errors: location | n/a | 6.6 ± 2.2 | 6 | 10 | 5 |
| Errors: grade | n/a | 18.4 ± 2.3 | 15 | 16 | 21 |
| Errors: total | n/a | 29.6 ± 3.0 | 29 | 31 | 30 |

### R6b_affine vs the article's final model (YOLOv8s-seg @1024)

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | n/a | 0.998 ± 0.001 | 0.981 | 0.997 | 0.998 |
| 6-class accuracy (%) | n/a | 85.3 ± 0.7 | 86.2 | 86.2 | 86.2 |
| Fracture sensitivity | n/a | 0.974 ± 0.003 | 0.952 | 0.972 | 0.986 |
| Fracture specificity | n/a | 0.988 ± 0.013 | 0.985 | 0.985 | 0.954 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.763 ± 0.024 | 0.814 | 0.774 | 0.792 |
| Dice (article: definition not stated; ours: detected fractures only) | n/a | 0.749 ± 0.004 | 0.77 | 0.748 | 0.753 |
| Recall NF No fracture | n/a | 0.988 ± 0.013 | 0.985 | 0.985 | 0.954 |
| Recall 2AB Neck Garden I-II | n/a | 0.287 ± 0.105 | 0.500 | 0.188 | 0.312 |
| Recall 2CD Neck Garden III-IV | n/a | 0.990 ± 0.013 | 0.951 | 1.000 | 1.000 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.918 ± 0.048 | 0.882 | 0.941 | 0.971 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.767 ± 0.052 | 0.738 | 0.786 | 0.762 |
| Recall 4 Subtrochanteric | n/a | 0.533 ± 0.126 | 0.750 | 0.667 | 0.667 |
| Errors: detection | n/a | 4.6 ± 1.1 | 8 | 5 | 5 |
| Errors: location | n/a | 8.6 ± 1.5 | 6 | 6 | 6 |
| Errors: grade | n/a | 17.6 ± 2.2 | 15 | 18 | 18 |
| Errors: total | n/a | 30.8 ± 1.5 | 29 | 29 | 29 |

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

| Definition | R4_refrac | R1_aasn | R5_concat | R6a_no_align | R6b_affine |
|---|---|---|---|---|---|
| Detected fractures only, top instance (our main definition) | 0.744 ± 0.001 | 0.738 ± 0.003 | 0.747 ± 0.003 | 0.746 ± 0.004 | 0.749 ± 0.004 |
| Same at Ultralytics' default threshold 0.25 | n/a | n/a | n/a | n/a | n/a |
| Same at 0.25, union of all instances ≥ 0.25 | n/a | n/a | n/a | n/a | n/a |
| All fractured femurs, missed = 0 | 0.730 ± 0.004 | 0.728 ± 0.003 | 0.729 ± 0.006 | 0.727 ± 0.004 | 0.729 ± 0.004 |
| All 210 femurs, no-fracture femur = 1 if nothing called else 0 | 0.813 ± 0.003 | 0.806 ± 0.007 | 0.811 ± 0.006 | 0.807 ± 0.004 | 0.809 ± 0.006 |
| Global (pooled pixels over fractured femurs) | 0.766 ± 0.004 | 0.763 ± 0.003 | 0.765 ± 0.004 | 0.764 ± 0.003 | 0.766 ± 0.005 |
| Per-femur spread, detected only: mean ± 1.96 SD (compare the article's 0.56–0.98) | 0.49–1.00 (ensemble) | 0.50–1.00 (ensemble) | 0.51–0.99 (ensemble) | 0.50–1.00 (ensemble) | 0.51–0.99 (ensemble) |

## OOF versus test segmentation

| Run | Dice detected, OOF pooled | Dice detected, test models | Predicted / true area, OOF (median, IQR) | Predicted / true area, test models (median, IQR, pooled over models) |
|---|---|---|---|---|
| R4_refrac | 0.797 | 0.744 ± 0.001 | 0.97 (0.80–1.18) | 1.28 (0.94–1.76) |
| R1_aasn | 0.795 | 0.738 ± 0.003 | 0.97 (0.79–1.17) | 1.26 (0.95–1.77) |
| R5_concat | 0.802 | 0.747 ± 0.003 | 0.97 (0.79–1.16) | 1.27 (0.95–1.72) |
| R6a_no_align | 0.798 | 0.746 ± 0.004 | 0.98 (0.81–1.17) | 1.27 (0.95–1.76) |
| R6b_affine | 0.799 | 0.749 ± 0.004 | 0.97 (0.80–1.17) | 1.28 (0.96–1.76) |

Median mask area in pixels by class: true masks of the training femurs and of the test femurs (the article says its test labels were reviewed by a committee), and the masks the models predict for called fractures (OOF; test pooled over the 5 fold models).

| Class | True, train | True, test | Test / train | Predicted OOF, R4_refrac | Predicted test, R4_refrac | Predicted OOF, R1_aasn | Predicted test, R1_aasn | Predicted OOF, R5_concat | Predicted test, R5_concat | Predicted OOF, R6a_no_align | Predicted test, R6a_no_align | Predicted OOF, R6b_affine | Predicted test, R6b_affine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 45,963 | 29,797 | 0.65 | 43,444 | 52,675 | 45,192 | 53,628 | 44,057 | 53,176 | 45,403 | 52,468 | 45,318 | 51,780 |
| 2CD Neck Garden III-IV | 75,612 | 45,844 | 0.61 | 75,635 | 79,142 | 74,768 | 80,595 | 73,894 | 79,924 | 74,572 | 78,138 | 74,124 | 77,503 |
| 3AB Pertroch Evans I-II | 85,540 | 88,794 | 1.04 | 96,462 | 93,149 | 93,943 | 93,365 | 91,622 | 93,253 | 92,627 | 94,361 | 95,912 | 91,932 |
| 3CDE Pertroch Evans III-V | 135,306 | 120,095 | 0.89 | 132,074 | 137,379 | 130,031 | 136,494 | 128,679 | 135,850 | 132,240 | 136,556 | 132,209 | 136,968 |
| 4 Subtrochanteric | 126,204 | 149,960 | 1.19 | 134,393 | 121,561 | 128,701 | 122,392 | 137,833 | 125,073 | 140,164 | 123,466 | 138,048 | 121,838 |
