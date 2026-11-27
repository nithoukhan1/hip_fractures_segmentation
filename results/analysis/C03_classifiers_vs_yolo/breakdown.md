# Breakdown: S01_yolov8s_seg_1024, S04_densenet121, S05_efficientnet_b4, S06_convnext_tiny

Made by `scripts/run_breakdown.py` from the filed run folders. Decisions use each run's OOF threshold. Mean ± SD is over the 5 held-out folds (OOF) or the 5 fold models (test); SD uses n − 1.

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

## Runs side by side (test, 5 fold models mean ± SD; OOF pooled in brackets)

| Metric | S01_yolov8s_seg_1024 | S04_densenet121 | S05_efficientnet_b4 | S06_convnext_tiny |
|---|---|---|---|---|
| 6-class accuracy (%) | 84.6 ± 2.1 (83.2) | 84.8 ± 2.0 (86.8) | 80.6 ± 1.5 (84.8) | 83.5 ± 1.6 (86.4) |
| Macro-F1 | 0.791 ± 0.018 (0.744) | 0.749 ± 0.049 (0.802) | 0.736 ± 0.021 (0.755) | 0.732 ± 0.024 (0.784) |
| Dice, fractured (missed = 0) | 0.693 ± 0.016 (0.740) | n/a (n/a) | n/a (n/a) | n/a (n/a) |
| Recall NF | 0.978 ± 0.008 (0.970) | 0.988 ± 0.013 (0.983) | 0.972 ± 0.013 (0.975) | 0.994 ± 0.014 (0.978) |
| Recall 2AB | 0.438 ± 0.117 (0.509) | 0.312 ± 0.133 (0.579) | 0.338 ± 0.114 (0.386) | 0.275 ± 0.034 (0.474) |
| Recall 2CD | 0.961 ± 0.028 (0.831) | 0.971 ± 0.011 (0.890) | 0.956 ± 0.011 (0.909) | 0.980 ± 0.020 (0.921) |
| Recall 3AB | 0.888 ± 0.060 (0.716) | 0.876 ± 0.057 (0.761) | 0.671 ± 0.038 (0.709) | 0.853 ± 0.075 (0.761) |
| Recall 3CDE | 0.681 ± 0.087 (0.794) | 0.814 ± 0.020 (0.814) | 0.757 ± 0.035 (0.794) | 0.767 ± 0.052 (0.804) |
| Recall 4 | 0.733 ± 0.124 (0.611) | 0.417 ± 0.102 (0.694) | 0.567 ± 0.109 (0.667) | 0.417 ± 0.132 (0.639) |
| Dice 2AB | 0.524 ± 0.036 (0.534) | n/a (n/a) | n/a (n/a) | n/a (n/a) |
| Dice 2CD | 0.677 ± 0.013 (0.751) | n/a (n/a) | n/a (n/a) | n/a (n/a) |
| Dice 3AB | 0.713 ± 0.039 (0.721) | n/a (n/a) | n/a (n/a) | n/a (n/a) |
| Dice 3CDE | 0.781 ± 0.019 (0.822) | n/a (n/a) | n/a (n/a) | n/a (n/a) |
| Dice 4 | 0.608 ± 0.024 (0.601) | n/a (n/a) | n/a (n/a) | n/a (n/a) |

Paired tests between the runs (same femurs) are in `stats.md` of the same folder, if it was made with `scripts/run_stats.py ... --compare`.

## Comparison with the base article

Source: González et al., Insights Imaging 2025;16:86 (doi 10.1186/s13244-025-01958-y). Table 2 gives mean ± SD of six training runs per setting on the same 210 test femurs; the final model is the run with the best validation accuracy (Table 3, confusion matrix in Fig. 3). Our fold models are trained on 4/5 of the 1,043 training femurs (about 834), the article's on all 1,043 plus a 201-femur validation set; our counterpart of its selected model is the fold model with the best accuracy on its own held-out fold.

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

| Definition | S01_yolov8s_seg_1024 | S04_densenet121 | S05_efficientnet_b4 | S06_convnext_tiny |
|---|---|---|---|---|
| Detected fractures only, top instance (our main definition) | 0.741 ± 0.009 | n/a | n/a | n/a |
| Same at Ultralytics' default threshold 0.25 | 0.744 ± 0.007 | n/a | n/a | n/a |
| Same at 0.25, union of all instances ≥ 0.25 | 0.745 ± 0.007 | n/a | n/a | n/a |
| All fractured femurs, missed = 0 | 0.693 ± 0.016 | n/a | n/a | n/a |
| All 210 femurs, no-fracture femur = 1 if nothing called else 0 | 0.781 ± 0.011 | n/a | n/a | n/a |
| Global (pooled pixels over fractured femurs) | 0.733 ± 0.013 | n/a | n/a | n/a |
| Per-femur spread, detected only: mean ± 1.96 SD (compare the article's 0.56–0.98) | 0.52–0.99 (ensemble) | n/a | n/a | n/a |

## OOF versus test segmentation

| Run | Dice detected, OOF pooled | Dice detected, test models | Predicted / true area, OOF (median, IQR) | Predicted / true area, test models (median, IQR, pooled over models) |
|---|---|---|---|---|
| S01_yolov8s_seg_1024 | 0.800 | 0.741 ± 0.009 | 0.99 (0.84–1.22) | 1.27 (0.94–1.80) |
| S04_densenet121 | n/a | n/a | n/a | n/a |
| S05_efficientnet_b4 | n/a | n/a | n/a | n/a |
| S06_convnext_tiny | n/a | n/a | n/a | n/a |

Median mask area in pixels by class: true masks of the training femurs and of the test femurs (the article says its test labels were reviewed by a committee), and the masks the models predict for called fractures (OOF; test pooled over the 5 fold models).

| Class | True, train | True, test | Test / train | Predicted OOF, S01_yolov8s_seg_1024 | Predicted test, S01_yolov8s_seg_1024 | Predicted OOF, S04_densenet121 | Predicted test, S04_densenet121 | Predicted OOF, S05_efficientnet_b4 | Predicted test, S05_efficientnet_b4 | Predicted OOF, S06_convnext_tiny | Predicted test, S06_convnext_tiny |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 45,963 | 29,797 | 0.65 | 54,558 | 57,046 | n/a | n/a | n/a | n/a | n/a | n/a |
| 2CD Neck Garden III-IV | 75,612 | 45,844 | 0.61 | 75,926 | 76,820 | n/a | n/a | n/a | n/a | n/a | n/a |
| 3AB Pertroch Evans I-II | 85,540 | 88,794 | 1.04 | 93,944 | 96,513 | n/a | n/a | n/a | n/a | n/a | n/a |
| 3CDE Pertroch Evans III-V | 135,306 | 120,095 | 0.89 | 131,154 | 130,396 | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 Subtrochanteric | 126,204 | 149,960 | 1.19 | 144,031 | 125,186 | n/a | n/a | n/a | n/a | n/a | n/a |
