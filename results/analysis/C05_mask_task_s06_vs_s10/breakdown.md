# Breakdown: S06_convnext_tiny, S10_m0_convnext_unet

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

## Runs side by side (test, 5 fold models mean ± SD; OOF pooled in brackets)

| Metric | S06_convnext_tiny | S10_m0_convnext_unet |
|---|---|---|
| 6-class accuracy (%) | 83.5 ± 1.6 (86.4) | 86.0 ± 0.5 (88.4) |
| Macro-F1 | 0.732 ± 0.024 (0.784) | 0.776 ± 0.017 (0.825) |
| Dice, fractured (missed = 0) | n/a (n/a) | 0.734 ± 0.005 (0.777) |
| Recall NF | 0.994 ± 0.014 (0.978) | 1.000 ± 0.000 (0.992) |
| Recall 2AB | 0.275 ± 0.034 (0.474) | 0.312 ± 0.077 (0.544) |
| Recall 2CD | 0.980 ± 0.020 (0.921) | 0.990 ± 0.013 (0.909) |
| Recall 3AB | 0.853 ± 0.075 (0.761) | 0.953 ± 0.016 (0.791) |
| Recall 3CDE | 0.767 ± 0.052 (0.804) | 0.738 ± 0.034 (0.839) |
| Recall 4 | 0.417 ± 0.132 (0.639) | 0.550 ± 0.046 (0.750) |
| Dice 2AB | n/a (n/a) | 0.578 ± 0.032 (0.624) |
| Dice 2CD | n/a (n/a) | 0.699 ± 0.009 (0.773) |
| Dice 3AB | n/a (n/a) | 0.750 ± 0.008 (0.763) |
| Dice 3CDE | n/a (n/a) | 0.836 ± 0.004 (0.852) |
| Dice 4 | n/a (n/a) | 0.655 ± 0.039 (0.682) |

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

| Definition | S06_convnext_tiny | S10_m0_convnext_unet |
|---|---|---|
| Detected fractures only, top instance (our main definition) | n/a | 0.746 ± 0.002 |
| Same at Ultralytics' default threshold 0.25 | n/a | n/a |
| Same at 0.25, union of all instances ≥ 0.25 | n/a | n/a |
| All fractured femurs, missed = 0 | n/a | 0.734 ± 0.005 |
| All 210 femurs, no-fracture femur = 1 if nothing called else 0 | n/a | 0.816 ± 0.004 |
| Global (pooled pixels over fractured femurs) | n/a | 0.767 ± 0.004 |
| Per-femur spread, detected only: mean ± 1.96 SD (compare the article's 0.56–0.98) | n/a | 0.48–1.00 (ensemble) |

## OOF versus test segmentation

| Run | Dice detected, OOF pooled | Dice detected, test models | Predicted / true area, OOF (median, IQR) | Predicted / true area, test models (median, IQR, pooled over models) |
|---|---|---|---|---|
| S06_convnext_tiny | n/a | n/a | n/a | n/a |
| S10_m0_convnext_unet | 0.799 | 0.746 ± 0.002 | 0.98 (0.79–1.18) | 1.27 (0.94–1.74) |

Median mask area in pixels by class: true masks of the training femurs and of the test femurs (the article says its test labels were reviewed by a committee), and the masks the models predict for called fractures (OOF; test pooled over the 5 fold models).

| Class | True, train | True, test | Test / train | Predicted OOF, S06_convnext_tiny | Predicted test, S06_convnext_tiny | Predicted OOF, S10_m0_convnext_unet | Predicted test, S10_m0_convnext_unet |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 45,963 | 29,797 | 0.65 | n/a | n/a | 42,584 | 54,829 |
| 2CD Neck Garden III-IV | 75,612 | 45,844 | 0.61 | n/a | n/a | 74,660 | 77,694 |
| 3AB Pertroch Evans I-II | 85,540 | 88,794 | 1.04 | n/a | n/a | 91,848 | 89,882 |
| 3CDE Pertroch Evans III-V | 135,306 | 120,095 | 0.89 | n/a | n/a | 133,085 | 138,130 |
| 4 Subtrochanteric | 126,204 | 149,960 | 1.19 | n/a | n/a | 136,438 | 124,222 |
