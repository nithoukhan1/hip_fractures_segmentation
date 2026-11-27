# Breakdown: S01_yolov8s_seg_1024, S03_yolo11s_seg_1024

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

## S03_yolo11s_seg_1024

Model `yolo11s-seg.pt` at 1024 px; fracture threshold 0.1329 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 83.3%) is fold 3.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.974 | 0.974 ± 0.013 | 0.988 ± 0.006 | 0.993 | 0.994 |
| Fracture sensitivity | 0.928 | 0.928 ± 0.014 | 0.941 ± 0.008 | 0.938 | 0.972 |
| Fracture specificity | 0.956 | 0.956 ± 0.025 | 0.948 ± 0.028 | 0.985 | 0.969 |
| 6-class accuracy (%) | 82.2 | 82.2 ± 1.1 | 81.1 ± 4.0 | 88.1 | 85.2 |
| Macro-F1 | 0.719 | 0.717 ± 0.036 | 0.750 ± 0.062 | 0.853 | 0.801 |
| Balanced accuracy | 0.717 | 0.715 ± 0.036 | 0.743 ± 0.060 | 0.842 | 0.801 |
| Cohen's kappa | 0.765 | 0.765 ± 0.013 | 0.760 ± 0.051 | 0.848 | 0.813 |
| Location accuracy (fractures called) | 0.929 | 0.929 ± 0.013 | 0.952 ± 0.017 | 0.978 | 0.972 |
| Grade accuracy, location right | 0.865 | 0.865 ± 0.025 | 0.828 ± 0.042 | 0.902 | 0.835 |
| Dice, fractured femurs (missed = 0) | 0.743 | 0.743 ± 0.015 | 0.693 ± 0.007 | 0.691 | 0.730 |
| Dice, detected fractures only | 0.800 | 0.800 ± 0.009 | 0.737 ± 0.005 | 0.737 | 0.750 |
| Predicted / true area, median | 1.01 | 1.01 ± 0.01 | 1.32 ± 0.04 | 1.27 | 1.31 |
| HD95 (px), median, detected | 62.6 | 62.6 ± 2.1 | 72.0 ± 2.9 | 73.2 | 68.6 |
| Errors: detection | 65.0 | 13.0 ± 3.2 | 12.0 ± 1.9 | 10.0 | 6.0 |
| Errors: location | 45.0 | 9.0 ± 1.7 | 6.6 ± 2.3 | 3.0 | 4.0 |
| Errors: grade | 76.0 | 15.2 ± 2.9 | 21.0 ± 5.2 | 12.0 | 21.0 |
| Errors: total | 186.0 | 37.2 ± 2.2 | 39.6 ± 8.4 | 25.0 | 31.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.956 | 0.956 ± 0.025 | 0.948 ± 0.028 | 0.985 | 0.969 |
| 2AB Neck Garden I-II | 57 / 16 | 0.386 | 0.383 ± 0.105 | 0.463 ± 0.137 | 0.562 | 0.438 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.858 | 0.858 ± 0.034 | 0.946 ± 0.020 | 0.976 | 0.976 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.746 | 0.746 ± 0.095 | 0.847 ± 0.073 | 0.853 | 0.971 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.744 | 0.744 ± 0.075 | 0.624 ± 0.091 | 0.762 | 0.619 |
| 4 Subtrochanteric | 36 / 12 | 0.611 | 0.604 ± 0.217 | 0.633 ± 0.192 | 0.917 | 0.833 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.506 | 0.505 ± 0.049 | 0.544 ± 0.020 | 0.583 | 0.702 / 0.875 ± 0.044 | 0.720 / 0.622 ± 0.022 | 1.26 / 2.13 ± 0.18 |
| 2CD Neck Garden III-IV | 0.749 | 0.749 ± 0.039 | 0.673 ± 0.024 | 0.691 | 0.949 / 0.985 ± 0.013 | 0.789 / 0.683 ± 0.019 | 0.99 / 1.74 ± 0.16 |
| 3AB Pertroch Evans I-II | 0.735 | 0.734 ± 0.042 | 0.725 ± 0.025 | 0.772 | 0.933 / 0.959 ± 0.026 | 0.788 / 0.756 ± 0.010 | 0.99 / 1.20 ± 0.05 |
| 3CDE Pertroch Evans III-V | 0.829 | 0.829 ± 0.020 | 0.789 ± 0.022 | 0.804 | 0.980 / 0.957 ± 0.039 | 0.846 / 0.825 ± 0.014 | 1.00 / 1.10 ± 0.05 |
| 4 Subtrochanteric | 0.633 | 0.631 ± 0.104 | 0.538 ± 0.094 | 0.675 | 0.833 / 0.767 ± 0.109 | 0.759 / 0.700 ± 0.054 | 1.10 / 0.98 ± 0.14 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 347 | 8 | 4 | 2 | 0 | 2 |
| 2AB | 17 | 22 | 14 | 4 | 0 | 0 |
| 2CD | 13 | 14 | 218 | 7 | 2 | 0 |
| 3AB | 9 | 2 | 2 | 100 | 19 | 2 |
| 3CDE | 4 | 1 | 4 | 29 | 148 | 13 |
| 4 | 6 | 0 | 0 | 1 | 7 | 22 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 63 | 2 | 0 | 0 | 0 | 0 |
| 2AB | 1 | 7 | 8 | 0 | 0 | 0 |
| 2CD | 0 | 1 | 40 | 0 | 0 | 0 |
| 3AB | 0 | 0 | 0 | 33 | 1 | 0 |
| 3CDE | 2 | 0 | 1 | 11 | 26 | 2 |
| 4 | 1 | 0 | 0 | 0 | 1 | 10 |

## Runs side by side (test, 5 fold models mean ± SD; OOF pooled in brackets)

| Metric | S01_yolov8s_seg_1024 | S03_yolo11s_seg_1024 |
|---|---|---|
| 6-class accuracy (%) | 84.6 ± 2.1 (83.2) | 81.1 ± 4.0 (82.2) |
| Macro-F1 | 0.791 ± 0.018 (0.744) | 0.750 ± 0.062 (0.719) |
| Dice, fractured (missed = 0) | 0.693 ± 0.016 (0.740) | 0.693 ± 0.007 (0.743) |
| Recall NF | 0.978 ± 0.008 (0.970) | 0.948 ± 0.028 (0.956) |
| Recall 2AB | 0.438 ± 0.117 (0.509) | 0.463 ± 0.137 (0.386) |
| Recall 2CD | 0.961 ± 0.028 (0.831) | 0.946 ± 0.020 (0.858) |
| Recall 3AB | 0.888 ± 0.060 (0.716) | 0.847 ± 0.073 (0.746) |
| Recall 3CDE | 0.681 ± 0.087 (0.794) | 0.624 ± 0.091 (0.744) |
| Recall 4 | 0.733 ± 0.124 (0.611) | 0.633 ± 0.192 (0.611) |
| Dice 2AB | 0.524 ± 0.036 (0.534) | 0.544 ± 0.020 (0.506) |
| Dice 2CD | 0.677 ± 0.013 (0.751) | 0.673 ± 0.024 (0.749) |
| Dice 3AB | 0.713 ± 0.039 (0.721) | 0.725 ± 0.025 (0.735) |
| Dice 3CDE | 0.781 ± 0.019 (0.822) | 0.789 ± 0.022 (0.829) |
| Dice 4 | 0.608 ± 0.024 (0.601) | 0.538 ± 0.094 (0.633) |

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

### S03_yolo11s_seg_1024 vs the article's YOLOv8-seg small @ 1024

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | 0.986 ± 0.008 | 0.988 ± 0.006 | 0.981 | 0.993 | 0.994 |
| 6-class accuracy (%) | 84.4 ± 2.5 | 81.1 ± 4.0 | 86.2 | 88.1 | 85.2 |
| Fracture sensitivity | n/a | 0.941 ± 0.008 | 0.952 | 0.938 | 0.972 |
| Fracture specificity | n/a | 0.948 ± 0.028 | 0.985 | 0.985 | 0.969 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.750 ± 0.062 | 0.814 | 0.853 | 0.801 |
| Dice (article: definition not stated; ours: detected fractures only) | 0.761 ± 0.008 | 0.737 ± 0.005 | 0.77 | 0.737 | 0.750 |
| Recall NF No fracture | n/a | 0.948 ± 0.028 | 0.985 | 0.985 | 0.969 |
| Recall 2AB Neck Garden I-II | n/a | 0.463 ± 0.137 | 0.500 | 0.562 | 0.438 |
| Recall 2CD Neck Garden III-IV | n/a | 0.946 ± 0.020 | 0.951 | 0.976 | 0.976 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.847 ± 0.073 | 0.882 | 0.853 | 0.971 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.624 ± 0.091 | 0.738 | 0.762 | 0.619 |
| Recall 4 Subtrochanteric | n/a | 0.633 ± 0.192 | 0.750 | 0.917 | 0.833 |
| Errors: detection | n/a | 12.0 ± 1.9 | 8 | 10 | 6 |
| Errors: location | n/a | 6.6 ± 2.3 | 6 | 3 | 4 |
| Errors: grade | n/a | 21.0 ± 5.2 | 15 | 12 | 21 |
| Errors: total | n/a | 39.6 ± 8.4 | 29 | 25 | 31 |

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

| Definition | S01_yolov8s_seg_1024 | S03_yolo11s_seg_1024 |
|---|---|---|
| Detected fractures only, top instance (our main definition) | 0.741 ± 0.009 | 0.737 ± 0.005 |
| Same at Ultralytics' default threshold 0.25 | 0.744 ± 0.007 | 0.740 ± 0.007 |
| Same at 0.25, union of all instances ≥ 0.25 | 0.745 ± 0.007 | 0.740 ± 0.007 |
| All fractured femurs, missed = 0 | 0.693 ± 0.016 | 0.693 ± 0.007 |
| All 210 femurs, no-fracture femur = 1 if nothing called else 0 | 0.781 ± 0.011 | 0.772 ± 0.011 |
| Global (pooled pixels over fractured femurs) | 0.733 ± 0.013 | 0.732 ± 0.006 |
| Per-femur spread, detected only: mean ± 1.96 SD (compare the article's 0.56–0.98) | 0.52–0.99 (ensemble) | 0.50–1.00 (ensemble) |

## OOF versus test segmentation

| Run | Dice detected, OOF pooled | Dice detected, test models | Predicted / true area, OOF (median, IQR) | Predicted / true area, test models (median, IQR, pooled over models) |
|---|---|---|---|---|
| S01_yolov8s_seg_1024 | 0.800 | 0.741 ± 0.009 | 0.99 (0.84–1.22) | 1.27 (0.94–1.80) |
| S03_yolo11s_seg_1024 | 0.800 | 0.737 ± 0.005 | 1.01 (0.85–1.25) | 1.33 (1.01–1.88) |

Median mask area in pixels by class: true masks of the training femurs and of the test femurs (the article says its test labels were reviewed by a committee), and the masks the models predict for called fractures (OOF; test pooled over the 5 fold models).

| Class | True, train | True, test | Test / train | Predicted OOF, S01_yolov8s_seg_1024 | Predicted test, S01_yolov8s_seg_1024 | Predicted OOF, S03_yolo11s_seg_1024 | Predicted test, S03_yolo11s_seg_1024 |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 45,963 | 29,797 | 0.65 | 54,558 | 57,046 | 56,359 | 58,120 |
| 2CD Neck Garden III-IV | 75,612 | 45,844 | 0.61 | 75,926 | 76,820 | 76,283 | 78,805 |
| 3AB Pertroch Evans I-II | 85,540 | 88,794 | 1.04 | 93,944 | 96,513 | 88,943 | 98,966 |
| 3CDE Pertroch Evans III-V | 135,306 | 120,095 | 0.89 | 131,154 | 130,396 | 134,283 | 137,007 |
| 4 Subtrochanteric | 126,204 | 149,960 | 1.19 | 144,031 | 125,186 | 148,676 | 130,334 |
