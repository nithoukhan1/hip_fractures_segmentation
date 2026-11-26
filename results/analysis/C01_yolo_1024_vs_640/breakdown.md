# Breakdown: S01_yolov8s_seg_1024, S02_yolov8s_seg_640

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

## S02_yolov8s_seg_640

Model `yolov8s-seg.pt` at 640 px; fracture threshold 0.0452 (Youden on the pooled OOF predictions). The fold model the article's selection rule picks (best accuracy on its own held-out fold, 85.2%) is fold 2.

### Headline metrics

| Metric | OOF pooled (n=1,043) | OOF, 5 folds mean ± SD | Test, 5 fold models mean ± SD | Test, selected fold model | Test, 5-model ensemble (n=210) |
|---|---|---|---|---|---|
| Fracture AUC | 0.975 | 0.976 ± 0.008 | 0.979 ± 0.007 | 0.980 | 0.994 |
| Fracture sensitivity | 0.935 | 0.935 ± 0.024 | 0.939 ± 0.013 | 0.945 | 0.972 |
| Fracture specificity | 0.970 | 0.970 ± 0.012 | 0.972 ± 0.007 | 0.969 | 0.938 |
| 6-class accuracy (%) | 82.6 | 82.6 ± 2.1 | 82.2 ± 2.1 | 85.7 | 83.8 |
| Macro-F1 | 0.751 | 0.748 ± 0.034 | 0.758 ± 0.036 | 0.807 | 0.787 |
| Balanced accuracy | 0.765 | 0.766 ± 0.046 | 0.744 ± 0.030 | 0.786 | 0.774 |
| Cohen's kappa | 0.772 | 0.772 ± 0.028 | 0.772 ± 0.027 | 0.818 | 0.795 |
| Location accuracy (fractures called) | 0.931 | 0.931 ± 0.012 | 0.963 ± 0.017 | 0.971 | 0.965 |
| Grade accuracy, location right | 0.854 | 0.854 ± 0.023 | 0.824 ± 0.032 | 0.873 | 0.837 |
| Dice, fractured femurs (missed = 0) | 0.746 | 0.746 ± 0.019 | 0.696 ± 0.009 | 0.703 | 0.731 |
| Dice, detected fractures only | 0.798 | 0.798 ± 0.006 | 0.741 ± 0.002 | 0.744 | 0.752 |
| Predicted / true area, median | 0.97 | 0.98 ± 0.02 | 1.29 ± 0.02 | 1.29 | 1.26 |
| HD95 (px), median, detected | 63.1 | 63.0 ± 1.5 | 69.9 ± 3.4 | 69.3 | 67.6 |
| Errors: detection | 55.0 | 11.0 ± 3.9 | 10.6 ± 2.3 | 10.0 | 8.0 |
| Errors: location | 44.0 | 8.8 ± 1.5 | 5.0 ± 2.3 | 4.0 | 5.0 |
| Errors: grade | 82.0 | 16.4 ± 2.7 | 21.8 ± 4.0 | 16.0 | 21.0 |
| Errors: total | 181.0 | 36.2 ± 4.4 | 37.4 ± 4.3 | 30.0 | 34.0 |

Errors are counted like the article: detection = fracture vs no fracture wrong; location = fracture found but wrong location; grade = right location, wrong grade.

### Recall per class

| Class | n OOF / test | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, selected model | Test, ensemble |
|---|---|---|---|---|---|---|
| NF No fracture | 363 / 65 | 0.970 | 0.970 ± 0.012 | 0.972 ± 0.007 | 0.969 | 0.938 |
| 2AB Neck Garden I-II | 57 / 16 | 0.544 | 0.544 ± 0.195 | 0.412 ± 0.201 | 0.562 | 0.562 |
| 2CD Neck Garden III-IV | 254 / 41 | 0.815 | 0.815 ± 0.059 | 0.941 ± 0.037 | 0.976 | 0.976 |
| 3AB Pertroch Evans I-II | 134 / 34 | 0.724 | 0.724 ± 0.088 | 0.871 ± 0.087 | 0.912 | 0.941 |
| 3CDE Pertroch Evans III-V | 199 / 42 | 0.734 | 0.734 ± 0.097 | 0.648 ± 0.066 | 0.714 | 0.643 |
| 4 Subtrochanteric | 36 / 12 | 0.806 | 0.807 ± 0.188 | 0.617 ± 0.046 | 0.583 | 0.583 |

### Dice per class (fractured femurs; a missed fracture counts as 0)

| Class | OOF pooled | OOF, 5 folds mean ± SD | Test, 5 models mean ± SD | Test, ensemble | Detected (OOF / test models) | Dice, detected only (OOF / test models) | Predicted / true area, median (OOF / test models) |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 0.530 | 0.531 ± 0.106 | 0.538 ± 0.051 | 0.592 | 0.737 / 0.863 ± 0.081 | 0.719 / 0.624 ± 0.029 | 1.03 / 2.08 ± 0.12 |
| 2CD Neck Garden III-IV | 0.753 | 0.753 ± 0.031 | 0.694 ± 0.011 | 0.702 | 0.957 / 1.000 ± 0.000 | 0.787 / 0.694 ± 0.011 | 0.95 / 1.69 ± 0.08 |
| 3AB Pertroch Evans I-II | 0.729 | 0.729 ± 0.052 | 0.730 ± 0.014 | 0.773 | 0.925 / 0.959 ± 0.026 | 0.788 / 0.761 ± 0.011 | 1.02 / 1.18 ± 0.07 |
| 3CDE Pertroch Evans III-V | 0.826 | 0.826 ± 0.021 | 0.775 ± 0.023 | 0.814 | 0.980 / 0.938 ± 0.027 | 0.843 / 0.826 ± 0.004 | 0.96 / 1.07 ± 0.02 |
| 4 Subtrochanteric | 0.659 | 0.658 ± 0.095 | 0.544 ± 0.036 | 0.602 | 0.889 / 0.783 ± 0.046 | 0.742 / 0.695 ± 0.032 | 1.10 / 0.99 ± 0.08 |

### Confusion matrices (rows = true class)

OOF pooled (1,043 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 352 | 3 | 4 | 1 | 0 | 3 |
| 2AB | 15 | 31 | 9 | 2 | 0 | 0 |
| 2CD | 11 | 23 | 207 | 13 | 0 | 0 |
| 3AB | 10 | 4 | 1 | 97 | 19 | 3 |
| 3CDE | 4 | 2 | 5 | 31 | 146 | 11 |
| 4 | 4 | 0 | 0 | 0 | 3 | 29 |

Test, 5-model ensemble (210 femurs):

| True \ predicted | NF | 2AB | 2CD | 3AB | 3CDE | 4 |
|---|---|---|---|---|---|---|
| NF | 61 | 3 | 0 | 1 | 0 | 0 |
| 2AB | 1 | 9 | 6 | 0 | 0 | 0 |
| 2CD | 0 | 1 | 40 | 0 | 0 | 0 |
| 3AB | 0 | 1 | 0 | 32 | 1 | 0 |
| 3CDE | 1 | 0 | 1 | 13 | 27 | 0 |
| 4 | 2 | 0 | 0 | 1 | 2 | 7 |

## Runs side by side (test, 5 fold models mean ± SD; OOF pooled in brackets)

| Metric | S01_yolov8s_seg_1024 | S02_yolov8s_seg_640 |
|---|---|---|
| 6-class accuracy (%) | 84.6 ± 2.1 (83.2) | 82.2 ± 2.1 (82.6) |
| Macro-F1 | 0.791 ± 0.018 (0.744) | 0.758 ± 0.036 (0.751) |
| Dice, fractured (missed = 0) | 0.693 ± 0.016 (0.740) | 0.696 ± 0.009 (0.746) |
| Recall NF | 0.978 ± 0.008 (0.970) | 0.972 ± 0.007 (0.970) |
| Recall 2AB | 0.438 ± 0.117 (0.509) | 0.412 ± 0.201 (0.544) |
| Recall 2CD | 0.961 ± 0.028 (0.831) | 0.941 ± 0.037 (0.815) |
| Recall 3AB | 0.888 ± 0.060 (0.716) | 0.871 ± 0.087 (0.724) |
| Recall 3CDE | 0.681 ± 0.087 (0.794) | 0.648 ± 0.066 (0.734) |
| Recall 4 | 0.733 ± 0.124 (0.611) | 0.617 ± 0.046 (0.806) |
| Dice 2AB | 0.524 ± 0.036 (0.534) | 0.538 ± 0.051 (0.530) |
| Dice 2CD | 0.677 ± 0.013 (0.751) | 0.694 ± 0.011 (0.753) |
| Dice 3AB | 0.713 ± 0.039 (0.721) | 0.730 ± 0.014 (0.729) |
| Dice 3CDE | 0.781 ± 0.019 (0.822) | 0.775 ± 0.023 (0.826) |
| Dice 4 | 0.608 ± 0.024 (0.601) | 0.544 ± 0.036 (0.659) |

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

### S02_yolov8s_seg_640 vs the article's YOLOv8-seg small @ 640

| Metric (test set) | Article, 6 runs mean ± SD | Ours, 5 fold models mean ± SD | Article, final model | Ours, selected fold model | Ours, 5-model ensemble |
|---|---|---|---|---|---|
| Fracture AUC | 0.985 ± 0.006 | 0.979 ± 0.007 | (other setting) | 0.980 | 0.994 |
| 6-class accuracy (%) | 82.0 ± 3.0 | 82.2 ± 2.1 | (other setting) | 85.7 | 83.8 |
| Fracture sensitivity | n/a | 0.939 ± 0.013 | (other setting) | 0.945 | 0.972 |
| Fracture specificity | n/a | 0.972 ± 0.007 | (other setting) | 0.969 | 0.938 |
| Macro-F1 (article: from Fig. 3) | n/a | 0.758 ± 0.036 | (other setting) | 0.807 | 0.787 |
| Dice (article: definition not stated; ours: detected fractures only) | 0.780 ± 0.011 | 0.741 ± 0.002 | (other setting) | 0.744 | 0.752 |
| Recall NF No fracture | n/a | 0.972 ± 0.007 | (other setting) | 0.969 | 0.938 |
| Recall 2AB Neck Garden I-II | n/a | 0.412 ± 0.201 | (other setting) | 0.562 | 0.562 |
| Recall 2CD Neck Garden III-IV | n/a | 0.941 ± 0.037 | (other setting) | 0.976 | 0.976 |
| Recall 3AB Pertroch Evans I-II | n/a | 0.871 ± 0.087 | (other setting) | 0.912 | 0.941 |
| Recall 3CDE Pertroch Evans III-V | n/a | 0.648 ± 0.066 | (other setting) | 0.714 | 0.643 |
| Recall 4 Subtrochanteric | n/a | 0.617 ± 0.046 | (other setting) | 0.583 | 0.583 |
| Errors: detection | n/a | 10.6 ± 2.3 | (other setting) | 10 | 8 |
| Errors: location | n/a | 5.0 ± 2.3 | (other setting) | 4 | 5 |
| Errors: grade | n/a | 21.8 ± 4.0 | (other setting) | 16 | 21 |
| Errors: total | n/a | 37.4 ± 4.3 | (other setting) | 30 | 34 |

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

| Definition | S01_yolov8s_seg_1024 | S02_yolov8s_seg_640 |
|---|---|---|
| Detected fractures only, top instance (our main definition) | 0.741 ± 0.009 | 0.741 ± 0.002 |
| Same at Ultralytics' default threshold 0.25 | 0.744 ± 0.007 | 0.746 ± 0.003 |
| Same at 0.25, union of all instances ≥ 0.25 | 0.745 ± 0.007 | 0.747 ± 0.004 |
| All fractured femurs, missed = 0 | 0.693 ± 0.016 | 0.696 ± 0.009 |
| All 210 femurs, no-fracture femur = 1 if nothing called else 0 | 0.781 ± 0.011 | 0.782 ± 0.008 |
| Global (pooled pixels over fractured femurs) | 0.733 ± 0.013 | 0.732 ± 0.007 |
| Per-femur spread, detected only: mean ± 1.96 SD (compare the article's 0.56–0.98) | 0.52–0.99 (ensemble) | 0.51–0.99 (ensemble) |

## OOF versus test segmentation

| Run | Dice detected, OOF pooled | Dice detected, test models | Predicted / true area, OOF (median, IQR) | Predicted / true area, test models (median, IQR, pooled over models) |
|---|---|---|---|---|
| S01_yolov8s_seg_1024 | 0.800 | 0.741 ± 0.009 | 0.99 (0.84–1.22) | 1.27 (0.94–1.80) |
| S02_yolov8s_seg_640 | 0.798 | 0.741 ± 0.002 | 0.97 (0.80–1.21) | 1.29 (0.99–1.81) |

Median mask area in pixels by class: true masks of the training femurs and of the test femurs (the article says its test labels were reviewed by a committee), and the masks the models predict for called fractures (OOF; test pooled over the 5 fold models).

| Class | True, train | True, test | Test / train | Predicted OOF, S01_yolov8s_seg_1024 | Predicted test, S01_yolov8s_seg_1024 | Predicted OOF, S02_yolov8s_seg_640 | Predicted test, S02_yolov8s_seg_640 |
|---|---|---|---|---|---|---|---|
| 2AB Neck Garden I-II | 45,963 | 29,797 | 0.65 | 54,558 | 57,046 | 50,575 | 58,298 |
| 2CD Neck Garden III-IV | 75,612 | 45,844 | 0.61 | 75,926 | 76,820 | 74,079 | 74,244 |
| 3AB Pertroch Evans I-II | 85,540 | 88,794 | 1.04 | 93,944 | 96,513 | 97,424 | 99,157 |
| 3CDE Pertroch Evans III-V | 135,306 | 120,095 | 0.89 | 131,154 | 130,396 | 129,554 | 137,388 |
| 4 Subtrochanteric | 126,204 | 149,960 | 1.19 | 144,031 | 125,186 | 143,938 | 125,769 |
