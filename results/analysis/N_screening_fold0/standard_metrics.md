# Standard metrics of every model

Classification: accuracy, macro precision, macro recall and macro-F1 over the 6 classes, fracture sensitivity and specificity, fracture AUC and the one-vs-rest macro AUC of the 6 classes. Segmentation (fractured femurs): Dice and IoU, where a missed fracture counts 0, and HD95 in pixels of the 1024 × 1536 image (median over the fractures the model calls; the number missed is shown beside it). 95% intervals in brackets: bootstrap over whole x-rays (2000 resamples) for accuracy, macro-F1 and Dice, DeLong for the fracture AUC. Each run uses its own out-of-fold fracture threshold. Script: `scripts/run_standard_metrics.py`.

**Only the held-out femurs of fold(s) 0**, as in the fold-0 screening (Analysis/11 section 7): each run's fracture threshold is fitted again on exactly these femurs, and the test set is not used.

## Out of fold, fold(s) 0 (208 femurs)

### Baselines: classification

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Sensitivity | Specificity | Fracture AUC | 6-class AUC |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt U-Net | 184/208 = 0.885 (0.837–0.927) | 0.853 | 0.809 | 0.822 (0.735–0.889) | 0.985 | 0.972 | 0.996 (0.990–1.000) | 0.963 |
| ConvNeXt U-Net, second run | 184/208 = 0.885 (0.839–0.925) | 0.907 | 0.770 | 0.811 (0.706–0.891) | 0.956 | 1.000 | 0.994 (0.987–1.000) | 0.949 |

### Baselines: segmentation

| Model | Dice | IoU | HD95, px: median (IQR) | Fractures missed |
|---|---|---|---|---|
| ConvNeXt U-Net | 0.749 (0.710–0.783) | 0.633 | 61 (47–90) | 2 of 136 |
| ConvNeXt U-Net, second run | 0.761 (0.725–0.791) | 0.643 | 60 (47–89) | 6 of 136 |

### New method candidates (fold-0 screening): classification

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Sensitivity | Specificity | Fracture AUC | 6-class AUC |
|---|---|---|---|---|---|---|---|---|
| Segment-and-zoom network, seed 2026 | 184/208 = 0.885 (0.839–0.928) | 0.864 | 0.797 | 0.816 (0.717–0.889) | 0.985 | 0.972 | 0.991 (0.979–1.000) | 0.958 |
| Segment-and-zoom network, seed 2027 | 186/208 = 0.894 (0.850–0.935) | 0.865 | 0.832 | 0.843 (0.766–0.906) | 0.985 | 0.986 | 0.997 (0.993–1.000) | 0.961 |
| ConvNeXt U-Net at 1152 × 768, seed 2026 | 183/208 = 0.880 (0.834–0.922) | 0.849 | 0.764 | 0.783 (0.673–0.860) | 0.978 | 0.986 | 0.997 (0.991–1.000) | 0.973 |
| ConvNeXt U-Net at 1152 × 768, seed 2027 | 181/208 = 0.870 (0.822–0.915) | 0.840 | 0.772 | 0.790 (0.687–0.864) | 0.993 | 1.000 | 0.999 (0.996–1.000) | 0.953 |
| ConvNeXt U-Net with ConvNeXt-S, seed 2026 | 186/208 = 0.894 (0.850–0.936) | 0.890 | 0.781 | 0.814 (0.706–0.888) | 0.978 | 0.986 | 0.996 (0.989–1.000) | 0.965 |
| ConvNeXt U-Net with ConvNeXt-S, seed 2027 | 180/208 = 0.865 (0.821–0.910) | 0.828 | 0.769 | 0.788 (0.686–0.863) | 0.993 | 0.986 | 0.998 (0.996–1.000) | 0.967 |

### New method candidates (fold-0 screening): segmentation

| Model | Dice | IoU | HD95, px: median (IQR) | Fractures missed |
|---|---|---|---|---|
| Segment-and-zoom network, seed 2026 | 0.763 (0.729–0.791) | 0.642 | 61 (49–89) | 2 of 136 |
| Segment-and-zoom network, seed 2027 | 0.763 (0.731–0.793) | 0.645 | 62 (47–85) | 2 of 136 |
| ConvNeXt U-Net at 1152 × 768, seed 2026 | 0.768 (0.739–0.794) | 0.645 | 62 (48–86) | 3 of 136 |
| ConvNeXt U-Net at 1152 × 768, seed 2027 | 0.779 (0.755–0.800) | 0.655 | 63 (47–91) | 1 of 136 |
| ConvNeXt U-Net with ConvNeXt-S, seed 2026 | 0.759 (0.725–0.790) | 0.641 | 64 (47–87) | 3 of 136 |
| ConvNeXt U-Net with ConvNeXt-S, seed 2027 | 0.761 (0.728–0.790) | 0.641 | 60 (45–90) | 1 of 136 |

### Per class: precision / recall / F1 (out of fold, fold(s) 0)

| Model | No fracture | Neck Garden I-II | Neck Garden III-IV | Pertroch Evans I-II | Pertroch Evans III-V | Subtrochanteric |
|---|---|---|---|---|---|---|
| ConvNeXt U-Net | 0.97 / 0.97 / 0.97 | 0.60 / 0.55 / 0.57 | 0.90 / 0.92 / 0.91 | 0.68 / 0.93 / 0.78 | 0.97 / 0.78 / 0.86 | 1.00 / 0.71 / 0.83 |
| ConvNeXt U-Net, second run | 0.92 / 1.00 / 0.96 | 1.00 / 0.64 / 0.78 | 0.87 / 0.94 / 0.91 | 0.73 / 0.81 / 0.77 | 0.91 / 0.80 / 0.85 | 1.00 / 0.43 / 0.60 |
| Segment-and-zoom network, seed 2026 | 0.97 / 0.97 / 0.97 | 0.70 / 0.64 / 0.67 | 0.91 / 0.96 / 0.93 | 0.67 / 0.89 / 0.76 | 0.94 / 0.75 / 0.83 | 1.00 / 0.57 / 0.73 |
| Segment-and-zoom network, seed 2027 | 0.97 / 0.99 / 0.98 | 0.67 / 0.55 / 0.60 | 0.92 / 0.94 / 0.93 | 0.69 / 0.89 / 0.77 | 0.94 / 0.78 / 0.85 | 1.00 / 0.86 / 0.92 |
| ConvNeXt U-Net at 1152 × 768, seed 2026 | 0.96 / 0.99 / 0.97 | 0.60 / 0.55 / 0.57 | 0.92 / 0.92 / 0.92 | 0.68 / 0.93 / 0.78 | 0.94 / 0.78 / 0.85 | 1.00 / 0.43 / 0.60 |
| ConvNeXt U-Net at 1152 × 768, seed 2027 | 0.99 / 1.00 / 0.99 | 0.60 / 0.55 / 0.57 | 0.90 / 0.90 / 0.90 | 0.62 / 0.89 / 0.73 | 0.94 / 0.72 / 0.82 | 1.00 / 0.57 / 0.73 |
| ConvNeXt U-Net with ConvNeXt-S, seed 2026 | 0.96 / 0.99 / 0.97 | 0.83 / 0.45 / 0.59 | 0.88 / 0.96 / 0.92 | 0.73 / 0.89 / 0.80 | 0.94 / 0.82 / 0.88 | 1.00 / 0.57 / 0.73 |
| ConvNeXt U-Net with ConvNeXt-S, seed 2027 | 0.99 / 0.99 / 0.99 | 0.58 / 0.64 / 0.61 | 0.86 / 0.94 / 0.90 | 0.68 / 0.70 / 0.69 | 0.86 / 0.78 / 0.82 | 1.00 / 0.57 / 0.73 |

## Confusion matrices

Rows = true class, columns = predicted class, in the order No fracture, Neck Garden I-II, Neck Garden III-IV, Pertroch Evans I-II, Pertroch Evans III-V, Subtrochanteric. All matrices are in `standard_metrics.json`.

**ConvNeXt U-Net, out of fold, fold(s) 0**

| True \ predicted | No fracture | Neck Garden I-II | Neck Garden III-IV | Pertroch Evans I-II | Pertroch Evans III-V | Subtrochanteric |
|---|---|---|---|---|---|---|
| No fracture | 70 | 2 | 0 | 0 | 0 | 0 |
| Neck Garden I-II | 0 | 6 | 2 | 3 | 0 | 0 |
| Neck Garden III-IV | 1 | 2 | 47 | 1 | 0 | 0 |
| Pertroch Evans I-II | 1 | 0 | 0 | 25 | 1 | 0 |
| Pertroch Evans III-V | 0 | 0 | 3 | 6 | 31 | 0 |
| Subtrochanteric | 0 | 0 | 0 | 2 | 0 | 5 |

