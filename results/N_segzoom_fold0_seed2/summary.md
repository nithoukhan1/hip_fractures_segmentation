# N_segzoom_fold0_seed2: summary

Network `zoom:convnext_tiny.fb_in22k_ft_in1k`, input 1536 x 1024, 60 epochs, batch 16 (micro-batch 8 x 2), lr 0.0002, final weights. Segment-and-zoom: crop 640 x 640 input pixels around the predicted mask, resampled to 512 x 512; teacher forcing of the crop centre over 30 epochs.
Folds completed [0], skipped [], failed []. Runtime 75.9 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0030

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.997 |
| Fracture sensitivity | 0.985 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 89.4 |
| 6-class macro-F1 | 0.843 |
| Garden I-II recall | 0.545 |
| Dice, fractured femurs (missed = 0) | 0.763 |
| Dice, detected fractures only | 0.774 |
| NSD at 10 px, fractured femurs | 0.233 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.997 |
| Fracture sensitivity | 0.926 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 86.5 |
| 6-class macro-F1 | 0.795 |
| Garden I-II recall | 0.545 |
| Dice, fractured femurs (missed = 0) | 0.747 |
| Dice, detected fractures only | 0.806 |
| NSD at 10 px, fractured femurs | 0.228 |

Pointing game of the zoom crop (fractured femurs: the window holds the centroid of the released mask): OOF 0.993 (mean share of the mask inside 0.990, n = 136).

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
