# N_segzoom_fold0: summary

Network `zoom:convnext_tiny.fb_in22k_ft_in1k`, input 1536 x 1024, 60 epochs, batch 16 (micro-batch 8 x 2), lr 0.0002, final weights. Segment-and-zoom: crop 640 x 640 input pixels around the predicted mask, resampled to 512 x 512; teacher forcing of the crop centre over 30 epochs.
Folds completed [0], skipped [], failed []. Runtime 72.5 min on Tesla T4.

## Fracture threshold fitted on the OOF predictions (Youden): 0.0005

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.991 |
| Fracture sensitivity | 0.985 |
| Fracture specificity | 0.972 |
| 6-class accuracy (%) | 88.5 |
| 6-class macro-F1 | 0.816 |
| Garden I-II recall | 0.636 |
| Dice, fractured femurs (missed = 0) | 0.763 |
| Dice, detected fractures only | 0.774 |
| NSD at 10 px, fractured femurs | 0.234 |

## Plain threshold 0.5 on the fracture probability

| Metric | OOF (n=208) |
|---|---|
| Fracture AUC | 0.991 |
| Fracture sensitivity | 0.949 |
| Fracture specificity | 0.986 |
| 6-class accuracy (%) | 87.5 |
| 6-class macro-F1 | 0.798 |
| Garden I-II recall | 0.636 |
| Dice, fractured femurs (missed = 0) | 0.747 |
| Dice, detected fractures only | 0.788 |
| NSD at 10 px, fractured femurs | 0.229 |

Pointing game of the zoom crop (fractured femurs: the window holds the centroid of the released mask): OOF 0.993 (mean share of the mask inside 0.986, n = 136).

Development run (`--no-test`): the held-out folds only; the test set was not predicted.
