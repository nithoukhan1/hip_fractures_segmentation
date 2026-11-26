# Experiment register

Every Kaggle run gets one line in the table at the **bottom** of this file. `scripts/register_run.py` appends the line automatically when a run is filed from `results/_inbox/`.
- **Track:** Shared (S runs), Paper A (A runs, SLG-Net), Paper B (R runs, RefFrac).
- **Metrics** are out-of-fold or validation, unless the run is a frozen *final* run or a fixed baseline (then test-set results are included).
- **Analyses** that combine runs (intervals, paired tests, per-class breakdowns) are in `results/analysis/<name>/`: C01 = S01 vs S02.
- Details of each run are in `results/<RUN>/NOTES.md`.

## Planned next runs (from Analysis/04 §7)
| RUN | Track | Step | What | Expected GPU time |
|---|---|---|---|---|
| ~~S00_env_check~~ | Shared | 0 | Done 2026-11-26 (register line 1) | ~4 min |
| ~~S00_env_check_r2~~ | Shared | 0 | Done 2026-11-26 (register line 2): data, folds and file list identical to `splits/folds_v1.csv` | ~4 min |
| ~~S01_yolov8s_seg_1024~~ | Shared | 1 | Done 2026-11-26 (register line 3): reproduces the article's accuracy and AUC; analysis in `results/analysis/C01_yolo_1024_vs_640/` | 2.9 h measured |
| ~~S02_yolov8s_seg_640~~ | Shared | 1 | Done 2026-11-26 (register line 4): same as S01 within noise; the article's Dice gain at 640 px is not reproduced | 1.35 h measured |
| S03_yolo11s_seg_1024 | Shared | 2 | Baseline 3: YOLO11s-seg @1024 with exactly S01's protocol (ready in `kaggle/runs.json`) | ~3 h (one launch) |
| S04–S06 | Shared | 2 | Classifiers DenseNet121, EfficientNet-B4, ConvNeXt-T (S06 = A0), 768 × 512 input | about 1.5–2 h each (estimate) |
| S07–S09 | Shared | 2 | Segmenters U-Net, DeepLabV3+, SegFormer, 768 × 512 input | about 1.5–2 h each (estimate) |
| M0 | Shared | 2 | ConvNeXt-T + U-Net decoder + flat 6-class head (= A1 in Paper A = R0 in Paper B) | about 2 h (estimate) |

## Register
| # | Date | RUN | Track | Step | Git commit | Kaggle version | GPU | Runtime (min) | Exit | Key result | Purpose / decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-26 | S00_env_check | Shared | 0 | 37d8d0f | not given | Tesla T4 x2 | 3.49 | 0 | same data: False; same folds: False; val empty: False (Kaggle's val/ holds 104 of the 1,043 training femurs; the other 1,149 files are pixel- and label-identical); fp16 TFLOPs: 24.0, 22.6 | Step 0: check Kaggle GPU and packages, and that Kaggle's data and folds match the local copy. Decision: pool val/ into train in all code; confirm with S00_env_check_r2 |
| 2 | 2026-11-26 | S00_env_check_r2 | Shared | 0 | 2758d45 | not given | Tesla T4 x2 | 3.7 | 0 | same data: True; same folds: True; val empty: False; val pooled into train: 104; all files found: True; fp16 TFLOPs: 23.7, 23.5 | Confirm Kaggle data and folds after pooling val/ into train |
| 3 | 2026-11-26 | S01_yolov8s_seg_1024 | Shared | 1 | 29e1852 | not given | Tesla T4 x2 | 176.1 | 0 | OOF: AUC 0.975, 6-class acc 0.832, macro-F1 0.744, Dice fractured 0.740, Garden I-II recall 0.509; test ensemble: AUC 0.994, 6-class acc 0.881, macro-F1 0.845, Dice fractured 0.725, Garden I-II recall 0.500; threshold (OOF Youden) 0.127 | Baseline 1: the paper's final model, YOLOv8s-seg @1024, retrained on our 5 folds (paper protocol, 100 epochs, final weights). Decision: reference YOLO baseline for both papers; report Dice per location, because the test neck masks are 35–40% smaller than the training ones (C01) |
| 4 | 2026-11-26 | S02_yolov8s_seg_640 | Shared | 1 | 29e1852 | not given | Tesla T4 x2 | 81.12 | 0 | OOF: AUC 0.975, 6-class acc 0.826, macro-F1 0.751, Dice fractured 0.746, Garden I-II recall 0.544; test ensemble: AUC 0.994, 6-class acc 0.838, macro-F1 0.787, Dice fractured 0.731, Garden I-II recall 0.562; threshold (OOF Youden) 0.045 | Baseline 2: the paper's best-Dice setting, YOLOv8s-seg @640, same protocol. Decision: no significant difference from S01 after Holm (C01); S03 (YOLO11) runs at 1024 px |
