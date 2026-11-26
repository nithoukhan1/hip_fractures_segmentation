# Experiment register

Every Kaggle run gets one line in the table at the **bottom** of this file. `scripts/register_run.py` appends the line automatically when a run is filed from `results/_inbox/`.
- **Track:** Shared (S runs), Paper A (A runs, SLG-Net), Paper B (R runs, RefFrac).
- **Metrics** are out-of-fold or validation, unless the run is a frozen *final* run (then test-set results are included).
- Details of each run are in `results/<RUN>/NOTES.md`.

## Planned next runs (from Analysis/04 §7)
| RUN | Track | Step | What | Expected GPU time |
|---|---|---|---|---|
| ~~S00_env_check~~ | Shared | 0 | Done 2026-11-26 (register line 1) | ~4 min |
| ~~S00_env_check_r2~~ | Shared | 0 | Done 2026-11-26 (register line 2): data, folds and file list identical to `splits/folds_v1.csv` | ~4 min |
| S01_yolov8s_seg_1024 | Shared | 1 | Baseline 1: the paper's final YOLOv8s-seg @1024, retrained on our 5 folds (two folds at a time on 2 × T4) | ~5–6 h (one launch) |
| S02_yolov8s_seg_640 | Shared | 1 | Baseline 2: the paper's best-Dice setting, YOLOv8s-seg @640, 5 folds | ~2–3 h (one launch) |

## Register
| # | Date | RUN | Track | Step | Git commit | Kaggle version | GPU | Runtime (min) | Exit | Key result | Purpose / decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-26 | S00_env_check | Shared | 0 | 37d8d0f | not given | Tesla T4 x2 | 3.49 | 0 | same data: False; same folds: False; val empty: False (Kaggle's val/ holds 104 of the 1,043 training femurs; the other 1,149 files are pixel- and label-identical); fp16 TFLOPs: 24.0, 22.6 | Step 0: check Kaggle GPU and packages, and that Kaggle's data and folds match the local copy. Decision: pool val/ into train in all code; confirm with S00_env_check_r2 |
| 2 | 2026-11-26 | S00_env_check_r2 | Shared | 0 | 2758d45 | not given | Tesla T4 x2 | 3.7 | 0 | same data: True; same folds: True; val empty: False; val pooled into train: 104; all files found: True; fp16 TFLOPs: 23.7, 23.5 | Confirm Kaggle data and folds after pooling val/ into train |
