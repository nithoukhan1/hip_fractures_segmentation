# Experiment register

Every Kaggle run gets one line in the table at the **bottom** of this file. `scripts/register_run.py` appends the line automatically when a run is filed from `results/_inbox/`.
- **Track:** Shared (S runs), Paper A (A runs, SLG-Net), Paper B (R runs, RefFrac).
- **Metrics** are out-of-fold or validation, unless the run is a frozen *final* run (then test-set results are included).
- Details of each run are in `results/<RUN>/NOTES.md`.

## Planned next runs (from Analysis/04 §7)
| RUN | Track | Step | What | Expected GPU time |
|---|---|---|---|---|
| S00_env_check | Shared | 0 | GPU, packages, data location, val empty, Kaggle data and folds identical to `splits/folds_v1.csv` | ~3 min |
| S01_yolov8s_seg_1024 | Shared | 1 | Reproduce the paper's final YOLOv8s-seg @1024, 5 folds | ~8–10 h (2–3 launches) |
| S02_yolov8n_seg_640 | Shared | 1 | Paper's best-Dice setting, 5 folds | ~3 h |

## Register
| # | Date | RUN | Track | Step | Git commit | Kaggle version | GPU | Runtime (min) | Exit | Key result | Purpose / decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
