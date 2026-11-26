# results/

Small, versioned outputs of every Kaggle run: metrics, per-case predictions, logs and run notes. **No model weights** (they stay in the Kaggle version outputs, which Kaggle keeps).

## Layout
```
results/
├── EXPERIMENTS.md            the experiment register: one line per run (planned runs at the top)
├── RUN_NOTES_TEMPLATE.md     template for each run's NOTES.md
├── _inbox/                   drop results_<RUN>.zip files from Kaggle here (zips are not committed)
│   └── filed/                zips that were already filed
├── analysis/<name>/          95% intervals and paired comparisons computed from filed runs by
│                             scripts/run_stats.py (stats.json, stats.md); recomputable, never edits a run folder
└── <RUN>/                    one folder per run, filed from its zip
    ├── run_meta.json         git commit, command, start time, runtime, exit code
    ├── NOTES.md              why the run was made, what changed, result, what we learnt, decision
    ├── config.json           exact settings, git commit, start-weights checksum (training runs)
    ├── metrics.json          summary metrics (out-of-fold, per fold, test when frozen), thresholds, folds done
    ├── summary.md            headline table, next to the base article's numbers where they exist
    ├── per_case_*.csv        one row per femur: per-class confidences, prediction, Dice, HD95, NSD
    └── folds/fold<k>/        per-fold predictions, training curve, Ultralytics settings, log tail
```

## Run names
- **S** = shared runs (environment check, baselines S1–S9, core model M0)
- **A** = Paper A (SLG-Net)
- **R** = Paper B (RefFrac)

Example: `S01_yolov8s_seg_1024`. The plan for every run is in `Analysis/04_Execution_Plan_A_and_B.md`.

## How a finished run gets here
1. On Kaggle: open the finished version → *Output* → download `results_<RUN>.zip`.
2. Put the zip in `results/_inbox/` in the local folder, and tell Claude in the project thread.
3. Filing (Claude does it, or run it yourself from the project folder):
   ```
   python scripts/register_run.py results/_inbox/results_<RUN>.zip --version <Kaggle version number> --purpose "<one line why>"
   ```
   This copies the run into `results/<RUN>/`, creates `NOTES.md` from the template, appends a line to `EXPERIMENTS.md` and moves the zip to `_inbox/filed/`.
4. `NOTES.md` is filled in (5–10 lines), and the same files are pushed to the repository, so the local folder and GitHub stay identical.

## Rules
1. A run folder is never edited after it is filed, except its `NOTES.md`. A re-run gets a new name, for example `S01_yolov8s_seg_1024_r2`.
2. Every number in a manuscript must come from a file in this folder.
3. Test-set metrics appear only for frozen configurations: fixed baselines such as S01 and S02, and runs marked `final`.
4. Failed runs are filed too (exit code ≠ 0), with a note on what went wrong. They show what was tried.
