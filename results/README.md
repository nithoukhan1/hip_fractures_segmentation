# results/

Small, versioned outputs of every Kaggle run: metrics, per-case predictions, logs and run notes. **No model weights** (they stay in the Kaggle version outputs, which Kaggle keeps).

## Layout
```
results/
├── EXPERIMENTS.md            the experiment register: one line per run (planned runs at the top)
├── RUN_NOTES_TEMPLATE.md     template for each run's NOTES.md
├── _inbox/                   drop results_<RUN>.zip files from Kaggle here (zips are not committed)
│   └── filed/                zips that were already filed
└── <RUN>/                    one folder per run, filed from its zip
    ├── run_meta.json         git commit, command, start time, runtime, exit code
    ├── NOTES.md              why the run was made, what changed, result, what we learnt, decision
    ├── config.yaml           exact settings (training runs)
    ├── metrics.json          summary metrics (validation, out-of-fold, test when final)
    ├── per_case_*.csv        one row per femur: prediction, probability, Dice, etc.
    └── log.txt / *.png       training log and small figures
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
3. Test-set metrics appear only in runs marked `final` (frozen configurations).
4. Failed runs are filed too (exit code ≠ 0), with a note on what went wrong. They show what was tried.
