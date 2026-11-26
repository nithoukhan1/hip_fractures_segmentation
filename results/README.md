# results/

Small, versioned outputs of every Kaggle run: metrics, per-case predictions and logs. **No model weights** (they stay in the Kaggle version outputs, which are kept by Kaggle).

## Layout
```
results/
├── EXPERIMENTS.md            one line per run: the experiment register
└── <RUN>/                    unzipped from results_<RUN>.zip
    ├── run_meta.json         git commit, command, start time, runtime, exit code
    ├── config.yaml           exact settings (training runs)
    ├── metrics.json          summary metrics (validation, out-of-fold, test when final)
    ├── per_case_*.csv        one row per femur: prediction, probability, Dice, etc.
    └── log.txt / *.png       training log and small figures
```

## Rules
1. A run folder is never edited after it is added. A re-run gets a new name, for example `B1_yolov8s_seg_1024_r2`.
2. Every number in the manuscript must come from a file in this folder.
3. Test-set metrics appear only in runs marked `final` (frozen configurations).
