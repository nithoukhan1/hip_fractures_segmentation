# Hip fracture segmentation and grading on the HUSJA-UA dataset

Research code, documentation and results for a PhD project. The aim is new methods for **segmenting and grading proximal femoral fractures on hip x-rays**, built on the public HUSJA-UA femoral fracture dataset, targeting SCI (JCR Q2/Q3) journals. Two papers are planned in parallel:
- **Paper A, SLG-Net (Segment, Locate, Grade):** the fracture region drives a location → grade decision.
- **Paper B, RefFrac:** a healthy reference femur helps find and outline subtle fractures.

Sources:
- **Base article:** González G, et al. *Classification and segmentation of hip fractures in x-rays: highlighting fracture regions for interpretable diagnosis.* Insights into Imaging 2025;16:86. [doi:10.1186/s13244-025-01958-y](https://doi.org/10.1186/s13244-025-01958-y)
- **Dataset:** [HUSJA-UA Femoral Fracture, IEEE DataPort](https://ieee-dataport.org/documents/husja-ua-femoral-fracture) (doi:10.21227/xny1-zz29, CC BY 4.0). On Kaggle: `utopianstar/dataset`. **The images are not stored in this repository.**

## Current status

| Phase | Status |
|---|---|
| Base article reading and dataset EDA | Done |
| Literature: who used the dataset, state of the art | Done |
| Ranking of directions D1–D9 and research plan | Done (Report 03) |
| Decision: Papers A and B in parallel; execution plan | Done (Report 04) |
| Fixed 5-fold split of the training set | Done (`splits/folds_v1.csv`) |
| Kaggle environment check | Done: data and folds on Kaggle are identical to `splits/folds_v1.csv` (Kaggle's val/ pooled into train) |
| Data loader, metrics, statistics, tests | Done (`src/femurseg/data.py`, `metrics.py`, `stats.py`, `tests/`) |
| Shared baselines S1–S9 and core model M0 | In progress: YOLO baselines S01 and S02 ready to run on Kaggle |
| Paper A and Paper B methods, experiments, manuscripts | Not started |

Details: [`PROJECT_TRACKER.md`](PROJECT_TRACKER.md) (overall) and [`PROGRESS_LOG.md`](PROGRESS_LOG.md) (step by step).

## Repository layout

```
.
├── README.md               this file
├── PROJECT_TRACKER.md      overall tracker: phases per track, milestones, decisions, open questions
├── PROGRESS_LOG.md         step-by-step log, newest first
├── Analysis/               reports
│   ├── 01_Base_Article_and_EDA_Report.md
│   ├── 02_How_to_Use_the_Dataset.md        nine directions D1–D9
│   ├── 03_Idea_Ranking_and_Research_Plan.md ranking, Option A and B designs
│   └── 04_Execution_Plan_A_and_B.md         A and B in parallel: runs, validation, compute, timeline
├── research/               literature notes
├── EDA/                    EDA script, per-image statistics, figures
├── splits/                 fixed train folds (official val folder is empty; test is untouched)
├── src/femurseg/           Python package: classes, data paths, data loader (data.py), metrics (metrics.py),
│                           confidence intervals and paired tests (stats.py)
├── scripts/                command-line tools: make_splits.py, register_run.py, train_yolo_seg.py (S01/S02),
│                           run_stats.py (95% intervals and paired comparisons of filed runs)
├── tests/                  checks of metrics, statistics and folds, and a CPU smoke test of the YOLO script
├── configs/                one YAML file per experiment
├── kaggle/                 Kaggle launcher notebook, run list and Kaggle-only scripts
├── results/                every run's metrics, notes and small outputs (no model weights)
│   ├── EXPERIMENTS.md      the experiment register: one line per run
│   └── _inbox/             drop zips downloaded from Kaggle here
└── manuscript/             paper drafts, tables and figures (later)
```

Not in git (see `.gitignore`): the `Femoral Fractures/` data folder, PDFs, model weights, zips and Kaggle run folders.

## Workflow: local folder ↔ GitHub ↔ Kaggle

```
 Local folder (E:\PhD\...\Segmentation Project)  <── same files ──>  GitHub repo (branch main)
        holds the data + this repo's files                                 │
              ^                                                            │ git clone (in a Kaggle notebook)
              │ results_<RUN>.zip → results/_inbox/                        v
              └──────────────────────────────────── Kaggle "Save Version" run (GPU, background)
```

1. **Code and documents** are written here and in the local folder, and kept identical.
2. **Training** runs on Kaggle: one launcher notebook clones this repository and runs one named run from `kaggle/runs.json`. You start it with **Save Version → Save & Run All (Commit)**, so it runs in the background.
3. **Results:** download `results_<RUN>.zip` from the version's Output tab into `results/_inbox/`. `scripts/register_run.py` files it into `results/<RUN>/`, adds a `NOTES.md` and a line in `results/EXPERIMENTS.md`. Weights stay in Kaggle outputs.

### Optional: make the local folder a git working copy (one time, Windows PowerShell)
This lets you pull updates and push your own edits without copying files by hand. It keeps your data folder untouched, because the data is in `.gitignore`.
```powershell
cd "E:\PhD\Admitted\Research\Project 3\Segmentation Project"
git clone --no-checkout https://github.com/nithoukhan1/hip_fractures_segmentation.git _tmp_repo
Move-Item _tmp_repo\.git .git
Remove-Item _tmp_repo
git reset            # adopts the repository history and keeps your files as they are
git status           # should list no differences, or only files you changed
```
After that: `git pull` to get updates, and `git add`, `git commit` and `git push` to send changes.

## Running on Kaggle (Save Version)

See [`kaggle/README.md`](kaggle/README.md). In short:
1. Create a new Kaggle notebook, **import** `kaggle/launcher.ipynb` (File → Import Notebook), **attach the dataset** `utopianstar/dataset`, set **Accelerator = GPU** and **Internet = On** (Internet needs a phone-verified Kaggle account).
2. In the settings cell set `RUN` (for example `S00_env_check`) and `BRANCH = "main"`.
3. Click **Save Version → Save & Run All (Commit)**. When it finishes, download `results_<RUN>.zip` from the Output tab into `results/_inbox/`.

## Ground rules for every experiment
1. The **official test set (210 femurs) is frozen.** It is never used for tuning, and only frozen configurations are evaluated on it.
2. Validation comes from the **5 fixed folds** in `splits/folds_v1.csv` (grouped by x-ray, stratified by class). The Kaggle copy's `val/` folder holds 104 of the training femurs; all code pools it back into train. Every configuration is trained 5 times; each training femur is predicted once by the model that did not train on it (out-of-fold).
3. Fixed training length, final weights: the held-out fold is never used to pick an epoch. Thresholds and calibration are fitted on out-of-fold predictions only. Development uses fold 0 only.
4. Every run records the git commit, config, seed or fold, GPU, runtime and metrics, and gets a line in `results/EXPERIMENTS.md`.
5. Primary endpoints are fixed in advance:
   - Paper A: 6-class macro-F1 and mean Dice on fractured femurs.
   - Paper B: mean Dice on fractured femurs and recall of Garden I–II fractures.

## Citation of the data and base work
If you use this code, also cite the base article and the dataset:
González G, Galant J, Salinas JM, et al. Classification and segmentation of hip fractures in x-rays: highlighting fracture regions for interpretable diagnosis. *Insights Imaging* 16, 86 (2025). https://doi.org/10.1186/s13244-025-01958-y
