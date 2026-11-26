# Hip fracture segmentation and grading on the HUSJA-UA dataset

Research code, documentation and results for a PhD project. The aim is a new method for **segmenting and grading proximal femoral fractures on hip x-rays**, built on the public HUSJA-UA femoral fracture dataset, targeting an SCI (JCR Q2/Q3) journal.

- **Base article:** González G, et al. *Classification and segmentation of hip fractures in x-rays: highlighting fracture regions for interpretable diagnosis.* Insights into Imaging 2025;16:86. [doi:10.1186/s13244-025-01958-y](https://doi.org/10.1186/s13244-025-01958-y)
- **Dataset:** [HUSJA-UA Femoral Fracture, IEEE DataPort](https://ieee-dataport.org/documents/husja-ua-femoral-fracture) (doi:10.21227/xny1-zz29, CC BY 4.0). On Kaggle: `utopianstar/dataset`. **The images are not stored in this repository.**

## Current status

| Phase | Status |
|---|---|
| Base article reading and dataset EDA | Done |
| Literature: who used the dataset, state of the art | Done |
| Ranking of directions D1–D9 and research plan | Done, **waiting for the choice of option** |
| Fixed 5-fold split of the training set | Done (`splits/folds_v1.csv`) |
| Baselines, method, experiments, manuscript | Not started |

Details: [`PROJECT_TRACKER.md`](PROJECT_TRACKER.md) (overall) and [`PROGRESS_LOG.md`](PROGRESS_LOG.md) (step by step).

## Repository layout

```
.
├── README.md               this file
├── PROJECT_TRACKER.md      overall tracker: phases, status, decisions, open questions
├── PROGRESS_LOG.md         step-by-step log, newest first
├── Analysis/               reports
│   ├── 01_Base_Article_and_EDA_Report.md
│   ├── 02_How_to_Use_the_Dataset.md        nine directions D1–D9
│   └── 03_Idea_Ranking_and_Research_Plan.md ranking, Options A/B, experiments, compute, timeline
├── research/               literature notes
├── EDA/                    EDA script, per-image statistics, figures
├── splits/                 fixed train folds (official val folder is empty; test is untouched)
├── src/femurseg/           Python package (class definitions, data paths; models and metrics come next)
├── scripts/                command-line tools (make_splits.py; training and evaluation scripts come next)
├── configs/                one YAML file per experiment
├── kaggle/                 Kaggle launcher notebook and Kaggle-only scripts
├── results/                metrics and small outputs of every run (no model weights)
└── manuscript/             paper drafts, tables and figures (later)
```

Not in git (see `.gitignore`): the `Femoral Fractures/` data folder, PDFs, model weights and Kaggle run folders.

## Workflow: local folder ↔ GitHub ↔ Kaggle

```
 Local folder (E:\PhD\...\Segmentation Project)  <── same files ──>  GitHub repo
        holds the data + this repo's files                                 │
                                                                           │ git clone (in a Kaggle notebook)
                                                                           v
                                         Kaggle "Save Version" run (GPU, background)
                                                                           │
                                         results in /kaggle/working ───────┘ copied into results/<run-id>/
```

1. **Code and documents** are written here and in the local folder, and kept identical.
2. **Training** runs on Kaggle: a small launcher notebook clones this repository, installs what is missing and runs one script. You start it with **Save Version → Save & Run All (Commit)**, so it runs in the background.
3. **Results** (metrics JSON/CSV, logs, small figures) come back into `results/<run-id>/` and are logged in `results/EXPERIMENTS.md`. Weights stay in Kaggle outputs.

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
1. Create a new Kaggle notebook, **attach the dataset** `utopianstar/dataset`, set **Accelerator = GPU** and **Internet = On** (Internet needs a phone-verified Kaggle account).
2. Import `kaggle/launcher.ipynb` (File → Import Notebook) and set `RUN` in the first cell (for example `env_check`).
3. Click **Save Version → Save & Run All (Commit)**. When it finishes, download the output and copy the run folder into `results/`.

## Ground rules for every experiment
1. The **official test set (210 femurs) is frozen.** It is never used for tuning.
2. Validation comes from the **5 fixed folds** in `splits/folds_v1.csv` (grouped by x-ray, stratified by class). Fold 0 is the development fold.
3. Every run records the git commit, config, seed or fold, GPU, runtime and metrics.
4. Primary endpoints are fixed in advance: 6-class macro-F1 and mean Dice on fractured femurs (Report 03, Section 7.5).

## Citation of the data and base work
If you use this code, also cite the base article and the dataset:
González G, Galant J, Salinas JM, et al. Classification and segmentation of hip fractures in x-rays: highlighting fracture regions for interpretable diagnosis. *Insights Imaging* 16, 86 (2025). https://doi.org/10.1186/s13244-025-01958-y
