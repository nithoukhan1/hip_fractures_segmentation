# kaggle/: running experiments on Kaggle with "Save Version"

## Files
| File | What it is |
|---|---|
| `launcher.ipynb` | The only notebook you need. Import it into Kaggle once, then change `RUN` for each run. |
| `runs.json` | The list of runs: script, arguments, extra packages, expected time. New runs are added here, not in the notebook. |
| `run.py` | Starts one run from `runs.json`, writes `run_meta.json` (git commit, command, runtime, exit code). |
| `env_check.py` | Step 0 (`S00_env_check`): checks GPU, packages, internet and data, and confirms the Kaggle data and folds match `splits/folds_v1.csv`. |

## One-time set-up (about 5 minutes)
1. On Kaggle: **Create → New Notebook**, then **File → Import Notebook** and upload `kaggle/launcher.ipynb` from this repository.
2. **Add Input** (right panel) → search for and attach the dataset **utopianstar/dataset**.
3. **Settings** (right panel):
   - **Accelerator:** GPU (T4 x2 or P100).
   - **Internet:** On. This needs a phone-verified Kaggle account.
4. Rename the notebook, for example `hipfx-launcher`.

## Every run (step by step)
1. In the settings cell, set `RUN` to a name from `runs.json` (first time: `S00_env_check`). Keep `BRANCH = "main"` (before the setup pull request is merged, use `claude/project-thread-kges8j`), or use a tag such as `A-v1-frozen` for final runs.
2. Click **Save Version** → choose **Save & Run All (Commit)** → **Save**. Kaggle runs the whole notebook in the background (up to 12 hours) and you can close the browser.
3. When the version shows *Complete*, open it → **Output** → download `results_<RUN>.zip`.
4. Put the zip in `results/_inbox/` in your local folder and tell Claude, or run `python scripts/register_run.py results/_inbox/results_<RUN>.zip --version <n>`. It files the run into `results/<RUN>/`, adds a notes stub and a line to `results/EXPERIMENTS.md`. Then the same files go to the repository.

## Runs available now
| RUN | Step | GPU time | What it does |
|---|---|---|---|
| `S00_env_check` | 0 | ~4 min | Hardware and package report, dataset location and counts, val-empty check, split identity check, GPU speed test (done 2026-11-26: found Kaggle's val/ folder holds 104 training femurs) |
| `S00_env_check_r2` | 0 | ~4 min | Same check after pooling val/ back into train: data, folds and file list must now be identical to `splits/folds_v1.csv` |

**About Kaggle's val/ folder:** the Kaggle copy moved 104 of the 1,043 training femurs into `val/`, without keeping both femurs of an x-ray together. All code pools `val/` back into train (`src/femurseg/paths.py`), so Kaggle and the local copy give the same 1,043 training femurs and the same 5 folds.

Run names: **S** = shared (baselines S1–S9, core model M0), **A** = Paper A (SLG-Net), **R** = Paper B (RefFrac). Shared baselines are added in Step 1 (`Analysis/04_Execution_Plan_A_and_B.md` §7).

## Good habits that save quota
- Kaggle's free GPU quota is commonly reported as about 30 h per week, with 12 h per session. Your exact remaining quota is shown in your Kaggle account settings.
- One Save Version run can chain several folds. The training scripts (from Step 1) will save a checkpoint every epoch, so a crashed run can resume.
- Use fold 0 for development. Run all 5 folds only for frozen configurations.
- Never tune anything on the test set. The training scripts will evaluate on test only when a run is marked `final`.
