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
| Ranking of the nine directions and research plan | Done (Report 03) |
| Decision: Papers A and B in parallel; execution plan | Done (Report 04) |
| Fixed 5-fold split of the training set | Done (`splits/folds_v1.csv`) |
| Kaggle environment check | Done: data and folds on Kaggle are identical to `splits/folds_v1.csv` (Kaggle's val/ pooled into train) |
| Data loader, metrics, statistics, tests | Done (`src/femurseg/data.py`, `metrics.py`, `stats.py`, `tests/`) |
| Shared nine baselines and core model ConvNeXt U-Net | **Done (the baseline-table milestone, 2026-11-27):** baseline table v1 in `results/analysis/M1_baseline_table/report.md`. YOLOv8s-seg and YOLOv8s-seg at 640 px reproduce the article's accuracy and AUC (`results/analysis/C01_yolo_1024_vs_640/report.md`); YOLO11s-seg adds nothing (`C02_yolov8_vs_yolo11`). The image the three classifiers better than YOLO out of fold and reproduce the article's DenseNet121 and EfficientNet-B4 (`C03_classifiers_vs_yolo`). The core model ConvNeXt U-Net classifies better than the ConvNeXt-T classifier and segments better than YOLO (`C05_mask_task_s06_vs_s10`). Among the segmenters (`C04_segmenters_vs_yolo`), ConvNeXt U-Net has the best Dice: out of fold it beats YOLO and every standard segmenter; on the test set its lead over YOLO is not significant. SegFormer (SegFormer-B2) is the best standard segmenter. Protocol in `Analysis/05_Step2_Baselines_Protocol.md` |
| Paper A (SLG-Net) | Fold-0 development done (`results/analysis/C06_paperA_dev_fold0/report.md`): the base setting missed the go/no-go line (macro-F1 .776 < .79); the design review traced this to the 1/16 region features, and region pooling at 1/32 only reaches ConvNeXt U-Net's level (.827). **Freeze approved 2026-11-27** (teacher forcing over 30 epochs, end-to-end mask, region pooling at 1/32 only; tag `A-v1-frozen`). The second-seed check confirmed the freeze (`results/analysis/C11_paperA_seed2/report.md`: +.054 macro-F1 over two seeds). **The five final the five SLG-Net chain runs are done and analysed** (2026-11-27, all from the tag, all 5 folds; `Analysis/06` section 12): out of fold, SLG-Net beats the base article's model (macro-F1 +.061) but is not more accurate than ConvNeXt U-Net (−.020, not significant; `results/analysis/C07_paperA_ablation_chain/report.md`, `C09_paperA_main_table`). Its location and grade decisions depend on the predicted fracture region: deleting the region loses 65% of the correct decisions, the control 1% (`C08_paperA_interpretability`); its grades follow its own region, but no better than ConvNeXt U-Net's (geometry test, the SLG-Net interpretability analysis, section 4). **2026-11-28: SLG-Net is kept as pre-registered**, and ConvNeXt U-Net gets the same deletion test (`A1_m0_deletion`, `Analysis/06` section 13) so that "more region-dependent than ConvNeXt U-Net" can be tested. **Result (2026-11-28, the SLG-Net interpretability analysis, section 2b): it is** (64% vs 48% of correct decisions lost when the region is removed, p < .001; the test set agrees). The plan of the combined paper (SLG-Net and RefFrac in one paper) is `Analysis/09_Combined_Paper_Plan.md` |
| Paper B (RefFrac) | Step 1 of Paper B (`Analysis/07_Step4_PaperB_Reference_Bank_Protocol.md`): pair index and bank of healthy reference femurs built (reference bank). Its mirror check failed because ImageNet features hardly see left and right; version 2 of the orientation rule reads the pelvis side from the image (the two femurs of an x-ray get opposite sides in 402 of 405 x-rays, `results/analysis/C10_paperB_orientation_review/`) and passes as `R00_reference_bank_r2` (804 of 810 partners, 99.3%): **step 1 of Paper B is complete** (`Analysis/07` section 9). **Step 2 of Paper B started 2026-11-27:** the protocol was written before any R run (`Analysis/08_Step5_PaperB_RefFrac_Protocol.md`: the Siamese RefFrac model, the eight reference runs, the reference tests and a fold-0 decision rule), the model is coded (`src/femurseg/refnet.py`) and tested on the CPU, and the fold-0 development is done (`results/analysis/C12_paperB_dev_fold0/report.md`): RefFrac passes the go/no-go (fold-0 Dice .770 against ConvNeXt U-Net's .749), clearly uses its references, and neither alternative setting wins. **The settings are frozen as written:** the eight final the eight reference runs are in `kaggle/runs.json` and start from the tag `B-v1-frozen` (created 2026-11-27). **The eight final runs are done and analysed** (2026-11-28; `Analysis/08` section 10): a healthy reference does not improve Dice; Garden I-II recall rises out of fold (RefFrac .702 vs ConvNeXt U-Net .544) but not after Holm and not on the test set (`results/analysis/C13_paperB_reference/report.md`). The model uses the comparison but not the match of the reference; the patient's own hip is no better, and there is no pairing shortcut (`C14_paperB_partner`). Alignment and fusion change how the reference is used, not accuracy (`C15_paperB_components`). RefFrac ties ConvNeXt U-Net and beats the base article's model out of fold (`C16_paperB_main_table`). **2026-11-28: no separate Paper B;** its findings join Paper A as one paper (`Analysis/08` section 11) |

Details: [`PROJECT_TRACKER.md`](PROJECT_TRACKER.md) (overall) and [`PROGRESS_LOG.md`](PROGRESS_LOG.md) (step by step).

## Repository layout

```
.
├── README.md               this file
├── GLOSSARY.md             folder and run names, and the full names the documents use for them
├── PROJECT_TRACKER.md      overall tracker: phases per track, milestones, decisions, open questions
├── PROGRESS_LOG.md         step-by-step log, newest first
├── Analysis/               reports
│   ├── 01_Base_Article_and_EDA_Report.md
│   ├── 02_How_to_Use_the_Dataset.md        nine research directions
│   ├── 03_Idea_Ranking_and_Research_Plan.md ranking, Option A and B designs
│   ├── 04_Execution_Plan_A_and_B.md         A and B in parallel: runs, validation, compute, timeline
│   ├── 05_Step2_Baselines_Protocol.md       how the classifiers, segmenters and ConvNeXt U-Net are trained and scored
│   ├── 06_Step3_PaperA_Protocol.md          Paper A: the SLG-Net ablation runs, fold-0 development, interpretability tests
│   ├── 07_Step4_PaperB_Reference_Bank_Protocol.md   Paper B, step 1: pair index and bank of healthy references
│   └── 08_Step5_PaperB_RefFrac_Protocol.md          Paper B, step 2: RefFrac model and its runs, fold-0 development
├── research/               literature notes
├── EDA/                    EDA script, per-image statistics, figures
├── splits/                 fixed train folds (official val folder is empty; test is untouched)
├── src/femurseg/           Python package: classes, data paths, data loader (data.py), metrics (metrics.py),
│                           confidence intervals and paired tests (stats.py), the article's numbers (reference.py),
│                           per-case files and scoring shared by all models (evaluation.py), parallel folds with
│                           a time guard (runner.py), PyTorch networks (nets.py), SLG-Net of Paper A (slg.py),
│                           Paper B's pair index and reference bank (refbank.py), RefFrac of Paper B (refnet.py)
│                           and image cache (torchdata.py)
├── scripts/                command-line tools: make_splits.py, register_run.py, train_yolo_seg.py (the YOLO baselines),
│                           train_net.py (classifiers, segmenters, ConvNeXt U-Net, the network check, SLG-Net
│                           and RefFrac runs),
│                           run_stats.py (95% intervals, paired comparisons and calibration of filed runs),
│                           run_breakdown.py (per class, mean ± SD over folds, comparison with the article),
│                           build_reference_bank.py (Paper B's reference bank of healthy femurs),
│                           check_orientation.py (Paper B's orientation review; runs on a laptop),
│                           run_interpretability.py (SLG-Net's oracle, deletion, risk–coverage and
│                           geometry tests), map_geometry.py (region geometry from saved maps; runs on a laptop),
│                           run_refstats.py (RefFrac's Garden I-II tests, subgroups, reference tests and the
│                           own-hip model's modes), run_ensemble.py (ensembles of saved predictions)
├── tests/                  checks of metrics, statistics, folds, the article's numbers, the breakdown, Paper A's
│                           interpretability analysis and region geometry, Paper B's analysis tools, the networks, SLG-Net and RefFrac,
│                           Paper B's reference bank, the Kaggle launcher's data check, and CPU smoke tests of
│                           both training scripts, both analysis scripts, the reference bank, RefFrac and the
│                           launcher's parallel groups
├── configs/                one YAML file per experiment
├── kaggle/                 Kaggle launcher notebook, run list and Kaggle-only scripts
├── results/                every run's metrics, notes and small outputs (no model weights)
│   ├── EXPERIMENTS.md      the experiment register: one line per run
│   ├── analysis/           analyses across runs, one folder each (folder names and what each analysis is
│   │                       called in the documents: GLOSSARY.md)
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
6. Documents call every model, run and analysis by its full name (for example ConvNeXt U-Net, RefFrac, the reference analysis), never by a short code. Folder names keep their codes because the scripts depend on them; `GLOSSARY.md` links the two.

## Citation of the data and base work
If you use this code, also cite the base article and the dataset:
González G, Galant J, Salinas JM, et al. Classification and segmentation of hip fractures in x-rays: highlighting fracture regions for interpretable diagnosis. *Insights Imaging* 16, 86 (2025). https://doi.org/10.1186/s13244-025-01958-y
