# Segmentation Project: Overall Project Tracker

**Goal:** Two papers with novel, feasible methods on the HUSJA-UA Femoral Fracture dataset (González et al., *Insights into Imaging* 2025, 16:86, doi 10.1186/s13244-025-01958-y), aimed at SCI JCR Q2/Q3 journals (Q4 acceptable).
- **Paper A, SLG-Net (Segment, Locate, Grade).**
- **Paper B, RefFrac (healthy reference femur).**

**Local folder:** `E:\PhD\Admitted\Research\Project 3\Segmentation Project` (same files as the repository, plus the data folder)
**Repository:** https://github.com/nithoukhan1/hip_fractures_segmentation (public)
**Training:** Kaggle dataset `utopianstar/dataset`, run through `kaggle/launcher.ipynb` with Save Version (background).
**Plans:** `Analysis/03_Idea_Ranking_and_Research_Plan.md` (ideas and designs) and `Analysis/04_Execution_Plan_A_and_B.md` (runs, validation, compute, timeline: this one is current).
**Experiment register:** `results/EXPERIMENTS.md`

## Phases

### Shared foundation
| # | Phase | Status | Output / location | Notes |
|---|-------|--------|-------------------|-------|
| 1 | Deep reading of base article | Done (2026-11-26) | `Analysis/01_…` §1 | Methods, numbers, limitations, future work extracted |
| 2 | Dataset EDA | Done (2026-11-26) | `EDA/` + `Analysis/01_…` §2 | Official val split is empty (local copy and IEEE DataPort comment agree). The Kaggle copy differs: see phase 7 |
| 3 | Literature: who has used this dataset, state of the art | Done (2026-11-26) | `research/01_prior_use_and_sota_review.md` | Nobody else has published on HUSJA-UA |
| 4 | Gap analysis and idea selection | Done (2026-11-26): **A and B in parallel** | `Analysis/02_…` (D1–D9), `Analysis/03_…` (ranking, designs) | |
| 5 | Execution plan for both papers | Done (2026-11-26) | `Analysis/04_Execution_Plan_A_and_B.md` | Runs, validation rules, compute, dual timeline |
| 6 | Repository, fixed splits, Kaggle launcher, run register | Done; merged into `main` 2026-11-26 (PR #1); default branch = `main` | README, `splits/folds_v1.csv`, `kaggle/`, `results/`, `scripts/register_run.py` | 5 grouped, stratified folds; counts match paper Table 1 |
| 7 | Kaggle environment check | Done 2026-11-26: run 1 found Kaggle's val/ holds 104 training femurs (now pooled into train); `S00_env_check_r2` confirmed same data, same folds, all files found | `results/S00_env_check/`, `results/S00_env_check_r2/` | 2 × T4, 4 CPU cores, 32 GB RAM; all 1,253 files identical to local |
| 8 | Data loader, metrics and statistics modules | Done 2026-11-26: loader, metrics (13 checks incl. scikit-learn), statistics (9 checks; reproduces the paper's accuracy interval 80.77–90.55 exactly), fold checks, CPU smoke test of the YOLO script and the statistics script | `src/femurseg/data.py`, `metrics.py`, `stats.py`, `scripts/run_stats.py`, `tests/` | Same code for both papers |
| 9 | Baselines S1–S9 and core model M0 (5 folds each) | In progress: **S01 and S02 done 2026-11-26** (register lines 3–4; analysis `results/analysis/C01_yolo_1024_vs_640/`): classification reproduces the article, Dice .02–.04 lower. S03 (YOLO11s-seg @1024) ready on Kaggle; S04–S09 and M0 need the Step 2 training script → **M1** | `results/S*`, `results/analysis/` | M0 = A1 in Paper A = R0 in Paper B |

### Track A: Paper 1, SLG-Net
| # | Phase | Weeks | Status |
|---|-------|-------|--------|
| A-1 | Build region pooling, geometry and location → grade heads; fold-0 development | 3–4 | Not started |
| A-2 | Rare classes, calibration; freeze the method (git tag) → **M2-A** | 5 | Not started |
| A-3 | Final runs and ablations A2–A6, oracle and region-deletion tests, sensitivity → **M3-A** | 6–7 | Not started |
| A-4 | Statistics and figures → **M4-A** | 8 | Not started |
| A-5 | Manuscript A, internal review, submission → **M5-A** | 9–12 | Not started |

### Track B: Paper 2, RefFrac
| # | Phase | Weeks | Status |
|---|-------|-------|--------|
| B-1 | Pair index and retrieval bank (CPU), nearest-neighbour sanity figure | 1–2 | Not started |
| B-2 | Siamese encoder, alignment and difference fusion; fold-0 development | 3–4 | Not started |
| B-3 | Alignment variants, k, R1 (AASN-style) baseline; freeze → **M2-B** | 5–6 | Not started |
| B-4 | Final runs R2–R6 | 7–8 | Not started |
| B-5 | R7 shortcut study (partner and swap tests), sensitivity → **M3-B** | 9 | Not started |
| B-6 | Statistics and figures → **M4-B** | 10 | Not started |
| B-7 | Manuscript B, internal review, submission → **M5-B** | 11–15 | Not started |

## Milestones
| ID | Milestone | Target date | Status |
|---|---|---|---|
| M0 | Setup merged into `main`, Kaggle environment check passed | 29 Nov 2026 | **Reached 26 Nov 2026** |
| M1 | Baseline table v1 (S1–S9 + M0) | 13 Dec 2026 | |
| M2-A | SLG-Net frozen | 3 Jan 2027 | |
| M2-B | RefFrac frozen | 10 Jan 2027 | |
| M3-A | Paper A final runs, ablations and tests done | 17 Jan 2027 | |
| M4-A | Paper A results package (statistics, figures) | 24 Jan 2027 | |
| M3-B | Paper B final runs and shortcut study done | 31 Jan 2027 | |
| M4-B | Paper B results package | 7 Feb 2027 | |
| M5-A | **Paper A submitted** | 21 Feb 2027 (buffer to 7 Mar) | |
| M5-B | **Paper B submitted** | 14 Mar 2027 (buffer to 28 Mar) | |

## Compute budget (Kaggle GPU-hours, from Report 04 §6)
| Block | GPU-hours | Used so far |
|---|---|---|
| Shared | 28–34 | 4.4 h of session time (S00 + S00_r2 0.12, S01 2.94, S02 1.35; T4 × 2) |
| Track A only | 35–38 | 0 |
| Track B only | 55–60 | 0 |
| **Total** | **120–135** (peak about 20 h in one week) | 4.4 |

## Key decisions log
| Date | Decision | Why |
|------|----------|-----|
| 2026-11-26 | Keep the official train / test split; never move images across it | Patient IDs are not released, so a re-split cannot be guaranteed patient-level |
| 2026-11-26 | Build validation from train: 5 folds grouped by x-ray ID (duplicate pair merged), stratified by class, seed 2026 (`splits/folds_v1.csv`) | Official val folder is empty; folds of about 209 femurs match the paper's 201-femur val size; out-of-fold predictions give 1,043 cases for paired statistics |
| 2026-11-26 | Each configuration = 5 fold-models; report OOF, test mean ± SD, and 5-model ensemble on test | Uses all training data, gives uncertainty for free, more statistical power than 210 test cases alone |
| 2026-11-26 | Four honesty rules: fixed training length with final weights; threshold and temperature fitted on pooled OOF only; development on fold 0 only (folds 1–4 reported as a check); test only for frozen configurations | The held-out fold and the test set never influence a choice |
| 2026-11-26 | **Run Option A (SLG-Net) and Option B (RefFrac) in parallel as two papers**, sharing split, code, baselines and M0 | Most work is shared; different research questions; risk is spread across two papers |
| 2026-11-26 | Paper B's main reference = k = 3 retrieved healthy femurs from the fold's own training bank. The contralateral partner is used only in a study with swap tests | In every two-femur x-ray exactly one femur is fractured, so a partner reference invites a shortcut |
| 2026-11-26 | Primary endpoints fixed in advance. A: 6-class macro-F1 + mean Dice on fractured femurs. B: mean Dice on fractured femurs + Garden I–II recall | Avoids metric shopping |
| 2026-11-26 | **No clinician available:** no reader study, no re-annotation, no clinical claims. Interpretability shown by measurements (oracle, region deletion, geometry consistency, pointing game, risk–coverage). Target technical and informatics journals | Nothing in either plan needs a radiologist; limitations stated openly |
| 2026-11-26 | Run names: S = shared (baselines B1–B9 renamed S1–S9), A = Paper A, R = Paper B | One register for both papers |
| 2026-11-26 | Documentation protocol: every run gets `results/<RUN>/` with `NOTES.md` and a register line; zips go through `results/_inbox/` and `scripts/register_run.py`; local folder = repository | Every number in a paper must trace to a file |
| 2026-11-26 | Repository layout mirrors the local folder; data, PDFs, zips and weights are not committed | Local folder and repo stay identical; Kaggle clones the repo |
| 2026-11-26 | `main` created as an empty base branch, so the whole setup can be reviewed in one pull request | The repository started with only the setup branch |
| 2026-11-26 | **Kaggle's val/ folder (104 of the 1,043 training femurs) is pooled back into train in all code** (`src/femurseg/paths.py`); the folds stay `splits/folds_v1.csv` | Kaggle's val/ is not the paper's missing validation set, it splits 57 x-rays across train and val (leak), and pooling gives the same data and folds on every copy (S00_env_check) |
| 2026-11-26 | Training scripts load all images into RAM once per session | Reading from /kaggle/input takes 98 ms per image; all 1,253 images take about 2 GB of the 32 GB RAM |
| 2026-11-26 | YOLO baselines (S01, S02): Ultralytics 8.4.219, COCO weights, 100 epochs, Ultralytics defaults (optimizer auto = AdamW for this data size), final weights with `val=False`; paper rule (fracture = any instance, class and mask of the top-confidence instance); fracture threshold = Youden on OOF (ties → higher sensitivity), plus results at Ultralytics' 0.25; test = mean ± SD of the 5 fold models and their ensemble | Same rules as every other run; the paper kept the best-validation checkpoint and did not report hyper-parameters, so differences are stated |
| 2026-11-26 | **Correction:** the paper's best-Dice setting is YOLOv8-**small** @640 (test Dice .780), not nano; S02 is `S02_yolov8s_seg_640` | Reports 03 and 04 said nano; fixed |
| 2026-11-26 | Statistics: 95% intervals by cluster bootstrap over x-rays (2,000 resamples), DeLong for AUC, exact Clopper–Pearson for accuracy; paired tests DeLong (AUC), exact McNemar + mid-p (decisions), randomisation test swapping per x-ray (macro-F1), Wilcoxon signed-rank with Pratt zeros (Dice); Holm within each comparison family | The two femurs of an x-ray are not independent; the paper's own accuracy interval is Clopper–Pearson (181/210 → 80.77–90.55), so our numbers are directly comparable |
| 2026-11-26 | **S01 (YOLOv8s-seg @1024) is the reference YOLO baseline** for both papers: its 5 fold models (mean ± SD) and their ensemble on the test set, and its pooled OOF predictions for development. The article's final model is quoted beside them | It reproduces the article's accuracy and AUC. The article's selection rule applied to our folds picks a model with 83.8% on test, so one selected run is a noisy estimate |
| 2026-11-26 | Dice is reported per location, OOF and test side by side; nothing learned from the test set is used to adjust masks | Test neck masks are 35–40% smaller than training neck masks (test labels were reviewed by a committee); every model over-segments them about 2 times on the test set (C01 report §4) |
| 2026-11-26 | New training code also records NSD at 10 and 20 px | NSD at 2 px is about .05 for every model (median HD95 63–72 px), so it cannot separate methods |
| 2026-11-26 | S03 = YOLO11s-seg at 1024 px with exactly S01's protocol | 640 vs 1024 px made no significant difference (C01); a like-for-like comparison with S01 isolates the architecture |
| 2026-11-26 | The article's published numbers live in `src/femurseg/reference.py`, checked by `tests/test_reference.py` | One checked source for every comparison table |

## Open questions
- Resolved 2026-11-26 (`S00_env_check_r2`): Kaggle's 104 val/ images are identical to the local ones; data, folds and file list all match.
- Is a T4 x2 session metered like one GPU? Steps 0–1 used 4.4 h of session time: if Kaggle shows about 4.4 h used this week, a T4 × 2 session counts once; about 8.8 h means it counts per GPU.
- Resolved 2026-11-26: real run times on T4 × 2 (two folds at a time): YOLOv8s-seg 2.9 h at 1024 px (about 58 min per fold), 1.35 h at 640 px (about 27 min per fold). The plan's estimates were about twice too high.
- Repository visibility: it is public now. Consider making it private until submission, so the methods are not visible before the papers; Kaggle can still clone it with a token (see `kaggle/launcher.ipynb`).
- Resolved 2026-11-26: `data.yml` codes are `2AB, 2CD, 3AB, 3CDE, 4` (location digit + grade letters); `readme.md` only asks to cite the base paper. No patient, device or DX/CR metadata is in the release.
