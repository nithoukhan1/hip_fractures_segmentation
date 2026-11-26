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
| 2 | Dataset EDA | Done (2026-11-26) | `EDA/` + `Analysis/01_…` §2 | Official val split is empty (local copy and IEEE DataPort comment agree) |
| 3 | Literature: who has used this dataset, state of the art | Done (2026-11-26) | `research/01_prior_use_and_sota_review.md` | Nobody else has published on HUSJA-UA |
| 4 | Gap analysis and idea selection | Done (2026-11-26): **A and B in parallel** | `Analysis/02_…` (D1–D9), `Analysis/03_…` (ranking, designs) | |
| 5 | Execution plan for both papers | Done (2026-11-26) | `Analysis/04_Execution_Plan_A_and_B.md` | Runs, validation rules, compute, dual timeline |
| 6 | Repository, fixed splits, Kaggle launcher, run register | Done (2026-11-26); **pull request to `main` waiting for your review** | README, `splits/folds_v1.csv`, `kaggle/`, `results/`, `scripts/register_run.py` | 5 grouped, stratified folds; counts match paper Table 1 |
| 7 | Kaggle environment check `S00_env_check` | **Next: your run on Kaggle** | `results/S00_env_check/` | Confirms data and folds identical to local, GPU type and speed |
| 8 | Data loader, metrics and statistics modules | Not started (Week 1) | `src/femurseg/` | Same code for both papers |
| 9 | Baselines S1–S9 and core model M0 (5 folds each) | Not started (Weeks 1–2) → **M1** | `results/S*` | M0 = A1 in Paper A = R0 in Paper B |

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
| M0 | Setup merged into `main`, Kaggle environment check passed | 29 Nov 2026 | Waiting for your PR review and Kaggle run |
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
| Shared | 28–34 | 0 |
| Track A only | 35–38 | 0 |
| Track B only | 55–60 | 0 |
| **Total** | **120–135** (peak about 20 h in one week) | 0 |

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

## Open questions
- Run `S00_env_check` on Kaggle: Kaggle data identical to the local copy (fingerprint `733a70aa…`)? Data path? GPU type, count and speed?
- Is a T4 x2 session metered like one GPU? Compare your quota before and after one run. If yes, A and B jobs can share one launch.
- Real hours per run on Kaggle (Week 1), to update the compute table in Report 04 §6.
- After the pull request is merged: set `main` as the default branch (GitHub → Settings → General → Default branch).
- Repository visibility: it is public now. Consider making it private until submission, so the methods are not visible before the papers; Kaggle can still clone it with a token (see `kaggle/launcher.ipynb`).
- Resolved 2026-11-26: `data.yml` codes are `2AB, 2CD, 3AB, 3CDE, 4` (location digit + grade letters); `readme.md` only asks to cite the base paper. No patient, device or DX/CR metadata is in the release.
