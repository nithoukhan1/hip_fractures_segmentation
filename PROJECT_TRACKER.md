# Segmentation Project: Overall Project Tracker

**Goal:** A novel, feasible method on the HUSJA-UA Femoral Fracture dataset (González et al., *Insights into Imaging* 2025, 16:86, doi 10.1186/s13244-025-01958-y), aimed at an SCI JCR Q2/Q3 journal (Q4 acceptable).
**Local folder:** `E:\PhD\Admitted\Research\Project 3\Segmentation Project` (same files as the repository, plus the data folder)
**Repository:** https://github.com/nithoukhan1/hip_fractures_segmentation (public)
**Training:** Kaggle dataset `utopianstar/dataset`, run through `kaggle/launcher.ipynb` with Save Version (background).
**Plan:** `Analysis/03_Idea_Ranking_and_Research_Plan.md`

| # | Phase | Status | Output / location | Notes |
|---|-------|--------|-------------------|-------|
| 1 | Deep reading of base article | Done (2026-11-26) | `Analysis/01_Base_Article_and_EDA_Report.md` §1 | Methods, numbers, limitations, future work extracted |
| 2 | Dataset EDA | Done (2026-11-26) | `EDA/` + report §2 | Official val split is empty (local copy and IEEE DataPort comment agree) |
| 3 | Literature: who has used this dataset, state of the art | Done (2026-11-26) | `research/01_prior_use_and_sota_review.md` | Nobody else has published on HUSJA-UA |
| 4 | Gap analysis and idea selection | **Ranking done; waiting for your choice** | `Analysis/02_…` (D1–D9), `Analysis/03_…` (ranking, Options A/B) | Recommended: Option A, SLG-Net (D3 + D2, with D1, D7, D9) |
| 4b | Repository set-up and fixed splits | Done (2026-11-26) | README, `splits/folds_v1.csv`, `kaggle/`, `results/` | 5 grouped, stratified folds; counts match paper Table 1 |
| 5 | Method framework design | Drafted (Option A §3, Option B §4) | `Analysis/03_…` | Final after the choice |
| 6 | Baselines reproduction on Kaggle (YOLOv8-seg, classifiers, segmenters) | Not started (Step 1–2, Weeks 1–2) | `results/B*` | Same for both options |
| 7 | Implementation of proposed method | Not started (Steps 3–4, Weeks 3–5) | `src/femurseg/`, `configs/` | |
| 8 | Experiments and ablations | Not started (Step 5, Weeks 6–7) | `results/` | 5 folds per configuration |
| 9 | Statistical analysis and figures | Not started (Step 6, Week 8) | `results/`, `manuscript/figures` | DeLong, McNemar, bootstrap, Wilcoxon, Holm |
| 10 | Manuscript writing | Not started (Step 7, Weeks 9–11) | `manuscript/` | CLAIM 2024 checklist |
| 11 | Journal selection and submission | Not started (Step 8, Week 12) | | Target late Feb 2027 (buffer to early Mar) |

## Milestones
| ID | Milestone | Target date | Status |
|---|---|---|---|
| M0 | Option chosen, repo live, Kaggle env check passed | 29 Nov 2026 | Waiting for decision |
| M1 | Baseline table v1 (B1–B9) | 13 Dec 2026 | |
| M2 | Method frozen (git tag) | 3 Jan 2027 | |
| M3 | All final runs and ablations done | 17 Jan 2027 | |
| M4 | Results package (stats, figures) | 24 Jan 2027 | |
| M5 | Manuscript submitted | 21 Feb 2027 | |

## Key decisions log
| Date | Decision | Why |
|------|----------|-----|
| 2026-11-26 | Keep the official train / test split; never move images across it | Patient IDs are not released, so a re-split cannot be guaranteed patient-level |
| 2026-11-26 | Build validation from train: 5 folds grouped by x-ray ID (duplicate pair merged), stratified by class, seed 2026 (`splits/folds_v1.csv`) | Official val folder is empty; folds of ~209 femurs match the paper's 201-femur val size; out-of-fold predictions give 1,043 cases for paired statistics |
| 2026-11-26 | Each configuration = 5 fold-models; report OOF, test mean ± SD, and 5-model ensemble on test | Uses all training data, gives uncertainty for free, more statistical power than 210 test cases alone |
| 2026-11-26 | Primary endpoints fixed in advance: 6-class macro-F1 and mean Dice on fractured femurs | Avoids metric shopping; rare classes weighted fairly |
| 2026-11-26 | Repository layout mirrors the local folder; data, PDFs and weights are not committed | Local folder and repo stay identical; Kaggle clones the repo |
| pending | Choose Option A (SLG-Net), Option B (RefFrac) or A then B | See `Analysis/03_…` §11 |

## Open questions
- Resolved 2026-11-26: `data.yml` codes are `2AB, 2CD, 3AB, 3CDE, 4` (location digit + grade letters); `readme.md` only asks to cite the base paper. No patient, device or DX/CR metadata is in the release.
- Run `env_check` on Kaggle to confirm the Kaggle copy is identical to the local one (fingerprint `733a70aa…`), the Kaggle data path, and the GPU type and speed.
- Real hours per run on Kaggle (Week 1) to update the compute table in Report 03 §8.
- Repository visibility: it is public now. Consider making it private until submission, so the method is not visible before the paper; Kaggle can still clone it with a token (see `kaggle/launcher.ipynb`).
