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
| 9 | Baselines S1–S9 and core model M0 (5 folds each) | **Done 2026-11-27 (milestone M1).** **S01 and S02 done 2026-11-26** (register lines 3–4; analysis `results/analysis/C01_yolo_1024_vs_640/`): classification reproduces the article, Dice .02–.04 lower. **Step 2 code merged 2026-11-26 (PR #5):** `scripts/train_net.py` for S04–S10 (protocol `Analysis/05_…`); **S00_net_check done** (register line 5). **S03, S06 and S10 done 2026-11-27** (register lines 6–8, PR #7 merged): YOLO11 = YOLOv8 (C02); the mask task helps classification (C05: OOF accuracy +2.0 points, macro-F1 +.042, Holm p ≤ .03); M0's Dice is above YOLO's. **S04, S05, S07 and S08 done 2026-11-27** (register lines 9–12): both article classifiers reproduced; DenseNet121 and ConvNeXt-T beat YOLO out of fold, YOLO's ensemble leads on the test set, not significantly (C03, `results/analysis/C03_classifiers_vs_yolo/`). **S09 done 2026-11-27** (register line 18). C04 (`results/analysis/C04_segmenters_vs_yolo/`): M0 has the best Dice (OOF .777; above every other model out of fold, Holm p ≤ .0003; above YOLO on the test set +.018, not significant); SegFormer is the best standard segmenter. **Baseline table v1:** `results/analysis/M1_baseline_table/` | `results/S*`, `results/analysis/` | M0 = run S10 = A1 in Paper A = R0 in Paper B |

### Track A: Paper 1, SLG-Net
| # | Phase | Weeks | Status |
|---|-------|-------|--------|
| A-1 | Build region pooling, geometry and location → grade heads; fold-0 development | 3–4 | **Done 2026-11-27:** protocol (`Analysis/06_…`), code (`src/femurseg/slg.py`) and the fold-0 development runs (register lines 13–16). C06 (`results/analysis/C06_paperA_dev_fold0/`): the base failed the go/no-go (macro-F1 .776 < .79); design review: the 1/16 region features caused it; teacher forcing and the end-to-end mask stay, **pooling moves to 1/32 only** (.827, M0's level) |
| A-2 | Rare classes, calibration; freeze the method (git tag) → **M2-A** | 5 | In progress: calibration in `scripts/run_stats.py` (S06 temperature 3.43, M0 3.14; development runs 3.1–3.4). **Freeze approved by the person 2026-11-27** (Analysis/06 §10): the person creates the tag `A-v1-frozen` on `main` after PR #12. Second-seed check `A_dev_seed2` (seed 2027) runs in parallel; its rule is fixed (§10, analysis C11) |
| A-3 | Final runs and ablations A2–A6, oracle and region-deletion tests, sensitivity → **M3-A** | 6–7 | Launching 2026-11-27: `A2_region` … `A6_slgnet` from the tag (about 13 h); results analysed after C11 (C07–C09) |
| A-4 | Statistics and figures → **M4-A** | 8 | Not started |
| A-5 | Manuscript A, internal review, submission → **M5-A** | 9–12 | Not started |

### Track B: Paper 2, RefFrac
| # | Phase | Weeks | Status |
|---|-------|-------|--------|
| B-1 | Pair index and retrieval bank (CPU), nearest-neighbour sanity figure | 1–2 | **Done 2026-11-27.** `R00_reference_bank` (register line 17): check 1 failed (58.4% < 60%), so the orientation rule was reviewed (Analysis/07 §8; C10). **`R00_reference_bank_r2`** (register line 19): version 2 (pelvis side read from the image) passes, 804 of 810 (99.3%); every R run reads its tables (Analysis/07 §9) |
| B-2 | Siamese encoder, alignment and difference fusion; fold-0 development | 3–4 | Next: the protocol (written before any R run), then the code and fold-0 development |
| B-3 | Alignment variants, k, R1 (AASN-style) baseline; freeze → **M2-B** | 5–6 | Not started |
| B-4 | Final runs R2–R6 | 7–8 | Not started |
| B-5 | R7 shortcut study (partner and swap tests), sensitivity → **M3-B** | 9 | Not started |
| B-6 | Statistics and figures → **M4-B** | 10 | Not started |
| B-7 | Manuscript B, internal review, submission → **M5-B** | 11–15 | Not started |

## Milestones
| ID | Milestone | Target date | Status |
|---|---|---|---|
| M0 | Setup merged into `main`, Kaggle environment check passed | 29 Nov 2026 | **Reached 26 Nov 2026** |
| M1 | Baseline table v1 (S1–S9 + M0) | 13 Dec 2026 | **Reached 27 Nov 2026** (`results/analysis/M1_baseline_table/`) |
| M2-A | SLG-Net frozen | 3 Jan 2027 | Approved 27 Nov 2026; reached when the tag `A-v1-frozen` exists, subject to the second-seed check (C11) |
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
| Shared | 28–34 | 20.0 h of session time (S00 + S00_r2 0.12, S01 2.94, S02 1.35, S00_net_check 0.06, S03 2.96, S06 1.46, S10 2.33, S04 1.37, S05 1.92, S07 1.45, S08 1.22, S09 2.81; T4 × 2). Baselines complete |
| Track A only | 35–38 | 1.7 h (A_dev_round1 0.87, A_dev_round2 0.84; one run per T4). Next: A_dev_seed2 about 0.9 h and the final runs about 13 h |
| Track B only | 55–60 | 0.09 h (R00_reference_bank 0.04, R00_reference_bank_r2 0.04) |
| **Total** | **120–135** (peak about 20 h in one week) | 21.8 |

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
| 2026-11-26 | **Step 2 recipe, the same for S04–S10** (`Analysis/05_Step2_Baselines_Protocol.md`): 768 × 512 input, ImageNet weights, AdamW lr 2e-4 wd 0.05, 2 warm-up epochs + cosine, 60 epochs, final weights, batch 16 (micro-batches with gradient accumulation if memory is short), fp16; no per-model tuning | Fixed before any result; fair to all baselines; fits Kaggle's T4s |
| 2026-11-26 | M0 = S06's ConvNeXt-T classifier + a U-Net decoder (binary mask); its 6-class head is S06's own head on the same features (checked identical in `tests/test_nets.py`). M0 runs as `S10_m0_convnext_unet` | S06 vs S10 isolates the effect of learning the mask |
| 2026-11-26 | Segmenters (S07–S09): 6-class pixel labels; fracture score = mean of the 500 highest pixel fracture probabilities; class = largest share of the predicted region. All models are scored by one module (`src/femurseg/evaluation.py`), which re-scores S01/S02 to exactly their `metrics.json` | One decision rule and one scoring code for every model |
| 2026-11-26 | Use one Kaggle account; run parallel sessions on it, or on a colleague's own account | Kaggle allows one account per person; a second own account risks a ban of both |
| 2026-11-27 | YOLO11s-seg (S03) adds nothing over YOLOv8s-seg (C02, smallest Holm p = .72): S01 stays the YOLO baseline; no further YOLO variants | The article's own model is the fairer reference; saves quota |
| 2026-11-27 | The mask task helps classification (C05: S10 vs S06, OOF accuracy +2.0 points, Holm p = .026; macro-F1 +.042, Holm p = .013; same direction on the test set): M0 stays the starting point of both papers | Confirms step A0 → A1; grade errors (65 of M0's 121 OOF errors) and calibration (OOF threshold .018, ECE .108) are what A2–A6 must fix |
| 2026-11-27 | **Paper A protocol fixed before any SLG-Net result** (`Analysis/06_Step3_PaperA_Protocol.md`): A2–A6 each add one component on top of M0 (region pooling, geometry, location → grade hierarchy, shared grade trunk, logit adjustment) with the Step 2 recipe unchanged; loss weights, τ = 1, teacher forcing 1 → 0 over 30 epochs and pooling at 1/16 + 1/32 fixed now | Each step of the ablation answers one question, and nothing is tuned after seeing results |
| 2026-11-27 | Fold-0 development = three questions (D1 teacher forcing, D2 detached mask, D3 pooling at 1/32 only), one alternative run each against the A6 base, compared with M0's own fold-0 model (macro-F1 .822). A default changes only if macro-F1 rises by ≥ .02 with Dice down by ≤ .01; if the A6 base scores below .79, the design is reviewed first | Fold 0 has 208 femurs: even M0's gain over S06 (+.026) has p = .51 there, so only clear effects may change a default |
| 2026-11-27 | The oracle and region-deletion tests are run by the training run itself (`--extra-tests`), with per-case files per test; the deletion control is the same shape at a random place on equally bright tissue, seeded by the file name | One code path for every variant; analysis C08 needs no extra runs |
| 2026-11-27 | Fold-0 development runs share a Kaggle session, one per T4 (parallel groups `A_dev_round1`, `A_dev_round2`) | A one-fold run uses one GPU, so two of them fit one session |
| 2026-11-27 | `kaggle/run.py` checks for the dataset before installing anything; without it a training run stops within seconds (exit code 2, **NO DATASET FOUND**, what `/kaggle/input` holds, how to attach the dataset). `register_run.py` does not file a run that never started, but still files the other run of a parallel group | S09's first launch had no dataset; a clear message replaces a traceback, and an empty zip must not take a run name or a register line |
| 2026-11-27 | Calibration = one temperature per run, fitted by NLL on its OOF femurs and applied unchanged to the test set; ECE (15 bins), Brier and NLL before and after; reported only, never used for a decision. Not applicable to the YOLO runs (S01–S03) | Protocol §6. YOLO gives probability 0 to the true class of 73–99 OOF femurs, so the NLL that fits the temperature is undefined |
| 2026-11-27 | **Paper B's reference bank fixed before any feature** (`Analysis/07_…`): the bank of fold model k = the healthy femurs of its 4 training folds (290–291); never a reference from the femur's own group; cosine of frozen ImageNet ConvNeXt-T features, centred on the mean of the fold model's training femurs, best of as released / mirrored; kept: the 5 nearest, one random healthy and one random fractured femur (seeded by md5); one fixed table read by every R run | No patient's own other hip during training, nothing from the held-out fold or the test set, and R2, R3, R4 and R7 differ only in the reference they test |
| 2026-11-27 | C03: both article classifiers are reproduced; DenseNet121 (S04) joins Paper A's main table as the strongest plain classifier; ConvNeXt-T stays the backbone of M0 and both papers | DenseNet121 = ConvNeXt-T (no significant difference), and the mask task was shown on ConvNeXt-T (C05) |
| 2026-11-27 | C06 (Paper A, fold 0): the base failed the go/no-go (.776 < .79), so the design was reviewed. Teacher forcing (D1) and the end-to-end mask (D2) stay; region pooling moves to 1/32 only (D3: +.052, Dice +.006). **Freeze proposed, waiting for the person's approval** | Fixed rule of Analysis/06 §4; with 1/32 pooling the heads match M0 (.827 vs .822), so the 1/16 features caused the shortfall |
| 2026-11-27 | Paper B's orientation rule, version 2 (Analysis/07 §8): a reference is mirrored when its pelvis side (the brighter half of the upper image) differs from the target's; re-run as `R00_reference_bank_r2` | Check 1 failed (58.4%): ImageNet features hardly see left and right. The pelvis side gives partners opposite sides in 402 of 405 x-rays (C10) |
| 2026-11-27 | **C04 (RQ1): M0 has the best Dice** of all segmenters; out of fold it beats YOLO (+.037), SegFormer (+.011), U-Net and DeepLabV3+ (Holm p ≤ .0003); on the test set it beats U-Net and DeepLabV3+, and its lead over YOLO (+.018, p = .36) is not significant. SegFormer (S09) joins Paper A's main table as the strongest standard segmenter; U-Net and DeepLabV3+ stay in the baseline table only | YOLO's Dice deficit comes mostly from missed fractures (51 of 680 OOF); the papers state both the OOF and the test result |
| 2026-11-27 | **Milestone M1 reached:** all ten baselines filed and analysed (C01–C05); baseline table v1 in `results/analysis/M1_baseline_table/` | Planned for 13 Dec 2026 |
| 2026-11-27 | **Paper A frozen (approved by the person):** teacher forcing over 30 epochs, end-to-end mask, region pooling at 1/32 only; tag `A-v1-frozen` on `main`, created by the person (this session cannot push tags). At the same time, a second-seed check of the base and of 1/32 pooling (`A_dev_seed2`, seed 2027) with a rule fixed before it runs (Analysis/06 §10): 1/32 stays if the mean of two seeds beats the base by ≥ .02 macro-F1, loses ≤ .01 Dice and reaches ≥ .79; otherwise the freeze goes back to the person before any final result is opened | The person chose to run the final runs and the check together; the check guards against a single-seed result on 208 femurs |
| 2026-11-27 | **Paper B step B-1 complete:** `R00_reference_bank_r2` passes check 1 (804 of 810, 99.3%); every R run reads its tables | Version 2 changed mainly the orientation of the references (same nearest femur in 87.1% of rows, same orientation in 57.6%) |
| 2026-11-27 | `run_stats.py`: a network run's true-class probability written as 0 (below the per-case file's 8 decimals) is read as 10⁻¹² like every other written 0, so its temperature is fitted; YOLO stays not applicable | It affected one femur of S05 and one of A6_dev_detach; no earlier value changed |
| 2026-11-26 | Micro-batches set by the pre-flight (`S00_net_check`): 16 × 1 for DenseNet121, ConvNeXt-T, U-Net, DeepLabV3+ and M0; 8 × 2 for EfficientNet-B4 and SegFormer | The protocol's rule (largest micro-batch under 85% of a T4); the optimiser still sees 16 images per update |

## Open questions
- Resolved 2026-11-26 (`S00_env_check_r2`): Kaggle's 104 val/ images are identical to the local ones; data, folds and file list all match.
- Resolved 2026-11-27: Paper A's freeze is approved, and the second-seed check runs at the same time as the final runs (Analysis/06 §10).
- Is a T4 x2 session metered like one GPU? All runs to 2026-11-27 took 11.2 h of session time: if Kaggle shows about 11 h used, a T4 × 2 session counts once; about 22 h means it counts per GPU.
- Resolved 2026-11-26: real run times on T4 × 2 (two folds at a time): YOLOv8s-seg 2.9 h at 1024 px (about 58 min per fold), 1.35 h at 640 px (about 27 min per fold). The plan's estimates were about twice too high.
- Repository visibility: it is public now. Consider making it private until submission, so the methods are not visible before the papers; Kaggle can still clone it with a token (see `kaggle/launcher.ipynb`).
- Resolved 2026-11-26: `data.yml` codes are `2AB, 2CD, 3AB, 3CDE, 4` (location digit + grade letters); `readme.md` only asks to cite the base paper. No patient, device or DX/CR metadata is in the release.
