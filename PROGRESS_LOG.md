# Progress Log (step-by-step)

Newest entries at the top. Each entry: what was done, what it produced, what comes next.

---

## 2026-11-26 · Step 6: Setup merged; first Kaggle run (environment check)

**Done (you)**
- Merged PR #1 into `main` and made `main` the default branch.
- Created the Kaggle notebook `hipfx-launcher` and ran `S00_env_check` on main (commit `37d8d0f`).

**Result** (`results/S00_env_check/`, register line 1)
- Kaggle gives 2 × Tesla T4 (15.6 GB each, about 23–24 fp16 TFLOPs), 4 CPU cores and 32 GB RAM, with Python 3.13, torch 2.11 and CUDA 13.0. timm and albumentations are installed; ultralytics and segmentation-models-pytorch are not.
- **Surprise:** Kaggle's `val/` folder is not empty. It holds 104 femurs, so train has 939 instead of 1,043, and the split check said "same data False".
- File-by-file comparison showed:
  - all 939 Kaggle train files and all 210 test files are pixel- and label-identical to the local copy;
  - the 104 local training femurs missing from Kaggle's train match Kaggle's val/ in count and labels.
  - So Kaggle's val/ is part of the official training set, not the paper's missing validation set.
  - It also splits 57 x-rays between train and val, which would leak if used for validation.
- Reading an image from Kaggle's input folder takes 98 ms, against 17 ms locally.

**Fix**
- All code now pools `val/` back into train (`src/femurseg/paths.py`).
- `scripts/make_splits.py` and `kaggle/env_check.py` use it. The env check also confirms that every file in `splits/folds_v1.csv` is found.
- Tested on the local data: the CSV is byte-identical to `splits/folds_v1.csv`.
- Tested on a copy of Kaggle's exact folder layout: same data True, same folds True, all files found True.

**Next**
- You: merge PR #2, then run `S00_env_check_r2` with Save Version to confirm on the real Kaggle files.
- Me: Step 1. The data loader caches all images in RAM because of the slow reads. Also the metrics and statistics modules, and the YOLO reproduction runs S01 and S02 with `ultralytics` installed through `kaggle/runs.json`.

---

## 2026-11-26 · Step 5: Decision for Papers A and B in parallel; execution plan; documentation tools

**Decided (by you)**
- Run **Option A (SLG-Net)** and **Option B (RefFrac)** in parallel, as two papers.
- No radiologist or other clinician is available.
- Every experiment and its result must be documented in the local folder and in the repository.

**Done**
1. Wrote `Analysis/04_Execution_Plan_A_and_B.md`:
   - **Shared foundation:** split, loader, metrics, baselines S1–S9 and the core model M0. M0 is Paper A's ablation A1 and Paper B's "no reference" model R0, so it is trained once and used twice.
   - **No-clinician plan:** no reader study, re-annotation or clinical claims. Interpretability is shown by five measurable tests (oracle, region deletion, geometry consistency, pointing game, risk–coverage). Technical and informatics journals are the targets.
   - **5-fold out-of-fold validation** explained step by step, with four honesty rules.
   - **Track B:** research questions RQ-B1 to RQ-B4. Main method: retrieved healthy references, k = 3, from the fold's own training bank. Runs R1–R7, with the contralateral shortcut study and swap tests. Primary endpoints: Dice on fractured femurs and Garden I–II recall.
   - **Compute:** A + B ≈ 120–135 GPU-h (A alone ≈ 65–72), about 4–4.5 weeks of free quota spread over 10 weeks, with a peak of about 20 h per week.
   - **Timeline:** Paper A submission about 21 Feb 2027, Paper B about 14 Mar 2027, each with a 2-week buffer.
2. Updated Report 03: decision banner, new run names (baselines B1–B9 are now S1–S9), and validation wording matching Report 04.
3. Documentation tools:
   - `results/RUN_NOTES_TEMPLATE.md`, the per-run notes: why, what changed, result, what we learnt, decision.
   - `results/_inbox/` for zips downloaded from Kaggle.
   - `scripts/register_run.py`, which files a zip into `results/<RUN>/`, creates `NOTES.md`, appends the register line and moves the zip to `_inbox/filed/`. It was tested on a mock run zip.
   - `results/EXPERIMENTS.md` restructured: planned runs on top, the register at the bottom.
4. Kaggle: the environment check is now run `S00_env_check`. Updated the launcher, `runs.json` and `kaggle/README.md`.
5. Updated README, `results/README.md` and `PROJECT_TRACKER.md` (shared / Track A / Track B phases, milestones for both papers, decisions log).
6. Created an empty `main` branch as the base, and opened a pull request with the whole setup for your review.

**Next**
- You: review and merge the pull request, then run `S00_env_check` on Kaggle and put `results_S00_env_check.zip` in `results/_inbox/`.
- Me (Step 1, shared): data loader, metrics and statistics modules, then the YOLO reproduction runs S01 and S02.

---

## 2026-11-26 · Step 4: Ranking the directions, research plan, repository set-up

**Done**
1. Scored D1–D9 on novelty, evidence in our data, chance of a clear result, Kaggle feasibility and reviewer value. Ranking: D3 > D7 > D2 > D4 > D9 > D1 = D8 > D6 > D5 (`Analysis/03_Idea_Ranking_and_Research_Plan.md` §2).
2. Checked the literature behind the top directions:
   - Hierarchical proximal femur classification exists only as classifier cascades or boxes, without fracture masks (Tanzi 2020; Jiménez-Sánchez 2019).
   - Left-right comparison exists only for pelvic fracture detection (AASN, ECCV 2020).
   - No hip paper we could read combines fracture masks with location → grade decisions, or uses the contralateral femur for segmentation or grading.
3. New evidence from the ground-truth masks:
   - Mask height alone separates Garden I–II vs III–IV with AUC 0.81 (train) / 0.84 (test).
   - Mask area separates Evans I–II vs III–V with AUC 0.80 / 0.75.
   - Centroid height alone predicts the location of test fractures with 80% accuracy.
4. Designed two options:
   - **A, SLG-Net (Segment, Locate, Grade), recommended.**
   - **B, RefFrac** (healthy reference femur), higher risk; suggested as Paper 2.
   - Wrote the full experiment plan: baselines B1–B9, ablations A0–A6 and an oracle study, metrics, statistics, pre-specified endpoints, Kaggle compute (about 70–80 GPU-h core), and a 12-week timeline.
5. Wrote and ran `scripts/make_splits.py` on the local data. It produced `splits/folds_v1.csv`: 5 folds of 208–209 femurs, grouped by x-ray, stratified by class, with the duplicate pair merged. Class counts match the paper exactly.
6. Read the dataset's own `data.yml` and `readme.md`. Class codes are `2AB, 2CD, 3AB, 3CDE, 4`: the location digit plus grade letters, so the labels are hierarchical. `data.yml` points val to `valid/images` while the folder is `val/`. This is harmless, because we write our own YOLO data files.
7. Set up the repository: README (workflow local ↔ GitHub ↔ Kaggle), `.gitignore`, `requirements.txt`, `src/femurseg/` (class table, data finder), `kaggle/` (launcher notebook, run registry, `env_check.py`), `results/` (experiment register), and updated trackers.

**Next**
- You choose Option A, B, or A then B (Report 03 §11).
- Run `env_check` on Kaggle (Step 0) and log it in `results/EXPERIMENTS.md`.
- Step 1: data loader, metrics module and the YOLOv8-seg baseline reproduction (identical for both options).

---

## 2026-11-26 · Step 3: Guided walkthrough and dataset-usage directions

**Done**
1. Gave a full step-by-step walkthrough of the article (7 parts) and the dataset (4 parts) in the project thread.
2. Wrote `Analysis/02_How_to_Use_the_Dataset.md`: why the dataset is worth using, what to avoid, and 9 evidence-backed directions (D1–D9), cited from our EDA, the base paper and the literature review.

**Next**
- Choose the core novelty (likely among D2 segmentation-first, D3 hierarchical grading, D4 contralateral comparison, D5 label-noise-aware), plus supporting pieces D1, D7, D9.

---

## 2026-11-26 · Step 1 and Step 2: Base article reading and dataset EDA

**Done**
1. Read the base article (González et al. 2025) page by page and extracted every method detail, number, limitation and stated future work.
2. Inspected the local dataset folder `Femoral Fractures/` and decoded the label format.
3. Wrote and ran `EDA/eda_femoral.py` (works locally and on Kaggle). Produced `EDA/per_image_stats.csv`, `EDA/summary.json`, `EDA/figures/01..07_*.png`.
4. Wrote the combined report `Analysis/01_Base_Article_and_EDA_Report.md`.

**Key findings**
- Local data matches the paper's train (1043) and test (210) femur counts exactly; class counts match Table 1 exactly.
- The validation split (201 femurs) is **missing** (empty `val/`, also reported on IEEE DataPort).
- One pixel-identical image pair in train has two different masks (IoU 0.70): evidence of annotation variability.
- Test neck-fracture masks are 35–40% smaller than train ones (committee-reviewed test labels): a train/test annotation shift.
- 398 x-rays contain a fractured femur plus the healthy contralateral femur; no x-ray has two fractured femurs.
- Left and right femurs are not mirrored to one side.
- Labels: YOLO-seg polygons, class 0–4; masks store class+1; no-fracture femurs have no label file.

**Next**
- Confirm whether the Kaggle copy has `val/`, and read `data.yml` / `readme.md`.
- Combine with the literature-search thread, then move to gap analysis and idea selection.
