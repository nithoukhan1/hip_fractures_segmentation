# Progress Log (step-by-step)

Newest entries at the top. Each entry: what was done, what it produced, what comes next.

---

## 2026-11-26 · Step 9: Step 2 code, one training script for S04–S10 (classifiers, segmenters, M0)

**Done (you)**
- Merged PR #4 (`b1a2719`) and started `S03_yolo11s_seg_1024` on Kaggle.

**Done (me)**
1. `scripts/train_net.py`: one script for the 7 remaining shared runs:
   - S04 DenseNet121, S05 EfficientNet-B4 and S06 ConvNeXt-T (= A0) image classifiers;
   - S07 U-Net, S08 DeepLabV3+ (ResNet34) and S09 SegFormer (MiT-B2) segmenters;
   - S10 = the core model M0 (= A1 = R0).
   It trains 5 folds, two at a time on the two T4s, with the same rules as S01: fixed length, final weights, OOF threshold, test mean ± SD and ensemble.
2. A pre-flight before training: it downloads the weights, finds the largest batch that fits the GPU (otherwise smaller batches with gradient accumulation to 16) and times a step. `S00_net_check` runs only this, for all 7 networks.
3. Shared modules:
   - `src/femurseg/evaluation.py`: the per-case files and scores, now used by every model. It re-scores S01 and S02 to exactly their `metrics.json`, and records NSD at 10 and 20 px as well.
   - `runner.py`: parallel folds with the time guard, also used by the YOLO script.
   - `torchdata.py`: every image read once into a resized cache.
   - `nets.py`: networks, losses and the decision rule.
4. Tests:
   - `tests/test_nets.py`: M0's class output is identical to the ConvNeXt-T classifier's with the same weights; decision rules, losses and the cache.
   - `tests/smoke_train_net.py`: full 5-fold CPU runs of a classifier, a segmenter and M0, plus both analysis scripts.
   - A learning check on synthetic data reached Dice 0.97.
   - The image cache was built from the real local data (1,253 femurs, 43.5 s). Every mask holds only its femur's class, and the smallest fracture mask has 1,722 pixels at 768 × 512 (median 23,045), well above the 500 pixels the segmenters' fracture score averages.
   - The YOLO smoke test still passes.
5. `run_stats.py` and `run_breakdown.py` handle runs without masks: classifiers get no Dice, not Dice 0.
6. The full protocol is written down before any result: `Analysis/05_Step2_Baselines_Protocol.md`.
7. Sent everything as PR #5 and synced your local folder.

**Next**
- You: merge PR #5, then run `S00_net_check` (about 15 min) in a second session on your main account with `BRANCH = "main"` (before the merge, `"claude/project-thread-kges8j"`). Put the zip in `results/_inbox/`.
- Me: file it and set the measured batch sizes and run times in `kaggle/runs.json`. The first training runs are then S06 and S10.
- When S03 finishes: I file it and write analysis C02 (S01 vs S03).

---

## 2026-11-26 · Step 8: YOLO baselines S01 and S02 filed and analysed

**Done (you)**
- Merged PR #3 and ran `S01_yolov8s_seg_1024` and `S02_yolov8s_seg_640` with Save Version, in two parallel sessions (commit `29e1852`). Both finished all 5 folds with exit code 0: S01 in 176 min, S02 in 81 min.

**Done (me)**
1. Filed both runs with `scripts/register_run.py` (`results/S01_yolov8s_seg_1024/`, `results/S02_yolov8s_seg_640/`, register lines 3 and 4) and wrote their `NOTES.md`.
2. 95% intervals and paired S01 vs S02 tests with `scripts/run_stats.py` → `results/analysis/C01_yolo_1024_vs_640/stats.md`.
3. New `scripts/run_breakdown.py` → `breakdown.md` in the same folder:
   - recall, precision and Dice per class;
   - every metric as mean ± SD over the 5 held-out folds and over the 5 fold models on the test set;
   - errors split like the article (detection, location, grade);
   - mask areas, and Dice under several definitions;
   - side by side with the article.
4. New `src/femurseg/reference.py`: the article's numbers (Table 2, Table 3, Fig. 3 confusion matrix). `tests/test_reference.py` checks that the typed confusion matrix gives every number the article derives from it (181/210, sensitivity, specificity, PPV, NPV, 8/6/15 errors). `tests/test_breakdown.py` checks the breakdown code; the smoke test now also runs both analysis scripts.
5. Wrote the interpretation: `results/analysis/C01_yolo_1024_vs_640/report.md`.

**Result**
- **Classification reproduces the article.** S01 fold models on the test set: AUC .985 ± .005 and 6-class accuracy 84.6 ± 2.1%, against .986 ± .008 and 84.4 ± 2.5% (article, 6 runs). S02: 82.2 ± 2.1% against 82.0 ± 3.0%.
- **The 5-model ensemble of S01 is the strongest baseline:** 88.1% (185/210, 95% CI 82.9–92.1), AUC .994, 25 errors against the article's 29.
- **Dice is .02 to .04 lower than the article's** (detected fractures: .741 at both sizes vs .761 and .780). The article's "640 px gives better masks" is not reproduced: S01 and S02 do not differ on any endpoint after the Holm correction.
- **Errors are mostly grades:** location is right for 96.6% of found fractures, the grade for 86.1% of those. Weak spots: Garden I–II (recall .44 ± .12) and Evans III–V called Evans I–II.
- **Finding:** the test masks of neck fractures are 35–40% smaller than the training ones (the article's test labels were reviewed by a committee). The models draw the same mask size on both, so they over-segment test neck fractures about 2 times. This explains most of the OOF-to-test Dice drop (.800 → .741).

**Decisions**
- S01 is the reference YOLO baseline for both papers: fold models mean ± SD, ensemble, and OOF for development.
- Dice is reported per location, OOF and test side by side. Nothing learned from the test set is used to correct it.
- New training code records NSD at 10 and 20 px as well (2 px is uninformative: about .05).
- S03 = YOLO11s-seg at 1024 px with S01's protocol (added to `kaggle/runs.json`).

**Next**
- You: merge PR #4, then run `S03_yolo11s_seg_1024` with Save Version (about 3 h). Check how much GPU quota Kaggle shows as used this week: it tells us whether a T4 × 2 session counts once or twice.
- Me: Step 2, one training script for the classifiers S04–S06, the segmenters S07–S09 and the core model M0 (768 × 512 input), tested on the CPU first.

---

## 2026-11-26 · Step 7: Kaggle check confirmed (M0 reached); Step 1 code: data loader, metrics, YOLO baselines

**Done (you)**
- Merged PR #2 and ran `S00_env_check_r2` on main (commit `2758d45`, 3.7 min, exit 0).

**Result** (`results/S00_env_check_r2/`, register line 2)
- Same data True, same folds True, all 1,253 files found, and the 104 femurs of Kaggle's val/ pooled into train.
- So the Kaggle data and folds are exactly the local ones. **Milestone M0 reached.**

**Done (me): Step 1 code, shared by both papers**
1. `src/femurseg/data.py`:
   - reads the fold file into records;
   - `fold_split` gives the training and held-out femurs of each fold;
   - `ImageStore` finds every file on any copy of the data and keeps images and masks in RAM.
2. `src/femurseg/metrics.py`:
   - segmentation: Dice, IoU, HD95 and normalised surface Dice, with the empty-mask conventions stated;
   - classification: AUC, sensitivity and specificity, 6-class accuracy, macro-F1, balanced accuracy, kappa;
   - also: the paper's error breakdown (detection, location, grade), calibration (ECE, Brier) and the Youden threshold. Ties go to the higher sensitivity.
3. `scripts/train_yolo_seg.py`, for baselines S01 and S02:
   - training: YOLOv8-seg from COCO weights, 100 epochs, Ultralytics 8.4.219 defaults, final weights. The held-out fold is never used to pick an epoch.
   - prediction: the paper's rule (fracture = any instance, class and mask of the top instance).
   - threshold: fitted on the out-of-fold predictions only, and also reported at Ultralytics' default 0.25.
   - test set: mean ± SD of the 5 fold models, and their ensemble.
   - Kaggle: two folds run at a time, one per T4, with an 11-hour time guard.
   - outputs: `metrics.json`, `summary.md` (with the paper's numbers alongside), and per-case CSVs for the statistics later.
4. `tests/`:
   - 13 metric checks against brute-force definitions and scikit-learn;
   - 4 checks of the fixed folds;
   - a CPU smoke test of the whole YOLO pipeline on a synthetic dataset with Kaggle's folder layout (val/ pooling, 5 folds two at a time, export, time guard).
   - All pass.
5. `kaggle/runs.json`: added `S01_yolov8s_seg_1024` (about 5–6 h) and `S02_yolov8s_seg_640` (about 2–3 h). `ultralytics==8.4.219` is pinned.
6. **Correction:** the paper's best-Dice setting is YOLOv8-*small* at 640 px (test Dice .780), not nano. S02 is now `S02_yolov8s_seg_640`, and Reports 03 and 04 are fixed.
7. `src/femurseg/stats.py` (statistics module) and `scripts/run_stats.py`:
   - 95% intervals: bootstrap over whole x-rays, because the two femurs of an x-ray are not independent. AUC also gets a DeLong interval, and accuracy an exact one.
   - Paired tests between runs on the same femurs: DeLong for AUC, McNemar for right/wrong decisions, a randomisation test for macro-F1, and Wilcoxon for Dice. Holm correction is applied.
   - Results go to `results/analysis/<name>/`, so run folders stay untouched.
   - 9 checks pass. One of them reproduces the paper's accuracy interval exactly: 181 of 210 correct gives 80.77–90.55 (exact Clopper–Pearson). So our intervals and the paper's are directly comparable.

**Differences from the paper** (stated in each run's NOTES and later in the papers)
- We use final weights, where the paper kept the best-validation checkpoint.
- We train 5 fold models, where the paper trained 6 seeds on one split.
- We use Ultralytics 8.4 defaults; the paper did not report its version or hyper-parameters.
- Our fracture threshold comes from out-of-fold predictions.

**Next**
- You: merge PR #3, then run `S01_yolov8s_seg_1024` with Save Version (guide in the thread), then `S02_yolov8s_seg_640`.
- Me, while S01 runs: the next baselines (S03 YOLO11s-seg, classifiers S04–S06). When S01 is back: file it and add 95% intervals with `scripts/run_stats.py`.

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
