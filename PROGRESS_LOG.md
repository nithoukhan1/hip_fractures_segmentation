# Progress Log (step-by-step)

Newest entries at the top. Each entry: what was done, what it produced, what comes next.

---

## 2026-11-27 · Step 13: S09's failed launch, a data check in the launcher, and calibration

**Done (you)**
- Merged PR #8 (`f342b8a`). Started six Kaggle sessions at the same time: `S07_unet_r34`, `S04_densenet121`, `S08_deeplabv3p_r34`, `S05_efficientnet_b4`, `A_dev_round1` and `A_dev_round2`.
- `S09_segformer_b2` stopped at once with "no folder with train/images and test/images found under /kaggle/input" (exit code 1, 0.0 min, commit `4820f6b`).

**Done (me)**
1. **Why S09 failed:** that session found no dataset under `/kaggle/input`, most likely because no dataset was attached to its notebook. The code is not at fault: the other sessions find the data, and nothing was trained. The zip holds only `run_meta.json`. It is renamed `results/_inbox/filed/results_S09_segformer_b2_failed_no_data.zip` and not filed, so the name `S09_segformer_b2` stays free for the relaunch.
2. **The launcher now checks the data first** (`kaggle/run.py`):
   - before installing anything, a training run (or a parallel group) looks for `train/images` and `test/images`;
   - if they are missing, the log shows **NO DATASET FOUND**, what `/kaggle/input` holds, and how to attach the dataset; the run stops within seconds with exit code 2, and `run_meta.json` records the reason;
   - `scripts/register_run.py` does not file a run that never started; in a parallel group, the other run is still filed;
   - 6 new tests (`tests/test_kaggle_run.py`).
3. **Calibration** (`scripts/run_stats.py`, protocol `Analysis/06_…` §6):
   - one temperature per run, fitted on its out-of-fold femurs, then applied unchanged to the test set;
   - ECE, Brier score and NLL are reported before and after, for the OOF femurs, the test ensemble and the fold models;
   - it is reported only: no threshold or decision uses it, and each femur keeps its most probable class.
4. **First calibration values** (from the filed runs, no new training):

   | Run | Temperature | ECE, OOF | ECE, test ensemble | ECE, test fold models |
   |---|---|---|---|---|
   | S06 (ConvNeXt-T) | 3.43 | .127 → .046 | .102 → .066 | .156 → .066 |
   | S10 (M0) | 3.14 | .108 → .040 | .103 → .064 | .131 → .062 |

   - Both models are over-confident by about the same factor, and one temperature fitted on OOF also works on the test set.
   - Not applicable to the YOLO runs: YOLO gives probability 0 to every class without a detected instance. For 73–99 of the 1,043 OOF femurs that is the true class, so the NLL is undefined. Their ECE and Brier are reported before calibration only.
5. **Checks:**
   - 8 new unit tests (6 launcher, 2 calibration); all 57 unit tests pass.
   - The CPU smoke test now also checks calibration in both statistics steps, and every step passed. One new check was too strict (it did not allow the note for a temperature at the end of its search range); it was corrected and holds on the same outputs.

**Next**
- You: in the S09 notebook, open the right panel → **Input** → **Add Input**, add `utopianstar/dataset`, keep `RUN = "S09_segformer_b2"` and `BRANCH = "main"`, and Save Version (about 3.3 h). Put each zip in `results/_inbox/` as it finishes.
- Me: file each zip; C03 and C04 → **M1**; C06 from the two development rounds; meanwhile Paper B step B-1 (protocol for the pair index and the retrieval bank).

---

## 2026-11-27 · Step 12: Paper A's protocol (Step 3) and SLG-Net code, tested on the CPU

**Done (you)**
- Merged PR #7 (`4820f6b`), which filed S03, S06 and S10 with analyses C02 and C05. The baselines S07, S04, S08, S09 and S05 are next on Kaggle.

**Done (me)**
1. **Protocol** `Analysis/06_Step3_PaperA_Protocol.md`, fixed before any SLG-Net result:
   - what each of A2–A6 adds on top of M0, with the Step 2 recipe unchanged;
   - three fold-0 development questions: D1 teacher forcing, D2 a detached mask, D3 region pooling at 1/32 only;
   - their decision rule against M0's own fold-0 model, the freeze (tag `A-v1-frozen`), the interpretability tests, and the final runs and analyses C06–C09.
2. **SLG-Net code** `src/femurseg/slg.py`, used as `--net slg:convnext_tiny.fb_in22k_ft_in1k:<A2..A6>` in `scripts/train_net.py`:
   - region pooling inside the predicted mask at 1/16 and 1/32, weighted by the mask confidence;
   - 8 geometry numbers of the mask (area, centroid, height, width, orientation, confidence);
   - fracture → location → grade heads whose 6-class probabilities come from the chain rule; one grade trunk per location (A4) or a shared one (A5, A6); logit adjustment with the training fold's priors (A6, training only);
   - teacher forcing: the heads get the released mask with a probability that falls from 1 to 0 over the first 30 epochs.
3. **Interpretability tests inside the training run** (`--extra-tests oracle,deletion`): every held-out and test femur is predicted again with the released mask (oracle), with the predicted region blurred out (deletion), and with the same shape blurred elsewhere (control). Each test gets its own per-case files and metrics.
4. **Kaggle:** `kaggle/run.py` can start two runs in one session, one per T4 (a *parallel group*, for example `A_dev_round1`). Their image cache is built once. `register_run.py` files the group's zip as separate runs, and `run_stats.py --folds 0` compares runs on fold 0 alone.
5. **Checks:**
   - 10 new unit tests (`tests/test_slg.py`): chain-rule probabilities, the geometry of a drawn ellipse, region pooling, logit adjustment, teacher forcing, a detached mask, the deletion images. The 39 earlier unit tests still pass.
   - The CPU smoke test now also trains SLG-Net A6 on 5 folds with both tests, runs a parallel group of two fold-0 runs, files the group's zip and compares the runs with M0 on fold 0. It passed.
6. **M0's fold-0 reference** (from the filed S10, no new run): macro-F1 .822 (.735–.889), accuracy 88.5% (184 of 208), Dice on fractured femurs .749.

**Next**
- You: continue the baselines (S07 + S04, then S08 + S09, then S05). After this PR is merged, run `A_dev_round1` and then `A_dev_round2` (about 55 min each) whenever a session is free; they do not depend on the baselines.
- Me: file each zip; C03 and C04 → **M1**; C06 decides D1–D3 → freeze → **M2-A**.

---

## 2026-11-27 · Step 11: S03, S06 and S10 filed; analyses C02 and C05

**Done (you)**
- Merged PR #6 (`18e4774`) and ran `S03_yolo11s_seg_1024` (from `b1a2719`, 178 min), `S10_m0_convnext_unet` (140 min) and `S06_convnext_tiny` (88 min) on Kaggle. All 5 folds of each finished with exit code 0.

**Done (me)**
1. Filed the three runs with `scripts/register_run.py` (register lines 6–8) and wrote their `NOTES.md`. Checked each run: every training femur is predicted once, by the model that did not train on it, and `metrics.json` equals a re-evaluation of the per-case files.
2. **C02, YOLO11s-seg (S03) vs YOLOv8s-seg (S01)** (`results/analysis/C02_yolov8_vs_yolo11/report.md`): no difference on any endpoint (smallest Holm-corrected p = .72). S01 stays the YOLO baseline.
3. **C05, does learning the mask help the classifier? (S06 vs S10)** (`results/analysis/C05_mask_task_s06_vs_s10/report.md`):
   - Yes on the 1,043 OOF femurs: accuracy 86.4% → 88.4% (Holm p = .026), macro-F1 .784 → .825 (Holm p = .013). M0 is better on all 5 held-out folds.
   - Same direction on the 210 test femurs (fold models 83.5% → 86.0%), not significant.
   - The gain comes from fewer detection and location errors. Grade errors are unchanged (65 of M0's 121 OOF errors), with Garden I-II the weakest class.
   - M0's mask is already better than YOLO's: Dice on fractured femurs .777 against .740 OOF (formal test in C04).
   - Both ConvNeXt models are over-confident (M0's OOF threshold .018), so calibration matters.
4. `scripts/run_breakdown.py` now reports classifier and segmenter runs cleanly. Their header names the network, they are compared with the article's final model (and with its own DenseNet121 or EfficientNet-B4 where it has one), and "n/a" replaces "nan". C01's report is unchanged.

**Next**
- You: run the remaining 5 baselines, two at a time: `S07_unet_r34` and `S04_densenet121` first, then `S08_deeplabv3p_r34` and `S09_segformer_b2`, then `S05_efficientnet_b4` (about 10 h in all). Put each zip in `results/_inbox/`.
- Me: file them, then C03 (classifiers vs YOLO) and C04 (segmenters vs YOLO), which complete the baseline table (milestone M1). Meanwhile, Step 3 code: Paper A's SLG-Net heads on top of M0 (A2–A6, tested on the CPU).

---

## 2026-11-26 · Step 10: S00_net_check filed; measured run times for S04–S10

**Done (you)**
- Merged PR #5 (`a15364b`) and ran `S00_net_check` on Kaggle (3.5 min, exit code 0).

**Done (me)**
1. Filed it with `scripts/register_run.py` (`results/S00_net_check/`, register line 5) and wrote its `NOTES.md`.
2. Result: all 7 networks downloaded their ImageNet weights and trained on a T4.
   - Micro-batch 16: DenseNet121, ConvNeXt-T, U-Net, DeepLabV3+ and M0.
   - Micro-batch 8 with 2 accumulated steps (still 16 images per update): EfficientNet-B4 (16 needed 14.6 GB) and SegFormer (16 ran out of memory).
3. Set the measured run times in `kaggle/runs.json`: S04 1.6 h, S05 2.1 h, S06 1.7 h, S07 1.7 h, S08 1.5 h, S09 3.3 h, S10 2.8 h (about 14.6 h in all).
4. Added the result to the protocol as an addendum (`Analysis/05_Step2_Baselines_Protocol.md` §7). The recipe is unchanged.
5. `scripts/register_run.py` now writes "Tesla T4 x2" for a net check that used both GPUs.

**Next**
- You: run `S10_m0_convnext_unet` (about 2.8 h) and `S06_convnext_tiny` (about 1.7 h), in two sessions if two are free. Put each zip in `results/_inbox/`.
- Me: file each run; with both in, analysis C05 (S06 vs S10: does learning the mask help the classifier?). When S03 finishes: file it and write C02 (S01 vs S03).

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
