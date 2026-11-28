# Progress Log (step-by-step)

Newest entries at the top. Each entry: what was done, what it produced, what comes next.

---

## 2026-11-28 · Step 27: Fold-0 screening results: no candidate passes

**Done (you)**
- Merged PR #24; ran all three screening sessions (`N_screen_round1`–`N_screen_round3`) and put their zips in the inbox; looked at the FracAtlas check sheets.

**Done (me)**
1. **Filed the six screening runs** with `register_run.py` (register lines 40–45, `results/N_*_fold0*`, each with its NOTES.md).
2. **Scored them** on fold 0 against the base, by the rule fixed before the runs (`Analysis/11` section 7.5; `results/analysis/N_screening_fold0/`):

   | Candidate (mean of two seeds) | Macro-F1 | Dice |
   |---|---|---|
   | The base: ConvNeXt U-Net, two runs | .817 | .755 |
   | Segment-and-zoom network | .829 (+.013) | .763 |
   | ConvNeXt U-Net at 1152 × 768 | .786 (−.031) | .773 |
   | ConvNeXt U-Net with ConvNeXt-S | .801 (−.016) | .760 |

   - **No candidate passes** (pass mark: macro-F1 ≥ .837 and Dice ≥ .745). None of the differences is significant (p ≥ .33).
   - The segment-and-zoom network is the only candidate above the base on both measures. Its crop holds the fracture centre in 135 of 136 fractured femurs.
   - Fold 0 is noisy: one subtrochanteric femur (7 on fold 0) moves macro-F1 by about .02, which is the size of the pass margin.
3. `scripts/run_standard_metrics.py` gained `--folds` (as `run_stats.py`), so the screening uses the same standard metrics.
4. Checked the FracAtlas check sheets: every crop shows a half pelvis with the femoral head in the upper third, and every red outline sits on the fractured hip. Two outlines lie low (images 3341 left and 3342 left, subtrochanteric), as noted in `Analysis/11` section 8.2a. In four healthy images (0275, 0781, 2197, 2255) a bright band (the edge of a lead shield or the table) crosses the lower part of the crop. They are kept, because they are part of the real images.
5. Defined the 5-fold segment-and-zoom run `N1_segzoom` in `kaggle/runs.json`, to be launched only if you choose it.

**Next**
- You: decide what comes after the screening (decision card in the thread).
- Me: the FracAtlas zero-shot test job, which every option needs.

---

## 2026-11-28 · Step 26: Standard-metric tables for every model, and the FracAtlas rules

**Done (you)**
- Merged PR #23 (the screening code and runs).

**Done (me)**
1. **Standard-metric tables** (`results/analysis/standard_metrics_all_runs/standard_metrics.md`, made by `scripts/run_standard_metrics.py`):
   - all 24 models, out of fold and on the test set;
   - classification: accuracy, macro precision, macro recall, macro-F1, sensitivity, specificity, fracture AUC and the 6-class AUC; per-class precision, recall and F1; confusion matrices;
   - segmentation: Dice, IoU and HD95;
   - 95% intervals.
   - The numbers equal those of the earlier analyses: for example, ConvNeXt U-Net has 922/1043 correct, macro-F1 .825 and Dice .777 out of fold.
2. **FracAtlas rules** (`Analysis/11` section 8), written before any model sees a FracAtlas image:
   - which hips are included: adult, frontal view, no implant, proximal femur visible;
   - how each hip is labelled;
   - the crop, placed by the femoral head so that it matches the geometry of a HUSJA-UA image;
   - what is reported.
3. **FracAtlas external test set** (`Analysis/11` sections 8.2 and 8.2a, `results/fracatlas/`):
   - screened the hip images by eye: all 22 usable fractured images and the first 60 eligible healthy images in a fixed random order (seed 2026); 82 images, 148 hips;
   - read two landmarks per hip by eye (the pelvic midline and the centre of the femoral head, without the fracture outlines): `results/fracatlas/landmarks.csv`; the 40 hips left out and why: `results/fracatlas/excluded.csv`;
   - the crop: a half of the pelvis, the femoral head placed where it lies in a HUSJA-UA image (calibrated on 20 of them), resized to 1024 x 1536 and windowed; made by `scripts/make_fracatlas_crops.py` (2 unit tests in `tests/test_fracatlas_crops.py`);
   - result: **148 hips, 23 fractured and 125 healthy** (`results/fracatlas/cases.csv`). The crops and masks stay on your PC in `FracAtlas_crops\` (not in the repo), with three check sheets.

**Next**
- You: look at `FracAtlas_crops\check_sheet_1.jpg` to `check_sheet_3.jpg` (every crop with its fracture outline in red); merge PR #24; launch `N_screen_round1` if it is not running yet, and send me its zip.
- Me: score round 1; then plan the FracAtlas test job on Kaggle (you upload the crops as a private Kaggle dataset).

---

## 2026-11-28 · Step 25: Fold-0 screening of the new method candidates (code and runs ready)

**Done (you)**
- Merged PR #22 and approved the plan of `Analysis/11`.

**Done (me)**
1. **Screening rules, written before any run** (`Analysis/11` section 7):
   - the three candidates as built: the segment-and-zoom network, the ConvNeXt U-Net at 1152 × 768, and the ConvNeXt U-Net with a ConvNeXt-S encoder;
   - the base to beat, from the two filed ConvNeXt U-Net runs on fold 0: macro-F1 .817, Dice .755;
   - the pass rule: macro-F1 ≥ .837 and Dice ≥ .745, mean of two seeds.
   - One change from the plan, stated before the runs: the higher-resolution candidate is screened at 1152 × 768 instead of 1536 × 1024, which would take about 3 h for one fold.
2. **The segment-and-zoom network** (`src/femurseg/zoomnet.py`, `--net zoom:<model>` in `scripts/train_net.py`):
   - the ConvNeXt U-Net finds the fracture;
   - a second ConvNeXt-T grades it from a 640 × 640 full-resolution crop around the predicted mask;
   - every femur's crop is saved, with the pointing game of the crop, for explainability.
3. **Tests:** 5 new unit tests (`tests/test_zoomnet.py`) and an end-to-end training check on synthetic data (`tests/smoke_train_zoomnet.py`). All 97 unit tests pass, and the existing end-to-end check passes too.
4. **Kaggle runs** (`kaggle/runs.json`): three sessions of two runs each, `N_screen_round1` to `N_screen_round3`. The register files them under the new track "New method".

**Next**
- You: merge the PR, then launch `N_screen_round1` (about 1.8 h on T4 x2).
- Me, while it runs: the standard-metric tables for all filed runs, and the FracAtlas inclusion and crop rules.

---

## 2026-11-28 · Step 24: New direction: standard metrics, FracAtlas, and a method that must beat our baselines (plan only)

**Done (you)**
- Merged PR #21.
- Set the new requirements: the proposed method must beat our own trained baselines, generalise to FracAtlas and be explainable, with standard metrics only.

**Done (me)**
1. **`Analysis/11_New_Direction_Metrics_FracAtlas_Method_Plan.md`:**
   - why the comparison with our retrained YOLOv8s-seg is like-for-like;
   - the standard-only metric set;
   - what FracAtlas can test;
   - the candidate methods with their Kaggle hours and risks;
   - the steps.
2. **FracAtlas inspected on the PC** (read-only):
   - 338 images flagged "hip", 63 of them fractured.
   - After looking at all 63, only about 22 are adult proximal femur fractures without an implant. The rest are operated hips, children, femoral shafts or forearms.
   - FracAtlas has no fracture type or grade, so it can test fracture yes/no, localisation and segmentation only.

**Next**
- You: approve the plan, or change it. Nothing runs before that.
- Me, after approval: the written rules, the FracAtlas crops and the standard-metrics tables, then the fold-0 development runs.

---

## 2026-11-28 · Step 23: Full names instead of short codes in every document

**Done (you)**
- Merged PR #20 and asked for the short codes (for example C13, M0, R4, S01, D1) to be replaced by their real names everywhere in the documentation.

**Done (me)**
1. **Every document now uses full names:** the trackers, this log, the Analysis reports, every run's notes, every analysis report, the experiment register and the READMEs.
   - Examples: ConvNeXt U-Net (was M0 and S10), YOLOv8s-seg (the published method, was S01), RefFrac (was R4), the reference analysis (was C13), "section 4" (was §4).
   - About 4,500 codes in 101 documents were replaced. The literature notes in `research/` were left alone, because their A1–A3 are AO fracture types, not our codes.
2. **New `GLOSSARY.md`:** the folder and run names, which keep their codes because the scripts and the Kaggle run list depend on them, next to the names the documents use.
3. **New ground rule 6** in `README.md`: full names in documents, never short codes.
4. No code, result or number changed.

**Next**
- You: merge this PR, and answer the second-backbone question when you are ready.
- Me: the figures, then the manuscript draft.

---

## 2026-11-28 · Step 22: The combined paper's idea, novelty and defence; ensembles tested

**Done (you)**
- Merged PR #19 and asked: what is the combined idea, its novelty and contribution, how do we defend it when neither extension beat ConvNeXt U-Net, and what next.

**Done (me)**
1. **Ensemble analysis, ensembles of saved predictions** (`scripts/run_ensemble.py`, 2 new tests; no GPU; `results/analysis/C17_ensembles/report.md`):
   - averaging ConvNeXt U-Net with SLG-Net and RefFrac gives the best out-of-fold macro-F1 so far (.842 vs .825), but it is not significant (p .12) and hardly more than averaging two ConvNeXt U-Nets (.828);
   - on the test set ConvNeXt U-Net + SLG-Net gets 189 of 210 right (macro-F1 .843), but it is the weakest ensemble out of fold, so this is not evidence. Exploratory, supplement at most.
2. **Analysis/10** (`Analysis/10_Novelty_Defense_and_Next_Plan.md`):
   - the combined idea: segment first (ConvNeXt U-Net), then make the decisions use the right evidence (region grounding, healthy references), each tested;
   - five points of novelty and five contributions;
   - a candid view of Q2/Q3 chances, with reviewers' likely objections and our answers;
   - every accuracy option with its Kaggle hours;
   - the test set cannot detect differences under about 4–5 accuracy points (9–10 femurs);
   - the numbered plan.
3. **Trackers:** EXPERIMENTS.md, PROJECT_TRACKER.md, README.md.

**Next**
- You:
  1. merge this PR;
  2. answer the second-backbone card (recommended: write first, run it later).
- Me: the figures (Fig. 2, Fig. 3), then the manuscript draft in `manuscript/`.

---

## 2026-11-28 · Step 21: ConvNeXt U-Net's deletion test, and the plan of the combined paper

**Done (you)**
- Merged PRs #17 and #18 and chose one paper (Paper B's findings join Paper A).
- Ran `A1_m0_deletion` from `main` and put its zip in `results\_inbox\filed\`.

**Done (me)**
1. **Filed `A1_m0_deletion`** (register line 39): from `080a0c5`, 5/5 folds, 2.46 h. It replicates ConvNeXt U-Net (921 vs 922 correct out of fold, macro-F1 .825 vs .825).
2. **The test fixed in Analysis/06 section 13** (`scripts/run_interpretability.py --compare-deletion`, 2 new tests):
   - of the 525 fractured femurs that both models get right, removing the predicted region makes 64.2% wrong for SLG-Net and 48.2% for ConvNeXt U-Net (117 vs 33 femurs, p < .001);
   - test set 65.5% vs 52.6% (p .002), the same direction; both controls lose under 1.5%;
   - **verdict: SLG-Net's location and grade decisions depend on the fracture region more than ConvNeXt U-Net's.** This is now SLG-Net's main claim (SLG-Net interpretability analysis, section 2b, Analysis/06 section 14).
3. **Plan of the combined paper** (`Analysis/09_Combined_Paper_Plan.md`): what "one paper" means, what it claims (8 claims with their evidence), outline, tables and figures, and what is left.
4. **Trackers:** EXPERIMENTS.md, PROJECT_TRACKER.md (Paper A milestone 3 reached; compute 69.2 h), README.md, kaggle/README.md.

**Next**
- You:
  1. merge this PR;
  2. decide on the optional second-backbone check (about 6–8 h; see our thread).
- Me: the figures from the filed results (Fig. 2, Fig. 3), then the manuscript draft (Methods and Results first).

---

## 2026-11-28 · Step 20: SLG-Net geometry, the ConvNeXt U-Net deletion test prepared, Paper B's final runs analysed (RefFrac analyses)

**Done (you)**
- Ran the eight Paper B final runs from `B-v1-frozen` and put their zips in `results\_inbox\filed\`, with `A6_slgnet.zip` (SLG-Net's maps).
- Chose option A for Paper A: keep SLG-Net as pre-registered and add ConvNeXt U-Net's deletion test. Merged PR #16 (the run `A1_m0_deletion` and its rule, `Analysis/06` section 13).

**Done (me)**
1. **Filed the eight runs** (register lines 31–38): all from `1c88912`, 5/5 folds, exit 0, 28.6 h in all. Each has its `NOTES.md`.
2. **SLG-Net's geometry test** (SLG-Net interpretability analysis, section 4): its decided grades follow its own region (Garden height AUC .87 OOF, .98 test; location 82.4% OOF) but no better than ConvNeXt U-Net (.92, .94; 83.8%).
3. **New tool `scripts/run_refstats.py`** (6 new tests): Garden I-II McNemar between runs, the reference analysis's subgroups, every R run's reference tests, and the own-hip model's modes with their own thresholds.
4. **Reference analysis, does a reference help?** (`results/analysis/C13_paperB_reference/report.md`):
   - Dice: no (every difference ≤ .005, also for small masks and for femurs with or without a partner);
   - Garden I-II recall out of fold rises with every reference model: ConvNeXt U-Net .544, the random-reference model .614, the nearest-reference model .684, RefFrac .702. RefFrac vs ConvNeXt U-Net is 12 vs 3 femurs (p .035) but Holm across the 4 pairs gives .14, the 16 test femurs show no gain, and folds 1–4 shrink it to +.109;
   - retrieval does not matter, and a random reference at prediction time changes nothing;
   - the model does use the comparison: with itself as the reference, RefFrac loses 201 of 663 found fractures.
5. **Own-hip analysis, the own other hip (own-hip model)**: not better than a retrieved femur; no pairing shortcut learnt; a fractured reference lowers P(fracture) of fractured femurs by .116.
6. **RefFrac component analysis, components**: alignment, fusion and AASN change how the reference is used, not accuracy; Garden I-II RefFrac > the no-alignment model (no alignment), Holm .031.
7. **RefFrac main-table analysis, main table**: RefFrac ties ConvNeXt U-Net (macro-F1 .826, Dice .777 OOF) and beats the base article's model YOLOv8s-seg out of fold (+.081 macro-F1, +.037 Dice); on the test set it is not better than ConvNeXt U-Net or YOLOv8s-seg.
8. **Trackers:** `Analysis/08` section 10, `EXPERIMENTS.md`, `PROJECT_TRACKER.md` (Paper B step 4 to Paper B step 6, Paper B milestone 3, compute 66.7 h), `README.md`.

**After this step (2026-11-28):** you merged PR #17 and chose to merge Paper B into Paper A as one paper (Analysis/08 section 11; no R sensitivity runs).

**Next**
- You:
  1. merge the small tracker PR for the one-paper decision;
  2. launch `A1_m0_deletion` from `main` (about 2.9 h) and send its zip.
- Me: file `A1_m0_deletion` and run the section 13 test; then plan the one combined paper (outline, main table, figures).

---

## 2026-11-27 · Step 19: analyses of Paper A's final runs (SLG-Net ablation analysis, the SLG-Net interpretability analysis, the SLG-Net main-table analysis)

**Done (you)**
- Merged PR #14 and created the tag `B-v1-frozen` (on `1c88912`, the merge of PR #14). Paper B milestone 2 is reached.

**Done (me)**
1. **Analysis tools:**
   - `scripts/run_stats.py`: `--pairs` compares chosen pairs instead of every pair, and `--family-holm` adds Holm across the compared pairs (each endpoint and scope separately). Re-running the SLG-Net second-seed check gives an identical `stats.json`.
   - `scripts/run_interpretability.py` (new, the SLG-Net interpretability analysis): the oracle test, region deletion against its control, risk–coverage and geometry consistency, from the per-case files. Every change it computes equals what the run wrote in `metrics.json`.
   - `scripts/map_geometry.py` (new): the geometry of each femur's predicted region from a run's saved probability maps. It needs only numpy and OpenCV, so it runs on your PC next to the large Kaggle zips.
   - 5 new tests (82 in all), all passing.
2. **SLG-Net ablation analysis, the ablation chain** (`results/analysis/C07_paperA_ablation_chain/report.md`):
   - only the mask task helps significantly (ConvNeXt U-Net vs ConvNeXt-T classifier: macro-F1 +.042, Holm within the chain p = .022);
   - **SLG-Net is not more accurate than ConvNeXt U-Net out of fold:** macro-F1 .806 vs .825 (p = .17), Dice .774 vs .777 (p = .08). On the test set it is ahead (+.041, 186 vs 182 correct), not significantly;
   - Geometry model (geometry) has the best out-of-fold macro-F1 (.838), but SLG-Net was named the final model before any result, so the geometry model is exploratory only;
   - folds 1–4 alone move every value by .009 or less, so the fold-0 development did not inflate the results.
3. **SLG-Net interpretability analysis, interpretability** (`results/analysis/C08_paperA_interpretability/report.md`):
   - **deletion:** removing SLG-Net's predicted region loses 64.7% of its correct fractured decisions; the same shape elsewhere loses 1.3% (p < .001). Location and grade depend on the region; the fracture decision does not;
   - **oracle:** the released mask fixes 8 of SLG-Net's 130 wrong fractured femurs and breaks 3, so better masks would help little;
   - **risk–coverage:** the 10% least confident femurs hold 32–40% of the errors for every model; SLG-Net (32.4%) is not better than ConvNeXt U-Net (35.5%);
   - **geometry:** done for ConvNeXt U-Net (its decided grades follow its own region, AUC .92 for Garden height). SLG-Net waits for its maps.
4. **SLG-Net main-table analysis, the main table** (`results/analysis/C09_paperA_main_table/report.md`, Holm across 8 comparisons):
   - out of fold, SLG-Net beats the base article's model YOLOv8s-seg (macro-F1 +.061, Holm p = .006; Dice +.034) and EfficientNet-B4, and has a higher Dice than U-Net, DeepLabV3+ and SegFormer;
   - it is not significantly different from DenseNet-121, ConvNeXt-T, U-Net, DeepLabV3+, SegFormer or ConvNeXt U-Net in macro-F1;
   - test set: no macro-F1 difference survives Holm; SLG-Net has the most correct femurs (186 of 210).
5. **Trackers:** Analysis/06 section 12 (results and how Holm was applied), Analysis/08 section 9 (the tag `B-v1-frozen`), decisions on register lines 22–26, `EXPERIMENTS.md`, `PROJECT_TRACKER.md`, `README.md`, `kaggle/README.md`.

**Next**
- You:
  1. merge this PR;
  2. choose how Paper A presents SLG-Net (the decision in our thread);
  3. send `A6_slgnet.zip`, the full Kaggle output of the SLG-Net version (it holds `work/A6_slgnet/prob_maps/`), for the geometry test;
  4. keep launching the Paper B final runs from `B-v1-frozen`, and send their zips.
- Me: SLG-Net's geometry test when its maps arrive; the RefFrac analyses as the R runs arrive.

---

## 2026-11-27 · Step 18: 11 runs filed; the SLG-Net second-seed check (Paper A's freeze stands); the RefFrac development analysis (Paper B's settings frozen)

**Done (you)**
- Merged PR #12 and created the tag `A-v1-frozen`, then merged PR #13.
- Ran eight Kaggle sessions:
  - `A_dev_seed2` from `main`;
  - the five final runs `A2_region` … `A6_slgnet` from the tag;
  - `R_dev_round1` and `R_dev_round2` from `main`.
- Put the zips and notebooks in `results\_inbox\filed\`.

**Done (me)**
1. **Verified the tag and the launches:**
   - `A-v1-frozen` points at `f96f84c`, the merge of PR #12;
   - each final run's log shows that commit, and all five trained 5 of 5 folds with exit code 0 (2.38–2.91 h each);
   - `A_dev_seed2` cloned `f96f84c`, and the R development groups cloned `2619b39` (the merge of PR #13).
2. **Filed 11 runs** (register lines 20–30): the eight zips hold 11 runs, because each group zip holds two. Every run was checked:
   - all folds present;
   - every out-of-fold femur predicted once, by the model that did not train on it;
   - `metrics.json` equal to a re-evaluation of the per-case files.
   The second-seed runs were filed and judged first; **no final result was opened before the SLG-Net second-seed check.**
3. **SLG-Net second-seed check, Paper A's second-seed check** (`results/analysis/C11_paperA_seed2/report.md`), by the rule of Analysis/06 section 10:
   - over the two seeds, pooling at 1/32 only beats the base by **+.054 macro-F1** (.827 vs .773), with Dice +.016 and .827 ≥ .79;
   - all three conditions hold, so **the freeze stands** and the five final runs are valid;
   - seed noise: macro-F1 moved by at most .004 between seeds, and Dice by .010.
4. **RefFrac development analysis, Paper B's fold-0 development** (`results/analysis/C12_paperB_dev_fold0/report.md`), by the rule of Analysis/08 section 4:
   - **go / no-go passes:** the base (mean of two seeds) has Dice .770 on fractured femurs (ConvNeXt U-Net .749) and macro-F1 .818 (ConvNeXt U-Net .822);
   - Whole-reference question (attention over the whole reference): Dice +.009, below .01 and within the seed noise N = .009, and macro-F1 −.022, so the local window stays;
   - Margin-loss question (AASN's margin loss): Dice −.014, so there is no margin loss;
   - **the model uses its references:** with the target as its own reference, Dice falls by .090 and P(fracture) on fractured femurs by .49. A random healthy reference instead of the 3 nearest changes Dice by only .004–.005 (a first hint for the reference question);
   - the difference map hits the fracture in 64% and 92% of fractured femurs (two seeds), against 8% by chance.
5. **Paper B's freeze, ready** (Analysis/08 section 8): no setting changes.
   - The eight final runs are in `kaggle/runs.json`: `R4_refrac`, `R3_nearest_ref`, `R2_random_ref`, `R7_partner`, `R1_aasn`, `R5_concat`, `R6a_no_align` and `R6b_affine`.
   - The training code is unchanged since the development runs (`2619b39`).
   - Measured cost: an RefFrac fold takes 81 min on one T4. So the k = 3 runs take about 4.2 h each, the k = 1 runs about 3 h: **about 29 h in all.**
6. **Notes and trackers:** run notes for all 11 runs; register decisions; addenda Analysis/06 section 11 and Analysis/08 section 8; `kaggle/README.md` (measured times, the R final runs, the steps for the tag `B-v1-frozen`), `EXPERIMENTS.md`, `PROJECT_TRACKER.md`, `README.md`.

**Next**
- You:
  1. merge this PR;
  2. create the tag `B-v1-frozen` on `main` (GitHub → Releases, as for `A-v1-frozen`);
  3. launch `R4_refrac` with `BRANCH = "B-v1-frozen"`; then `R3_nearest_ref` and `R2_random_ref`, `R7_partner`, `R1_aasn`, and `R5_concat`, `R6a_no_align`, `R6b_affine`, as your GPU quota allows;
  4. send the zips as they finish.
- Me: the SLG-Net analyses, the analyses of Paper A's final runs (Step 19); then the RefFrac analyses as the R runs arrive.

---

## 2026-11-27 · Step 17: Paper B step 2 of Paper B: the RefFrac protocol, the model and the fold-0 development runs

**Done (you)**
- Merged PR #12 (`f96f84c`) and created the tag `A-v1-frozen` on it, so Paper A's final runs start from the frozen code.

**Done (me)**
1. **The Paper B step 2 protocol, written before any R run** (`Analysis/08_Step5_PaperB_RefFrac_Protocol.md`):
   - **the eight reference runs** (section 1): each differs from RefFrac (the main method) in one component, and all use ConvNeXt U-Net's recipe unchanged;
   - **the model exactly** (section 2):
     - the references go through the same encoder, without gradients;
     - alignment at 1/16 and 1/32: local cross-attention, none, or a global affine transform;
     - fusion [T, T − R̃, |T − R̃|] or concatenation, started as the identity, so the untrained network computes ConvNeXt U-Net's function;
     - AASN-style model = AASN adapted to femur crops (its margin loss on projected features);
     - Own-hip model = the patient's own healthy hip, tested with five kinds of reference to measure the pairing shortcut;
   - **the reference tests** every run records (section 2.6): the target as its own reference (`self`) and a random healthy femur (`random`); the own-hip model instead `blind`, `retrieved`, `random` and `fractured`; and a pointing game for the difference map;
   - **a fact from the reference table alone:** fractured femurs get less similar references than healthy ones (nearest centred cosine .574 vs .612 out of fold), the signal RefFrac is meant to use;
   - the honesty rules (section 3), the fold-0 development and its decision rule (section 4), the final runs, the RefFrac analyses and compute (section 5), and the known limitations (section 6).
2. **The code** (`src/femurseg/refnet.py`, new; `--net ref:<timm model>:R1..R7` in `scripts/train_net.py`):
   - the reference table is checked against the fold file and the bank rules before training;
   - the references are loaded and mirrored as the table says, flipped with the target, and augmented on their own;
   - each reference test writes its own per-case files and probability maps; the summary adds the tests and the pointing game.
   - **ConvNeXt U-Net and SLG-Net are untouched:** on the smoke data their outputs are byte-identical to `main` (`f96f84c`). So `A_dev_seed2`, launched from `main`, trains the same code before or after this PR.
3. **Tests on the CPU:**
   - 11 new unit tests (`tests/test_refnet.py`): the untrained RefFrac equals ConvNeXt U-Net, no gradient reaches the references, the attention window, the alignments, the losses, the reference policies, bad tables refused, the references flipped with the target, and the pointing game. **All 73 unit tests pass** (every file except `tests/test_data.py`, which needs the dataset);
   - **the end-to-end smoke test passed** (`tests/smoke_train_refnet.py`, 9 min): the pre-flight of all 8 variants with their references; a 5-fold RefFrac run and a 5-fold the own-hip model run with their reference tests; a development group run through the Kaggle launcher, filed by `register_run.py` and compared with ConvNeXt U-Net by `run_stats.py --folds 0`.
4. **Fold-0 development runs** in `kaggle/runs.json` (Analysis/08 section 4), about 80 min per session, `BRANCH = "main"`:
   - `R_dev_round1` = `R4_dev_base` (GPU 0) + `R4_dev_base_s2` (the same with seed 2027, GPU 1);
   - `R_dev_round2` = `R4_dev_global` (whole-reference question: attention over the whole reference) + `R4_dev_margin` (margin-loss question: plus AASN's margin loss).
   - **The rule was written before the runs** (RefFrac development analysis):
     - base = the mean of the two seeds; go/no-go: Dice on fractured femurs ≥ .739 and macro-F1 ≥ .79 (ConvNeXt U-Net's fold 0: .749 and .822);
     - an alternative is adopted only if its Dice beats the base by ≥ .01 and by more than the seed noise, with macro-F1 at most .02 lower.
5. **Trackers:** `kaggle/README.md`, `results/EXPERIMENTS.md` (planned runs; the RefFrac analyses), `PROJECT_TRACKER.md` (Paper B step 2 in progress, Paper A milestone 2 reached with the tag, two decisions), `README.md`.

**Next**
- You:
  1. merge this PR;
  2. keep Paper A first: `A_dev_seed2` (`BRANCH = "main"`) and the five final runs (`BRANCH = "A-v1-frozen"`);
  3. then launch `R_dev_round1` and `R_dev_round2` (`BRANCH = "main"`), as your GPU quota allows;
  4. send the zips as they finish.
- Me:
  - SLG-Net second-seed check as soon as `A_dev_seed2` is filed, then the SLG-Net analyses when the final runs are in;
  - RefFrac development analysis when both R_dev rounds are filed; then the freeze of RefFrac (tag `B-v1-frozen`, made by you) and the final the eight reference runs in `kaggle/runs.json`.

---

## 2026-11-27 · Step 16: SegFormer-B2 and R00_r2 filed; the segmenter analysis; baseline table v1 (baseline table); Paper A's freeze approved; second-seed check

**Done (you)**
- Merged PR #11 (`d3d73af`). `R00_reference_bank_r2` and the relaunched `S09_segformer_b2` finished, and you sent both zips.
- Approved the freeze of Paper A's settings and the five final runs. You also asked for the second-seed check of the base and of 1/32 pooling, to run at the same time.

**Done (me)**
1. **Filed** the two runs (register lines 18 and 19). SegFormer-B2 was checked like the earlier runs: all 5 folds, every out-of-fold femur predicted once by the model that did not train on it, `metrics.json` equal to a re-evaluation of the per-case files.
2. **R00_reference_bank_r2, Paper B:** check 1 = **804 of 810 (99.3%)**, exactly as the review predicted, and the figure shows every reference with the pelvis on the target's side. **Step 1 of Paper B is complete** (Analysis/07 section 9). Compared with version 1, the rule changed mainly the orientation of the references (same nearest femur in 87.1% of rows, same orientation in 57.6%).
3. **Segmenter analysis, the segmenters vs YOLO** (`results/analysis/C04_segmenters_vs_yolo/report.md`), research the segmentation question:
   - **ConvNeXt U-Net has the best Dice** in every view (OOF .777, test ensemble .744). Out of fold it beats YOLO (+.037), SegFormer (+.011), U-Net (+.029) and DeepLabV3+ (+.028), all Holm p ≤ .0003.
   - **On the 210 test femurs** ConvNeXt U-Net beats U-Net and DeepLabV3+ (Holm p ≤ .0005), but its lead over YOLO (+.018, p = .36) and SegFormer (+.006, p = .44) is not significant.
   - YOLO's Dice deficit comes mostly from missed fractures (51 of 680 out of fold, against 16–28): on the fractures it finds, its masks are as good as ConvNeXt U-Net's.
   - **SegFormer (SegFormer-B2) is the best standard segmenter** and classifies like ConvNeXt U-Net. U-Net and DeepLabV3+ are no better than YOLO.
   - All five models draw the test neck fractures 1.7–2.0 times larger than the test masks (the annotation shift of the YOLO resolution analysis).
4. **Baseline table v1 = the baseline-table milestone** (`results/analysis/M1_baseline_table/report.md`): all ten baselines side by side, with intervals, test fold models and calibration. Planned for 13 Dec, reached on 27 Nov.
5. **Paper A's freeze** (Analysis/06 section 10):
   - The tag `A-v1-frozen` goes on `main` after this PR. GitHub refused my push of the tag (this session may push only its own branch), so you create it on GitHub in one step.
   - The training code at that commit is identical to `f342b8a`, which trained the development runs.
6. **Second-seed check** (`kaggle/runs.json`, group `A_dev_seed2`, about 55 min, `BRANCH = "main"`): `A6_dev_base_s2` and `A6_dev_scale32_s2` repeat the two development runs with seed 2027. **The rule was written before the runs** (Analysis/06 section 10, analysis the SLG-Net second-seed check):
   - 1/32 pooling stays if, on the mean of the two seeds, it beats the base by ≥ .02 macro-F1, loses ≤ .01 Dice and reaches macro-F1 ≥ .79;
   - otherwise the freeze goes back to you before any final-run result is opened.
7. **Notes and checks:** run notes for SegFormer-B2 and R00_r2, register decisions, the protocol addenda (Analysis/06 section 10, Analysis/07 section 9). Only `runs.json` changed among the code files; its tests pass (6 of 6).

**Next**
- You:
  1. merge this PR;
  2. create the tag `A-v1-frozen` on `main` (GitHub → Releases);
  3. launch `A_dev_seed2` (`BRANCH = "main"`) and `A6_slgnet` (`BRANCH = "A-v1-frozen"`); then `A2_region`, `A3_geometry`, `A4_hierarchy` and `A5_shared_grade` (same tag), two sessions at a time, as your GPU quota allows;
  4. send the zips as they finish.
- Me:
  - SLG-Net second-seed check (second seed) as soon as `A_dev_seed2` is filed, then the SLG-Net analyses when the final runs are in;
  - Paper B step 2's protocol: the Siamese RefFrac model and the eight reference runs.

---

## 2026-11-27 · Step 15: seven runs filed; the classifier analysis and the SLG-Net development analysis; Paper B's orientation rule reviewed

**Done (you)**
- Merged PR #10 (`dc152a9`). Seven Kaggle sessions finished: `S04_densenet121`, `S05_efficientnet_b4`, `S07_unet_r34`, `S08_deeplabv3p_r34`, `A_dev_round1`, `A_dev_round2` and `R00_reference_bank`.
- Relaunched `S09_segformer_b2` with the dataset attached. It is still running.

**Done (me)**
1. **Filed** the seven zips as nine runs (register lines 9–17; each group zip holds two development runs). Every run was checked:
   - all folds present;
   - every out-of-fold femur predicted once, by the model that did not train on it;
   - `metrics.json` equal to a re-evaluation of the per-case files.
2. **SLG-Net development analysis, Paper A's fold-0 development** (`results/analysis/C06_paperA_dev_fold0/report.md`), by the fixed rule of Analysis/06 section 4:
   - **go / no-go fails:** `A6_dev_base` has macro-F1 .776 < .79 (ConvNeXt U-Net .822). Accuracy is the same (184/208); two of the 7 subtrochanteric femurs make the gap (p = .25);
   - Teacher-forcing question: teacher forcing stays (+.008);
   - Detached-mask question: the end-to-end mask stays (+.019, short of .02);
   - **Pooling-scale question: region pooling at 1/32 only** (+.052, Dice +.006). With it the heads match ConvNeXt U-Net (.827);
   - design review: the 1/16 region features caused the shortfall, not the heads. The proposed freeze is teacher forcing 30 epochs, end-to-end mask, pooling at 1/32 (Analysis/06 section 9). The five final runs `A2_region` … `A6_slgnet` are in `kaggle/runs.json` and wait for your approval of the freeze.
3. **Classifier analysis, the classifiers vs YOLO** (`results/analysis/C03_classifiers_vs_yolo/report.md`):
   - both article classifiers are reproduced (DenseNet121 84.8 ± 2.0% against 83.8%; EfficientNet-B4 80.6 ± 1.5% against 79.0%);
   - out of fold, DenseNet121 and ConvNeXt-T beat YOLO (DenseNet121: accuracy +3.5 points, macro-F1 +.058, Holm p = .002), mostly by half as many detection errors;
   - on the test set YOLO's ensemble is ahead (88.1%), not significantly, thanks to better location (2 errors against 8–9).
4. **Reference bank, Paper B's reference bank:** check 1 failed (mirror check 58.4% < 60%). The ImageNet features hardly see left and right, because ImageNet training uses random flips. So the rule required a review of the orientation rule.
   - **Review** (`scripts/check_orientation.py`, run on your PC's images, no labels; `results/analysis/C10_paperB_orientation_review/`): a femur's pelvis side, read as the brighter half of the upper image, gives the two femurs of an x-ray opposite sides in 402 of 405 x-rays (99.3%). It agrees with pixel alignment for 99.2% of random pairs of different patients.
   - **Version 2** (Analysis/07 section 8): a reference is mirrored when its pelvis side differs from the target's. Code `refbank.py` and `build_reference_bank.py` changed; new run `R00_reference_bank_r2` (about 3 min).
5. **Calibration fix:** `run_stats.py` now fits a temperature for network runs whose per-case file writes a true-class probability as 0 (below 8 decimals). This affected one femur each of EfficientNet-B4 and A6_dev_detach. YOLO stays "not applicable", and no earlier value changed.
6. **Notes and tests:**
   - run notes for the nine runs;
   - 3 new unit tests (pelvis side, given orientation, the calibration of a written 0): 66 unit tests in all, and the 62 that need no data pass;
   - the reference-bank smoke test passes.

**Next**
- You:
  1. merge this PR;
  2. run `R00_reference_bank_r2` (`BRANCH = "main"`, about 3 min);
  3. say whether Paper A's settings may be frozen. If yes, I tag `A-v1-frozen` and you launch `A2_region` … `A6_slgnet` with `BRANCH = "A-v1-frozen"` (about 13 h in all);
  4. send SegFormer-B2's zip when it finishes.
- Me:
  - Segmenter analysis when SegFormer-B2 is filed;
  - the checks of `R00_reference_bank_r2`, then Paper B step 2's protocol (the Siamese model).

---

## 2026-11-27 · Step 14: Paper B step 1 of Paper B, the pair index and the bank of healthy reference femurs

**Done (you)**
- Merged PR #9 (`b35eb97`): the launcher's data check and calibration.

**Done (me)**
1. **Protocol** `Analysis/07_Step4_PaperB_Reference_Bank_Protocol.md`, fixed before any feature is computed:
   - the bank of fold model k is the healthy femurs of its 4 training folds (290 or 291); held-out and test femurs are never in it;
   - a femur never gets a reference from its own group (its x-ray, its partner, the duplicate x-ray 00262);
   - similarity is the cosine of frozen ImageNet ConvNeXt-T features (ConvNeXt U-Net's starting weights), centred on the mean of the fold model's training femurs; each bank femur is tried as released and mirrored, and used in the better orientation;
   - kept per femur and fold model: the 5 nearest, one random healthy femur (run the random-reference model, swap test a) and one random fractured femur (swap test b), drawn with a seed made from the file name;
   - seven checks before Paper B step 2, with a decision rule for the mirror check (≥ 80% as written, 60–80% stated as a limitation, below 60% review).
2. **Code:** `src/femurseg/refbank.py` (the rules) and `scripts/build_reference_bank.py` (the run). Run `R00_reference_bank` is in `kaggle/runs.json` (about 10 min, one T4). The launcher's data check covers it too.
3. **Checks:**
   - 6 new unit tests (`tests/test_refbank.py`): the counts of the real fold file (banks of 291, 290, 290, 291 and 290 femurs; 334 + 64 fractured femurs with a healthy partner), the exclusion of the own group, the orientation rule, centring on training femurs only, the seeded draws and the checks. All 63 unit tests pass.
   - A smoke test (`tests/smoke_reference_bank.py`) runs the whole run through `kaggle/run.py` on the synthetic data, on the CPU, in under a minute. It passed.

**Next**
- You: run `R00_reference_bank` (`BRANCH = "main"`, about 10 min) in any free session once this PR is merged. It does not depend on the baselines or the development rounds.
- Me: file it and read the checks of section 4, then Paper B step 2's protocol (the Siamese model).

---

## 2026-11-27 · Step 13: SegFormer-B2's failed launch, a data check in the launcher, and calibration

**Done (you)**
- Merged PR #8 (`f342b8a`). Started six Kaggle sessions at the same time: `S07_unet_r34`, `S04_densenet121`, `S08_deeplabv3p_r34`, `S05_efficientnet_b4`, `A_dev_round1` and `A_dev_round2`.
- `S09_segformer_b2` stopped at once with "no folder with train/images and test/images found under /kaggle/input" (exit code 1, 0.0 min, commit `4820f6b`).

**Done (me)**
1. **Why SegFormer-B2 failed:** that session found no dataset under `/kaggle/input`, most likely because no dataset was attached to its notebook. The code is not at fault: the other sessions find the data, and nothing was trained. The zip holds only `run_meta.json`. It is renamed `results/_inbox/filed/results_S09_segformer_b2_failed_no_data.zip` and not filed, so the name `S09_segformer_b2` stays free for the relaunch.
2. **The launcher now checks the data first** (`kaggle/run.py`):
   - before installing anything, a training run (or a parallel group) looks for `train/images` and `test/images`;
   - if they are missing, the log shows **NO DATASET FOUND**, what `/kaggle/input` holds, and how to attach the dataset; the run stops within seconds with exit code 2, and `run_meta.json` records the reason;
   - `scripts/register_run.py` does not file a run that never started; in a parallel group, the other run is still filed;
   - 6 new tests (`tests/test_kaggle_run.py`).
3. **Calibration** (`scripts/run_stats.py`, protocol `Analysis/06_…` section 6):
   - one temperature per run, fitted on its out-of-fold femurs, then applied unchanged to the test set;
   - ECE, Brier score and NLL are reported before and after, for the OOF femurs, the test ensemble and the fold models;
   - it is reported only: no threshold or decision uses it, and each femur keeps its most probable class.
4. **First calibration values** (from the filed runs, no new training):

   | Run | Temperature | ECE, OOF | ECE, test ensemble | ECE, test fold models |
   |---|---|---|---|---|
   | ConvNeXt-T classifier | 3.43 | .127 → .046 | .102 → .066 | .156 → .066 |
   | ConvNeXt U-Net | 3.14 | .108 → .040 | .103 → .064 | .131 → .062 |

   - Both models are over-confident by about the same factor, and one temperature fitted on OOF also works on the test set.
   - Not applicable to the YOLO runs: YOLO gives probability 0 to every class without a detected instance. For 73–99 of the 1,043 OOF femurs that is the true class, so the NLL is undefined. Their ECE and Brier are reported before calibration only.
5. **Checks:**
   - 8 new unit tests (6 launcher, 2 calibration); all 57 unit tests pass.
   - The CPU smoke test now also checks calibration in both statistics steps, and every step passed. One new check was too strict (it did not allow the note for a temperature at the end of its search range); it was corrected and holds on the same outputs.

**Next**
- You: in the SegFormer-B2 notebook, open the right panel → **Input** → **Add Input**, add `utopianstar/dataset`, keep `RUN = "S09_segformer_b2"` and `BRANCH = "main"`, and Save Version (about 3.3 h). Put each zip in `results/_inbox/` as it finishes.
- Me: file each zip; the classifier analysis and the segmenter analysis → **the baseline table**; the SLG-Net development analysis from the two development rounds; meanwhile Paper B step 1 of Paper B (protocol for the pair index and the retrieval bank).

---

## 2026-11-27 · Step 12: Paper A's protocol (Step 3) and SLG-Net code, tested on the CPU

**Done (you)**
- Merged PR #7 (`4820f6b`), which filed YOLO11s-seg, ConvNeXt-T classifier and ConvNeXt U-Net with the YOLO version analysis and the mask-task analysis. The baselines ResNet-34 U-Net, DenseNet-121, ResNet-34 DeepLabV3+, SegFormer-B2 and EfficientNet-B4 are next on Kaggle.

**Done (me)**
1. **Protocol** `Analysis/06_Step3_PaperA_Protocol.md`, fixed before any SLG-Net result:
   - what each of the five SLG-Net chain runs adds on top of ConvNeXt U-Net, with the Step 2 recipe unchanged;
   - three fold-0 development questions: the teacher-forcing question forcing, the detached-mask question a detached mask, the pooling-scale question region pooling at 1/32 only;
   - their decision rule against ConvNeXt U-Net's own fold-0 model, the freeze (tag `A-v1-frozen`), the interpretability tests, and the final runs and the SLG-Net analyses.
2. **SLG-Net code** `src/femurseg/slg.py`, used as `--net slg:convnext_tiny.fb_in22k_ft_in1k:<A2..A6>` in `scripts/train_net.py`:
   - region pooling inside the predicted mask at 1/16 and 1/32, weighted by the mask confidence;
   - 8 geometry numbers of the mask (area, centroid, height, width, orientation, confidence);
   - fracture → location → grade heads whose 6-class probabilities come from the chain rule; one grade trunk per location (hierarchy model) or a shared one (shared-trunk model, SLG-Net); logit adjustment with the training fold's priors (SLG-Net, training only);
   - teacher forcing: the heads get the released mask with a probability that falls from 1 to 0 over the first 30 epochs.
3. **Interpretability tests inside the training run** (`--extra-tests oracle,deletion`): every held-out and test femur is predicted again with the released mask (oracle), with the predicted region blurred out (deletion), and with the same shape blurred elsewhere (control). Each test gets its own per-case files and metrics.
4. **Kaggle:** `kaggle/run.py` can start two runs in one session, one per T4 (a *parallel group*, for example `A_dev_round1`). Their image cache is built once. `register_run.py` files the group's zip as separate runs, and `run_stats.py --folds 0` compares runs on fold 0 alone.
5. **Checks:**
   - 10 new unit tests (`tests/test_slg.py`): chain-rule probabilities, the geometry of a drawn ellipse, region pooling, logit adjustment, teacher forcing, a detached mask, the deletion images. The 39 earlier unit tests still pass.
   - The CPU smoke test now also trains SLG-Net on 5 folds with both tests, runs a parallel group of two fold-0 runs, files the group's zip and compares the runs with ConvNeXt U-Net on fold 0. It passed.
6. **ConvNeXt U-Net's fold-0 reference** (from the filed ConvNeXt U-Net, no new run): macro-F1 .822 (.735–.889), accuracy 88.5% (184 of 208), Dice on fractured femurs .749.

**Next**
- You: continue the baselines (ResNet-34 U-Net + DenseNet-121, then ResNet-34 DeepLabV3+ + SegFormer-B2, then EfficientNet-B4). After this PR is merged, run `A_dev_round1` and then `A_dev_round2` (about 55 min each) whenever a session is free; they do not depend on the baselines.
- Me: file each zip; the classifier analysis and the segmenter analysis → **the baseline table**; the SLG-Net development analysis decides the three development questions → freeze → **Paper A milestone 2**.

---

## 2026-11-27 · Step 11: YOLO11s-seg, ConvNeXt-T classifier and ConvNeXt U-Net filed; the YOLO version analysis and the mask-task analysis

**Done (you)**
- Merged PR #6 (`18e4774`) and ran `S03_yolo11s_seg_1024` (from `b1a2719`, 178 min), `S10_m0_convnext_unet` (140 min) and `S06_convnext_tiny` (88 min) on Kaggle. All 5 folds of each finished with exit code 0.

**Done (me)**
1. Filed the three runs with `scripts/register_run.py` (register lines 6–8) and wrote their `NOTES.md`. Checked each run: every training femur is predicted once, by the model that did not train on it, and `metrics.json` equals a re-evaluation of the per-case files.
2. **YOLO version analysis, YOLO11s-seg vs YOLOv8s-seg** (`results/analysis/C02_yolov8_vs_yolo11/report.md`): no difference on any endpoint (smallest Holm-corrected p = .72). YOLOv8s-seg stays the YOLO baseline.
3. **Mask-task analysis, does learning the mask help the classifier? (ConvNeXt-T classifier vs ConvNeXt U-Net)** (`results/analysis/C05_mask_task_s06_vs_s10/report.md`):
   - Yes on the 1,043 OOF femurs: accuracy 86.4% → 88.4% (Holm p = .026), macro-F1 .784 → .825 (Holm p = .013). ConvNeXt U-Net is better on all 5 held-out folds.
   - Same direction on the 210 test femurs (fold models 83.5% → 86.0%), not significant.
   - The gain comes from fewer detection and location errors. Grade errors are unchanged (65 of ConvNeXt U-Net's 121 OOF errors), with Garden I-II the weakest class.
   - ConvNeXt U-Net's mask is already better than YOLO's: Dice on fractured femurs .777 against .740 OOF (formal test in the segmenter analysis).
   - Both ConvNeXt models are over-confident (ConvNeXt U-Net's OOF threshold .018), so calibration matters.
4. `scripts/run_breakdown.py` now reports classifier and segmenter runs cleanly. Their header names the network, they are compared with the article's final model (and with its own DenseNet121 or EfficientNet-B4 where it has one), and "n/a" replaces "nan". The YOLO resolution analysis's report is unchanged.

**Next**
- You: run the remaining 5 baselines, two at a time: `S07_unet_r34` and `S04_densenet121` first, then `S08_deeplabv3p_r34` and `S09_segformer_b2`, then `S05_efficientnet_b4` (about 10 h in all). Put each zip in `results/_inbox/`.
- Me: file them, then the classifier analysis (classifiers vs YOLO) and the segmenter analysis (segmenters vs YOLO), which complete the baseline table (the baseline-table milestone). Meanwhile, Step 3 code: Paper A's SLG-Net heads on top of ConvNeXt U-Net (five SLG-Net chain runs, tested on the CPU).

---

## 2026-11-26 · Step 10: S00_net_check filed; measured run times for the seven other baselines

**Done (you)**
- Merged PR #5 (`a15364b`) and ran `S00_net_check` on Kaggle (3.5 min, exit code 0).

**Done (me)**
1. Filed it with `scripts/register_run.py` (`results/S00_net_check/`, register line 5) and wrote its `NOTES.md`.
2. Result: all 7 networks downloaded their ImageNet weights and trained on a T4.
   - Micro-batch 16: DenseNet121, ConvNeXt-T, U-Net, DeepLabV3+ and ConvNeXt U-Net.
   - Micro-batch 8 with 2 accumulated steps (still 16 images per update): EfficientNet-B4 (16 needed 14.6 GB) and SegFormer (16 ran out of memory).
3. Set the measured run times in `kaggle/runs.json`: DenseNet-121 1.6 h, EfficientNet-B4 2.1 h, ConvNeXt-T classifier 1.7 h, ResNet-34 U-Net 1.7 h, ResNet-34 DeepLabV3+ 1.5 h, SegFormer-B2 3.3 h, ConvNeXt U-Net 2.8 h (about 14.6 h in all).
4. Added the result to the protocol as an addendum (`Analysis/05_Step2_Baselines_Protocol.md` section 7). The recipe is unchanged.
5. `scripts/register_run.py` now writes "Tesla T4 x2" for a net check that used both GPUs.

**Next**
- You: run `S10_m0_convnext_unet` (about 2.8 h) and `S06_convnext_tiny` (about 1.7 h), in two sessions if two are free. Put each zip in `results/_inbox/`.
- Me: file each run; with both in, the mask-task analysis (ConvNeXt-T classifier vs ConvNeXt U-Net: does learning the mask help the classifier?). When YOLO11s-seg finishes: file it and write the YOLO version analysis (YOLOv8s-seg vs YOLO11s-seg).

---

## 2026-11-26 · Step 9: Step 2 code, one training script for the seven other baselines (classifiers, segmenters, ConvNeXt U-Net)

**Done (you)**
- Merged PR #4 (`b1a2719`) and started `S03_yolo11s_seg_1024` on Kaggle.

**Done (me)**
1. `scripts/train_net.py`: one script for the 7 remaining shared runs:
   - DenseNet-121, EfficientNet-B4 and ConvNeXt-T classifier image classifiers;
   - ResNet-34 U-Net, ResNet-34 DeepLabV3+ (ResNet34) and SegFormer-B2 (MiT-B2) segmenters;
   - the core model ConvNeXt U-Net.
   It trains 5 folds, two at a time on the two T4s, with the same rules as YOLOv8s-seg: fixed length, final weights, OOF threshold, test mean ± SD and ensemble.
2. A pre-flight before training: it downloads the weights, finds the largest batch that fits the GPU (otherwise smaller batches with gradient accumulation to 16) and times a step. `S00_net_check` runs only this, for all 7 networks.
3. Shared modules:
   - `src/femurseg/evaluation.py`: the per-case files and scores, now used by every model. It re-scores YOLOv8s-seg and YOLOv8s-seg at 640 px to exactly their `metrics.json`, and records NSD at 10 and 20 px as well.
   - `runner.py`: parallel folds with the time guard, also used by the YOLO script.
   - `torchdata.py`: every image read once into a resized cache.
   - `nets.py`: networks, losses and the decision rule.
4. Tests:
   - `tests/test_nets.py`: ConvNeXt U-Net's class output is identical to the ConvNeXt-T classifier's with the same weights; decision rules, losses and the cache.
   - `tests/smoke_train_net.py`: full 5-fold CPU runs of a classifier, a segmenter and ConvNeXt U-Net, plus both analysis scripts.
   - A learning check on synthetic data reached Dice 0.97.
   - The image cache was built from the real local data (1,253 femurs, 43.5 s). Every mask holds only its femur's class, and the smallest fracture mask has 1,722 pixels at 768 × 512 (median 23,045), well above the 500 pixels the segmenters' fracture score averages.
   - The YOLO smoke test still passes.
5. `run_stats.py` and `run_breakdown.py` handle runs without masks: classifiers get no Dice, not Dice 0.
6. The full protocol is written down before any result: `Analysis/05_Step2_Baselines_Protocol.md`.
7. Sent everything as PR #5 and synced your local folder.

**Next**
- You: merge PR #5, then run `S00_net_check` (about 15 min) in a second session on your main account with `BRANCH = "main"` (before the merge, `"claude/project-thread-kges8j"`). Put the zip in `results/_inbox/`.
- Me: file it and set the measured batch sizes and run times in `kaggle/runs.json`. The first training runs are then ConvNeXt-T classifier and ConvNeXt U-Net.
- When YOLO11s-seg finishes: I file it and write the YOLO version analysis (YOLOv8s-seg vs YOLO11s-seg).

---

## 2026-11-26 · Step 8: YOLO baselines YOLOv8s-seg and YOLOv8s-seg at 640 px filed and analysed

**Done (you)**
- Merged PR #3 and ran `S01_yolov8s_seg_1024` and `S02_yolov8s_seg_640` with Save Version, in two parallel sessions (commit `29e1852`). Both finished all 5 folds with exit code 0: YOLOv8s-seg in 176 min, YOLOv8s-seg at 640 px in 81 min.

**Done (me)**
1. Filed both runs with `scripts/register_run.py` (`results/S01_yolov8s_seg_1024/`, `results/S02_yolov8s_seg_640/`, register lines 3 and 4) and wrote their `NOTES.md`.
2. 95% intervals and paired YOLOv8s-seg vs YOLOv8s-seg at 640 px tests with `scripts/run_stats.py` → `results/analysis/C01_yolo_1024_vs_640/stats.md`.
3. New `scripts/run_breakdown.py` → `breakdown.md` in the same folder:
   - recall, precision and Dice per class;
   - every metric as mean ± SD over the 5 held-out folds and over the 5 fold models on the test set;
   - errors split like the article (detection, location, grade);
   - mask areas, and Dice under several definitions;
   - side by side with the article.
4. New `src/femurseg/reference.py`: the article's numbers (Table 2, Table 3, Fig. 3 confusion matrix). `tests/test_reference.py` checks that the typed confusion matrix gives every number the article derives from it (181/210, sensitivity, specificity, PPV, NPV, 8/6/15 errors). `tests/test_breakdown.py` checks the breakdown code; the smoke test now also runs both analysis scripts.
5. Wrote the interpretation: `results/analysis/C01_yolo_1024_vs_640/report.md`.

**Result**
- **Classification reproduces the article.** YOLOv8s-seg fold models on the test set: AUC .985 ± .005 and 6-class accuracy 84.6 ± 2.1%, against .986 ± .008 and 84.4 ± 2.5% (article, 6 runs). YOLOv8s-seg at 640 px: 82.2 ± 2.1% against 82.0 ± 3.0%.
- **The 5-model ensemble of YOLOv8s-seg is the strongest baseline:** 88.1% (185/210, 95% CI 82.9–92.1), AUC .994, 25 errors against the article's 29.
- **Dice is .02 to .04 lower than the article's** (detected fractures: .741 at both sizes vs .761 and .780). The article's "640 px gives better masks" is not reproduced: YOLOv8s-seg and YOLOv8s-seg at 640 px do not differ on any endpoint after the Holm correction.
- **Errors are mostly grades:** location is right for 96.6% of found fractures, the grade for 86.1% of those. Weak spots: Garden I–II (recall .44 ± .12) and Evans III–V called Evans I–II.
- **Finding:** the test masks of neck fractures are 35–40% smaller than the training ones (the article's test labels were reviewed by a committee). The models draw the same mask size on both, so they over-segment test neck fractures about 2 times. This explains most of the OOF-to-test Dice drop (.800 → .741).

**Decisions**
- YOLOv8s-seg is the reference YOLO baseline for both papers: fold models mean ± SD, ensemble, and OOF for development.
- Dice is reported per location, OOF and test side by side. Nothing learned from the test set is used to correct it.
- New training code records NSD at 10 and 20 px as well (2 px is uninformative: about .05).
- YOLO11s-seg at 1024 px with YOLOv8s-seg's protocol (added to `kaggle/runs.json`).

**Next**
- You: merge PR #4, then run `S03_yolo11s_seg_1024` with Save Version (about 3 h). Check how much GPU quota Kaggle shows as used this week: it tells us whether a T4 × 2 session counts once or twice.
- Me: Step 2, one training script for the three classifiers, the three segmenters and the core model ConvNeXt U-Net (768 × 512 input), tested on the CPU first.

---

## 2026-11-26 · Step 7: Kaggle check confirmed (ConvNeXt U-Net reached); Step 1 code: data loader, metrics, YOLO baselines

**Done (you)**
- Merged PR #2 and ran `S00_env_check_r2` on main (commit `2758d45`, 3.7 min, exit 0).

**Result** (`results/S00_env_check_r2/`, register line 2)
- Same data True, same folds True, all 1,253 files found, and the 104 femurs of Kaggle's val/ pooled into train.
- So the Kaggle data and folds are exactly the local ones. **Milestone ConvNeXt U-Net reached.**

**Done (me): Step 1 code, shared by both papers**
1. `src/femurseg/data.py`:
   - reads the fold file into records;
   - `fold_split` gives the training and held-out femurs of each fold;
   - `ImageStore` finds every file on any copy of the data and keeps images and masks in RAM.
2. `src/femurseg/metrics.py`:
   - segmentation: Dice, IoU, HD95 and normalised surface Dice, with the empty-mask conventions stated;
   - classification: AUC, sensitivity and specificity, 6-class accuracy, macro-F1, balanced accuracy, kappa;
   - also: the paper's error breakdown (detection, location, grade), calibration (ECE, Brier) and the Youden threshold. Ties go to the higher sensitivity.
3. `scripts/train_yolo_seg.py`, for baselines YOLOv8s-seg and YOLOv8s-seg at 640 px:
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
6. **Correction:** the paper's best-Dice setting is YOLOv8-*small* at 640 px (test Dice .780), not nano. YOLOv8s-seg at 640 px is now `S02_yolov8s_seg_640`, and Reports 03 and 04 are fixed.
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
- Me, while YOLOv8s-seg runs: the next baselines (YOLO11s-seg, the three classifiers). When YOLOv8s-seg is back: file it and add 95% intervals with `scripts/run_stats.py`.

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
- Me: Step 1. The data loader caches all images in RAM because of the slow reads. Also the metrics and statistics modules, and the YOLO reproduction runs YOLOv8s-seg and YOLOv8s-seg at 640 px with `ultralytics` installed through `kaggle/runs.json`.

---

## 2026-11-26 · Step 5: Decision for Papers A and B in parallel; execution plan; documentation tools

**Decided (by you)**
- Run **Option A (SLG-Net)** and **Option B (RefFrac)** in parallel, as two papers.
- No radiologist or other clinician is available.
- Every experiment and its result must be documented in the local folder and in the repository.

**Done**
1. Wrote `Analysis/04_Execution_Plan_A_and_B.md`:
   - **Shared foundation:** split, loader, metrics, the nine baselines and the core model ConvNeXt U-Net. ConvNeXt U-Net is the second step of the SLG-Net ablation chain and the "no reference" starting point of RefFrac, so it is trained once and used twice.
   - **No-clinician plan:** no reader study, re-annotation or clinical claims. Interpretability is shown by five measurable tests (oracle, region deletion, geometry consistency, pointing game, risk–coverage). Technical and informatics journals are the targets.
   - **5-fold out-of-fold validation** explained step by step, with four honesty rules.
   - **Track B:** research the reference question to the alignment-and-fusion question. Main method: retrieved healthy references, k = 3, from the fold's own training bank. The eight reference runs, with the contralateral shortcut study and swap tests. Primary endpoints: Dice on fractured femurs and Garden I–II recall.
   - **Compute:** A + B ≈ 120–135 GPU-h (A alone ≈ 65–72), about 4–4.5 weeks of free quota spread over 10 weeks, with a peak of about 20 h per week.
   - **Timeline:** Paper A submission about 21 Feb 2027, Paper B about 14 Mar 2027, each with a 2-week buffer.
2. Updated Report 03: decision banner, new run names for the nine baselines, and validation wording matching Report 04.
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
- Me (Step 1, shared): data loader, metrics and statistics modules, then the YOLO reproduction runs YOLOv8s-seg and YOLOv8s-seg at 640 px.

---

## 2026-11-26 · Step 4: Ranking the directions, research plan, repository set-up

**Done**
1. Scored the nine directions on novelty, evidence in our data, chance of a clear result, Kaggle feasibility and reviewer value. Ranking: the hierarchical-grading direction > the uncertainty direction > the segmentation-first direction > the healthy-femur-comparison direction > the benchmark direction > the rare-class direction = external-test direction > the foundation-model direction > the label-noise direction (`Analysis/03_Idea_Ranking_and_Research_Plan.md` section 2).
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
   - Wrote the full experiment plan: the nine baselines, ablations the SLG-Net ablation chain and an oracle study, metrics, statistics, pre-specified endpoints, Kaggle compute (about 70–80 GPU-h core), and a 12-week timeline.
5. Wrote and ran `scripts/make_splits.py` on the local data. It produced `splits/folds_v1.csv`: 5 folds of 208–209 femurs, grouped by x-ray, stratified by class, with the duplicate pair merged. Class counts match the paper exactly.
6. Read the dataset's own `data.yml` and `readme.md`. Class codes are `2AB, 2CD, 3AB, 3CDE, 4`: the location digit plus grade letters, so the labels are hierarchical. `data.yml` points val to `valid/images` while the folder is `val/`. This is harmless, because we write our own YOLO data files.
7. Set up the repository: README (workflow local ↔ GitHub ↔ Kaggle), `.gitignore`, `requirements.txt`, `src/femurseg/` (class table, data finder), `kaggle/` (launcher notebook, run registry, `env_check.py`), `results/` (experiment register), and updated trackers.

**Next**
- You choose Option A, B, or A then B (Report 03 section 11).
- Run `env_check` on Kaggle (Step 0) and log it in `results/EXPERIMENTS.md`.
- Step 1: data loader, metrics module and the YOLOv8-seg baseline reproduction (identical for both options).

---

## 2026-11-26 · Step 3: Guided walkthrough and dataset-usage directions

**Done**
1. Gave a full step-by-step walkthrough of the article (7 parts) and the dataset (4 parts) in the project thread.
2. Wrote `Analysis/02_How_to_Use_the_Dataset.md`: why the dataset is worth using, what to avoid, and 9 evidence-backed directions (nine directions), cited from our EDA, the base paper and the literature review.

**Next**
- Choose the core novelty (likely among the segmentation-first direction, the hierarchical-grading direction grading, the healthy-femur-comparison direction contralateral comparison, the label-noise direction), plus supporting pieces the rare-class direction, the uncertainty direction, the benchmark direction.

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
