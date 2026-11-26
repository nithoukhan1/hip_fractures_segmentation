# Report 04: Execution plan, Papers A and B in parallel

**Date:** 2026-11-26 · **Decision:** run **Option A (SLG-Net)** and **Option B (RefFrac)** in parallel, as **two papers** that share one codebase, one data split and one set of baselines.
**Constraint:** no access to a radiologist or any other clinician. **Rule:** every experiment and its result is documented in the local folder and in the repository.
**This report replaces** the compute (§8) and timeline (§9) sections of Report 03. The ranking and the method ideas in Report 03 stay valid.

**Run names from now on:**
- **S** = shared runs (the baselines B1–B9 of Report 03 are now S1–S9).
- **A** = Paper A runs.
- **R** = Paper B (RefFrac) runs.

---

## 1. Step 1: Is running A and B in parallel possible? Yes, because most of the work is shared

```
                    SHARED FOUNDATION (built once, Weeks 0-2)
   data split -> data loader -> metrics + statistics -> baselines S1-S9 -> core model M0
   (M0 = ConvNeXt-T encoder + U-Net decoder + flat 6-class head)
                                   |
             +---------------------+----------------------+
             v                                            v
   TRACK A (Paper 1): SLG-Net                 TRACK B (Paper 2): RefFrac
   M0 + new HEADS:                             M0 + new INPUT side:
   region pooling, region geometry,            a healthy reference femur, Siamese
   location -> grade hierarchy                 encoder, alignment, difference fusion
   "How should the network DECIDE?"            "What should the network COMPARE WITH?"
```

- **Shared:** everything before the fork. That covers the split, data loader, augmentations, training loop, metrics, statistics code, Kaggle launcher, all nine baselines, and M0. M0 is ablation A1 in Paper A and the "no reference" model R0 in Paper B, so it is trained once and used twice.
- **Separate:** only each paper's own module, its ablations and its manuscript.
- **The two papers answer different questions:**
  - Paper A changes how the network *decides*.
  - Paper B changes what it *compares with*.
  - So they are two distinct contributions, not one study cut in two. Each paper will name the other in its cover letter as a related submission, which journals expect.
- **Parallel tracks also spread the risk.** If one method gives only small gains, the other paper can still carry the project.

## 2. Step 2: The constraint of no clinician access, and what changes

**Nothing in either plan needs a radiologist.** Both use only the labels already released by the authors. The train labels were made by two radiologists, and the test labels were reviewed by a committee.

| We will NOT do | What we do instead |
|---|---|
| Reader study (do doctors improve with the AI?) | State it as a limitation and future work, exactly as the base paper did |
| New, corrected or adjudicated annotations | Use the released labels as they are. Report label variability from the data itself (duplicate image pair: IoU 0.70, about Dice 0.82) |
| Clinical judgement of failure cases | Describe failures with the released labels and the published Garden and Evans definitions ("labelled Garden I–II, predicted III–IV"), with no expert opinion |
| Claims of clinical benefit | Claim technical performance, and support interpretability with **measurements** (below) |

**How interpretability is shown without a clinician (measurable tests):**
1. **Oracle test (A):** give the heads the released (ground-truth) mask instead of the predicted one. If grading improves, the decision really depends on the region.
2. **Region-deletion test (A):** blank out the predicted region. If grade confidence drops, the model is looking at the fracture, not elsewhere.
3. **Geometry consistency (A):** check that predicted region size and position agree with the predicted grade and location, which our data shows they should (Report 03 §2).
4. **Difference maps (B):** check whether "target minus reference" features peak inside the released (ground-truth) fracture mask (we measure this with a pointing-game score).
5. **Uncertainty (A and B):** check whether flagging the most uncertain 10% of cases removes a large share of the errors (risk–coverage curves).

**Journal choice:** without a clinician co-author we aim at **technical and informatics journals**. Examples are J Imaging Informatics in Medicine, BMC Medical Imaging, IJIST, MBEC, Diagnostics, Bioengineering, BSPC and Computers in Biology and Medicine. Clinical radiology journals such as Insights into Imaging, Eur J Radiol and Skeletal Radiology usually expect clinical validation or clinician authors. If a clinician collaborator joins later, a small reader study could strengthen a revision, but no part of the plan depends on it.

## 3. Step 3: The validation approach (5-fold, out-of-fold), explained

### 3.1 The problem
The authors released train (1,043 femurs) and test (210), but **no validation set**. A validation set is needed to:
- choose settings (learning rate, number of epochs, loss weights),
- set thresholds,
- calibrate probabilities.

The test set must never be used for any of these, or its results stop being honest.

### 3.2 The solution, step by step
1. Cut the 1,043 training femurs into **5 equal parts** ("folds"). Both femurs of one x-ray always stay in the same part, and each part has the same mix of the 6 classes. This is already done: `splits/folds_v1.csv`, with 208–209 femurs per part.
2. Train **5 models** of the same configuration. Each model learns from 4 parts and is checked on the 5th:

```
              part 1   part 2   part 3   part 4   part 5      (in the files: fold 0 ... fold 4)
   Model 1:   CHECK    train    train    train    train
   Model 2:   train    CHECK    train    train    train
   Model 3:   train    train    CHECK    train    train
   Model 4:   train    train    train    CHECK    train
   Model 5:   train    train    train    train    CHECK
```

3. Every training femur is predicted **exactly once, by the one model that never trained on it**. Together these are the **out-of-fold (OOF) predictions**. They are an honest extra test of **1,043 cases, 5 times bigger than the official test set**, so comparisons between methods become much more reliable statistically.
4. The **official test set (210)** is predicted by all 5 models. We report the average ± SD (how stable the method is) and the **ensemble** (the 5 models' probabilities averaged). The ensemble is the headline number compared with the paper.

### 3.3 Is it correct?
Yes. k-fold cross-validation is the standard approach when a dataset is small or has no validation set. nnU-Net, the reference method in medical segmentation, uses 5 folds by default. It also uses all 1,043 training femurs, whereas a single fixed validation part would waste 20% of them for model building.

### 3.4 Four rules that keep the numbers honest
1. **Fixed training length.** Every model trains for a fixed, pre-decided number of epochs and keeps its final weights. The checked part is **never** used to pick the "best epoch", because that would let it leak into the result.
2. **Post-hoc choices come from OOF only.** The fracture threshold and the calibration temperature are fitted on the pooled OOF predictions, then applied unchanged to the test set.
3. **Tuning happens on part 1 only.** All development uses fold 0. Final results are reported on all 5 parts and, as a check, on parts 2–5 only. If the two agree, the tuning did not inflate anything.
4. **The test set is touched only by frozen configurations**, once per configuration.

**Known limitation, written into the paper:** patient IDs were not released, so two different x-rays of the same patient could land in different parts. This can make OOF numbers slightly optimistic. It does **not** affect the official test set, which the authors split by patient.

## 4. Step 4: Track A (Paper 1), SLG-Net, in short

The full design is in Report 03 §3 and §7. The only changes are:
- the run names,
- the shared M0,
- the interpretability tests from §2 of this report.

| Run | What | Folds | Reuses |
|---|---|---|---|
| A0 | ConvNeXt-T flat classifier | 5 | = S6 (shared) |
| A1 | + segmentation decoder, flat head | 5 | = M0 (shared) |
| A2 | + region pooling | 5 | |
| A3 | + region geometry | 5 | |
| A4 | + location → grade hierarchy | 5 | |
| A5 | + shared severity trunk | 5 | |
| A6 | + logit adjustment = **SLG-Net** (final) | 5 | |
| A-oracle, A-deletion | Interpretability tests | inference only | A6 models |
| A-sens | Second backbone; input 1152×768 | 5 + 5 | |

**Primary endpoints (fixed in advance):** 6-class macro-F1 and mean Dice on fractured femurs.

## 5. Step 5: Track B (Paper 2), RefFrac, full plan

### 5.1 Research questions
- **RQ-B1:** Does a healthy reference femur improve fracture segmentation and detection, **especially for subtle fractures** (undisplaced Garden I–II, the smallest masks)?
- **RQ-B2:** Which reference is better: the patient's **own contralateral femur** or a **retrieved healthy femur** from other patients?
- **RQ-B3:** Do bilateral models learn the dataset's **shortcut**? Here exactly one femur of every two-femur x-ray is fractured, and 64 of the 65 healthy test femurs are partners of a fractured one. How do we prevent it? This methodological warning is a contribution in itself.
- **RQ-B4:** How should the reference be aligned and fused (difference vs concatenation; none vs spatial transformer vs cross-attention)?

### 5.2 The data B uses (from `splits/folds_v1.csv`)
- Fractured femurs whose healthy partner is present: train 334 (66–68 per fold), test 64 of 145 (44%).
- **Retrieval bank:** the healthy femurs in each model's own 4 training parts (about 290), plus their mirror images. The target's own x-ray is excluded while training, so a model never "retrieves" the patient's own other hip during training. At test time it cannot, because test x-rays are not in the bank.

### 5.3 Method
1. **Reference choice (main method):** for every femur, retrieve the k = 3 most similar healthy femurs from the bank. Similarity is cosine distance of frozen ImageNet ConvNeXt-T features, trying both the image and its mirror and keeping the better match. This works for **100% of cases** and has **no shortcut**, because the reference is always a healthy femur of another patient.
2. **Siamese encoder:** the same ConvNeXt-T, with shared weights, encodes the target and the references.
3. **Alignment:** local cross-attention at the 1/16 and 1/32 scales, where each target position looks for its match in the reference. Ablations test none and a global spatial transformer (as in AASN).
4. **Fusion:** at each scale, [target, target − reference, |target − reference|] → 1×1 convolution → the same U-Net decoder and flat head as M0. The **only** difference from M0 is the reference, so gains can be attributed to it.
5. **Contralateral study (RQ-B2, RQ-B3):**
   - a variant uses the mirrored partner as the reference whenever it exists;
   - swap tests then replace the partner with (a) a random healthy femur and (b) a random fractured femur, and measure how much the prediction depends on the partner's state;
   - segmentation quality with own-partner vs retrieved reference is compared on the 64 paired test fractures.

### 5.4 Baselines and ablations
| Run | What | Tests |
|---|---|---|
| S1–S9, M0 (= R0) | Shared baselines; M0 has no reference | Starting point |
| R1 | AASN-style Siamese + global spatial transformer, adapted to femur crops | Closest published idea (Chen et al., ECCV 2020) |
| R2 | M0 + a **random** healthy reference | Does *any* second image help, or only a matched one? |
| R3 | Retrieved reference, k = 1 | Value of retrieval |
| R4 | Retrieved, k = 3, cross-attention + difference fusion = **RefFrac** (final) | Main method |
| R5 | R4 with concatenation instead of difference fusion | Fusion type |
| R6 | R4 without alignment; R4 with global spatial transformer instead | Alignment type |
| R7 | Partner reference when present (retrieved otherwise) + swap tests | Shortcut study (RQ-B3) |
| R-sens | Second backbone | Robustness |

### 5.5 Metrics and statistics
- Same metrics, rules and tests as Paper A.
- Extra subgroup reports:
  - Garden I–II recall;
  - Dice on the smallest third of masks;
  - paired vs unpaired fractures;
  - the pointing-game score of difference maps;
  - the swap-test shift in fracture probability.
- **Primary endpoints (fixed in advance):** mean Dice on fractured femurs and recall of Garden I–II (subtle) fractures.

## 6. Step 6: Kaggle compute budget (GPU-hours on one T4 or P100)

| Block | What | GPU-hours |
|---|---|---|
| Shared | env check; S1 YOLOv8s-seg @1024; S2 YOLOv8n-seg @640; S3 YOLO11s-seg; S4–S6 classifiers; S7–S9 segmenters; M0 (5 folds each) | **≈ 28–34** |
| Track A only | development (fold 0); A2–A5; A6 final; sensitivity; oracle and deletion tests | **≈ 35–38** |
| Track B only | development (fold 0); R1–R7 (5 folds each); sensitivity; retrieval and swap tests | **≈ 55–60** |
| **A + B total** | | **≈ 120–135** |
| A alone, for comparison | | ≈ 65–72 |

**What parallel work costs:**
- B adds about **55–60 GPU-hours**, roughly **2 more weeks of free quota**. Kaggle's free quota is commonly reported as about 30 h per week; your exact number is in your Kaggle settings.
- In total, A + B use about **4–4.5 weeks of quota, spread over 10 weeks** of experiments.
- **No week needs more than about 20 GPU-hours**, because A's heavy final runs (Weeks 6–7) and B's (Weeks 8–9) are staggered. That leaves about 10 h per week for re-runs.

**Planned load per week:**

| Week | Dates | Shared | A | B | Total |
|---|---|---|---|---|---|
| 0 | 26–29 Nov | 0.3 | | | ~0.3 |
| 1 | 30 Nov–6 Dec | 11–13 | | | ~12 |
| 2 | 7–13 Dec | 17–21 | | | ~19 |
| 3–5 | 14 Dec–3 Jan | | 12–15 | 16 + 4 | ~11 per week |
| 6–7 | 4–17 Jan | | 22–23 | 8 | ~15 per week |
| 8–9 | 18–31 Jan | | 1 | 28–32 | ~15 per week |

**Two ways to stretch the quota:**
1. If Kaggle counts a **T4 x2** session like one GPU, one launch can run an A job on GPU 0 and a B job on GPU 1 at the same time, halving the cost of parallel work. `S00_env_check` (26 Nov) confirmed 2 × T4 per session, 4 CPU cores and 32 GB RAM. We will compare your quota before and after one long T4 x2 run to find out how it is counted.
2. Several folds or variants are chained in one Save Version launch (up to about 10 hours), so you click **about 3–5 times per week**, not once per model.

## 7. Step 7: Timeline for both papers

| Wk | Dates | Shared | Paper A (SLG-Net) | Paper B (RefFrac) |
|---|---|---|---|---|
| 0 | 26–29 Nov | `S00_env_check`; `main` branch | | |
| 1 | 30 Nov–6 Dec | Loader, metrics, statistics, S1–S2 (YOLO reproduction) | | Pair index and retrieval-bank code (CPU) |
| 2 | 7–13 Dec | S3–S9 + M0 → **M1: baseline table** | | Retrieval sanity check (nearest-neighbour figure) |
| 3–4 | 14–27 Dec | | Build SLG-Net heads; fold-0 development | Build Siamese encoder + fusion; fold-0 development |
| 5 | 28 Dec–3 Jan | | Rare-class + calibration → **M2-A: frozen** | Alignment variants, k; R1 baseline |
| 6–7 | 4–17 Jan | | Final + ablations + tests → **M3-A** | Development ends → **M2-B: frozen**; R2–R3 |
| 8 | 18–24 Jan | | Statistics + figures → **M4-A** | R4 final, R5–R6 |
| 9 | 25–31 Jan | | Manuscript A | R7 shortcut study, sensitivity → **M3-B** |
| 10 | 1–7 Feb | | Manuscript A | Statistics + figures → **M4-B** |
| 11 | 8–14 Feb | | Manuscript A, internal review | Manuscript B |
| 12 | 15–21 Feb | | **M5-A: submit Paper A** | Manuscript B |
| 13–14 | 22 Feb–7 Mar | | Buffer | Manuscript B, internal review |
| 15 | 8–14 Mar | | | **M5-B: submit Paper B** |

Each paper keeps a 2-week buffer: Paper A by about 7 Mar 2027, Paper B by about 28 Mar 2027. Option: post Paper A as a preprint when it is submitted, so Paper B can cite it. Check each journal's preprint policy first.

## 8. Step 8: Documentation of every experiment (local folder = repository)

1. **One folder per run:** `results/<RUN>/` holds:
   - `run_meta.json` (git commit, command, runtime, exit code),
   - `config.yaml`, `metrics.json`, per-case CSVs and the log,
   - `NOTES.md` from `results/RUN_NOTES_TEMPLATE.md`: why we ran it, what changed, the result, the decision.
2. **One line per run** in the register `results/EXPERIMENTS.md`, with its track (S/A/R), purpose, key result and decision.
3. **Weekly:** an entry in `PROGRESS_LOG.md`, with milestone status updated in `PROJECT_TRACKER.md`.
4. **Getting Kaggle results in:**
   - Download `results_<RUN>.zip` from the Kaggle version's Output tab into `results/_inbox/` in the local folder, and tell me.
   - `scripts/register_run.py` files it into `results/<RUN>/`, adds the register line and a notes stub, and I push the same files to the repository.
   - Both places stay identical, and no run is ever edited after filing. A re-run gets a new name.
5. **Traceability:** every number in a manuscript must point to a file in `results/`.

## 9. Step 9: Risks of working in parallel, and how we handle them

| Risk | Handling |
|---|---|
| Quota contention in the final-run weeks | A's final runs have priority (Weeks 6–7); B's come in Weeks 8–9. T4 x2 sharing if it is metered as one session |
| Too much at once for you (launching runs, reading results) | Runs are batched (3–5 launches per week); I file results and update trackers |
| Papers seen as one study split in two | Different research questions, different primary endpoints, each paper stands alone, and each cover letter discloses the other |
| One track stalls | Tracks are independent after Week 2; the other paper continues on schedule |
| Scope creep (combining A and B) | A combined model is **future work**, not part of either paper |

## 10. What happens next
1. **You:** review and merge the setup pull request into `main`. Then import `kaggle/launcher.ipynb` into Kaggle and run `S00_env_check`.
   - Use `BRANCH = "main"` after the merge (before it, `"claude/project-thread-kges8j"`).
   - Then put the zip in `results/_inbox/`.
2. **Me (Step 1, shared):**
   - data loader with the same augmentations for all models;
   - metrics (Dice, IoU, HD95, NSD, AUC, macro-F1, ECE) and the statistics module;
   - the YOLO reproduction runs S1–S2 in `kaggle/runs.json`.
