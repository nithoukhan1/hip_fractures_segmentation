# Report 08: Step 5 protocol, Paper B (RefFrac) step B-2: the Siamese model, runs R1–R7 and the fold-0 development

**Date:** 2026-11-27 · **Code:** `scripts/train_net.py --net ref:convnext_tiny.fb_in22k_ft_in1k:<R1..R7>` with `src/femurseg/refnet.py` · **References:** the fixed tables of step B-1, `results/R00_reference_bank_r2/` · **Runs:** `kaggle/runs.json`
**Why this report:** it fixes, before any R run exists, what each run of Paper B is, the model exactly, how the fold-0 development is run and decided, what is then frozen, and how the final runs and the reference tests are scored. Paper B's methods section is written from it. The design comes from Report 03 §4 and Report 04 §5, and the references from Report 07.

**Starting point:**
- M0 (= R0, filed as S10) is the model every R run extends. On the 1,043 out-of-fold femurs it scores Dice .777 on fractured femurs, Garden I-II recall .544 and macro-F1 .825 (C04, C05). Its test-ensemble Dice is .744 (C04).
- **Paper B's primary endpoints** (fixed in Report 04 §5.5): mean Dice on fractured femurs, and Garden I-II recall (57 out-of-fold and 16 test femurs).
- Every femur already has its references in the fixed table of step B-1 (Report 07 §9). For each fold model, the table holds:
  - the 5 most similar healthy femurs of the 4 training folds;
  - one random healthy femur and one random fractured femur;
  - for each of them, whether it is mirrored, so that its pelvis is on the target's side.
- **A fact computed now from that table** (no model involved): fractured femurs get less similar references than healthy femurs.

  | Femurs | Centred cosine of the nearest reference, fractured | Healthy |
  |---|---|---|
  | Held out (680 fractured, 363 healthy) | .574 | .612 |
  | Test, 5 fold models (725 / 325 rows) | .573 | .599 |

  - A fracture makes a femur look less like healthy femurs. That is the signal RefFrac is meant to use.
  - It also means that how well the references match is itself weakly informative. This is legitimate, because it is computed the same way for any new patient.

---

## 1. The runs of Paper B

| Run | `--net` | References (k) | Alignment | Fusion | Question (Report 04 §5) |
|---|---|---|---|---|---|
| R0 | = M0 = S10 (filed) | none | – | – | Starting point |
| R1 | `ref:…:R1` | the nearest (1) | global affine | concatenation + AASN's margin loss | The closest published idea (AASN), adapted (§2.4) |
| R2 | `ref:…:R2` | a random healthy femur (1) | local cross-attention | difference | Does any healthy second image help? |
| R3 | `ref:…:R3` | the nearest (1) | local cross-attention | difference | Does retrieval help (R3 vs R2)? |
| R4 | `ref:…:R4` | the 3 nearest (3) | local cross-attention | difference | **RefFrac**, the main method; 3 references vs 1 (R4 vs R3) |
| R5 | `ref:…:R5` | the 3 nearest | local cross-attention | concatenation | Difference or concatenation? |
| R6a | `ref:…:R6a` | the 3 nearest | none | difference | Is alignment needed? |
| R6b | `ref:…:R6b` | the 3 nearest | global affine | difference | Local or global alignment? |
| R7 | `ref:…:R7` | own healthy partner, else the nearest (1) | local cross-attention | difference | Own hip vs retrieved (RQ-B2); the pairing shortcut (RQ-B3) (§2.5) |
| R-sens | later | | | | Second backbone, as A-sens |

**One recipe.** Every run uses M0's recipe unchanged (Report 05 §2):
- input 768 × 512, the same augmentations;
- AdamW (learning rate 2 × 10⁻⁴, weight decay 0.05), 2 warm-up epochs then cosine decay, **60 epochs, final weights**;
- batch 16 (micro-batch set by the pre-flight), fp16, seed 2026 + fold.

Every R run starts from the ImageNet weights, as M0 did, not from M0's trained weights. Only the reference path differs from M0, so a difference from M0 can be attributed to the reference.

## 2. The model, exactly

### 2.1 Shared by R1–R7
1. **Trunk and losses = M0's:**
   - the `segmentation_models_pytorch` U-Net with the `tu-convnext_tiny` encoder, the same decoder and binary mask head;
   - M0's classifier head (the timm head: global average pool, LayerNorm, dropout 0.2, linear to 6 classes) on the 1/32 features;
   - the loss: cross-entropy of the 6 classes + binary cross-entropy + Dice of the fracture mask (weight 1).
2. **Siamese encoding:** the target T and its k references R₁ … R_k go through the same encoder (shared weights).
   - The references are encoded **without gradient** (forward passes only). With gradient, the encoder's activation memory would grow k + 1 times (4 times for R4), and a step would cost about 2.5 times M0's. Without it, a step costs about 1.6 times M0's for k = 3 and 1.2 times for k = 1 (§5), and the memory stays close to M0's.
   - The encoder still learns from the target. The reference features are always computed with the current weights, so target and references stay in one feature space.
   - ConvNeXt's encoder has no batch statistics and no dropout, so a reference is encoded exactly as it would be with gradient.
   - Only the references' 1/16 (384 channels) and 1/32 (768 channels) features are kept.
3. **Fusion at 1/16 and 1/32:** the target's features T are replaced by F = W [T, T − R̃, |T − R̃|], a 1 × 1 convolution from 3C to C channels. R̃ is the reference feature aligned to each target position (§2.2).
   - W starts as [I, 0, 0], with bias 0. The untrained network therefore computes exactly M0's function, whatever the references. What the reference adds is learned from the first step (both checked by `tests/test_refnet.py`).
   - The decoder gets F at 1/16 and 1/32 instead of T. The 1/4 and 1/8 skip features stay the target's own. The classifier head gets F at 1/32.
4. **References as input:** the same cached 768 × 512 images as the target. Each is mirrored when its `_mirrored` flag in the table is 1, then copied to 3 channels and normalised like the target.
5. **Training augmentation:**
   - the target gets M0's augmentations;
   - its references get the target's left–right flip (Report 07 §3.6), so they keep the orientation the table chose;
   - each reference then gets its own, independent draw of the other augmentations (affine, brightness/contrast, gamma, noise or blur).

   Why the independent draw: the bank holds about 290 femurs. With k = 3, each serves as a reference about 8–9 times per epoch (834 × 3 / 290). Its own augmentation keeps the model from memorising the bank, and teaches it to cope with a reference that is not aligned. At prediction time nothing is augmented; the references are used as cached and mirrored as the table says.
6. **The table is fixed:** no run retrieves anything itself. `train_net.py` refuses a table that does not match the fold file (fold, class, group), or whose references break the bank rules (a healthy femur of the fold model's 4 training folds, outside the femur's own group).

### 2.2 Alignment
- **Local cross-attention (R2–R5, R7).** One attention head at each of the two scales:
  - query = W_q LN(T(p)) and key = W_k LN(R_j(p′)), of dimension 128;
  - value = R_j(p′) itself, with no projection, so that R̃ stays in the encoder's feature space and T − R̃ is meaningful;
  - each target position p attends to the positions p′ of **all k references** within ±1/8 of the image height of p (±96 px at 768 × 512: a 13 × 13 window at 1/16 and 7 × 7 at 1/32), with one softmax over all of them;
  - so R̃(p) is the best-matching healthy anatomy near p, among the k references.

  Why ±96 px: the crops are framed alike, so the same anatomy lies near the same place. The window allows shifts of 12.5% of the crop's height (19% of its width), and stops the attention from matching a fracture line to an unrelated edge far away. Development question D1 (§4) tests the alternative: the whole reference.
- **None (R6a):** R̃(p) = the mean of the k references at the same position.
- **Global affine (R1, R6b):** a learned spatial transformer (Jaderberg et al. 2015).
  - For each reference, a small localisation network reads [T, R_j] at 1/32: a 1 × 1 convolution to 256 channels, GELU, a 3 × 3 convolution with stride 2 to 128, GELU, a global average and a linear layer to 6 numbers.
  - These 6 numbers are an affine transform, initialised to the identity.
  - R_j is resampled with it at 1/16 and 1/32 (bilinear, zeros outside the reference), and R̃ is the mean over the k references.

### 2.3 Fusion variants
- **Difference** (all runs except R1 and R5): §2.1, point 3.
- **Concatenation** (R1, R5): F = W [T, R̃], with W starting as [I, 0]. Its linear part spans the same functions as the difference's. So R5 vs R4 mainly tests the |T − R̃| term and the starting point.

### 2.4 R1: AASN, adapted to femur crops
**What AASN does** (Chen et al., ECCV 2020; its method section was read on 2026-11-27, arXiv 2007.01464):
- The pelvic x-ray is flipped about its symmetry line. A thin-plate-spline warp, computed from 16 detected pelvic landmarks, aligns the flipped image's **features** to the original's, after encoding.
- DenseNet-121 with shared weights, split after its third dense block (1/16 of the input). The two feature maps are fused by **concatenation**, and the rest of the DenseNet continues from there.
- A **margin (contrastive) loss** on projected middle-level features (linear, BatchNorm, ReLU), with weight λ = 0.5. The squared distance between the two sides is pulled to 0 outside fractures, and pushed above a margin m = 0.5 inside them.
- The output is a fracture heat map. Image-level AUC was 97.7% on 2,359 pelvic x-rays. In its ablation, feature alignment and the margin loss each added 1.2–3.5 points of recall at 1% false positives.

**What R1 keeps:**
- the Siamese shared encoder;
- alignment of the reference's features after encoding, by one global warp;
- concatenation;
- the margin loss with λ = 0.5 and m = 0.5, on 1/16 features, with the same kind of projection (1 × 1 convolution to 128 channels, BatchNorm, ReLU).

**What R1 changes, and why:**
1. **A learned global affine transform** replaces the landmark-based thin-plate spline: the femur crops have no landmarks, and we have no landmark detector.
2. **The reference is the nearest retrieved healthy femur**, not the other side of the same x-ray: most femurs have no usable partner. R7 tests the partner.
3. **Fusion at 1/16 and 1/32 into M0's U-Net decoder and head**, so that R1 differs from RefFrac only in alignment, fusion, k and the loss, not in the trunk.
4. **The projected features are scaled to length 1 at each position.** The squared distance then lies in [0, 2], and the margin 0.5 has a fixed meaning (AASN does not state a normalisation).
5. **The fracture area is the target's released mask** (its reference is healthy), averaged onto the 1/16 grid. The loss is the mean over grid cells of (1 − a) d² + a max(0, 0.5 − d²), where a is the share of the cell inside the mask and d² the squared distance.

### 2.5 R7: the patient's own other hip, and the pairing shortcut (RQ-B2, RQ-B3)
**Training policy** (as planned in Report 03 §4.2):
- k = 1. The reference is the other femur of the same x-ray, always mirrored (it is the other hip), **when it is healthy**; otherwise it is the nearest retrieved femur. References are therefore always healthy in training.
- Of the 1,043 training femurs, 348 get their partner: 334 fractured femurs and the 14 femurs of healthy pairs. The 334 healthy femurs whose partner is fractured get a retrieved reference.

**The shortcut, stated before the run:**
- Under this policy, a femur whose reference is its own partner is fractured in 334 of 348 cases (96%).
- The partner is recognisable without any label: the same x-ray means the same exposure, scale and patient.
- So R7 can learn "my reference looks like my own x-ray, so I am fractured".
- The policy also needs the partner's label at prediction time, which a real system does not have.

**So R7 is evaluated in five modes** (prediction only, same weights; `--extra-tests blind,retrieved,random,fractured`):

| Mode | Reference at prediction time | Uses a label? | Role |
|---|---|---|---|
| base | the training policy: own partner if healthy, otherwise the nearest | **yes** (the partner's) | Reported only, to show the size of the leak |
| `blind` | own partner whenever there is one (healthy or not), otherwise the nearest | no | **R7's result**: what a real system could do |
| `retrieved` | the nearest retrieved femur, for every femur | no | The same model without partners |
| `random` | the femur's random healthy femur (swap test a) | no | Does the model read the reference's identity? |
| `fractured` | the femur's random fractured femur (swap test b) | no | Does the model read the reference's state? |

**RQ-B2, own hip vs retrieved.** On the fractured femurs with a healthy partner (334 out of fold, 64 test), we compare Dice:
- with the partner (`blind`) vs with the nearest retrieved femur (`retrieved`), in the same model;
- with the partner vs R3, which was trained with retrieved references only.

**RQ-B3, the shortcut.**
1. Healthy femurs whose partner is fractured (334 out of fold, 64 test): P(fracture) and the false-positive rate under `blind` (their fractured partner) vs `retrieved`.
2. Fractured femurs with a healthy partner: the change of P(fracture) and Dice when the partner is swapped for a random healthy femur (`random`), compared with the same swap in R3 and R4.
3. Under `fractured`: how much a fractured reference moves the prediction.

The thresholds of the label-free modes are fitted on their own out-of-fold predictions at the analysis stage (C14), following rule 4 of §3.

### 2.6 Tests recorded by every R run, and the pointing game
- `--extra-tests self,random` predicts the held-out fold (and the test set) again with other references:
  - **`self`:** the target itself as its reference, which carries no information. How much does the decision depend on the comparison? The model never saw this in training, so a change is read with care.
  - **`random`:** the femur's random healthy femur instead of its retrieved ones. How much does the match matter at prediction time?
  - R7 adds `blind`, `retrieved` and `fractured` (§2.5).
- **Every mode writes its own per-case files and probability maps.** Dice, macro-F1 and Garden I-II recall are therefore known in every mode, also for the test ensemble.
  - `metrics.json` reports, per mode, the change of P(fracture) on fractured **and on healthy** femurs, and the share of correct healthy femurs that become fractured.
  - The references each femur got in each mode are written to `per_case_<oof|test>_references.csv`.
- **Pointing game of the difference map** (base mode, every run). For each fractured femur: does the 1/16 cell with the largest |T − R̃| touch the released mask? It is reported as a hit rate next to chance, the share of cells that touch the mask.

## 3. Rules that keep Paper B's numbers honest (Report 04 §3.4)
1. **Development uses fold 0 only.** The development runs predict fold 0's 208 held-out femurs and never the test set (`--no-test`).
2. **Fixed now, before any R result:** everything in §1 and §2. That includes the variants, the recipe, the loss weights, the window, k = 3 for R4, the references without gradient and their augmentation. The only exceptions are the development questions of §4.
3. **Freeze (milestone M2-B):** the settings chosen in §4 are written into `kaggle/runs.json` and tagged `B-v1-frozen` (by you, on GitHub, as for Paper A). Every final R run starts from that tag.
4. **Thresholds and the calibration temperature** come from the pooled out-of-fold predictions only.
5. **Check on folds 1–4:** final results are also reported on folds 1–4 only. If those agree with all 5 folds, the fold-0 development did not inflate anything.
6. **References:**
   - they come only from the fixed table of step B-1;
   - no label is used to choose a reference at prediction time, except in R7's base mode, which is reported only as the leak (§2.5);
   - the reference of a held-out or test femur is a training femur of the fold model's own 4 folds, or, in R7, the femur's own partner from the same x-ray.

## 4. Fold-0 development: two questions and a second seed, two Kaggle sessions

| Question | Base (`R4_dev_base`) | Alternative run | Changed argument |
|---|---|---|---|
| D1 A local window, or the whole reference? | ±1/8 of the image height | `R4_dev_global` | `--ref-window 0` |
| D2 Does AASN's margin loss help RefFrac use its references? | no margin loss | `R4_dev_margin` | `--ref-margin 0.5` |
| Seed noise of the base | seed 2026 | `R4_dev_base_s2` | `--seed 2027` |

**How the runs are launched:**
- Each development run is R4 trained on fold 0 only, with the tests `self` and `random` and no test set.
- Each takes about 75 min on one T4: M0's fold 0 trained in 44.5 min, and the references add about 60%.
- Kaggle gives two T4s per session, so two runs share one session (a parallel group of `kaggle/run.py`):

| RUN to launch | GPU 0 | GPU 1 | Session time |
|---|---|---|---|
| `R_dev_round1` | `R4_dev_base` | `R4_dev_base_s2` | ≈ 80 min |
| `R_dev_round2` | `R4_dev_global` | `R4_dev_margin` | ≈ 80 min |

Why a second seed from the start: for Paper A you asked for one after the first round (Analysis/06 §10). Here it comes in the first session, at no extra session.

**Reference:** M0's own fold-0 model, from the filed run S10, as in Analysis/06 §4. It scores:
- macro-F1 .822;
- accuracy 184 of 208;
- Dice on fractured femurs .749;
- Garden I-II recall .545, from only 11 femurs.

**Decision rule, fixed now.** Analysis **C12** compares the runs with M0 on fold 0's 208 femurs (`scripts/run_stats.py --folds 0`, with thresholds fitted again on those femurs). "Base" means the mean of `R4_dev_base` and `R4_dev_base_s2`. The **seed noise** N is the difference between their two Dice values.
1. **Go / no-go:** if the base's Dice on fractured femurs is below .739 (M0 − .01), or its macro-F1 is below .79 (M0 − .03), the design goes back to you before any 5-fold run.
2. **Each question:** the alternative replaces the base setting only if both hold:
   - its Dice beats the base by at least .01 **and** by more than N;
   - its macro-F1 is not lower than the base's by more than .02.
3. **Two winners:** if D1 and D2 both pass, their combination is run on fold 0 (one more session) before freezing.
4. **Reported, not used:** Garden I-II recall (11 femurs), each seed's own numbers, the `self` and `random` tests, and the pointing game.
5. **Does the base use its references?** If `self` changes the base's fold-0 Dice by less than .005 and its mean P(fracture) on fractured femurs by less than .01, the base ignores its references. This is reported to you with the decision, because it would already answer RQ-B1 with "no" in development.

**Why Dice decides:** it is Paper B's first primary endpoint, and the second one (Garden I-II recall) has only 11 femurs in fold 0.
**Why so conservative:** fold 0 holds only 136 fractured femurs, and a change of .01 in mean Dice is within the noise of one seed. So the rule changes a default only for a clear effect, and it is not a search.

## 5. Final runs, analyses and compute

**Final runs (after M2-B, from the tag):**
- R1–R7, 5 folds and the test set, each with the tests `self` and `random`; R7 with `blind`, `retrieved`, `random` and `fractured` instead.
- Order:
  1. R4;
  2. R3 and R2 (RQ-B1);
  3. R7 (RQ-B2, RQ-B3);
  4. R1 (the published baseline);
  5. R5, R6a and R6b (RQ-B4).

**Compute** (estimated from S00_net_check; the pre-flight of the first development run measures it):
- **Per training step:** S06's ConvNeXt-T step takes 0.51 s per 16 images, and a forward pass about a third of that. The references' forward passes therefore make a step about 1.6 times M0's for k = 3, and 1.2 times for k = 1.
- **5-fold runs on T4 x2:** M0 took 2.3 h (140 min). So R4, R5, R6a and R6b take about 3.7 h each, and R1, R2, R3 and R7 about 2.8 h each: **about 26 h of session time**, plus about 3 h of development.
- **Budget:** Report 04 §6 planned 55–60 GPU-hours for Track B.
  - That fits if a T4 x2 session counts once in Kaggle's quota.
  - If it counts twice (still open), Track B needs about 60 GPU-hours, at the top of the plan.
  - Paper A's final runs keep priority (Report 04 §9).

**Analyses:**

| Analysis | Runs | Question |
|---|---|---|
| C12 | `R4_dev_*` vs M0 (fold 0) | Development decisions (§4) |
| C13 | M0 (R0), R2, R3, R4 | RQ-B1: does a reference help, and does retrieval matter? Primary endpoints: Dice and Garden I-II recall. Subgroups: the smallest third of masks; paired vs single femurs |
| C14 | R7, with R3 and R4 | RQ-B2 and RQ-B3 (§2.5) |
| C15 | R4 vs R1, R5, R6a, R6b | RQ-B4: alignment and fusion; the published baseline |
| C16 | R4 vs S01, S04–S10 | Paper B's main table |

Statistics as in Paper A: cluster bootstrap over x-rays, paired tests, and Holm within each analysis. Out of fold first, the test set as confirmation, and folds 1–4 only as a check.

## 6. Known limitations, stated now
- **References without gradient:** the encoder learns from the target only. A version with gradients through the references would cost about 2.5 times M0's compute and is not tested.
- **Generic retrieval:** the references were chosen by frozen ImageNet features, which describe framing and contrast more than fine anatomy (Report 07 §6).
- **A small bank:** about 290 healthy femurs per fold model. For an unusual femur, even the nearest reference may differ.
- **Wrong orientation:** about 0.5% of femurs may get a reference mirrored the wrong way (Report 07 §8). The cross-attention has to cope with it.
- **The pairing shortcut** is a property of this dataset: one fractured femur in each two-femur x-ray. R7's `blind` numbers do not carry over to populations where both hips are often healthy.
- **Development noise:** fold 0 has only 136 fractured femurs. Hence the second seed and the default-favouring rule of §4.
- **The train/test annotation shift** (C01, C04) affects every model's test Dice in the same way. As before, Dice is also reported per location.

## 7. Implementation (2026-11-27)
- **Code:**
  - `src/femurseg/refnet.py`: the variants, the network, the alignments, the fusion, the margin loss, the reference table and its policies, and the pointing game;
  - `src/femurseg/torchdata.py`: `RefFemurDataset`;
  - `src/femurseg/nets.py`: the `ref:` kind;
  - `scripts/train_net.py`: `--ref-table`, `--pairs-table`, `--ref-k`, `--ref-window`, `--ref-margin`, and the reference tests.
- **Size:**

  | Network | Parameters |
  |---|---|
  | M0 | 31.93 M |
  | R2, R3, R4, R7 | 34.45 M |
  | R5 | 33.71 M |
  | R6a | 34.15 M |
  | R6b | 34.84 M |
  | R1 | 34.15 M |
- **M0 and SLG-Net are unchanged.** A 1-epoch fold-0 run of M0, and one of SLG-Net A6 with the oracle and deletion tests, were run on the synthetic dataset with this code and with `main` before it (`f96f84c`). Their per-case files, `metrics.json` and `summary.md` are byte-identical; only the seconds per epoch differ. So Paper A's second-seed check (`A_dev_seed2`, launched from `main`) is not affected.
- **Tests:**
  - `tests/test_refnet.py`: 11 checks, including the untrained network = M0, the attention window, the alignments, the losses, the table's policies and rules, and the joint flip;
  - `tests/smoke_train_refnet.py`: end to end on the CPU. It covers the pre-flight of every variant, 5-fold runs of R4 and R7 with their tests, and the development group through `kaggle/run.py`, filed by `register_run.py` and compared with M0 by `run_stats.py --folds 0`.
  - **Both pass** (2026-11-27): the smoke test in 9 min, and all 73 unit tests of the repository (every file except `tests/test_data.py`, which needs the dataset).

## 8. Addendum (2026-11-27): result of the fold-0 development (C12) and the freeze

Written on 2026-11-27, after the four development runs and before any final R run. §1–§7 are unchanged.

- **Runs:** `R4_dev_base`, `R4_dev_base_s2`, `R4_dev_global` and `R4_dev_margin` (register lines 27–30), launched from `main` = `2619b39` in the groups `R_dev_round1` and `R_dev_round2` (83 and 87 min). Each trained fold 0 completely, at micro-batch 16 (12.6 GB of a T4's 15 GB).
- **Result** (`results/analysis/C12_paperB_dev_fold0/report.md`), on fold 0's 208 femurs:

  | Run | Dice, fractured femurs | Macro-F1 |
  |---|---|---|
  | M0 (S10, fold 0) | .749 | .822 |
  | Base (mean of seeds 2026 and 2027) | **.770** | **.818** |
  | D1 whole reference | .778 | .796 |
  | D2 margin loss | .755 | .829 |

- **Decisions by the rule of §4:**
  1. go / no-go: **passes** (Dice .770 ≥ .739; macro-F1 .818 ≥ .79);
  2. D1: Dice +.009, below .01 and within the seed noise N = .009, and macro-F1 −.022: **the local window stays**;
  3. D2: Dice −.014: **no margin loss** in R2–R7 (R1 keeps it, as AASN's own loss);
  4. no combination run is needed;
  5. the base uses its references: with itself as the reference, its Dice falls by .090 and its P(fracture) on fractured femurs by .49 (means of the two seeds).
- **Reported, not used:** a random healthy reference instead of the 3 nearest changes the base's Dice by only .004–.005 at prediction time (a first hint for RQ-B1); the pointing game hits the fracture in 64% and 92% of fractured femurs (two seeds; chance 8%).
- **Freeze (milestone M2-B, §3.3):** no setting of §1–§2 changes. The eight final runs (`R4_refrac`, `R3_nearest_ref`, `R2_random_ref`, `R7_partner`, `R1_aasn`, `R5_concat`, `R6a_no_align`, `R6b_affine`) are written into `kaggle/runs.json` with exactly these settings. After the pull request of Step 18 is merged, the person creates the tag `B-v1-frozen` on `main`, and every final R run is launched from it. The training code at that commit is the code that trained the development runs (`2619b39`): the pull request changes results, analyses, documents and run entries only.
- **Compute, measured:** an R4 fold took 81 min on one T4 (81 s per epoch, against M0's 45 s), 1.8 times M0's fold rather than the 1.6 times of §5; the pre-flight measured 1.40–1.55 s per micro-step (M0: 0.91 s). So R4, R5, R6a and R6b take about 4.2 h each on T4 x2, and R1, R2, R3 and R7 about 3 h each (not yet measured): **about 29 h** of session time, instead of the 26 h of §5.
- **Order of the final runs (§5):** R4 first; then R3 and R2; R7; R1; R5, R6a and R6b.

## 9. Addendum (2026-11-27): the tag `B-v1-frozen` exists

- **Created** by the person on 2026-11-27 on `1c88912`, the merge of PR #14 (Step 18). Milestone M2-B is reached.
- **Its files are identical to `3c52ed7`** (Step 18). Its training code is identical to `2619b39`, which trained the development runs: `scripts/train_net.py`, `src/femurseg/`, `kaggle/run.py` and `kaggle/launcher.ipynb` have no difference between the two commits.
- **Every final R run** is launched with `BRANCH = "B-v1-frozen"`, and its log must show `1c88912`.

## 10. Addendum (2026-11-28): results of the final runs (C13–C16)

All eight final runs (R1–R7, register lines 31–38) were trained from `1c88912` on 5/5 folds, about 28.6 h of T4 x2 session time in total. The analyses were run as §5 fixed them, with one tool added: `scripts/run_refstats.py` (Garden I-II McNemar, subgroups, reference tests and R7's modes), because `run_stats.py` has no test for Garden I-II recall. Reports: `results/analysis/C13_paperB_reference/`, `C14_paperB_partner/`, `C15_paperB_components/`, `C16_paperB_main_table/` (each `report.md`).

**Choices made at the analysis stage, before reading the affected numbers:**
- Garden I-II recall is compared with an exact McNemar test on the Garden I-II femurs (as accuracy elsewhere) and a cluster-bootstrap interval; Holm across the pairs of each analysis.
- "The smallest third of masks" is each part's own lower tertile of released mask area among fractured femurs (test masks are smaller, C01).
- A swap test is read at the threshold of the mode it replaces; R7's blind and retrieved modes get their own OOF Youden thresholds (rule 4 of §3).

**Results:**

| Question | Answer | Evidence |
|---|---|---|
| RQ-B1, Dice: does a healthy reference help? | **No** | every Dice difference ≤ .005, OOF and test, also in the subgroups (C13) |
| RQ-B1, Garden I-II recall | **Trend, not significant** | OOF M0 .544 < R2 .614 < R3 .684 < R4 .702; R4 vs M0 +.158, 12 vs 3, p .035, Holm .14; test .312 = .312; folds 1–4 +.109 (C13) |
| RQ-B1: does retrieval matter? | **No** | R3 vs R2 n.s.; a random reference at prediction time changes nothing (C13) |
| Is the reference used? | **Yes, as "different from healthy"** | self test: R4 loses 201 of 663 fractures, Garden I-II .702 → .158; pointing game 4–9 × chance (C13 §4) |
| RQ-B2: own hip vs retrieved | **No difference** | R7 blind vs retrieved Dice −.001 (Holm .10), vs R3 −.000 (C14) |
| RQ-B3: pairing shortcut | **Not learnt** | a fractured partner does not raise P(fracture) of healthy femurs (.0071 vs .0093); base = blind decisions (C14) |
| RQ-B4: alignment and fusion | **No effect on accuracy** | Dice and macro-F1 n.s.; Garden I-II R4 > R6a (8 vs 0, Holm .031), other variants Holm .06–.08; test none (C15) |
| Main table | **R4 ties M0** | OOF beats S01 (+.081 macro-F1, +.037 Dice), S05, S06, and S07–S09 on Dice; test: not better than M0 or S01 (C16) |

**Consequence for Paper B:** the pre-registered primary hypothesis (a reference improves Dice and Garden I-II recall) is not confirmed. The defensible paper is an analysis of reference-based comparison: it is used, its match does not matter, the patient's own hip is not better, and there is no pairing shortcut; the Garden I-II gain is an exploratory out-of-fold trend. How to frame Paper B is a decision for the person (as for Paper A).

## 11. Addendum (2026-11-28): the person's decision, one paper

- **Paper B is not written as a separate paper.** Its findings become part of Paper A, which reports both extensions of the shared core model M0: SLG-Net (segmentation-guided hierarchy) and RefFrac (comparison with healthy references).
- **What Paper A takes from Paper B:** the R runs as pre-registered (protocol and tag `B-v1-frozen`), the main results of C13 (no Dice gain; the Garden I-II trend as exploratory), the reference tests (the comparison is used, its match is not), C14 (the own hip is no better; no pairing shortcut) and C15 in short. C16's rows join Paper A's main table.
- **No new Paper B runs.** The sensitivity runs (R-sens) are dropped. Milestones M4-B and M5-B lapse; B-6 figures are made as part of Paper A's results package (M4-A).
