# Report 08: Step 5 protocol, Paper B (RefFrac) step 2 of Paper B: the Siamese model, the eight reference runs and the fold-0 development

**Date:** 2026-11-27 · **Code:** `scripts/train_net.py --net ref:convnext_tiny.fb_in22k_ft_in1k:<R1..R7>` with `src/femurseg/refnet.py` · **References:** the fixed tables of step 1 of Paper B, `results/R00_reference_bank_r2/` · **Runs:** `kaggle/runs.json`
**Why this report:** it fixes, before any R run exists, what each run of Paper B is, the model exactly, how the fold-0 development is run and decided, what is then frozen, and how the final runs and the reference tests are scored. Paper B's methods section is written from it. The design comes from Report 03 section 4 and Report 04 section 5, and the references from Report 07.

**Starting point:**
- ConvNeXt U-Net is the model every R run extends. On the 1,043 out-of-fold femurs it scores Dice .777 on fractured femurs, Garden I-II recall .544 and macro-F1 .825 (segmenter analysis, the mask-task analysis). Its test-ensemble Dice is .744 (segmenter analysis).
- **Paper B's primary endpoints** (fixed in Report 04 section 5.5): mean Dice on fractured femurs, and Garden I-II recall (57 out-of-fold and 16 test femurs).
- Every femur already has its references in the fixed table of step 1 of Paper B (Report 07 section 9). For each fold model, the table holds:
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

| Run | `--net` | References (k) | Alignment | Fusion | Question (Report 04 section 5) |
|---|---|---|---|---|---|
| ConvNeXt U-Net | = ConvNeXt U-Net (filed) | none | – | – | Starting point |
| AASN-style model | `ref:…:R1` | the nearest (1) | global affine | concatenation + AASN's margin loss | The closest published idea (AASN), adapted (section 2.4) |
| Random-reference model | `ref:…:R2` | a random healthy femur (1) | local cross-attention | difference | Does any healthy second image help? |
| Nearest-reference model | `ref:…:R3` | the nearest (1) | local cross-attention | difference | Does retrieval help (nearest-reference model vs the random-reference model)? |
| RefFrac | `ref:…:R4` | the 3 nearest (3) | local cross-attention | difference | **RefFrac**, the main method; 3 references vs 1 (RefFrac vs the nearest-reference model) |
| Concatenation model | `ref:…:R5` | the 3 nearest | local cross-attention | concatenation | Difference or concatenation? |
| No-alignment model | `ref:…:R6a` | the 3 nearest | none | difference | Is alignment needed? |
| Affine-alignment model | `ref:…:R6b` | the 3 nearest | global affine | difference | Local or global alignment? |
| Own-hip model | `ref:…:R7` | own healthy partner, else the nearest (1) | local cross-attention | difference | Own hip vs retrieved (own-hip question); the pairing shortcut (pairing-shortcut question) (section 2.5) |
| R-sens | later | | | | Second backbone, as A-sens |

**One recipe.** Every run uses ConvNeXt U-Net's recipe unchanged (Report 05 section 2):
- input 768 × 512, the same augmentations;
- AdamW (learning rate 2 × 10⁻⁴, weight decay 0.05), 2 warm-up epochs then cosine decay, **60 epochs, final weights**;
- batch 16 (micro-batch set by the pre-flight), fp16, seed 2026 + fold.

Every R run starts from the ImageNet weights, as ConvNeXt U-Net did, not from ConvNeXt U-Net's trained weights. Only the reference path differs from ConvNeXt U-Net, so a difference from ConvNeXt U-Net can be attributed to the reference.

## 2. The model, exactly

### 2.1 Shared by the eight reference runs
1. **Trunk and losses = ConvNeXt U-Net's:**
   - the `segmentation_models_pytorch` U-Net with the `tu-convnext_tiny` encoder, the same decoder and binary mask head;
   - ConvNeXt U-Net's classifier head (the timm head: global average pool, LayerNorm, dropout 0.2, linear to 6 classes) on the 1/32 features;
   - the loss: cross-entropy of the 6 classes + binary cross-entropy + Dice of the fracture mask (weight 1).
2. **Siamese encoding:** the target T and its k references R₁ … R_k go through the same encoder (shared weights).
   - The references are encoded **without gradient** (forward passes only). With gradient, the encoder's activation memory would grow k + 1 times (4 times for RefFrac), and a step would cost about 2.5 times ConvNeXt U-Net's. Without it, a step costs about 1.6 times ConvNeXt U-Net's for k = 3 and 1.2 times for k = 1 (section 5), and the memory stays close to ConvNeXt U-Net's.
   - The encoder still learns from the target. The reference features are always computed with the current weights, so target and references stay in one feature space.
   - ConvNeXt's encoder has no batch statistics and no dropout, so a reference is encoded exactly as it would be with gradient.
   - Only the references' 1/16 (384 channels) and 1/32 (768 channels) features are kept.
3. **Fusion at 1/16 and 1/32:** the target's features T are replaced by F = W [T, T − R̃, |T − R̃|], a 1 × 1 convolution from 3C to C channels. R̃ is the reference feature aligned to each target position (section 2.2).
   - W starts as [I, 0, 0], with bias 0. The untrained network therefore computes exactly ConvNeXt U-Net's function, whatever the references. What the reference adds is learned from the first step (both checked by `tests/test_refnet.py`).
   - The decoder gets F at 1/16 and 1/32 instead of T. The 1/4 and 1/8 skip features stay the target's own. The classifier head gets F at 1/32.
4. **References as input:** the same cached 768 × 512 images as the target. Each is mirrored when its `_mirrored` flag in the table is 1, then copied to 3 channels and normalised like the target.
5. **Training augmentation:**
   - the target gets ConvNeXt U-Net's augmentations;
   - its references get the target's left–right flip (Report 07 section 3.6), so they keep the orientation the table chose;
   - each reference then gets its own, independent draw of the other augmentations (affine, brightness/contrast, gamma, noise or blur).

   Why the independent draw: the bank holds about 290 femurs. With k = 3, each serves as a reference about 8–9 times per epoch (834 × 3 / 290). Its own augmentation keeps the model from memorising the bank, and teaches it to cope with a reference that is not aligned. At prediction time nothing is augmented; the references are used as cached and mirrored as the table says.
6. **The table is fixed:** no run retrieves anything itself. `train_net.py` refuses a table that does not match the fold file (fold, class, group), or whose references break the bank rules (a healthy femur of the fold model's 4 training folds, outside the femur's own group).

### 2.2 Alignment
- **Local cross-attention (reference runs, the own-hip model).** One attention head at each of the two scales:
  - query = W_q LN(T(p)) and key = W_k LN(R_j(p′)), of dimension 128;
  - value = R_j(p′) itself, with no projection, so that R̃ stays in the encoder's feature space and T − R̃ is meaningful;
  - each target position p attends to the positions p′ of **all k references** within ±1/8 of the image height of p (±96 px at 768 × 512: a 13 × 13 window at 1/16 and 7 × 7 at 1/32), with one softmax over all of them;
  - so R̃(p) is the best-matching healthy anatomy near p, among the k references.

  Why ±96 px: the crops are framed alike, so the same anatomy lies near the same place. The window allows shifts of 12.5% of the crop's height (19% of its width), and stops the attention from matching a fracture line to an unrelated edge far away. Development the whole-reference question (section 4) tests the alternative: the whole reference.
- **None (no-alignment model):** R̃(p) = the mean of the k references at the same position.
- **Global affine (AASN-style model, the affine-alignment model):** a learned spatial transformer (Jaderberg et al. 2015).
  - For each reference, a small localisation network reads [T, R_j] at 1/32: a 1 × 1 convolution to 256 channels, GELU, a 3 × 3 convolution with stride 2 to 128, GELU, a global average and a linear layer to 6 numbers.
  - These 6 numbers are an affine transform, initialised to the identity.
  - R_j is resampled with it at 1/16 and 1/32 (bilinear, zeros outside the reference), and R̃ is the mean over the k references.

### 2.3 Fusion variants
- **Difference** (all runs except the AASN-style model and the concatenation model): section 2.1, point 3.
- **Concatenation** (AASN-style model, the concatenation model): F = W [T, R̃], with W starting as [I, 0]. Its linear part spans the same functions as the difference's. So the concatenation model vs RefFrac mainly tests the |T − R̃| term and the starting point.

### 2.4 the AASN-style model: AASN, adapted to femur crops
**What AASN does** (Chen et al., ECCV 2020; its method section was read on 2026-11-27, arXiv 2007.01464):
- The pelvic x-ray is flipped about its symmetry line. A thin-plate-spline warp, computed from 16 detected pelvic landmarks, aligns the flipped image's **features** to the original's, after encoding.
- DenseNet-121 with shared weights, split after its third dense block (1/16 of the input). The two feature maps are fused by **concatenation**, and the rest of the DenseNet continues from there.
- A **margin (contrastive) loss** on projected middle-level features (linear, BatchNorm, ReLU), with weight λ = 0.5. The squared distance between the two sides is pulled to 0 outside fractures, and pushed above a margin m = 0.5 inside them.
- The output is a fracture heat map. Image-level AUC was 97.7% on 2,359 pelvic x-rays. In its ablation, feature alignment and the margin loss each added 1.2–3.5 points of recall at 1% false positives.

**What the AASN-style model keeps:**
- the Siamese shared encoder;
- alignment of the reference's features after encoding, by one global warp;
- concatenation;
- the margin loss with λ = 0.5 and m = 0.5, on 1/16 features, with the same kind of projection (1 × 1 convolution to 128 channels, BatchNorm, ReLU).

**What the AASN-style model changes, and why:**
1. **A learned global affine transform** replaces the landmark-based thin-plate spline: the femur crops have no landmarks, and we have no landmark detector.
2. **The reference is the nearest retrieved healthy femur**, not the other side of the same x-ray: most femurs have no usable partner. The own-hip model tests the partner.
3. **Fusion at 1/16 and 1/32 into ConvNeXt U-Net's U-Net decoder and head**, so that AASN-style model differs from RefFrac only in alignment, fusion, k and the loss, not in the trunk.
4. **The projected features are scaled to length 1 at each position.** The squared distance then lies in [0, 2], and the margin 0.5 has a fixed meaning (AASN does not state a normalisation).
5. **The fracture area is the target's released mask** (its reference is healthy), averaged onto the 1/16 grid. The loss is the mean over grid cells of (1 − a) d² + a max(0, 0.5 − d²), where a is the share of the cell inside the mask and d² the squared distance.

### 2.5 the own-hip model: the patient's own other hip, and the pairing shortcut (own-hip question, the pairing-shortcut question)
**Training policy** (as planned in Report 03 section 4.2):
- k = 1. The reference is the other femur of the same x-ray, always mirrored (it is the other hip), **when it is healthy**; otherwise it is the nearest retrieved femur. References are therefore always healthy in training.
- Of the 1,043 training femurs, 348 get their partner: 334 fractured femurs and the 14 femurs of healthy pairs. The 334 healthy femurs whose partner is fractured get a retrieved reference.

**The shortcut, stated before the run:**
- Under this policy, a femur whose reference is its own partner is fractured in 334 of 348 cases (96%).
- The partner is recognisable without any label: the same x-ray means the same exposure, scale and patient.
- So the own-hip model can learn "my reference looks like my own x-ray, so I am fractured".
- The policy also needs the partner's label at prediction time, which a real system does not have.

**So the own-hip model is evaluated in five modes** (prediction only, same weights; `--extra-tests blind,retrieved,random,fractured`):

| Mode | Reference at prediction time | Uses a label? | Role |
|---|---|---|---|
| base | the training policy: own partner if healthy, otherwise the nearest | **yes** (the partner's) | Reported only, to show the size of the leak |
| `blind` | own partner whenever there is one (healthy or not), otherwise the nearest | no | **Own-hip model's result**: what a real system could do |
| `retrieved` | the nearest retrieved femur, for every femur | no | The same model without partners |
| `random` | the femur's random healthy femur (swap test a) | no | Does the model read the reference's identity? |
| `fractured` | the femur's random fractured femur (swap test b) | no | Does the model read the reference's state? |

**Own-hip question, own hip vs retrieved.** On the fractured femurs with a healthy partner (334 out of fold, 64 test), we compare Dice:
- with the partner (`blind`) vs with the nearest retrieved femur (`retrieved`), in the same model;
- with the partner vs the nearest-reference model, which was trained with retrieved references only.

**Pairing-shortcut question, the shortcut.**
1. Healthy femurs whose partner is fractured (334 out of fold, 64 test): P(fracture) and the false-positive rate under `blind` (their fractured partner) vs `retrieved`.
2. Fractured femurs with a healthy partner: the change of P(fracture) and Dice when the partner is swapped for a random healthy femur (`random`), compared with the same swap in the nearest-reference model and RefFrac.
3. Under `fractured`: how much a fractured reference moves the prediction.

The thresholds of the label-free modes are fitted on their own out-of-fold predictions at the analysis stage (own-hip analysis), following rule 4 of section 3.

### 2.6 Tests recorded by every R run, and the pointing game
- `--extra-tests self,random` predicts the held-out fold (and the test set) again with other references:
  - **`self`:** the target itself as its reference, which carries no information. How much does the decision depend on the comparison? The model never saw this in training, so a change is read with care.
  - **`random`:** the femur's random healthy femur instead of its retrieved ones. How much does the match matter at prediction time?
  - Own-hip model adds `blind`, `retrieved` and `fractured` (section 2.5).
- **Every mode writes its own per-case files and probability maps.** Dice, macro-F1 and Garden I-II recall are therefore known in every mode, also for the test ensemble.
  - `metrics.json` reports, per mode, the change of P(fracture) on fractured **and on healthy** femurs, and the share of correct healthy femurs that become fractured.
  - The references each femur got in each mode are written to `per_case_<oof|test>_references.csv`.
- **Pointing game of the difference map** (base mode, every run). For each fractured femur: does the 1/16 cell with the largest |T − R̃| touch the released mask? It is reported as a hit rate next to chance, the share of cells that touch the mask.

## 3. Rules that keep Paper B's numbers honest (Report 04 section 3.4)
1. **Development uses fold 0 only.** The development runs predict fold 0's 208 held-out femurs and never the test set (`--no-test`).
2. **Fixed now, before any R result:** everything in sections 1 and 2. That includes the variants, the recipe, the loss weights, the window, k = 3 for RefFrac, the references without gradient and their augmentation. The only exceptions are the development questions of section 4.
3. **Freeze (Paper B milestone 2):** the settings chosen in section 4 are written into `kaggle/runs.json` and tagged `B-v1-frozen` (by you, on GitHub, as for Paper A). Every final R run starts from that tag.
4. **Thresholds and the calibration temperature** come from the pooled out-of-fold predictions only.
5. **Check on folds 1–4:** final results are also reported on folds 1–4 only. If those agree with all 5 folds, the fold-0 development did not inflate anything.
6. **References:**
   - they come only from the fixed table of step 1 of Paper B;
   - no label is used to choose a reference at prediction time, except in the own-hip model's base mode, which is reported only as the leak (section 2.5);
   - the reference of a held-out or test femur is a training femur of the fold model's own 4 folds, or, in the own-hip model, the femur's own partner from the same x-ray.

## 4. Fold-0 development: two questions and a second seed, two Kaggle sessions

| Question | Base (`R4_dev_base`) | Alternative run | Changed argument |
|---|---|---|---|
| Whole-reference question A local window, or the whole reference? | ±1/8 of the image height | `R4_dev_global` | `--ref-window 0` |
| Margin-loss question Does AASN's margin loss help RefFrac use its references? | no margin loss | `R4_dev_margin` | `--ref-margin 0.5` |
| Seed noise of the base | seed 2026 | `R4_dev_base_s2` | `--seed 2027` |

**How the runs are launched:**
- Each development run is RefFrac trained on fold 0 only, with the tests `self` and `random` and no test set.
- Each takes about 75 min on one T4: ConvNeXt U-Net's fold 0 trained in 44.5 min, and the references add about 60%.
- Kaggle gives two T4s per session, so two runs share one session (a parallel group of `kaggle/run.py`):

| RUN to launch | GPU 0 | GPU 1 | Session time |
|---|---|---|---|
| `R_dev_round1` | `R4_dev_base` | `R4_dev_base_s2` | ≈ 80 min |
| `R_dev_round2` | `R4_dev_global` | `R4_dev_margin` | ≈ 80 min |

Why a second seed from the start: for Paper A you asked for one after the first round (Analysis/06 section 10). Here it comes in the first session, at no extra session.

**Reference:** ConvNeXt U-Net's own fold-0 model, from the filed run ConvNeXt U-Net, as in Analysis/06 section 4. It scores:
- macro-F1 .822;
- accuracy 184 of 208;
- Dice on fractured femurs .749;
- Garden I-II recall .545, from only 11 femurs.

**Decision rule, fixed now.** Analysis **the RefFrac development analysis** compares the runs with ConvNeXt U-Net on fold 0's 208 femurs (`scripts/run_stats.py --folds 0`, with thresholds fitted again on those femurs). "Base" means the mean of `R4_dev_base` and `R4_dev_base_s2`. The **seed noise** N is the difference between their two Dice values.
1. **Go / no-go:** if the base's Dice on fractured femurs is below .739 (ConvNeXt U-Net − .01), or its macro-F1 is below .79 (ConvNeXt U-Net − .03), the design goes back to you before any 5-fold run.
2. **Each question:** the alternative replaces the base setting only if both hold:
   - its Dice beats the base by at least .01 **and** by more than N;
   - its macro-F1 is not lower than the base's by more than .02.
3. **Two winners:** if the whole-reference question and the margin-loss question both pass, their combination is run on fold 0 (one more session) before freezing.
4. **Reported, not used:** Garden I-II recall (11 femurs), each seed's own numbers, the `self` and `random` tests, and the pointing game.
5. **Does the base use its references?** If `self` changes the base's fold-0 Dice by less than .005 and its mean P(fracture) on fractured femurs by less than .01, the base ignores its references. This is reported to you with the decision, because it would already answer the reference question with "no" in development.

**Why Dice decides:** it is Paper B's first primary endpoint, and the second one (Garden I-II recall) has only 11 femurs in fold 0.
**Why so conservative:** fold 0 holds only 136 fractured femurs, and a change of .01 in mean Dice is within the noise of one seed. So the rule changes a default only for a clear effect, and it is not a search.

## 5. Final runs, analyses and compute

**Final runs (after Paper B milestone 2, from the tag):**
- Eight reference runs, 5 folds and the test set, each with the tests `self` and `random`; the own-hip model with `blind`, `retrieved`, `random` and `fractured` instead.
- Order:
  1. RefFrac;
  2. Nearest-reference model and the random-reference model (reference question);
  3. Own-hip model (own-hip question, the pairing-shortcut question);
  4. AASN-style model (the published baseline);
  5. Concatenation model, the no-alignment model and the affine-alignment model (alignment-and-fusion question).

**Compute** (estimated from S00_net_check; the pre-flight of the first development run measures it):
- **Per training step:** ConvNeXt-T classifier's ConvNeXt-T step takes 0.51 s per 16 images, and a forward pass about a third of that. The references' forward passes therefore make a step about 1.6 times ConvNeXt U-Net's for k = 3, and 1.2 times for k = 1.
- **5-fold runs on T4 x2:** ConvNeXt U-Net took 2.3 h (140 min). So RefFrac, the concatenation model, the no-alignment model and the affine-alignment model take about 3.7 h each, and the AASN-style model, the random-reference model, the nearest-reference model and the own-hip model about 2.8 h each: **about 26 h of session time**, plus about 3 h of development.
- **Budget:** Report 04 section 6 planned 55–60 GPU-hours for Track B.
  - That fits if a T4 x2 session counts once in Kaggle's quota.
  - If it counts twice (still open), Track B needs about 60 GPU-hours, at the top of the plan.
  - Paper A's final runs keep priority (Report 04 section 9).

**Analyses:**

| Analysis | Runs | Question |
|---|---|---|
| RefFrac development analysis | `R4_dev_*` vs ConvNeXt U-Net (fold 0) | Development decisions (section 4) |
| Reference analysis | ConvNeXt U-Net, the random-reference model, the nearest-reference model, RefFrac | Reference question: does a reference help, and does retrieval matter? Primary endpoints: Dice and Garden I-II recall. Subgroups: the smallest third of masks; paired vs single femurs |
| Own-hip analysis | Own-hip model, with the nearest-reference model and RefFrac | Own-hip question and the pairing-shortcut question (section 2.5) |
| RefFrac component analysis | RefFrac vs the AASN-style model, the concatenation model, the no-alignment model, the affine-alignment model | Alignment-and-fusion question: alignment and fusion; the published baseline |
| RefFrac main-table analysis | RefFrac vs YOLOv8s-seg, the seven other baselines | Paper B's main table |

Statistics as in Paper A: cluster bootstrap over x-rays, paired tests, and Holm within each analysis. Out of fold first, the test set as confirmation, and folds 1–4 only as a check.

## 6. Known limitations, stated now
- **References without gradient:** the encoder learns from the target only. A version with gradients through the references would cost about 2.5 times ConvNeXt U-Net's compute and is not tested.
- **Generic retrieval:** the references were chosen by frozen ImageNet features, which describe framing and contrast more than fine anatomy (Report 07 section 6).
- **A small bank:** about 290 healthy femurs per fold model. For an unusual femur, even the nearest reference may differ.
- **Wrong orientation:** about 0.5% of femurs may get a reference mirrored the wrong way (Report 07 section 8). The cross-attention has to cope with it.
- **The pairing shortcut** is a property of this dataset: one fractured femur in each two-femur x-ray. The own-hip model's `blind` numbers do not carry over to populations where both hips are often healthy.
- **Development noise:** fold 0 has only 136 fractured femurs. Hence the second seed and the default-favouring rule of section 4.
- **The train/test annotation shift** (YOLO resolution analysis, the segmenter analysis) affects every model's test Dice in the same way. As before, Dice is also reported per location.

## 7. Implementation (2026-11-27)
- **Code:**
  - `src/femurseg/refnet.py`: the variants, the network, the alignments, the fusion, the margin loss, the reference table and its policies, and the pointing game;
  - `src/femurseg/torchdata.py`: `RefFemurDataset`;
  - `src/femurseg/nets.py`: the `ref:` kind;
  - `scripts/train_net.py`: `--ref-table`, `--pairs-table`, `--ref-k`, `--ref-window`, `--ref-margin`, and the reference tests.
- **Size:**

  | Network | Parameters |
  |---|---|
  | ConvNeXt U-Net | 31.93 M |
  | Random-reference model, the nearest-reference model, RefFrac, the own-hip model | 34.45 M |
  | Concatenation model | 33.71 M |
  | No-alignment model | 34.15 M |
  | Affine-alignment model | 34.84 M |
  | AASN-style model | 34.15 M |
- **ConvNeXt U-Net and SLG-Net are unchanged.** A 1-epoch fold-0 run of ConvNeXt U-Net, and one of SLG-Net with the oracle and deletion tests, were run on the synthetic dataset with this code and with `main` before it (`f96f84c`). Their per-case files, `metrics.json` and `summary.md` are byte-identical; only the seconds per epoch differ. So Paper A's second-seed check (`A_dev_seed2`, launched from `main`) is not affected.
- **Tests:**
  - `tests/test_refnet.py`: 11 checks, including the untrained network = ConvNeXt U-Net, the attention window, the alignments, the losses, the table's policies and rules, and the joint flip;
  - `tests/smoke_train_refnet.py`: end to end on the CPU. It covers the pre-flight of every variant, 5-fold runs of RefFrac and the own-hip model with their tests, and the development group through `kaggle/run.py`, filed by `register_run.py` and compared with ConvNeXt U-Net by `run_stats.py --folds 0`.
  - **Both pass** (2026-11-27): the smoke test in 9 min, and all 73 unit tests of the repository (every file except `tests/test_data.py`, which needs the dataset).

## 8. Addendum (2026-11-27): result of the fold-0 development (RefFrac development analysis) and the freeze

Written on 2026-11-27, after the four development runs and before any final R run. sections 1–7 are unchanged.

- **Runs:** `R4_dev_base`, `R4_dev_base_s2`, `R4_dev_global` and `R4_dev_margin` (register lines 27–30), launched from `main` = `2619b39` in the groups `R_dev_round1` and `R_dev_round2` (83 and 87 min). Each trained fold 0 completely, at micro-batch 16 (12.6 GB of a T4's 15 GB).
- **Result** (`results/analysis/C12_paperB_dev_fold0/report.md`), on fold 0's 208 femurs:

  | Run | Dice, fractured femurs | Macro-F1 |
  |---|---|---|
  | ConvNeXt U-Net (fold 0) | .749 | .822 |
  | Base (mean of seeds 2026 and 2027) | **.770** | **.818** |
  | Whole-reference question reference | .778 | .796 |
  | Margin-loss question loss | .755 | .829 |

- **Decisions by the rule of section 4:**
  1. go / no-go: **passes** (Dice .770 ≥ .739; macro-F1 .818 ≥ .79);
  2. Whole-reference question: Dice +.009, below .01 and within the seed noise N = .009, and macro-F1 −.022: **the local window stays**;
  3. Margin-loss question: Dice −.014: **no margin loss** in the RefFrac-family runs (AASN-style model keeps it, as AASN's own loss);
  4. no combination run is needed;
  5. the base uses its references: with itself as the reference, its Dice falls by .090 and its P(fracture) on fractured femurs by .49 (means of the two seeds).
- **Reported, not used:** a random healthy reference instead of the 3 nearest changes the base's Dice by only .004–.005 at prediction time (a first hint for the reference question); the pointing game hits the fracture in 64% and 92% of fractured femurs (two seeds; chance 8%).
- **Freeze (Paper B milestone 2, section 3.3):** no setting of sections 1–2 changes. The eight final runs (`R4_refrac`, `R3_nearest_ref`, `R2_random_ref`, `R7_partner`, `R1_aasn`, `R5_concat`, `R6a_no_align`, `R6b_affine`) are written into `kaggle/runs.json` with exactly these settings. After the pull request of Step 18 is merged, the person creates the tag `B-v1-frozen` on `main`, and every final R run is launched from it. The training code at that commit is the code that trained the development runs (`2619b39`): the pull request changes results, analyses, documents and run entries only.
- **Compute, measured:** an RefFrac fold took 81 min on one T4 (81 s per epoch, against ConvNeXt U-Net's 45 s), 1.8 times ConvNeXt U-Net's fold rather than the 1.6 times of section 5; the pre-flight measured 1.40–1.55 s per micro-step (ConvNeXt U-Net: 0.91 s). So RefFrac, the concatenation model, the no-alignment model and the affine-alignment model take about 4.2 h each on T4 x2, and the AASN-style model, the random-reference model, the nearest-reference model and the own-hip model about 3 h each (not yet measured): **about 29 h** of session time, instead of the 26 h of section 5.
- **Order of the final runs (section 5):** RefFrac first; then the nearest-reference model and the random-reference model; the own-hip model; the AASN-style model; the concatenation model, the no-alignment model and the affine-alignment model.

## 9. Addendum (2026-11-27): the tag `B-v1-frozen` exists

- **Created** by the person on 2026-11-27 on `1c88912`, the merge of PR #14 (Step 18). Paper B milestone 2 is reached.
- **Its files are identical to `3c52ed7`** (Step 18). Its training code is identical to `2619b39`, which trained the development runs: `scripts/train_net.py`, `src/femurseg/`, `kaggle/run.py` and `kaggle/launcher.ipynb` have no difference between the two commits.
- **Every final R run** is launched with `BRANCH = "B-v1-frozen"`, and its log must show `1c88912`.

## 10. Addendum (2026-11-28): results of the final runs (RefFrac analyses)

All eight final runs (eight reference runs, register lines 31–38) were trained from `1c88912` on 5/5 folds, about 28.6 h of T4 x2 session time in total. The analyses were run as section 5 fixed them, with one tool added: `scripts/run_refstats.py` (Garden I-II McNemar, subgroups, reference tests and the own-hip model's modes), because `run_stats.py` has no test for Garden I-II recall. Reports: `results/analysis/C13_paperB_reference/`, `C14_paperB_partner/`, `C15_paperB_components/`, `C16_paperB_main_table/` (each `report.md`).

**Choices made at the analysis stage, before reading the affected numbers:**
- Garden I-II recall is compared with an exact McNemar test on the Garden I-II femurs (as accuracy elsewhere) and a cluster-bootstrap interval; Holm across the pairs of each analysis.
- "The smallest third of masks" is each part's own lower tertile of released mask area among fractured femurs (test masks are smaller, the YOLO resolution analysis).
- A swap test is read at the threshold of the mode it replaces; the own-hip model's blind and retrieved modes get their own OOF Youden thresholds (rule 4 of section 3).

**Results:**

| Question | Answer | Evidence |
|---|---|---|
| Reference question, Dice: does a healthy reference help? | **No** | every Dice difference ≤ .005, OOF and test, also in the subgroups (reference analysis) |
| Reference question, Garden I-II recall | **Trend, not significant** | OOF ConvNeXt U-Net .544 < the random-reference model .614 < the nearest-reference model .684 < RefFrac .702; RefFrac vs ConvNeXt U-Net +.158, 12 vs 3, p .035, Holm .14; test .312 = .312; folds 1–4 +.109 (reference analysis) |
| Reference question: does retrieval matter? | **No** | Nearest-reference model vs the random-reference model n.s.; a random reference at prediction time changes nothing (reference analysis) |
| Is the reference used? | **Yes, as "different from healthy"** | self test: RefFrac loses 201 of 663 fractures, Garden I-II .702 → .158; pointing game 4–9 × chance (reference analysis, section 4) |
| Own-hip question: own hip vs retrieved | **No difference** | Own-hip model blind vs retrieved Dice −.001 (Holm .10), vs the nearest-reference model −.000 (own-hip analysis) |
| Pairing-shortcut question: pairing shortcut | **Not learnt** | a fractured partner does not raise P(fracture) of healthy femurs (.0071 vs .0093); base = blind decisions (own-hip analysis) |
| Alignment-and-fusion question: alignment and fusion | **No effect on accuracy** | Dice and macro-F1 n.s.; Garden I-II RefFrac > the no-alignment model (8 vs 0, Holm .031), other variants Holm .06–.08; test none (RefFrac component analysis) |
| Main table | **RefFrac ties ConvNeXt U-Net** | OOF beats YOLOv8s-seg (+.081 macro-F1, +.037 Dice), EfficientNet-B4, ConvNeXt-T classifier, and the three segmenters on Dice; test: not better than ConvNeXt U-Net or YOLOv8s-seg (RefFrac main-table analysis) |

**Consequence for Paper B:** the pre-registered primary hypothesis (a reference improves Dice and Garden I-II recall) is not confirmed. The defensible paper is an analysis of reference-based comparison: it is used, its match does not matter, the patient's own hip is not better, and there is no pairing shortcut; the Garden I-II gain is an exploratory out-of-fold trend. How to frame Paper B is a decision for the person (as for Paper A).

## 11. Addendum (2026-11-28): the person's decision, one paper

- **Paper B is not written as a separate paper.** Its findings become part of Paper A, which reports both extensions of the shared core model ConvNeXt U-Net: SLG-Net (segmentation-guided hierarchy) and RefFrac (comparison with healthy references).
- **What Paper A takes from Paper B:** the R runs as pre-registered (protocol and tag `B-v1-frozen`), the main results of the reference analysis (no Dice gain; the Garden I-II trend as exploratory), the reference tests (the comparison is used, its match is not), the own-hip analysis (the own hip is no better; no pairing shortcut) and the RefFrac component analysis in short. The RefFrac main-table analysis's rows join Paper A's main table.
- **No new Paper B runs.** The sensitivity runs (R-sens) are dropped. Milestones Paper B milestone 4 and Paper B milestone 5 lapse; Paper B step 6 figures are made as part of Paper A's results package (Paper A milestone 4).
