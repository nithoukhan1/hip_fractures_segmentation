# Report 07: Step 4 protocol, Paper B (RefFrac) step 1 of Paper B: the pair index and the bank of healthy reference femurs

**Date:** 2026-11-27 · **Code:** `src/femurseg/refbank.py`, `scripts/build_reference_bank.py` · **Run:** `R00_reference_bank` (version 1), `R00_reference_bank_r2` (version 2 of the orientation rule, section 8) in `kaggle/runs.json`
**Why this report:** RefFrac compares every femur with healthy reference femurs. This report fixes, before any feature is computed, which femurs may serve as references for which femur, how they are chosen, and which checks the bank must pass before step 2 of Paper B (the Siamese model) starts. The design comes from Report 03 section 4 and Report 04 section 5.

---

## 1. What Paper B step 1 delivers

No model is trained. One short Kaggle run (`R00_reference_bank`, about 10 minutes on one T4) writes:

1. **The pair index** (`pairs.csv`): one row per femur with its partner (the other femur of the same x-ray), the pair type, and whether the partner may serve as a reference.
2. **The reference table** (`references.csv`): for each of the 5 fold models and each of the 1,253 femurs:
   - the 5 most similar healthy femurs of that model's bank, each with its orientation and similarity;
   - one random healthy femur;
   - one random fractured femur.
3. **The checks** of section 4 (`sanity.json`, `summary.md`) and a figure of nearest neighbours (`nn_fold0.png`).

Every R run of Paper B reads the same two tables, so the runs differ only in what the protocol says they differ in.

## 2. The data (counted from `splits/folds_v1.csv`, checked by `tests/test_refbank.py`)

| | Train (5 folds) | Test |
|---|---|---|
| Femurs | 1,043 | 210 |
| Healthy / fractured | 363 / 680 | 65 / 145 |
| Fractured femur, healthy partner | 334 (66–68 per fold) | 64 (44% of the fractures) |
| Healthy femur, fractured partner | 334 | 64 (all but 1 of the 65 healthy test femurs) |
| Healthy femur, healthy partner | 14 (7 x-rays) | 0 |
| Femur alone in its x-ray | 361 | 82 |

- No x-ray holds two fractured femurs.
- The one duplicate image of the dataset (`00399_00262` is a copy of `00397_00261`, both fractured) has its own x-ray ID. The fold file therefore puts it in the same **group** as x-ray 00261, so every rule below uses the group, not the x-ray ID.

## 3. The rules

### 3.1 The bank of each fold model
- **Fold model k** (trained on the 4 folds other than k) has one bank: **the healthy femurs of those 4 folds**, 290 or 291 femurs.
- The same bank serves every femur that model sees:
  - its training femurs;
  - its held-out fold (out-of-fold predictions);
  - the test set.
- Held-out and test femurs are never in the bank, so they are always compared with other patients.
- **References are healthy.** A fractured femur is a reference only in swap test (b) of run the own-hip model.

### 3.2 Exclusion
- A femur never gets a reference from **its own group**: its x-ray, its partner, and for x-ray 00261 the duplicate x-ray 00262.
- This matters for training femurs only. Without it, a model would "retrieve" the patient's own other hip during training but never at test time.

### 3.3 Similarity and orientation
1. **Features:**
   - the ImageNet weights of ConvNeXt U-Net's backbone (`convnext_tiny.fb_in22k_ft_in1k`), frozen, never trained here;
   - the input exactly as ConvNeXt U-Net sees it: 768 × 512, grey copied to 3 channels, ImageNet normalisation;
   - the network's own global average pooling and final normalisation, giving 768 numbers per image.
2. **Centring:** before the cosine, the mean feature of the fold model's training femurs (both orientations) is subtracted.
   - All femur crops look alike, so the raw features share a large common part that pushes every cosine towards 1.
   - The mean uses images only, no labels, and never the held-out fold or the test set.
   - The pair index belongs to no fold model, so it uses the mean of all 1,043 training femurs.
3. **Orientation (version 1; replaced by version 2 in section 8 after check 1 failed):** every femur is encoded as released and mirrored left–right.
   - The similarity of a femur T and a bank femur R is the larger of cos(T, R) and cos(T, mirrored R).
   - R is then used in that orientation.
   - Each bank femur counts once. Ties go to the earlier file of the fold file.

### 3.4 What is kept per femur and fold model
| Column | What | Used by |
|---|---|---|
| `ref1`–`ref5` | the 5 most similar bank femurs | Nearest-reference model (k = 1), RefFrac (k = 3), a sensitivity check of k |
| `random_healthy` | one bank femur outside the femur's group, drawn with a seed made from the femur's file name and the fold model (md5) | Random-reference model (any healthy reference) and swap test (a) |
| `random_fractured` | the same draw over the fractured femurs of the 4 training folds | swap test (b) only |

- Each column comes with its orientation (`_mirrored`) and similarity (`_cos`).
- The random references use the same orientation rule. The random-reference model then differs from RefFrac only in how well the reference matches, not in how it is oriented.

### 3.5 The partner (run the own-hip model only)
- The partner is the other femur of the same x-ray. It is the other hip, so it is always mirrored.
- Only a healthy partner is used as a reference. Without one, the own-hip model falls back to the retrieved reference (Report 04 section 5.3).
- `pairs.csv` also stores the cosine of each femur with its partner, mirrored and as released: check 1 below.

### 3.6 During training (from Paper B step 2 on)
- Training flips the target left–right at random (Step 2 recipe). Its references are then flipped too, so their orientation relative to the target stays as chosen here.
- The table is fixed: no R run retrieves anything itself.

## 4. Checks before Paper B step 2 starts

| # | Check | What is computed | Decision |
|---|---|---|---|
| 1 | **Mirror check** | For every femur with a partner (810 femurs), is the mirrored partner closer than the partner as released? | ≥ 80%: the features see left and right, and the orientation rule is used as written. 60–80%: used, and stated as a limitation. Below 60%: the orientation rule is reviewed before Paper B step 2 |
| 2 | **Retrieval beats chance** | Mean similarity of the 3 nearest vs the random healthy femur, held-out femurs of each fold model | Reported |
| 3 | **No duplicates** | A reference with the same pixels (`pixel_md5`) as its femur stops the run; references with cosine ≥ 0.995 are listed | Expect none. Any listed pair is looked at before Paper B step 2 |
| 4 | **Held-out vs test** | Similarity of the nearest reference: median and quartiles, held-out vs test femurs | Reported. A clearly lower test median means test femurs are harder to match, which is stated as a limitation |
| 5 | **Partner rank** (positive control) | For held-out and test femurs with a healthy partner: the rank their own partner would have in the bank | Reported. It shows how close the retrieved femurs come to the patient's own other hip, which is the comparison RefFrac vs the own-hip model makes |
| 6 | **Who is in the bank** | Pair types of each bank's femurs | Reported: 334 of the 363 healthy training femurs are partners of fractured femurs, so the bank mirrors the dataset |
| 7 | **Figure** | Fold model 0: one held-out femur per class (seeded) with its 3 nearest references | Looked at: similar framing and anatomy, orientation right |

## 5. Run and outputs
- **Run:** `R00_reference_bank` (step 4): `scripts/build_reference_bank.py`, about 10 minutes, one T4.
  - About 2 minutes to build the image cache.
  - Under 1 minute of features (1,253 femurs × 2 orientations).
  - The checks.
- **Filed in `results/R00_reference_bank/`:**
  - `pairs.csv` (1,253 rows);
  - `references.csv` (6,265 rows: 5 fold models × 1,253 femurs);
  - `sanity.json`, `summary.md`, `nn_fold0.png`, `config.json`.
- **Not filed:** `work/R00_reference_bank/features.npz`, the features themselves (about 4 MB). They are cheap to recompute.
- **Tests:**
  - `tests/test_refbank.py`: real fold counts, banks, exclusion, orientation, centring, seeded draws, checks;
  - `tests/smoke_reference_bank.py`: the whole run on the synthetic dataset through `kaggle/run.py`, on the CPU.

## 6. Limitations, stated now
- **Generic features:** ImageNet features describe framing, pose and contrast more than fine anatomy. The figure and check 5 show how far "similar" goes.
- **A small bank:** about 290 femurs. For an unusual femur, even the nearest one may differ.
- **Who the healthy femurs are:** most healthy femurs in the bank are partners of fractured femurs. Their x-rays were taken because of a fracture of the other hip.
- **No test femur is ever a reference.** Test femurs are matched against training femurs only, like any new patient.

## 7. Next: Paper B step 2
The Siamese model: ConvNeXt U-Net's encoder shared by the target and its references, cross-attention alignment at 1/16 and 1/32, and fusion [T, T − R, |T − R|] into ConvNeXt U-Net's decoder and head (Report 04 section 5.3). It gets its own protocol (`Analysis/08_…`) before any run, then fold-0 development, as for Paper A.

---

## 8. Addendum (2026-11-27): check 1 failed, so the orientation rule was reviewed (version 2)

**What happened.** `R00_reference_bank` ran on Kaggle (commit `dc152a9`, 2.46 min, register line 17). Check 1 failed: for only 473 of the 810 femurs with a partner (58.4%) was the mirrored partner closer than the partner as released. This is below the 60% line of section 4, so the rule required a review of the orientation rule before Paper B step 2. The figure agreed: in `nn_fold0.png`, some references are shown with the pelvis on the other side from the target's.

**Why.** ImageNet networks are trained with random left–right flips. Their pooled features therefore change very little when an image is mirrored.
- For the same reason, the similarity ranking is not affected. Retrieval still clearly beats chance (check 2: the 3 nearest have a cosine .539 higher than a random healthy femur), and a femur's own partner would rank 6th (median) among about 290 bank femurs (check 5).
- But "the better orientation" (section 3.3, point 3) was close to a coin toss.

**The review** (`scripts/check_orientation.py`, run on the released images of all 1,253 femurs; no label used; outputs and report in `results/analysis/C10_paperB_orientation_review/`):

| Test | Result |
|---|---|
| The partner correlates better mirrored, with 96 × 64 pixel thumbnails instead of ImageNet features | 796 of 810 femurs (98.3%) |
| **Pelvis side** (the brighter half of the upper image): the two femurs of an x-ray get opposite sides | **402 of 405 x-rays (99.3%)** |
| Variants, reported only: whole image / upper third | 383 (94.6%) / 404 (99.8%) |
| 20,000 random pairs of femurs from different x-ray groups: "mirror when the sides differ" agrees with the orientation the thumbnails prefer | 19,832 (99.2%) |
| Pelvis on the left: all / training / test | 49.7% / 49.6% / 50.5% |

- The pelvis side was the first rule tried, and it is kept as written. The upper-third variant does 2 x-rays better, but it was not chosen, so that no choice is fitted to these 405 x-rays.
- The 3 x-rays whose two femurs got the same side are 00022, 00273 and 00298. 13 femurs have an ambiguous side (the two upper halves differ by less than 5% in brightness).

**Version 2 of the rule (replaces section 3.3, point 3).**
1. Every femur's pelvis side is read from its own image: the half (left or right) of the upper image that is brighter on average. The pelvis is bone; the other side is the soft tissue of the thigh. The function is `refbank.pelvis_side`, applied to the 768 × 512 cached image.
2. A reference is used mirrored when its pelvis side differs from the target's, and as released otherwise. Target and reference then always show the same side.
3. The similarity is the centred cosine of the ImageNet features in that orientation. The rules for the bank, the exclusion, the centring, k and the seeds are unchanged.
4. The random references (random-reference model, swap tests a and b) follow the same rule.
5. The partner of the own-hip model stays "always mirrored": it is the other hip.

**Check 1, version 2.** Check 1 is now the share of femurs with a partner whose partner the rule mirrors. Its decision lines are unchanged (≥ 80%: use as written). The review predicts 804 of 810 (99.3%). The version-1 share (58.4%) stays in `sanity.json` and `summary.md` as `features_share`, reported only. Check 5 now uses the partner mirrored, as the own-hip model does.

**Run.** `R00_reference_bank_r2` (about 3 min, one T4): the same script with the version-2 rule.
- Every R run will read its `pairs.csv` and `references.csv`.
- `R00_reference_bank` stays filed as the record of version 1.
- Paper B step 2 starts once `R00_reference_bank_r2` passes check 1 and its figure shows every reference with the pelvis on the target's side.

**Limitation, stated now.** About 0.5% of femurs may get the wrong side: 3 of 405 x-rays failed, and 13 femurs are ambiguous. Their references are then mirrored the wrong way. The Siamese model's cross-attention alignment (Paper B step 2) must cope with such a reference; this is a small share of femurs.

## 9. Addendum (2026-11-27): result of version 2 (`R00_reference_bank_r2`); step 1 of Paper B is complete

`R00_reference_bank_r2` ran on Kaggle (commit `d3d73af`, 2.65 min, one T4; register line 19; `results/R00_reference_bank_r2/`). It uses the same script, bank, features, k and seeds as version 1, with the orientation rule of section 8.

| Check (section 4) | Version 1 (`R00_reference_bank`) | Version 2 (`R00_reference_bank_r2`) | Rule |
|---|---|---|---|
| 1. Mirror check | 473 of 810 (58.4%), by the features | **804 of 810 (99.3%)**, by the pelvis-side rule (the features alone: still 473, reported only) | ≥ 80%: **use as written** |
| 2. Retrieval vs chance (3 nearest minus random healthy) | +.539 cosine | +.547 cosine | clearly above chance |
| 3. Duplicates | none | none (0 references with cosine ≥ .995) | none |
| 5. Own partner's rank in the bank (median) | 6 (as released or mirrored, whichever is closer) | 6 (always mirrored); rank 1 for 25.1%, top 5 for 47.5% | reported |
| Figure `nn_fold0.png` | some references with the pelvis on the other side | the pelvis is on the target's side in all 18 references (checked by eye; one dark, grid-patterned reference is hard to read) | every reference on the target's side |

- The 804 matches the review's prediction exactly (section 8).
- **How much the references changed** (6,265 rows: 5 fold models × 1,253 femurs, computed from the two `references.csv` files):
  - the nearest reference is the same femur in 87.1% of rows, and the same femur in the same orientation in 57.6%;
  - the 5 references overlap by 89.5%, and the whole set of 5 is the same in 54.4%;
  - 48.3% of nearest references are mirrored (49.4% in version 1), and the mean cosine of the nearest is .585 (.592).
- So the rule mostly changed the **orientation** in which a reference is shown, not **which** femurs are retrieved. This is expected, because the ImageNet features hardly see left and right (section 8).

**Decision.**
- Check 1 passes and the figure is as required, so **step 1 of Paper B is complete.**
- Every R run reads `results/R00_reference_bank_r2/pairs.csv` and `references.csv`. `R00_reference_bank` stays filed as the record of version 1.
- **Next: Paper B step 2**, the protocol of RefFrac's model and the eight reference runs (Report 04): a Siamese ConvNeXt-T that encodes the femur and its references with shared weights, cross-attention at 1/16 and 1/32, fusion [T, T − R, |T − R|]. It is written before any R run, and fold-0 development follows, as for Paper A.
