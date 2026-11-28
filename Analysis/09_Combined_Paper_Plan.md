# Report 09: One paper instead of two: what it means, what it claims, and what is left to do

**Date:** 2026-11-28 · **Decision:** the person's choice of 2026-11-28 (Analysis/08 §11): Paper B is not written separately; its findings join Paper A.
**Inputs:** C01–C16 (`results/analysis/`), Analysis/06 §12–§14, Analysis/08 §10–§11.

---

## 1. Did Papers A and B "fail"?

Each paper had two kinds of hypotheses: one about **accuracy**, and one about **how the model decides**. They came out differently.

| Hypothesis | Paper A (SLG-Net) | Paper B (RefFrac) |
|---|---|---|
| **Accuracy:** more accurate than M0, the model both extend | **Not confirmed.** Macro-F1 .806 vs .825 OOF (n.s.), Dice equal (C07) | **Not confirmed.** Dice equal; Garden I-II recall .702 vs .544 OOF, not significant after Holm, absent on the test set (C13) |
| **Mechanism:** the model decides the way it was designed to | **Confirmed.** Its location and grade decisions depend on the fracture region, and significantly more than M0's: 64% vs 48% of correct decisions lost when the region is removed, p < .001, test set agrees (C08 §2b, rule fixed before the run) | **Partly.** The model does use the comparison with a healthy femur (without it, RefFrac loses 201 of 663 fractures), but the *match* of the reference does not matter, the patient's own hip is no better, and there is no pairing shortcut (C13 §4, C14) |
| **Against the published model** (the base article's YOLOv8s-seg, our S01) | Better out of fold (+.061 macro-F1, +.034 Dice; C09) | Better out of fold (+.081 macro-F1, +.037 Dice; C16) |

So the two ideas did not make the classifier more accurate. Both ideas changed how the model reaches its decision, and we can measure that. Two other results stand out:
- the shared core model M0 itself is strong: the best Dice of every baseline, and better than the published model out of fold;
- a retrained M0 (`A1_m0_deletion`) replicates M0 almost exactly (921 vs 922 correct), except for Garden I-II recall (.614 vs .544). That gap is 4 of 57 femurs and comes from the random seed alone. It matters for RefFrac: its Garden I-II gain over M0 (+.158) is only about twice the seed noise, so it must be reported as exploratory.

This is a common and publishable outcome when the experiments are pre-registered and honest. It does need a different story from "our new model is best".

## 2. What "one paper" means

**Before:** two papers, each claiming its own new model.
- Paper A: "SLG-Net improves classification by grounding it in the segmented region."
- Paper B: "RefFrac improves detection of subtle fractures by comparing with healthy femurs."

**Now:** one paper about **how to make a segment-and-classify model for hip fractures decide for the right reasons**, on the HUSJA-UA dataset. Its structure:
1. a reproducible benchmark (10 baselines, 5-fold out-of-fold evaluation, cluster bootstrap) and the core model M0;
2. two ways to extend M0, each with a pre-registered hypothesis:
   - **region grounding** (SLG-Net);
   - **comparison with healthy references** (RefFrac);
3. for each extension, what it does to accuracy (nothing significant) and to the decision mechanism (region grounding: measurably stronger dependence on the region; reference comparison: used, but the match does not matter);
4. what this means for the field: on this dataset, gains in accuracy are limited by the labels (train/test annotation shift, C01; grade errors), and interpretability must be measured, not assumed.

**What changes in practice:**
- **One manuscript, one submission.** There is no Paper B manuscript, cover letter or second journal.
- **No more training for Paper B.** All eight R runs are used as they are. The R sensitivity runs are dropped.
- **Paper A's protocol (Analysis/06) and Paper B's protocol (Analysis/08) both stay as written.** The paper cites both as pre-registered (they are dated in the repository before each result), so every claim keeps its pre-registration.
- **The paper's main table** combines C09 and C16: the 9 baselines, M0, SLG-Net and RefFrac on the same femurs.
- **Length:** a combined paper is longer. The RefFrac ablation (C15) and the R7 study (C14) move largely to the supplement.
- **Why this can be stronger:** two independent extensions that both fail to beat M0 make a convincing case that M0's level is a ceiling for this dataset. Two different, measured mechanisms are more interesting than one.

## 3. Working title and claims

**Working title:** *Grounding and comparing: two pre-registered extensions of a segment-and-classify model for hip fracture classification on radiographs*

**Claims, each with its evidence** (every number is OOF, n = 1,043, unless marked "test"):

| # | Claim | Evidence |
|---|---|---|
| 1 | We give the first independent benchmark on HUSJA-UA: 10 baselines, 5-fold OOF, x-ray-level bootstrap. We reproduce the published accuracy but not its Dice, and we describe a train/test annotation shift | C01–C05, M1 table |
| 2 | M0 (ConvNeXt-T U-Net with a classification head) beats the published model out of fold and has the best Dice | C04, C05, C09 |
| 3 | Neither region grounding nor reference comparison improves accuracy over M0 | C07, C13, C16 |
| 4 | Region grounding makes the location and grade decisions depend on the fracture region, more than in M0 (64% vs 48%, p < .001; test 66% vs 53%) | C08 §2b |
| 5 | The fracture/no-fracture decision of SLG-Net does not depend on the region (P(fracture) −.03), while M0's does (−.20) | C08 §2, §2b |
| 6 | A reference-comparison model uses the comparison (self test: R4 loses 201 of 663 fractures), but any healthy femur works: retrieval, a matched femur or the patient's own hip give no gain | C13 §4, C14 |
| 7 | No pairing shortcut is learnt, even when the training policy invites one | C14 |
| 8 | Garden I-II recall rises with references out of fold, but the gain is within seed noise and absent on the test set (exploratory) | C13 §2, `C08/m0_replicate` |

**What the paper does not claim:** better accuracy; clinical usefulness (there is no clinician and no reader study); that dependence on the region means the decision is "correct".

## 4. Outline (with the source of each part)

1. **Introduction:** hip fracture classification; the base article (González et al. 2025); interpretability claims usually untested; our two hypotheses.
2. **Data:** HUSJA-UA, the missing validation set, our fixed 5-fold split, the annotation shift (Analysis/01, C01).
3. **Methods**
   - 3.1 Evaluation protocol: OOF, thresholds from OOF only, cluster bootstrap, paired tests, Holm, pre-registration (Analysis/04 §3, 05, 06, 08).
   - 3.2 Baselines and M0 (Analysis/05).
   - 3.3 SLG-Net (Analysis/06 §1–2).
   - 3.4 RefFrac and the reference bank (Analysis/07, 08 §1–2).
   - 3.5 Mechanism tests: oracle, region deletion with control, geometry consistency, risk-coverage, the self and swap tests, the pointing game, R7's modes.
4. **Results**
   - 4.1 Benchmark and M0 (M1 table, C04/C05).
   - 4.2 Main table: all models (C09 + C16).
   - 4.3 Ablations: SLG chain (C07), RefFrac chain (C13, C15 in short).
   - 4.4 Region grounding: deletion vs M0, geometry (C08).
   - 4.5 Reference comparison: self/random tests, own hip, shortcut (C13 §4, C14).
5. **Discussion:** why accuracy does not move (label noise, annotation shift, 57 Garden I-II femurs); what the mechanism results mean; limitations (one dataset, no clinician, one backbone, seed noise in small classes).
6. **Conclusion.**
- **Supplement:** per-class tables (breakdowns), C15 in full, C14 in full, calibration, folds 1–4 checks, risk-coverage.

## 5. Tables and figures (planned)

| Item | Content | Source |
|---|---|---|
| Table 1 | Dataset and split | Analysis/01, folds_v1.csv |
| Table 2 | Main table: 8 baselines, M0, SLG-Net, RefFrac (OOF and test; macro-F1, Dice, Garden I-II, correct) | C09, C16 |
| Table 3 | Paired comparisons with M0 and with the published model | C07, C09, C13, C16 |
| Table 4 | Mechanism tests: deletion (SLG-Net vs M0), self and random tests (RefFrac), R7 modes | C08, C13, C14 |
| Fig. 1 | Overview: M0, SLG-Net and RefFrac | new drawing |
| Fig. 2 | Deletion: share of correct decisions lost, per class, SLG-Net vs M0 vs controls | C08 |
| Fig. 3 | Reference tests: Dice and P(fracture) under base, self, random, fractured reference | C13, C14 |
| Fig. 4 | Example radiographs: predicted region, deletion, difference map | run outputs (needs the Kaggle maps) |
| Fig. S1–S3 | Risk-coverage, geometry, per-class confusion | C08, breakdowns |

## 6. What is left, in order

1. **Optional: a second backbone** (the sensitivity runs of Analysis/06 §7: EfficientNetV2-S, about 8 h). Reviewers often ask "does it hold for another backbone?". The deletion result (claim 4) is the one that most needs it. **Recommendation:** run M0 and SLG-Net with the second backbone and the deletion test (about 6–8 h on T4 x2), so claim 4 is shown twice. It is the person's choice.
2. **Figures:** Fig. 2 and Fig. 3 from the filed results (no compute). Fig. 4 needs maps from the Kaggle work folders of A6, M0 and R4 (`A6_slgnet.zip` is on the PC; the other two would need their full Kaggle outputs).
3. **Manuscript draft** in `manuscript/` (draft, tables, claim checklist), section by section, starting with Methods and Results, which are fixed by the reports.
4. **Journal:** a technical/informatics journal (Analysis/04 §2). The combined paper suits J Imaging Informatics in Medicine, Computers in Biology and Medicine or BSPC.
5. **Timeline:** the M5-A target (21 Feb 2027) stands. With no Paper B, about 4 weeks are freed for the combined manuscript.
