# Report 09: One paper instead of two: what it means, what it claims, and what is left to do

**Date:** 2026-11-28 · **Decision:** the person's choice of 2026-11-28 (Analysis/08 section 11): Paper B is not written separately; its findings join Paper A.
**Inputs:** all analyses (`results/analysis/`), Analysis/06 sections 12–14, Analysis/08 sections 10–11.

---

## 1. Did Papers A and B "fail"?

Each paper had two kinds of hypotheses: one about **accuracy**, and one about **how the model decides**. They came out differently.

| Hypothesis | Paper A (SLG-Net) | Paper B (RefFrac) |
|---|---|---|
| **Accuracy:** more accurate than ConvNeXt U-Net, the model both extend | **Not confirmed.** Macro-F1 .806 vs .825 OOF (n.s.), Dice equal (SLG-Net ablation analysis) | **Not confirmed.** Dice equal; Garden I-II recall .702 vs .544 OOF, not significant after Holm, absent on the test set (reference analysis) |
| **Mechanism:** the model decides the way it was designed to | **Confirmed.** Its location and grade decisions depend on the fracture region, and significantly more than ConvNeXt U-Net's: 64% vs 48% of correct decisions lost when the region is removed, p < .001, test set agrees (SLG-Net interpretability analysis, section 2b, rule fixed before the run) | **Partly.** The model does use the comparison with a healthy femur (without it, RefFrac loses 201 of 663 fractures), but the *match* of the reference does not matter, the patient's own hip is no better, and there is no pairing shortcut (reference analysis, section 4, the own-hip analysis) |
| **Against the published model** (the base article's YOLOv8s-seg, our YOLOv8s-seg) | Better out of fold (+.061 macro-F1, +.034 Dice; the SLG-Net main-table analysis) | Better out of fold (+.081 macro-F1, +.037 Dice; the RefFrac main-table analysis) |

So the two ideas did not make the classifier more accurate. Both ideas changed how the model reaches its decision, and we can measure that. Two other results stand out:
- the shared core model ConvNeXt U-Net itself is strong: the best Dice of every baseline, and better than the published model out of fold;
- a retrained ConvNeXt U-Net (`A1_m0_deletion`) replicates ConvNeXt U-Net almost exactly (921 vs 922 correct), except for Garden I-II recall (.614 vs .544). That gap is 4 of 57 femurs and comes from the random seed alone. It matters for RefFrac: its Garden I-II gain over ConvNeXt U-Net (+.158) is only about twice the seed noise, so it must be reported as exploratory.

This is a common and publishable outcome when the experiments are pre-registered and honest. It does need a different story from "our new model is best".

## 2. What "one paper" means

**Before:** two papers, each claiming its own new model.
- Paper A: "SLG-Net improves classification by grounding it in the segmented region."
- Paper B: "RefFrac improves detection of subtle fractures by comparing with healthy femurs."

**Now:** one paper about **how to make a segment-and-classify model for hip fractures decide for the right reasons**, on the HUSJA-UA dataset. Its structure:
1. a reproducible benchmark (10 baselines, 5-fold out-of-fold evaluation, cluster bootstrap) and the core model ConvNeXt U-Net;
2. two ways to extend ConvNeXt U-Net, each with a pre-registered hypothesis:
   - **region grounding** (SLG-Net);
   - **comparison with healthy references** (RefFrac);
3. for each extension, what it does to accuracy (nothing significant) and to the decision mechanism (region grounding: measurably stronger dependence on the region; reference comparison: used, but the match does not matter);
4. what this means for the field: on this dataset, gains in accuracy are limited by the labels (train/test annotation shift, the YOLO resolution analysis; grade errors), and interpretability must be measured, not assumed.

**What changes in practice:**
- **One manuscript, one submission.** There is no Paper B manuscript, cover letter or second journal.
- **No more training for Paper B.** All eight R runs are used as they are. The R sensitivity runs are dropped.
- **Paper A's protocol (Analysis/06) and Paper B's protocol (Analysis/08) both stay as written.** The paper cites both as pre-registered (they are dated in the repository before each result), so every claim keeps its pre-registration.
- **The paper's main table** combines the SLG-Net main-table analysis and the RefFrac main-table analysis: the 9 baselines, ConvNeXt U-Net, SLG-Net and RefFrac on the same femurs.
- **Length:** a combined paper is longer. The RefFrac ablation (RefFrac component analysis) and the own-hip model study (own-hip analysis) move largely to the supplement.
- **Why this can be stronger:** two independent extensions that both fail to beat ConvNeXt U-Net make a convincing case that ConvNeXt U-Net's level is a ceiling for this dataset. Two different, measured mechanisms are more interesting than one.

## 3. Working title and claims

**Working title:** *Grounding and comparing: two pre-registered extensions of a segment-and-classify model for hip fracture classification on radiographs*

**Claims, each with its evidence** (every number is OOF, n = 1,043, unless marked "test"):

| # | Claim | Evidence |
|---|---|---|
| 1 | We give the first independent benchmark on HUSJA-UA: 10 baselines, 5-fold OOF, x-ray-level bootstrap. We reproduce the published accuracy but not its Dice, and we describe a train/test annotation shift | Baseline analyses, the baseline table |
| 2 | ConvNeXt U-Net (ConvNeXt-T U-Net with a classification head) beats the published model out of fold and has the best Dice | Segmenter analysis, the mask-task analysis, the SLG-Net main-table analysis |
| 3 | Neither region grounding nor reference comparison improves accuracy over ConvNeXt U-Net | SLG-Net ablation analysis, the reference analysis, the RefFrac main-table analysis |
| 4 | Region grounding makes the location and grade decisions depend on the fracture region, more than in ConvNeXt U-Net (64% vs 48%, p < .001; test 66% vs 53%) | SLG-Net interpretability analysis, section 2b |
| 5 | The fracture/no-fracture decision of SLG-Net does not depend on the region (P(fracture) −.03), while ConvNeXt U-Net's does (−.20) | SLG-Net interpretability analysis, sections 2, 2b |
| 6 | A reference-comparison model uses the comparison (self test: RefFrac loses 201 of 663 fractures), but any healthy femur works: retrieval, a matched femur or the patient's own hip give no gain | Reference analysis, section 4, the own-hip analysis |
| 7 | No pairing shortcut is learnt, even when the training policy invites one | Own-hip analysis |
| 8 | Garden I-II recall rises with references out of fold, but the gain is within seed noise and absent on the test set (exploratory) | Reference analysis, section 2, `C08/m0_replicate` |

**What the paper does not claim:** better accuracy; clinical usefulness (there is no clinician and no reader study); that dependence on the region means the decision is "correct".

## 4. Outline (with the source of each part)

1. **Introduction:** hip fracture classification; the base article (González et al. 2025); interpretability claims usually untested; our two hypotheses.
2. **Data:** HUSJA-UA, the missing validation set, our fixed 5-fold split, the annotation shift (Analysis/01, the YOLO resolution analysis).
3. **Methods**
   - 3.1 Evaluation protocol: OOF, thresholds from OOF only, cluster bootstrap, paired tests, Holm, pre-registration (Analysis/04 sections 3, 05, 06, 08).
   - 3.2 Baselines and ConvNeXt U-Net (Analysis/05).
   - 3.3 SLG-Net (Analysis/06 sections 1–2).
   - 3.4 RefFrac and the reference bank (Analysis/07, 08 sections 1–2).
   - 3.5 Mechanism tests: oracle, region deletion with control, geometry consistency, risk-coverage, the self and swap tests, the pointing game, the own-hip model's modes.
4. **Results**
   - 4.1 Benchmark and ConvNeXt U-Net (baseline table, the segmenter analysis/C05).
   - 4.2 Main table: all models (SLG-Net main-table analysis + RefFrac main-table analysis).
   - 4.3 Ablations: SLG chain (SLG-Net ablation analysis), RefFrac chain (reference analysis, the RefFrac component analysis in short).
   - 4.4 Region grounding: deletion vs ConvNeXt U-Net, geometry (SLG-Net interpretability analysis).
   - 4.5 Reference comparison: self/random tests, own hip, shortcut (reference analysis, section 4, the own-hip analysis).
5. **Discussion:** why accuracy does not move (label noise, annotation shift, 57 Garden I-II femurs); what the mechanism results mean; limitations (one dataset, no clinician, one backbone, seed noise in small classes).
6. **Conclusion.**
- **Supplement:** per-class tables (breakdowns), the RefFrac component analysis in full, the own-hip analysis in full, calibration, folds 1–4 checks, risk-coverage.

## 5. Tables and figures (planned)

| Item | Content | Source |
|---|---|---|
| Table 1 | Dataset and split | Analysis/01, folds_v1.csv |
| Table 2 | Main table: 8 baselines, ConvNeXt U-Net, SLG-Net, RefFrac (OOF and test; macro-F1, Dice, Garden I-II, correct) | SLG-Net main-table analysis, the RefFrac main-table analysis |
| Table 3 | Paired comparisons with ConvNeXt U-Net and with the published model | SLG-Net ablation analysis, the SLG-Net main-table analysis, the reference analysis, the RefFrac main-table analysis |
| Table 4 | Mechanism tests: deletion (SLG-Net vs ConvNeXt U-Net), self and random tests (RefFrac), the own-hip model modes | SLG-Net interpretability analysis, the reference analysis, the own-hip analysis |
| Fig. 1 | Overview: ConvNeXt U-Net, SLG-Net and RefFrac | new drawing |
| Fig. 2 | Deletion: share of correct decisions lost, per class, SLG-Net vs ConvNeXt U-Net vs controls | SLG-Net interpretability analysis |
| Fig. 3 | Reference tests: Dice and P(fracture) under base, self, random, fractured reference | Reference analysis, the own-hip analysis |
| Fig. 4 | Example radiographs: predicted region, deletion, difference map | run outputs (needs the Kaggle maps) |
| Supplementary Figs. 1–3 | Risk-coverage, geometry, per-class confusion | SLG-Net interpretability analysis, breakdowns |

## 6. What is left, in order

1. **Optional: a second backbone** (the sensitivity runs of Analysis/06 section 7: EfficientNetV2-S, about 8 h). Reviewers often ask "does it hold for another backbone?". The deletion result (claim 4) is the one that most needs it. **Recommendation:** run ConvNeXt U-Net and SLG-Net with the second backbone and the deletion test (about 6–8 h on T4 x2), so claim 4 is shown twice. It is the person's choice.
2. **Figures:** Fig. 2 and Fig. 3 from the filed results (no compute). Fig. 4 needs maps from the Kaggle work folders of SLG-Net, ConvNeXt U-Net and RefFrac (`A6_slgnet.zip` is on the PC; the other two would need their full Kaggle outputs).
3. **Manuscript draft** in `manuscript/` (draft, tables, claim checklist), section by section, starting with Methods and Results, which are fixed by the reports.
4. **Journal:** a technical/informatics journal (Analysis/04 section 2). The combined paper suits J Imaging Informatics in Medicine, Computers in Biology and Medicine or BSPC.
5. **Timeline:** the Paper A milestone 5 target (21 Feb 2027) stands. With no Paper B, about 4 weeks are freed for the combined manuscript.
