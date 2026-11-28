# Report 10: The combined paper: its idea, its novelty, whether it can be defended, and what to do next

**Date:** 2026-11-28 · **Builds on:** Analysis/09 (one paper), C01–C17 · **Question from the person:** "Paper A and B didn't beat our own baseline, so we haven't succeeded yet. What is the combined idea, the novelty and the contribution, how do we defend it, and what next?"

---

## 1. Where we really stand (the numbers that matter)

| | OOF (1,043 femurs) macro-F1 / Dice / correct | Test (210) macro-F1 / Dice / correct | Status |
|---|---|---|---|
| Published method (YOLOv8s-seg, our reproduction S01) | .744 / .740 / 868 | **.845** / .725 / 185 | Reproduces the article's accuracy (C01) |
| **M0** (ours: ConvNeXt-T U-Net + classification head) | **.825 / .777 / 922** | .779 / **.744** / 182 | Beats S01 out of fold on every endpoint (p < .002) |
| SLG-Net (A6) | .806 / .774 / 907 | .820 / .737 / **186** | = M0 (n.s.); **more region-grounded than M0** (C08 §2b) |
| RefFrac (R4) | .826 / .777 / 920 | .779 / .739 / 180 | = M0 (n.s.); uses the comparison, not the match (C13, C14) |
| Ensemble M0 + SLG-Net + RefFrac (C17, exploratory) | .842 / .777 / 929 | .819 / .744 / 187 | n.s. against M0 and against two M0s |

Four facts shape every option below:
1. **Our own model M0 is already the strongest model on this dataset out of fold**, by a clear and significant margin over the published method. "We did not beat our baseline" is true, but our baseline is our own contribution, and it beats the state of the art.
2. **The test set cannot separate good models.** With 210 femurs and the observed disagreement between models, a paired test has 80% power only for differences of about **4.2–4.6 accuracy points (9–10 femurs)**. Out of fold the threshold is about **2.1 points (22 femurs)**. Every difference between M0, SLG-Net, RefFrac and their ensembles is smaller than that. This is also why S01 "wins" test macro-F1 (.845) while losing clearly out of fold: test macro-F1 swings with a handful of femurs in the small classes (16 Garden I-II, 9 subtrochanteric).
3. **The remaining errors are mostly grade errors** (M0 OOF: 65 grade, 34 location, 22 detection), concentrated where the labels are least reliable: Garden I-II vs III-IV, and Evans I-II vs III-V. The released masks also show a train/test annotation shift (test neck masks are 35–40% smaller, C01), and a duplicated image carries two different masks (Dice .82). Part of the accuracy ceiling is in the labels, not the models.
4. **Seed noise is large in the rarest class:** retraining M0 identically moves Garden I-II recall from .544 to .614 (4 of 57 femurs; C08 `m0_replicate`).

## 2. The combined paper: idea, novelty, contributions

### 2.1 The proposed idea, in one sentence
**A segmentation-first framework for classifying hip fractures on radiographs: one model that delineates the fracture and classifies its type (M0), and two ways to make its decisions use the right evidence. These are region-grounded hierarchical decisions (SLG-Net) and comparison with healthy reference femurs (RefFrac). Each is tested with pre-registered experiments that measure whether the decisions really depend on that evidence.**

**Working title:** *Segment, ground, compare: a pre-registered evaluation of evidence-grounded deep learning for hip fracture classification and segmentation on radiographs*

### 2.2 How a reviewer would read the novelty

| # | Novelty | Why it is new | Evidence |
|---|---|---|---|
| N1 | **First independent study on HUSJA-UA**, with a reproducible benchmark of 10 models, 5-fold out-of-fold evaluation and x-ray-level statistics | No one else has published on this dataset (checked 2026-11-26; re-check before submission). The original study used a single split without paired tests | C01–C05, M1 |
| N2 | **A segmentation-first multi-task model (M0)** that beats the published YOLO pipeline out of fold: macro-F1 +.081, Dice +.037, accuracy +5 points, all p < .002 | The published method classifies by detection; ours segments and classifies jointly | C04, C05, C16 |
| N3 | **Region-grounded hierarchical classification (SLG-Net) whose location and grade decisions demonstrably depend on the delineated fracture**, more than the same model without grounding: 64% vs 48% of correct decisions lost when the region is removed (p < .001; test 66% vs 53%) | Most "interpretable" fracture models show heat maps without testing them. A deletion test against a matched baseline, with its rule fixed before the run, is rare in this field | C08 §2, §2b |
| N4 | **The first test of reference-based ("compare with a healthy femur") classification against its own shortcut and confounds:** the model uses the comparison (without it, 201 of 663 fractures are lost), but any healthy femur works. A matched femur and the patient's own other hip give no gain, and no "same x-ray means fractured" shortcut is learnt | Contralateral/reference methods (e.g. AASN) report gains but do not separate "any reference" from "matched reference", or test the pairing shortcut | C13 §4, C14, C15 |
| N5 | **Evaluation that is honest about what this dataset can show:** pre-registered protocols, OOF primary, minimal detectable differences, seed noise, annotation shift | Supports the reliability of every claim above | Analysis/04–08, this report |

### 2.3 The contributions as they would appear in the paper
1. A reproducible benchmark and evaluation protocol for HUSJA-UA (code, splits, per-femur predictions).
2. M0, a segmentation-first model that outperforms the published approach out of fold in classification and segmentation.
3. SLG-Net: region-grounded hierarchical decisions, shown by a pre-registered deletion test to depend on the fracture region more than M0's decisions do, at no loss of accuracy.
4. RefFrac: a healthy-reference comparison, with tests that show what such models actually use (the difference from any healthy femur) and that rule out the contralateral shortcut.
5. Evidence that, on this dataset, accuracy is bounded by label reliability and test-set size, so design choices should also be judged on how a model decides.

### 2.4 What the paper must not claim
- That SLG-Net or RefFrac are more accurate than M0, or than the published method on the test set.
- Clinical usefulness (there is no reader study and no clinician).
- That relying on the region makes a decision correct. It makes the decision traceable to the region.

## 3. Can this be defended for a Q2/Q3 journal? (candid)

**Yes, for technical and informatics journals, with the right framing. The risk is real.**

**What helps:**
- M0's clear out-of-fold win over the published method is an accuracy result. We do have an accuracy story; it is carried by M0, not by the extensions.
- The mechanism results are positive, significant and pre-registered (N3, N4). Many published "explainable" models have none.
- First independent work on a new public dataset, with full code and predictions.
- Pre-registration and honest statistics are increasingly valued by reviewers and editors.

**What reviewers will push on, and our answer:**

| Likely objection | Answer |
|---|---|
| "The proposed models do not beat the baseline" | The baseline is our own and beats the published method; the extensions add verified grounding at equal accuracy; the test set cannot detect differences under about 4 points (§1). |
| "On the test set, the published model has the highest macro-F1" | Test macro-F1 rests on 16 + 9 femurs in two classes; out of fold (1,043 femurs) M0 is ahead by .081 (p < .001); accuracy on the test set: M0 182, S01 185, SLG-Net 186, n.s. |
| "Is the grounding result specific to ConvNeXt?" | **Only answerable by a second backbone** (the optional runs). Without them, this is the weakest point. |
| "No external validation" | True for every work on this dataset. FracAtlas has no Garden/Evans grades. Stated as a limitation. |
| "Interpretability without clinicians" | We measure faithfulness (deletion, geometry, pointing), not plausibility to radiologists; stated explicitly. |

**Odds, plainly:** Q1 is unlikely without an accuracy gain or external data. A Q2 is plausible if the paper is framed around N2–N4 and the grounding result holds on a second backbone. Q3/Q4 is very likely. Check each journal's current JCR quartile at submission time. Candidates from Analysis/04 §2: BMC Medical Imaging, J Imaging Informatics in Medicine, IJIST, MBEC, Diagnostics, Bioengineering, BSPC.

## 4. Is there still a realistic way to get an accuracy gain?

| Option | Kaggle hours | Chance of a significant OOF gain over M0 | Comment |
|---|---|---|---|
| **Ensembles of saved predictions** | 0 (done, C17) | Tested: none significant | OOF +.016 macro-F1 at best; the test set looks better but disagrees with OOF. Supplementary at most |
| Test-time augmentation (flips) | ~1 per model (needs saved weights or re-runs) | Low | Usually +0.5–1 point, below the detectable 2 points |
| Higher input 1152 × 768 (already planned as a sensitivity run) | ~5 per model | Low–moderate | The grade errors are fine details; it may help Garden I-II. Pre-registered, so a fair attempt |
| Second backbone (EfficientNetV2-S; planned) | ~2.5–3 per model | Low for accuracy | Its real value is robustness of N3 |
| New loss for grades (ordinal, label smoothing) | ~3 per run, plus development | Low–moderate | A new hypothesis after seeing results: must be pre-registered and developed on fold 0 only, or it is p-hacking |
| Train on all 1,043 femurs for the test set | ~0.5 per model | None for OOF | Matches the article's protocol; affects only test numbers |

**Recommendation:** do not start new architectures to chase accuracy. The gains on offer are below what this dataset can detect, and every extra attempt after seeing results weakens the pre-registered claims. The only accuracy attempt worth paying for is the one already pre-registered: the sensitivity runs. They also answer the strongest reviewer objection.

## 5. The plan (numbered)

1. **Adopt the framing of §2** for the combined paper: segmentation-first plus evidence-grounded decisions. Accuracy is carried by M0, and grounding is verified by tests. Update Analysis/09 accordingly.
2. **Run the robustness check** (answer "Yes" or "Write first, run later" on the card):
   - M0 and SLG-Net with a second backbone (EfficientNetV2-S), each with the deletion test, 5 folds: about 5–6 h of T4 x2 time;
   - I write the rule before the runs: N3 is "replicated" if SLG-Net again loses a larger share than M0 out of fold, p < .05;
   - I first check that the code supports the second backbone (about 1 h of my work, no GPU).
3. **Optional, only if quota allows:** the 1152 × 768 resolution runs of M0 and SLG-Net (~10 h), with a pre-registered rule, as the one fair accuracy attempt.
4. **Figures from the filed results** (no GPU):
   - Fig. 2: deletion, SLG-Net vs M0 vs controls, per class;
   - Fig. 3: the reference tests;
   - Fig. 1: the overview drawing.
5. **Manuscript draft** in `manuscript/`: Methods and Results first (fixed by the reports), then Introduction and Discussion around §2–§3; supplement with C14, C15 and C17.
6. **Before submission:** re-check that no one else has published on HUSJA-UA; choose the journal (§3); publish the code and per-femur predictions with the paper.
