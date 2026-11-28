# Ensemble analysis: Can combining ConvNeXt U-Net, SLG-Net and RefFrac buy accuracy? Ensembles from saved predictions

*Written 2026-11-28. Exploratory: no protocol fixed this analysis before the runs existed. The five ensembles and the nine comparisons were fixed in `scripts/run_ensemble.py` before anything was computed, and all of them are reported.*

**Why:** none of the extensions beats ConvNeXt U-Net alone (SLG-Net ablation analysis, the reference analysis, the RefFrac main-table analysis). Their mistakes may differ, though, so averaging their probabilities might. This costs no GPU time.

**How:** `python scripts/run_ensemble.py --name C17_ensembles`. Output: `ensembles.md` / `ensembles.json`.
- The members' 6-class probabilities are averaged per femur.
- Each ensemble's threshold is fitted on its own OOF femurs.
- The mask comes from ConvNeXt U-Net, so Dice is ConvNeXt U-Net's, except for fractures the ensemble misses.
- **Control: the ensemble of two ConvNeXt U-Nets = two trainings of ConvNeXt U-Net** (first ConvNeXt U-Net run and `A1_m0_deletion`). Any ensemble gains a little from averaging alone, so the fair question is whether SLG-Net or RefFrac add more than a second ConvNeXt U-Net does.
- Holm runs across all 9 comparisons, per endpoint and scope.

## Results

| Model | OOF correct /1043 | OOF macro-F1 | OOF Garden I-II | Test correct /210 | Test macro-F1 | Test Garden I-II |
|---|---|---|---|---|---|---|
| ConvNeXt U-Net | 922 | .825 | .544 | 182 | .779 | .312 |
| ConvNeXt U-Net + ConvNeXt U-Net (control) | 924 | .828 | .632 | 180 | .787 | .438 |
| ConvNeXt U-Net + SLG-Net | 921 | .823 | .579 | **189** | **.843** | .375 |
| ConvNeXt U-Net + RefFrac | **931** | .838 | **.684** | 184 | .797 | .312 |
| ConvNeXt U-Net + SLG-Net + RefFrac | 929 | **.842** | .632 | 187 | .819 | .375 |
| All four | 925 | .832 | .649 | 186 | .814 | .375 |

**Paired tests** (`ensembles.md`):
- **Out of fold** (the primary scope), no ensemble beats ConvNeXt U-Net or the control after Holm.
  - The largest gains over ConvNeXt U-Net are the three-model ensemble macro-F1 +.016 (p .12) and the ConvNeXt U-Net + RefFrac ensemble Garden I-II +.140 (9 vs 1 femurs, p .021, Holm .19).
  - Against the control (two ConvNeXt U-Nets), the three-model ensemble adds +.014 macro-F1 (p .16).
- **On the test set,** the ConvNeXt U-Net + SLG-Net ensemble gets 189 of 210 right (90.0%), and its macro-F1 is .843 vs ConvNeXt U-Net's .779 (p .008, Holm .07). The three-model ensemble has 187 right, with macro-F1 +.040 (p .03, Holm .25).
- The two scopes disagree: the ConvNeXt U-Net + SLG-Net ensemble is the weakest ensemble out of fold and the strongest on the test set. Given that, and the test set's minimal detectable difference of about 4–5 accuracy points (Analysis/10 section 4), no ensemble can be claimed better.

## What the ensemble analysis shows
1. Combining the three designs gives the highest out-of-fold macro-F1 of any model so far (.842), but no gain is significant, and it is not clearly larger than what averaging two ConvNeXt U-Nets gives.
2. The test-set gains of the ConvNeXt U-Net + SLG-Net ensemble and the three-model ensemble are the largest seen against ConvNeXt U-Net (+.06 and +.04 macro-F1). They do not hold out of fold, so they are not evidence.
3. **For the paper:** an exploratory supplementary table at most. It supports "the designs make different mistakes", not "an ensemble is better".
