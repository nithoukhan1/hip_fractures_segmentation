# C17: Can combining M0, SLG-Net and RefFrac buy accuracy? Ensembles from saved predictions

*Written 2026-11-28. Exploratory: no protocol fixed this analysis before the runs existed. The five ensembles and the nine comparisons were fixed in `scripts/run_ensemble.py` before anything was computed, and all of them are reported.*

**Why:** none of the extensions beats M0 alone (C07, C13, C16). Their mistakes may differ, though, so averaging their probabilities might. This costs no GPU time.

**How:** `python scripts/run_ensemble.py --name C17_ensembles`. Output: `ensembles.md` / `ensembles.json`.
- The members' 6-class probabilities are averaged per femur.
- Each ensemble's threshold is fitted on its own OOF femurs.
- The mask comes from M0, so Dice is M0's, except for fractures the ensemble misses.
- **Control: E0 = two trainings of M0** (S10 and `A1_m0_deletion`). Any ensemble gains a little from averaging alone, so the fair question is whether SLG-Net or RefFrac add more than a second M0 does.
- Holm runs across all 9 comparisons, per endpoint and scope.

## Results

| Model | OOF correct /1043 | OOF macro-F1 | OOF Garden I-II | Test correct /210 | Test macro-F1 | Test Garden I-II |
|---|---|---|---|---|---|---|
| M0 (S10) | 922 | .825 | .544 | 182 | .779 | .312 |
| E0 = M0 + M0 (control) | 924 | .828 | .632 | 180 | .787 | .438 |
| E1 = M0 + SLG-Net | 921 | .823 | .579 | **189** | **.843** | .375 |
| E2 = M0 + RefFrac | **931** | .838 | **.684** | 184 | .797 | .312 |
| E3 = M0 + SLG-Net + RefFrac | 929 | **.842** | .632 | 187 | .819 | .375 |
| E4 = all four | 925 | .832 | .649 | 186 | .814 | .375 |

**Paired tests** (`ensembles.md`):
- **Out of fold** (the primary scope), no ensemble beats M0 or the control after Holm.
  - The largest gains over M0 are E3 macro-F1 +.016 (p .12) and E2 Garden I-II +.140 (9 vs 1 femurs, p .021, Holm .19).
  - Against the control E0, E3 adds +.014 macro-F1 (p .16).
- **On the test set,** E1 (M0 + SLG-Net) gets 189 of 210 right (90.0%), and its macro-F1 is .843 vs M0's .779 (p .008, Holm .07). E3 has 187 right, with macro-F1 +.040 (p .03, Holm .25).
- The two scopes disagree: E1 is the weakest ensemble out of fold and the strongest on the test set. Given that, and the test set's minimal detectable difference of about 4–5 accuracy points (Analysis/10 §4), no ensemble can be claimed better.

## What C17 shows
1. Combining the three designs gives the highest out-of-fold macro-F1 of any model so far (.842), but no gain is significant, and it is not clearly larger than what averaging two M0s gives.
2. The test-set gains of E1 and E3 are the largest seen against M0 (+.06 and +.04 macro-F1). They do not hold out of fold, so they are not evidence.
3. **For the paper:** an exploratory supplementary table at most. It supports "the designs make different mistakes", not "an ensemble is better".
