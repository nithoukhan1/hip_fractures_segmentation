# Own-hip analysis: The patient's own other hip as the reference (own-hip question) and the pairing shortcut (pairing-shortcut question)

*Written 2026-11-28 from these filed runs (tag `B-v1-frozen` = 1c88912, 5/5 folds):* *`R7_partner` (register line 34) in its five modes, with `R3_nearest_ref` (line 32), `R4_refrac` (line 31) and `S10_m0_convnext_unet` (ConvNeXt U-Net, line 8).*

**Plan:** `Analysis/08_Step5_PaperB_RefFrac_Protocol.md` section 2.5. The own-hip model was trained with the other femur of the same x-ray as its reference when that femur is healthy, otherwise with the nearest retrieved femur. It is evaluated in five modes with the same weights:
- `base`: the training policy. It needs the partner's label, so it is reported only as the size of the leak.
- **`blind`: the partner whenever there is one, healthy or not. This is the own-hip model's result**, what a real system could do.
- `retrieved`: the nearest retrieved femur for every femur.
- `random` and `fractured`: swap tests with a random healthy or a random fractured femur.

**Thresholds** (rule 4 of section 3): `blind` and `retrieved` are refitted on their own OOF predictions (Youden; .0031 and .0601). The swap tests use blind's threshold, so they show what the swap does to the same decision rule. The nearest-reference model and RefFrac use their own thresholds.

**Pair types** (fixed table `results/R00_reference_bank_r2/pairs.csv`): out of fold 334 fractured femurs with a healthy partner, 334 healthy femurs with a fractured partner, 361 single femurs and 14 femurs of healthy pairs; on the test set 64, 64 and 82.

**Where the numbers come from** (all in this folder):
- `partner.md` / `partner.json`: every test below, made by `python scripts/run_refstats.py partner --name C14_paperB_partner`.
- `stats.md` / `breakdown.md`: intervals and per-class errors of ConvNeXt U-Net, the nearest-reference model, RefFrac and the own-hip model (in its base mode), made by `python scripts/run_stats.py S10_m0_convnext_unet R3_nearest_ref R4_refrac R7_partner --name C14_paperB_partner` and `python scripts/run_breakdown.py <the same> --name C14_paperB_partner`.
- Own-hip model's swap tests next to the other R runs: `../C13_paperB_reference/reftests.md`.

## 1. The own-hip model in each mode

| Mode | OOF correct /1043 | OOF macro-F1 | OOF Dice | OOF Garden I-II | False positives | Test correct /210 | Test macro-F1 | Test Dice | Test Garden I-II |
|---|---|---|---|---|---|---|---|---|---|
| base (leak, not a result) | 916 | .819 | .780 | .632 | 3 | 182 | .796 | .741 | .375 |
| **blind (own-hip model's result)** | **916** | **.819** | **.780** | **.632** | 3 | **183** | **.808** | **.741** | **.438** |
| retrieved | 917 | .822 | .781 | .632 | 3 | 182 | .795 | .741 | .438 |
| random | 912 | .812 | .779 | .632 | 9 | 181 | .790 | .742 | .438 |
| fractured | 864 | .760 | .743 | .596 | 5 | 181 | .817 | .718 | .500 |

For comparison: ConvNeXt U-Net 922 / .825 / .777 / .544 out of fold and 182 / .779 / .744 / .312 on the test set; the nearest-reference model 918 / .826 / .775 / .684 and 177 / .765 / .742 / .375.

## 2. The own-hip question: is the patient's own hip a better reference than a retrieved one?

On the fractured femurs with a healthy partner (Dice, Wilcoxon; Holm across the 2 comparisons):

| Comparison | OOF (334): Dice A vs B; difference (95% CI); p; Holm | Test (64) |
|---|---|---|
| Own-hip model blind (own hip) vs the own-hip model retrieved | .783 vs .785; −.001 (−.003 to −.000); .051; .10 | .731 vs .731; −.000; .86 |
| Own-hip model blind (own hip) vs the nearest-reference model (trained with retrieved references) | .783 vs .784; −.000 (−.007 to +.007); .79; .79 | .731 vs .731; −.001; .67 |

Garden I-II recall, all femurs: blind = retrieved (.632 OOF, .438 test, no femur differs); blind vs the nearest-reference model −.053 OOF (p .45) and +.062 test (p 1.00).

**Answer: no.** The patient's own healthy hip is not a better reference than a retrieved healthy femur. This matches the reference analysis: the match of the reference does not matter.

## 3. The pairing-shortcut question: does the own-hip model learn the pairing shortcut?

Under the training policy, a femur whose reference is its own partner is fractured in 334 of 348 cases (96%), so the own-hip model could have learnt "my reference comes from my own x-ray, so I am fractured". Three tests:

1. **Healthy femurs whose partner is fractured** (334 OOF, 64 test). If the own-hip model used the shortcut, showing them their own partner (blind) would raise P(fracture).
   - P(fracture), mean: blind .0071 vs retrieved .0093 OOF (Wilcoxon p .008, so *lower* with the partner); .0001 vs .0031 on the test set (p .22).
   - False positives: 3 vs 3 OOF, 1 vs 1 test (McNemar p 1.00).
   - **No shortcut**: the own partner does not push a healthy femur towards fracture. If anything, it pushes it slightly towards healthy.
2. **Fractured femurs with a healthy partner: swap the partner for a random healthy femur**, compared with the same swap in the nearest-reference model and RefFrac.
   - Mean change of P(fracture): the own-hip model +.006, the nearest-reference model +.000, RefFrac +.001 OOF; no fracture is lost in the own-hip model or the nearest-reference model, 2 in RefFrac.
   - Change of Dice: the own-hip model +.001, the nearest-reference model −.000, RefFrac −.002.
   - Own-hip model's change differs from RefFrac's in Dice by +.003 (Holm .005) and from the nearest-reference model's in P(fracture) by +.006 (Holm .08). Both are tiny, and on the test set nothing differs (every Holm p ≥ .90).
   - **No shortcut**: losing the own partner does not cost the own-hip model its fracture decisions.
3. **A fractured reference** (`fractured`, against blind): P(fracture) of fractured femurs falls by .116 OOF (.108 test); Dice falls .780 → .743 (−.037, p < .001; test −.023, p .15); 11 fractures are lost and 2 healthy femurs turn fractured OOF.
   - Own-hip model reads the *state* of the reference: when the reference is itself fractured, the target looks less different and less fractured. That is the comparison working as designed, and the same as the self test in the reference analysis.

**The leak:** blind and base give the same decisions in all three pair types out of fold (273 / 331 / 298 correct) and differ by one femur on the test set. Using the partner's label to choose the reference therefore buys nothing.

## 4. What the own-hip analysis shows
1. **Own-hip question: the patient's own hip is not better than a retrieved healthy femur** (Dice difference ≤ .001, Garden I-II identical).
2. **Pairing-shortcut question: the own-hip model did not learn the pairing shortcut.** A fractured partner does not make a healthy femur look fractured; swapping the own partner for a random femur changes nothing; base and blind agree.
3. **Own-hip model reads the state of the reference**: a fractured reference makes fractured femurs look less fractured (P −.116, Dice −.037).
4. Own-hip model blind is a usable, label-free model: OOF macro-F1 .819 and Dice .780 (ConvNeXt U-Net .825, .777), test 183 correct (the most of any R run).
5. **For the paper:** this is a clean negative result for the "use the other hip" idea on this dataset, and a reassuring one for the shortcut concern. Limitation (section 6 of the protocol): one fractured femur per two-femur x-ray is a property of this dataset.
