# Run notes: A1_m0_deletion

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper A: ConvNeXt U-Net retrained exactly as the first ConvNeXt U-Net run with the region-deletion test (--extra-tests deletion), so SLG-Net's region dependence can be compared with ConvNeXt U-Net's (Analysis/06 section 13, the person's choice of 2026-11-28)

**2. What changed compared with the previous run** (config, code commit, data, folds): Same settings and seed as `S10_m0_convnext_unet` (ConvNeXt U-Net), plus `--extra-tests deletion`. Code `080a0c5` (the merge of PR #16; the launcher log shows it), T4 x2, all 5 folds trained, exit 0. 147.9 min.

**3. Result** (`metrics.json`, `summary.md`; threshold .0084 fitted on the OOF femurs only):
- Out of fold (1,043 femurs): AUC .994, accuracy 921/1043, macro-F1 .825, Dice fractured .778, Garden I-II recall .614. ConvNeXt U-Net: 922, .825, .777, .544.
- Test ensemble (210 femurs): AUC 1.000, accuracy 179/210, macro-F1 .783, Dice fractured .738, Garden I-II recall .438.
- Region deletion, OOF: 274 of 562 correct fractured decisions lost (48.8%), control 8 (1.4%); P(fracture) −.199, P(true class) −.442.

**4. What we learnt:** It replicates ConvNeXt U-Net (no paired difference significant, `../analysis/C08_paperA_interpretability/m0_replicate/`), except Garden I-II recall, which moves by 4 of 57 femurs with the seed alone. ConvNeXt U-Net also depends on the region, but less than SLG-Net: on the jointly correct fractured femurs 48.2% vs 64.2% are lost (p < .001; test 52.6% vs 65.5%, p .002). By the rule of Analysis/06 section 13, SLG-Net is more region-dependent than ConvNeXt U-Net (SLG-Net interpretability analysis, section 2b).

**5. Decision / next step:** Analysis/06 section 14. The combined paper is planned in Analysis/09.
