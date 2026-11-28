# SLG-Net main-table analysis: Paper A's main table: SLG-Net against every baseline

*Written 2026-11-27 from these filed runs:*
- *`A6_slgnet` (SLG-Net; register line 26; tag `A-v1-frozen`);*
- *the baselines `S01_yolov8s_seg_1024` (line 3), `S04_densenet121` and `S05_efficientnet_b4` (lines 9–10), `S06_convnext_tiny` (line 7), `S07_unet_r34` and `S08_deeplabv3p_r34` (lines 11–12), `S09_segformer_b2` (line 18) and `S10_m0_convnext_unet` (ConvNeXt U-Net, line 8).*

**Plan:** `Analysis/06_Step3_PaperA_Protocol.md` section 7: SLG-Net against YOLOv8s-seg and the seven other baselines.

**Where the numbers come from** (all in this folder):
- `stats.md` / `stats.json`, made by
  `python scripts/run_stats.py A6_slgnet S01_yolov8s_seg_1024 S04_densenet121 S05_efficientnet_b4 S06_convnext_tiny S07_unet_r34 S08_deeplabv3p_r34 S09_segformer_b2 S10_m0_convnext_unet --pairs A6_slgnet:<each baseline> --family-holm --name C09_paperA_main_table`.
  - The protocol fixed no adjustment for this table. The p-values below are therefore adjusted by Holm across the 8 comparisons (each endpoint and scope separately), which is the cautious choice.
- `breakdown.md`: per class, the error types and the comparison with the base article, made by `python scripts/run_breakdown.py <the same 9 runs> --name C09_paperA_main_table`.

**How decisions are made:** each run uses its own OOF threshold. Out of fold there are 1,043 femurs (680 fractured); the test set is scored with each run's 5-model ensemble on 210 femurs (145 fractured).

## 1. The main table

| Model | Type | OOF macro-F1 (95% CI) | OOF Dice, fractured (95% CI) | OOF correct | Test macro-F1 (95% CI) | Test Dice (95% CI) | Test correct |
|---|---|---|---|---|---|---|---|
| YOLOv8s-seg @1024 (the base article's model) | instance segmenter | .744 (.707–.779) | .740 (.722–.755) | 868 (83.2%) | **.845** (.780–.898) | .725 (.694–.754) | 185 |
| DenseNet-121 | classifier | .802 (.766–.833) | – | 905 (86.8%) | .760 (.670–.832) | – | 180 |
| EfficientNet-B4 | classifier | .755 (.717–.789) | – | 884 (84.8%) | .744 (.658–.814) | – | 170 |
| ConvNeXt-T classifier | classifier | .784 (.746–.816) | – | 901 (86.4%) | .742 (.651–.816) | – | 178 |
| ResNet-34 U-Net, ResNet-34 | segmenter + head | .800 (.766–.832) | .748 (.734–.762) | 900 (86.3%) | .799 (.714–.863) | .711 (.678–.740) | 182 |
| ResNet-34 DeepLabV3+, ResNet-34 | segmenter + head | .800 (.765–.830) | .749 (.734–.763) | 899 (86.2%) | .778 (.696–.847) | .698 (.661–.734) | 181 |
| SegFormer-B2 | segmenter + head | .817 (.784–.847) | .766 (.753–.778) | 917 (87.9%) | .804 (.721–.870) | .737 (.711–.760) | 185 |
| ConvNeXt U-Net, ConvNeXt-T U-Net | segmenter + head | **.825** (.793–.855) | **.777** (.764–.789) | **922 (88.4%)** | .779 (.695–.849) | **.744** (.718–.766) | 182 |
| **SLG-Net** | segmentation-guided hierarchy | .806 (.771–.836) | .774 (.761–.786) | 907 (87.0%) | .820 (.747–.879) | .737 (.711–.761) | **186** |

## 2. SLG-Net against each baseline (paired, same femurs)

| SLG-Net vs | OOF macro-F1: difference (95% CI); Holm p | OOF Dice: difference; Holm p | OOF accuracy: difference; Holm p | Test macro-F1: difference; p (Holm) | Test Dice: difference; p (Holm) |
|---|---|---|---|---|---|
| YOLOv8s-seg | **+.061 (+.028 to +.097); .006** | **+.034; < .001** | **+3.7 points; .003** | −.025; .51 (1.00) | +.012; .41 (.82) |
| DenseNet-121 | +.004 (−.025 to +.035); 1.00 | – | +0.2; 1.00 | +.061; .16 (.94) | – |
| EfficientNet-B4 | **+.050 (+.018 to +.083); .020** | – | +2.2; .16 | +.076; .090 (.63) | – |
| ConvNeXt-T classifier | +.022 (−.007 to +.052); .84 | – | +0.6; 1.00 | +.079; .006 (.051) | – |
| ResNet-34 U-Net | +.006 (−.020 to +.033); 1.00 | **+.026; < .001** | +0.7; 1.00 | +.021; .50 (1.00) | **+.026; < .001 (< .001)** |
| ResNet-34 DeepLabV3+ | +.006 (−.022 to +.036); 1.00 | **+.025; < .001** | +0.8; 1.00 | +.042; .19 (.96) | **+.039; .002 (.010)** |
| SegFormer-B2 | −.012 (−.043 to +.018); 1.00 | **+.008; .003** | −1.0; 1.00 | +.016; .57 (1.00) | −.000; .24 (.71) |
| ConvNeXt U-Net | −.020 (−.047 to +.008); .87 | −.002; .076 | −1.4; .46 | +.041; .21 (.96) | −.006; .66 (.82) |

## 3. Against the base article's final model (test set)

The base article's numbers come from its Table 3 and Fig. 3; ours are from `breakdown.md`.

| | Article, final model | SLG-Net, 5-model ensemble | SLG-Net, 5 fold models (mean ± SD) |
|---|---|---|---|
| 6-class accuracy | 86.2% | 88.6% | 86.3 ± 1.5% |
| Macro-F1 | .814 | .820 | .789 ± .015 |
| Fracture sensitivity / specificity | .952 / .985 | .986 / .985 | .972 / .991 |
| Errors: detection / location / grade | 8 / 6 / 15 | 3 / 4 / 17 | |
| Garden I-II recall (16 femurs) | .500 | .312 | .287 ± .034 |

- **Training data differs:** the article's model was trained on all 1,043 training femurs, plus a validation set. Each of our fold models saw about 834.
- **Selection rule:** our fold model picked by the article's rule (best accuracy on its own held-out fold) scores 84.3% and macro-F1 .769.

## 4. What the table shows
1. **Out of fold, SLG-Net clearly beats the base article's model** (our reproduction YOLOv8s-seg): +.061 macro-F1, +.034 Dice and +3.7 points accuracy, all significant after Holm. It also beats EfficientNet-B4 on macro-F1 (+.050). Its Dice beats every segmenter except ConvNeXt U-Net: U-Net +.026, DeepLabV3+ +.025, SegFormer +.008.
2. **It does not beat the strongest baselines.** Against ConvNeXt U-Net, SegFormer, DenseNet, U-Net, DeepLabV3+ and ConvNeXt, the macro-F1 differences lie between −.020 and +.022, and none is significant. ConvNeXt U-Net remains the best model out of fold.
3. **On the test set no macro-F1 difference survives Holm.**
   - SLG-Net has the most correct femurs (186 of 210).
   - YOLOv8s-seg has the highest macro-F1 (.845), although it is the weakest model out of fold (.744). The same reversal was seen for ConvNeXt U-Net in the segmenter analysis: on 210 femurs with smaller test masks (the annotation shift of the YOLO resolution analysis), the test set ranks the models differently from the 1,043 OOF femurs. The protocol makes the OOF result primary.
4. **For the paper:**
   - Out of fold, SLG-Net is **better than the published approach** (YOLOv8s-seg) and EfficientNet-B4 in macro-F1, and better than every segmenter except ConvNeXt U-Net in Dice.
   - It is **not significantly different from the other baselines** in macro-F1. No equivalence margin was fixed in advance, so this does not mean it is as good as they are.
   - What sets it apart is that its location and grade decisions demonstrably depend on the fracture region (SLG-Net interpretability analysis), not higher accuracy.
