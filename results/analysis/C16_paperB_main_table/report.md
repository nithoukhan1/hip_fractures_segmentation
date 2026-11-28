# RefFrac main-table analysis: Paper B's main table: RefFrac against every baseline

*Written 2026-11-28 from these filed runs:* *`R4_refrac` (RefFrac; register line 31; tag `B-v1-frozen`) and the baselines `S01_yolov8s_seg_1024` (line 3), `S04_densenet121` and `S05_efficientnet_b4` (lines 9–10), `S06_convnext_tiny` (line 7), `S07_unet_r34` and `S08_deeplabv3p_r34` (lines 11–12), `S09_segformer_b2` (line 18) and `S10_m0_convnext_unet` (ConvNeXt U-Net, line 8).*

**Plan:** `Analysis/08_Step5_PaperB_RefFrac_Protocol.md` section 5: RefFrac against YOLOv8s-seg and the seven other baselines, built like Paper A's the SLG-Net main-table analysis.

**Where the numbers come from** (all in this folder):
- `stats.md` / `stats.json`: `python scripts/run_stats.py R4_refrac S01_yolov8s_seg_1024 S04_densenet121 S05_efficientnet_b4 S06_convnext_tiny S07_unet_r34 S08_deeplabv3p_r34 S09_segformer_b2 S10_m0_convnext_unet --pairs R4_refrac:<each baseline> --family-holm --name C16_paperB_main_table`. Holm across the 8 comparisons, each endpoint and scope separately (as in the SLG-Net main-table analysis).
- `g12.md`: Garden I-II recall, `python scripts/run_refstats.py g12 <the same 9 runs> --pairs <the same 8 pairs> --name C16_paperB_main_table`.
- `breakdown.md`: per class, error types and the comparison with the base article.

**How decisions are made:** each run uses its own OOF threshold. 1,043 out-of-fold femurs (680 fractured, 57 Garden I-II); test set: each run's 5-model ensemble on 210 femurs (145 fractured, 16 Garden I-II).

## 1. The main table

| Model | Type | OOF macro-F1 (95% CI) | OOF Dice, fractured (95% CI) | OOF Garden I-II | OOF correct | Test macro-F1 (95% CI) | Test Dice (95% CI) | Test Garden I-II | Test correct |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv8s-seg @1024 (the base article's model) | instance segmenter | .744 (.707–.779) | .740 (.722–.755) | .509 | 868 | **.845** (.780–.898) | .725 (.694–.754) | **.500** | **185** |
| DenseNet-121 | classifier | .802 (.766–.833) | – | .579 | 905 | .760 (.670–.832) | – | .250 | 180 |
| EfficientNet-B4 | classifier | .755 (.717–.789) | – | .386 | 884 | .744 (.658–.814) | – | .312 | 170 |
| ConvNeXt-T classifier | classifier | .784 (.746–.816) | – | .474 | 901 | .742 (.651–.816) | – | .250 | 178 |
| ResNet-34 U-Net, ResNet-34 | segmenter + head | .800 (.766–.832) | .748 (.734–.762) | .614 | 900 | .799 (.714–.863) | .711 (.678–.740) | .375 | 182 |
| ResNet-34 DeepLabV3+, ResNet-34 | segmenter + head | .800 (.765–.830) | .749 (.734–.763) | .596 | 899 | .778 (.696–.847) | .698 (.661–.734) | .250 | 181 |
| SegFormer-B2 | segmenter + head | .817 (.784–.847) | .766 (.753–.778) | .579 | 917 | .804 (.721–.870) | .737 (.711–.760) | .375 | **185** |
| ConvNeXt U-Net, ConvNeXt-T U-Net | segmenter + head | .825 (.793–.855) | **.777** (.764–.789) | .544 | **922** | .779 (.695–.849) | **.744** (.718–.766) | .312 | 182 |
| **RefFrac** | ConvNeXt U-Net + 3 healthy references | **.826** (.792–.855) | **.777** (.764–.789) | **.702** | 920 | .779 (.696–.844) | .739 (.713–.763) | .312 | 180 |

## 2. RefFrac against each baseline (paired, same femurs)

| RefFrac vs | OOF macro-F1: diff; Holm | OOF Dice: diff; Holm | OOF accuracy: points; Holm | OOF Garden I-II: diff (95% CI); only RefFrac / only other; Holm | Test macro-F1: diff; p | Test Dice: diff; p (Holm) |
|---|---|---|---|---|---|---|
| YOLOv8s-seg | **+.081; .002** | **+.037; < .001** | **+5.0; < .001** | +.193 (+.053 to +.351); 16 / 5; .16 | −.066; .10 | +.014; .48 (.58) |
| DenseNet-121 | +.023; .31 | – | +1.4; .36 | +.123 (.000 to +.246); 10 / 3; .37 | +.020; .56 | – |
| EfficientNet-B4 | **+.070; .002** | – | **+3.5; .004** | **+.316 (+.175 to +.456); 20 / 2; < .001** | +.035; .34 | – |
| ConvNeXt-T classifier | **+.042; .025** | – | +1.8; .16 | **+.228 (+.105 to +.351); 15 / 2; .016** | +.038; .19 | – |
| ResNet-34 U-Net | +.025; .27 | **+.029; < .001** | +1.9; .13 | +.088 (−.018 to +.193); 8 / 3; .37 | −.019; .33 | **+.028; < .001 (.002)** |
| ResNet-34 DeepLabV3+ | +.026; .27 | **+.028; < .001** | +2.0; .11 | +.105 (.000 to +.211); 8 / 2; .37 | +.001; .97 | **+.040; .011 (.043)** |
| SegFormer-B2 | +.008; 1.00 | **+.011; .002** | +0.3; 1.00 | +.123 (−.018 to +.246); 11 / 4; .37 | −.025; .45 | +.002; .28 (.58) |
| ConvNeXt U-Net | +.000; 1.00 | +.000; .25 | −0.2; 1.00 | +.158 (+.035 to +.281); 12 / 3; .18 | +.000; .98 | −.005; .19 (.58) |

Garden I-II recall on the 16 test femurs: no difference against any baseline (every p ≥ .37).

## 3. Against the base article's final model (test set)

| | Article, final model | RefFrac, 5-model ensemble | RefFrac, 5 fold models (mean ± SD) |
|---|---|---|---|
| 6-class accuracy | 86.2% | 85.7% | 86.0 ± 0.9% |
| Macro-F1 | .814 | .779 | .777 ± .021 |
| Fracture sensitivity / specificity | .952 / .985 | .986 / .985 | .981 / .997 |
| Errors: detection / location / grade | 8 / 6 / 15 | 3 / 6 / 21 | |
| Garden I-II recall (16 femurs) | .500 | .312 | .325 ± .052 |

The training data differ as in the SLG-Net main-table analysis: each of our fold models saw about 834 femurs, the article's model all 1,043 plus a validation set.

## 4. What the table shows
1. **Out of fold, RefFrac ties ConvNeXt U-Net for the best macro-F1 and Dice**, and beats the base article's model (YOLOv8s-seg) on macro-F1 (+.081), Dice (+.037) and accuracy (+5.0 points), all after Holm. It also beats EfficientNet-B4 and ConvNeXt-T on macro-F1, and the three the three segmenters on Dice.
2. **Its distinctive number is Garden I-II recall out of fold: .702, the highest of all nine models.** After Holm it beats EfficientNet-B4 (+.316) and ConvNeXt-T (+.228); against ConvNeXt U-Net (+.158, Holm .18) and the others it is a trend.
3. **On the test set RefFrac is not better than ConvNeXt U-Net or YOLOv8s-seg.** Its test Dice beats U-Net and DeepLabV3+ only. Its Garden I-II recall (.312) equals ConvNeXt U-Net's and is below YOLOv8s-seg's (.500) and the article's (.500), on 16 femurs.
4. **The out-of-fold to test drop** (macro-F1 .826 → .779) is the same as ConvNeXt U-Net's (.825 → .779). It comes from the train/test annotation shift seen since the YOLO resolution analysis, not from the reference.
5. **For the paper:** like SLG-Net in the SLG-Net main-table analysis, RefFrac is a strong model but not better than ConvNeXt U-Net, the model it extends. Paper B's honest contribution is the reference analysis (the own-hip analysis): the model uses the comparison but not the match of the reference, the patient's own hip is no better, and there is no pairing shortcut. The Garden I-II trend out of fold is worth reporting as exploratory.
