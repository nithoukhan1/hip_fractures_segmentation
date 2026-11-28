# Glossary: folder names and the names used in the documents

The documents call every model, run and analysis by its full name. The folders and the code still use short folder names, because scripts, the Kaggle run list (`kaggle/runs.json`) and the register depend on them. This page links the two.

## Models and runs (`results/<folder>/`)

| Folder | Name in the documents | What it is |
|---|---|---|
| `S00_env_check`, `S00_env_check_r2` | environment check | Kaggle data and library check, no training |
| `S00_net_check` | network check | Memory and time check of every network on a T4 GPU, no training |
| `S01_yolov8s_seg_1024` | **YOLOv8s-seg** | The published method (base article), reproduced at 1024 px |
| `S02_yolov8s_seg_640` | YOLOv8s-seg at 640 px | The same model at the article's best-Dice size |
| `S03_yolo11s_seg_1024` | YOLO11s-seg | The newer YOLO version |
| `S04_densenet121` | DenseNet-121 | Image classifier, the article's best classifier |
| `S05_efficientnet_b4` | EfficientNet-B4 | Image classifier |
| `S06_convnext_tiny` | ConvNeXt-T classifier | Image classifier with our backbone; first step of the SLG-Net ablation chain |
| `S07_unet_r34` | ResNet-34 U-Net | Standard segmenter with a classification head |
| `S08_deeplabv3p_r34` | ResNet-34 DeepLabV3+ | Standard segmenter with a classification head |
| `S09_segformer_b2` | SegFormer-B2 | Standard segmenter with a classification head |
| `S10_m0_convnext_unet` | **ConvNeXt U-Net** | Our core model: ConvNeXt-T encoder, U-Net decoder (fracture mask) and a 6-class head. SLG-Net and RefFrac both extend it |
| `A1_m0_deletion` | second ConvNeXt U-Net run | ConvNeXt U-Net trained again with the same settings, plus the region-deletion test |
| `A2_region` | region-pooling model | ConvNeXt U-Net + region pooling (SLG-Net ablation chain) |
| `A3_geometry` | geometry model | + region geometry |
| `A4_hierarchy` | hierarchy model | + location → grade hierarchy |
| `A5_shared_grade` | shared-trunk model | + one grade trunk shared by the neck and pertrochanteric grades |
| `A6_slgnet` | **SLG-Net** | + logit adjustment: the full Paper A model |
| `A6_dev_*` | SLG-Net fold-0 development runs | Teacher-forcing, detached-mask and pooling-scale questions |
| `R00_reference_bank`, `R00_reference_bank_r2` | reference bank | Healthy reference femurs and the pair index (second version: pelvis-side rule) |
| `R1_aasn` | AASN-style model | The closest published idea (AASN), adapted |
| `R2_random_ref` | random-reference model | One random healthy reference |
| `R3_nearest_ref` | nearest-reference model | The single most similar healthy reference |
| `R4_refrac` | **RefFrac** | The 3 most similar healthy references, local cross-attention, difference fusion: the full Paper B model |
| `R4_dev_*` | RefFrac fold-0 development runs | Whole-reference and margin-loss questions |
| `R5_concat` | concatenation model | RefFrac with concatenation instead of the difference |
| `R6a_no_align` | no-alignment model | RefFrac without alignment |
| `R6b_affine` | affine-alignment model | RefFrac with one global affine alignment |
| `R7_partner` | own-hip model | Trained with the patient's own healthy other hip as the reference |
| `N_segzoom_fold0`, `N_segzoom_fold0_seed2` | segment-and-zoom network (screening) | ConvNeXt U-Net finds the fracture; a second ConvNeXt-T grades it from a full-resolution crop around the predicted mask. Fold 0, seeds 2026 and 2027 |
| `N_res1152_fold0`, `N_res1152_fold0_seed2` | ConvNeXt U-Net at 1152 × 768 (screening) | The ConvNeXt U-Net at 1.5 times the resolution. Fold 0, two seeds |
| `N_convnext_s_fold0`, `N_convnext_s_fold0_seed2` | ConvNeXt U-Net with ConvNeXt-S (screening) | The ConvNeXt U-Net with the larger ConvNeXt-S encoder. Fold 0, two seeds |
| `N1_segzoom` | segment-and-zoom network, 5 folds | The segment-and-zoom network on 5 folds and the test set, with the oracle and deletion tests. Defined 2026-11-28, run only if the person chooses it (`Analysis/11` section 7.5) |
| `N_fracatlas_zero_shot` | FracAtlas zero-shot test | Every baseline and SLG-Net, as trained on HUSJA-UA, on the 148 FracAtlas hips (`scripts/eval_fracatlas.py`, `Analysis/11` section 8.4) |

Run series in the register: **S** = shared baselines, **A** = Paper A (SLG-Net), **R** = Paper B (RefFrac), **N** = the new method (`Analysis/11`).

## Analyses (`results/analysis/<folder>/`)

| Folder | Name in the documents | Question |
|---|---|---|
| `C01_yolo_1024_vs_640` | YOLO resolution analysis | Does 1024 px beat 640 px for YOLOv8s-seg? Do we reproduce the article? |
| `C02_yolov8_vs_yolo11` | YOLO version analysis | YOLOv8s-seg vs YOLO11s-seg |
| `C03_classifiers_vs_yolo` | classifier analysis | The three classifiers vs YOLOv8s-seg |
| `C04_segmenters_vs_yolo` | segmenter analysis | The segmenters and ConvNeXt U-Net vs YOLOv8s-seg |
| `C05_mask_task_s06_vs_s10` | mask-task analysis | Does learning the mask help the classifier? (ConvNeXt-T classifier vs ConvNeXt U-Net) |
| `C06_paperA_dev_fold0` | SLG-Net development analysis | SLG-Net's fold-0 development decisions |
| `C07_paperA_ablation_chain` | SLG-Net ablation analysis | Each step of the SLG-Net chain against the step before |
| `C08_paperA_interpretability` | SLG-Net interpretability analysis | Oracle, region deletion, risk–coverage, geometry; SLG-Net vs ConvNeXt U-Net under deletion |
| `C09_paperA_main_table` | SLG-Net main-table analysis | SLG-Net against every baseline |
| `C10_paperB_orientation_review` | orientation review | Review of the left/right rule of the reference bank |
| `C11_paperA_seed2` | SLG-Net second-seed check | Does SLG-Net's frozen design hold with another seed? |
| `C12_paperB_dev_fold0` | RefFrac development analysis | RefFrac's fold-0 development decisions |
| `C13_paperB_reference` | reference analysis | Does a healthy reference help, and does retrieval matter? |
| `C14_paperB_partner` | own-hip analysis | The patient's own other hip, and the pairing shortcut |
| `C15_paperB_components` | RefFrac component analysis | Alignment, fusion and the AASN-style model |
| `C16_paperB_main_table` | RefFrac main-table analysis | RefFrac against every baseline |
| `C17_ensembles` | ensemble analysis | Do ensembles of saved predictions buy accuracy? |
| `M1_baseline_table` | baseline table | All ten baselines side by side |
| `standard_metrics_all_runs` | standard-metrics tables | Every model with the standard metrics only (`Analysis/11` section 2.2): accuracy, macro precision, recall and F1, per-class precision, recall and F1, sensitivity, specificity, AUC, confusion matrices, Dice, IoU, HD95 |
| `N_screening_fold0` | fold-0 screening scores | The three new method candidates (two seeds each) and the two ConvNeXt U-Net runs on fold 0: standard metrics (`standard_metrics.md`) and intervals with paired tests (`stats.md`). `Analysis/11` section 7.5 |

## Other terms
- **OOF (out of fold):** each of the 1,043 training femurs is predicted by the fold model that did not see it. This is the main evaluation scope.
- **Test:** the 210 femurs of the released test set, predicted by the ensemble of the 5 fold models.
- **Garden I-II recall:** share of the 57 (OOF) or 16 (test) undisplaced neck fractures that are found and graded correctly.
- **Paper A / Paper B:** the two original plans (SLG-Net and RefFrac). Since 2026-11-28 they form one combined paper.
