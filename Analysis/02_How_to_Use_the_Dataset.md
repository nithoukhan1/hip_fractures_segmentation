# Report 02: How we can use the HUSJA-UA dataset, and why

Date: 2026-11-26. Builds on `Analysis/01_Base_Article_and_EDA_Report.md` (our EDA) and `research/01_prior_use_and_sota_review.md` (literature thread). Each claim is tagged with its evidence: **[EDA]** for our measurements, **[Paper]** for the base article, and a linked source for literature.

## 1. Why this dataset is worth building on

1. **Nobody else has published on it yet.** It went public on IEEE DataPort on 11 Nov 2025. Of the 25 works citing the base article (OpenAlex), none uses the data; no GitHub repository or Kaggle notebook does either ([literature review §1, §4](../research/01_prior_use_and_sota_review.md)). A paper from us would be among the first independent benchmarks.
2. **It's unique.** It is the only public hip x-ray dataset with **fracture-region masks plus Garden/Evans grades**. FracAtlas has masks but only 63 fractured hip studies; PXR150 and PelvixNet have image-level labels only ([FracAtlas, Sci Data 2023](https://www.nature.com/articles/s41597-023-02432-4); review §5c).
3. **Segmentation of hip fractures is the thinnest part of the literature.** Almost all recent hip work is classification or bounding boxes: [OccuNet, Radiology 2026](https://pubs.rsna.org/doi/10.1148/radiol.253931); [Yıldız Potter, Sci Rep 2024](https://www.nature.com/articles/s41598-024-63001-2); [Shen, Front Med 2026](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1759383/full); [Zhao, JIIM 2026](https://doi.org/10.1007/s10278-026-02230-5). The only fracture-region masks are in González 2025 and FracAtlas-based work such as [Ahmed et al., Eng Appl AI 2026](https://doi.org/10.1016/j.engappai.2026.115849).
4. **The baseline leaves clear room for improvement** [Paper]:
   - Dice is 0.77 with a 95% CI of 0.56–0.98, and there is no per-class Dice or boundary metric.
   - Dice gets *worse* at higher resolution.
   - Accuracy (86.2%) is not significantly better than DenseNet121 (83.8%).
   - The rare classes fail most.
5. **It's feasible on Kaggle:** 1,043 training crops at 1024 px. YOLO-seg, U-Net-family models or LoRA-tuned SAM fit in a 12-hour T4/P100 session (review §6).

## 2. What reviewers already see too much of (what to avoid)

The citing works are mostly **small private single-centre studies that add one attention block to YOLO** and gain about 1 mAP point, with no variance reported ([Shen 2026](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1759383/full): 0.805 → 0.817; [Zhao 2026](https://doi.org/10.1007/s10278-026-02230-5)). A "YOLO + new module" paper is weak. Our method needs a **clear idea tied to a real property of this data**, plus repeated runs and proper statistics.

## 3. Ways to use the dataset, each tied to evidence

These are directions, not the final pick. Selection is the next step.

| # | Direction | Why (evidence) | Literature status | Feasibility |
|---|---|---|---|---|
| D1 | **Rare-class / long-tail learning** (class-balanced or focal loss, decoupled heads, targeted copy-paste of fracture regions, generative augmentation) | Garden I–II has 57 and subtroch 36 train femurs [EDA]; worst classes in the confusion matrix [Paper]; authors' own future work [Paper] | Not tried on this data (review §6.1) | High |
| D2 | **Segmentation-first architecture** (boundary-aware or CNN–transformer decoder, high-resolution mask head) with per-class Dice, HD95 and NSD | Dice drops with resolution, wide CI, no per-class report [Paper]; targets are small (2–10% of the crop) [EDA] | No dedicated segmentation model on hip-fracture masks (review §5b) | High |
| D3 | **Hierarchical / ordinal multi-task grading** (location first, then grade; mask features guide the grade) | 15 of 21 errors are the right location with the wrong grade [Paper]; mask size grows with grade [EDA]; centroid position encodes location [EDA]; Garden-only grading reaches 90.6% with private data ([Yang, Sci Rep 2025](https://www.nature.com/articles/s41598-025-27766-4)) | Hierarchical grading + segmentation not reported (review §6.3) | High |
| D4 | **Contralateral (bilateral) comparison**: use the healthy femur of the same x-ray as a reference | 398 x-rays have a fractured + healthy pair; no x-ray has two fractures [EDA]; radiologists compare sides; segmentation-guided reasoning helps undisplaced neck fractures ([Wang, Bioengineering 2025](https://mdpi.com/2306-5354/12/5/466)) | Not used in any paper we found | Medium (pairs exist for about half the cases; needs a fallback for single-femur x-rays) |
| D5 | **Label noise and annotation-shift-aware training** (soft or uncertain boundaries, consistency losses, boundary-tolerant losses) | Duplicate image with masks at IoU 0.70; test neck masks 35–40% smaller than train [EDA]; MedSAM label refinement helps on FracAtlas ([Ahmed 2026](https://doi.org/10.1016/j.engappai.2026.115849)) | Only Ahmed 2026, on other data | Medium–High |
| D6 | **Foundation-model segmentation** (SAM / MedSAM / SAM2 with LoRA adapters, prompt-free) | Barely explored for fracture-region masks on radiographs ([Magg, arXiv 2026](https://arxiv.org/pdf/2603.10541); review §5b) | Open | Medium (GPU memory on Kaggle) |
| D7 | **Uncertainty and calibration** (ensembles, MC dropout, mask uncertainty to flag cases for review) | The paper's title promises interpretability, but there is no calibration or failure analysis [Paper] | Open (review §6.6) | High (an add-on to any model) |
| D8 | **External transfer test** on FracAtlas hip images | Single centre is the authors' own limitation [Paper]; FracAtlas has 63 fractured hips with masks ([FracAtlas](https://www.nature.com/articles/s41597-023-02432-4)) | Not done | Medium (small, different labels) |
| D9 | **Reproducible benchmark** of modern models (YOLOv8/11-seg, Mask2Former, nnU-Net, SAM-based) with repeated runs | Nobody has benchmarked this data independently (review §1) | Open | High, but alone it is Q3/Q4 |

## 4. How these combine into a publishable story (Q2/Q3)

A single paper normally needs **one main idea plus solid evaluation**:
- **Core novelty:** one of D2, D3, D4 or D5. These address properties *specific to this dataset*, which is what separates us from the "YOLO + attention" crowd.
- **Supporting pieces:** D1 (rare classes), D7 (uncertainty) and D9 (benchmark table) strengthen any core idea at low extra cost.
- **Generalisation check:** D8, if time allows, answers the reviewers' standard "single-centre" objection.

## 5. Rules we'll follow regardless of the idea
1. Keep the official test set (210) frozen. Build our validation set from train, grouped by x-ray ID and stratified by class (the official val folder is empty).
2. Reproduce the paper's YOLOv8-seg baseline ourselves (their code is not public).
3. Report AUC, 6-class accuracy, macro-F1, balanced accuracy, **per-class Dice / HD95 / NSD**, and calibration (ECE), using 3–6 seeds with 95% CIs. Use DeLong for AUC and McNemar or bootstrap for accuracy and Dice.
4. State honestly the missing val set, the duplicate image, and the train/test annotation shift.
