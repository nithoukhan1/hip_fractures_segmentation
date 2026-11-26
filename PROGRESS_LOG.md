# Progress Log (step-by-step)

Newest entries at the top. Each entry: what was done, what it produced, what comes next.

---

## 2026-11-26 · Step 4: Ranking the directions, research plan, repository set-up

**Done**
1. Scored D1–D9 on novelty, evidence in our data, chance of a clear result, Kaggle feasibility and reviewer value. Ranking: D3 > D7 > D2 > D4 > D9 > D1 = D8 > D6 > D5 (`Analysis/03_Idea_Ranking_and_Research_Plan.md` §2).
2. Checked the literature behind the top directions:
   - Hierarchical proximal femur classification exists only as classifier cascades or boxes, without fracture masks (Tanzi 2020; Jiménez-Sánchez 2019).
   - Left-right comparison exists only for pelvic fracture detection (AASN, ECCV 2020).
   - No hip paper we could read combines fracture masks with location → grade decisions, or uses the contralateral femur for segmentation or grading.
3. New evidence from the ground-truth masks:
   - Mask height alone separates Garden I–II vs III–IV with AUC 0.81 (train) / 0.84 (test).
   - Mask area separates Evans I–II vs III–V with AUC 0.80 / 0.75.
   - Centroid height alone predicts the location of test fractures with 80% accuracy.
4. Designed two options:
   - **A, SLG-Net (Segment, Locate, Grade), recommended.**
   - **B, RefFrac** (healthy reference femur), higher risk; suggested as Paper 2.
   - Wrote the full experiment plan: baselines B1–B9, ablations A0–A6 and an oracle study, metrics, statistics, pre-specified endpoints, Kaggle compute (about 70–80 GPU-h core), and a 12-week timeline.
5. Wrote and ran `scripts/make_splits.py` on the local data. It produced `splits/folds_v1.csv`: 5 folds of 208–209 femurs, grouped by x-ray, stratified by class, with the duplicate pair merged. Class counts match the paper exactly.
6. Read the dataset's own `data.yml` and `readme.md`. Class codes are `2AB, 2CD, 3AB, 3CDE, 4`: the location digit plus grade letters, so the labels are hierarchical. `data.yml` points val to `valid/images` while the folder is `val/`. This is harmless, because we write our own YOLO data files.
7. Set up the repository: README (workflow local ↔ GitHub ↔ Kaggle), `.gitignore`, `requirements.txt`, `src/femurseg/` (class table, data finder), `kaggle/` (launcher notebook, run registry, `env_check.py`), `results/` (experiment register), and updated trackers.

**Next**
- You choose Option A, B, or A then B (Report 03 §11).
- Run `env_check` on Kaggle (Step 0) and log it in `results/EXPERIMENTS.md`.
- Step 1: data loader, metrics module and the YOLOv8-seg baseline reproduction (identical for both options).

---

## 2026-11-26 · Step 3: Guided walkthrough and dataset-usage directions

**Done**
1. Gave a full step-by-step walkthrough of the article (7 parts) and the dataset (4 parts) in the project thread.
2. Wrote `Analysis/02_How_to_Use_the_Dataset.md`: why the dataset is worth using, what to avoid, and 9 evidence-backed directions (D1–D9), cited from our EDA, the base paper and the literature review.

**Next**
- Choose the core novelty (likely among D2 segmentation-first, D3 hierarchical grading, D4 contralateral comparison, D5 label-noise-aware), plus supporting pieces D1, D7, D9.

---

## 2026-11-26 · Step 1 and Step 2: Base article reading and dataset EDA

**Done**
1. Read the base article (González et al. 2025) page by page and extracted every method detail, number, limitation and stated future work.
2. Inspected the local dataset folder `Femoral Fractures/` and decoded the label format.
3. Wrote and ran `EDA/eda_femoral.py` (works locally and on Kaggle). Produced `EDA/per_image_stats.csv`, `EDA/summary.json`, `EDA/figures/01..07_*.png`.
4. Wrote the combined report `Analysis/01_Base_Article_and_EDA_Report.md`.

**Key findings**
- Local data matches the paper's train (1043) and test (210) femur counts exactly; class counts match Table 1 exactly.
- The validation split (201 femurs) is **missing** (empty `val/`, also reported on IEEE DataPort).
- One pixel-identical image pair in train has two different masks (IoU 0.70): evidence of annotation variability.
- Test neck-fracture masks are 35–40% smaller than train ones (committee-reviewed test labels): a train/test annotation shift.
- 398 x-rays contain a fractured femur plus the healthy contralateral femur; no x-ray has two fractured femurs.
- Left and right femurs are not mirrored to one side.
- Labels: YOLO-seg polygons, class 0–4; masks store class+1; no-fracture femurs have no label file.

**Next**
- Confirm whether the Kaggle copy has `val/`, and read `data.yml` / `readme.md`.
- Combine with the literature-search thread, then move to gap analysis and idea selection.
