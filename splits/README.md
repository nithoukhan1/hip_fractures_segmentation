# splits/

## Why this folder exists
The official release has an **empty `val/` folder**, so the paper's 201 validation femurs are not available. We keep the **official test set (210 femurs) untouched** and build validation folds from the 1,043 training femurs.

## `folds_v1.csv` (made by `scripts/make_splits.py`, seed 2026, 5 folds)
- One row per femur (train and test). `fold` = 0–4 for train, and −1 for test.
- **Grouped by x-ray ID:** both femurs of one radiograph are always in the same fold.
- **Duplicate merged:** the pixel-identical pair `00397_00261.png` / `00399_00262.png` (masks IoU 0.70) is in one group (fold 0).
- **Stratified by the 6 classes.** Every fold holds 208–209 femurs from 139–141 x-rays:

| fold | NF | 2AB (Garden I–II) | 2CD (Garden III–IV) | 3AB (Evans I–II) | 3CDE (Evans III–V) | 4 (Subtroch) | total | x-rays |
|---|---|---|---|---|---|---|---|---|
| 0 | 72 | 11 | 51 | 27 | 40 | 7 | 208 | 140 |
| 1 | 73 | 11 | 51 | 27 | 39 | 8 | 209 | 141 |
| 2 | 73 | 12 | 51 | 26 | 40 | 7 | 209 | 141 |
| 3 | 72 | 12 | 51 | 27 | 40 | 7 | 209 | 141 |
| 4 | 73 | 11 | 50 | 27 | 40 | 7 | 208 | 139 |
| test | 65 | 16 | 41 | 34 | 42 | 12 | 210 | 146 |

Each fold also holds 66–68 fractured femurs whose healthy contralateral femur is present (test: 64). This was not forced; it happened, and it matters for Option B.

## Columns
`file, split, fold, femur_id, xray_id, group, cls6 (0 = no fracture, 1–5 = mask value), yolo_id, code (data.yml), class_name, location, grade, n_femurs_in_xray, pair_file, pair_cls6, duplicate_of, pixel_md5`

## Integrity
`folds_v1_summary.json` holds a fingerprint (SHA-1 over file names, classes and pixel hashes): `733a70aa5c970be9ae3e76d65c69636dbc0ecb5a`. The Kaggle `env_check` run recomputes it on Kaggle's copy of the data, so we know both copies are identical.

## Known limitation
Patient IDs are not released, so two different x-rays of the same patient could fall into different folds. This affects validation (model selection) only, not the official test set, which the authors split by patient. It will be stated in the paper.
