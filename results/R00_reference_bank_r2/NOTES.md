# Run notes: R00_reference_bank_r2

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B step 1 of Paper B again, with version 2 of the orientation rule (`Analysis/07` section 8): a reference is mirrored when its pelvis side, read from the image, differs from the femur's. Expected: check 1 = 804 of 810 partners mirrored (the prediction of the orientation review).

**2. What changed compared with the previous run** (config, code commit, data, folds):
- Only the orientation rule (`refbank.pelvis_side`, `refbank.orientation`), compared with `R00_reference_bank` (register line 17). Same script, bank, features, centring, k and seeds.
- Check 1 now counts the partners the rule mirrors; check 5 uses the partner mirrored.
- Code `d3d73af` (main after PR #11), one T4, 2.65 min. Trains nothing. Kaggle version number not recorded.

**3. Result** (`summary.md`, `sanity.json`, `nn_fold0.png`):
- Check 1: **804 of 810 (99.3%)**, exactly as predicted. With the ImageNet features alone (version 1), still 473 (58.4%), reported only.
- Check 2: the 3 nearest beat a random healthy femur by +.547 cosine (mean of the 5 fold models).
- Check 3: no duplicates. Check 5: the own partner, mirrored, would rank 6th (median) in the bank; rank 1 for 25.1%.
- The figure shows every reference with the pelvis on the target's side (checked by eye).

**4. What we learnt:**
- Compared with version 1 (6,265 rows of `references.csv`), the nearest reference is the same femur in 87.1% of rows, but in the same orientation in only 57.6%; the 5 references overlap by 89.5%.
- So the rule changed mainly the orientation of the references, not which femurs are retrieved, as expected from the orientation review.

**5. Decision / next step:** Check 1 passes (≥ 80%), so step 1 of Paper B is complete (`Analysis/07` section 9). Every R run reads this run's `pairs.csv` and `references.csv`. Next: the Paper B step 2 protocol (the Siamese RefFrac model, the eight reference runs), then fold-0 development.
