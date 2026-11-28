# Run notes: R00_reference_bank

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Paper B step 1 of Paper B (`Analysis/07_Step4_PaperB_Reference_Bank_Protocol.md`): the pair index and the table of healthy reference femurs for the 5 fold models, plus the checks of section 4 before Paper B step 2. Trains nothing.

**2. What changed compared with the previous run** (config, code commit, data, folds): First run of Paper B step 1 (orientation rule, version 1). Code `dc152a9`, one T4, 2.46 min (estimate 10). Kaggle version number not recorded.

**3. Result** (`summary.md`, `sanity.json`, `nn_fold0.png`):
- Check 1, mirror check: 473 of 810 femurs (58.4%), **below the 60% line**.
- Check 2: the 3 nearest beat a random healthy femur by .539 in cosine.
- Check 3: no duplicate.
- Check 4: held-out and test similarity are alike (nearest-reference median .57–.60 against .60–.61).
- Check 5: a femur's own partner would rank 6th (median) in the bank.
- Banks of 290–291 healthy femurs.

**4. What we learnt:** The frozen ImageNet features rank similar femurs well but hardly see left and right (ImageNet training uses random flips). So "the better orientation" was close to a coin toss, and the figure shows some references with the pelvis on the wrong side.

**5. Decision / next step:** The rule required a review of the orientation rule (Analysis/07 section 8; `results/analysis/C10_paperB_orientation_review/`).
- A pelvis-side rule read from the pixels gives the two femurs of an x-ray opposite sides in 402 of 405 x-rays.
- Version 2 is re-run as `R00_reference_bank_r2`. This folder stays as the record of version 1.
