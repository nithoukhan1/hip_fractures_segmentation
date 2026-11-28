# Run notes: S00_env_check_r2

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Confirm on the real Kaggle files that pooling Kaggle's val/ folder (104 femurs) back into train gives exactly the official 1,043 training femurs and our folds. Expected: same data True, same folds True, all files found True.

**2. What changed compared with the previous run** (config, code commit, data, folds): Code `2758d45` (main after PR #2): `src/femurseg/paths.py` pools val/ into train; `make_splits.py` and `env_check.py` use it. Same notebook, T4 x2, Internet on. The Kaggle version number was not recorded; the cell start times (seconds to minutes apart) again suggest the cells were run by hand in the editor.

**3. Result** (from `env_report.json`, `folds_check.csv`):
- Same data **True** (fingerprint `733a70aa…` on Kaggle and locally), same folds **True** (0 differences), all files found **True** (0 missing, 0 extra), class counts match the paper's Table 1.
- `folds_check.csv` written on Kaggle is byte-identical to `splits/folds_v1.csv`.
- 104 val images pooled, training pool 1,043. GPUs 2 × T4, fp16 23.7 and 23.5 TFLOPs. Image read 91 ms.

**4. What we learnt:** The hypothesis held. The 104 Kaggle val images are pixel- and label-identical to the corresponding official training femurs, so Kaggle and the local copy now give the same data and folds.

**5. Decision / next step:** Step 0 is complete (milestone ConvNeXt U-Net). Start Step 1: data and metrics modules, then the YOLO reproduction runs YOLOv8s-seg and YOLOv8s-seg at 640 px. Use Save Version for those, because they take hours.
