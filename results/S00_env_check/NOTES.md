# Run notes: S00_env_check

*Filled in after each run. Keep it short: 5–10 lines are enough. Numbers come from this folder's files.*

**1. Why (purpose and hypothesis):** Step 0. Before any training, check Kaggle's GPU and packages, and confirm that the Kaggle copy of the data and our folds are identical to the local copy (`splits/folds_v1.csv`).

**2. What changed compared with the previous run** (config, code commit, data, folds): First run. Code at `37d8d0f` (main after PR #1), notebook `hipfx-launcher`, accelerator T4 x2, Internet on. The Kaggle version number was not recorded. The cells started a few seconds apart, which suggests they were run by hand in the editor rather than through Save Version.

**3. Result** (from `env_report.json`, `folds_check.csv`, `make_splits_stdout.txt`):
- Hardware: 2 × Tesla T4 (15.6 GB each), fp16 matmul 24.0 and 22.6 TFLOPs; 4 CPU cores; 32 GB RAM; 20.9 GB free in /kaggle/working.
- Software: Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0, timm 1.0.30, albumentations 2.0.8, scikit-learn 1.6.1, OpenCV 4.14. **Not installed:** ultralytics, segmentation-models-pytorch, monai. GitHub and PyPI reachable.
- Data root `/kaggle/input/datasets/utopianstar/dataset/femoral/data`; images 1024 × 1536 grayscale; reading one image takes 98 ms (local PC: 17 ms).
- Folders on Kaggle: train 939 images (612 labels), **val 104 (68)**, test 210 (145). The local copy has train 1,043 (680), val 0, test 210 (145).
- Split check: same data **False**, same folds **False** (868 differences), because the split script then read only train/ and test/.

**4. What we learnt:**
- File-by-file comparison of `folds_check.csv` with `splits/folds_v1.csv`: all 939 Kaggle train images and all 210 test images are pixel-identical to the local copy, with the same classes.
- The 104 local training femurs missing from Kaggle's train/ match Kaggle's val/ in count and labels (104 images, 68 fractured; NF 36, 2AB 7, 2CD 31, 3AB 10, 3CDE 17, subtrochanteric 3). So Kaggle's val/ is part of the official training set, **not** the paper's missing 201-femur validation set.
- Kaggle's val/ does not keep x-rays together: 57 of the 101 x-rays it touches have their other femur in train/. Used as a validation set, it would leak.
- Kaggle's `data.yml` was edited (descriptive class names, different path).
- Image reads from /kaggle/input are slow: training scripts should load all 1,253 images into RAM once per session (about 2 GB of 32 GB).
- YOLO baselines need `ultralytics` installed through `kaggle/runs.json`; U-Net and DeepLab baselines need `segmentation-models-pytorch`.
- Two GPUs per session: one launch can run two jobs at once, one per GPU.

**5. Decision / next step:**
- All code now pools val/ (and valid/) back into train (`src/femurseg/paths.py`), so every copy gives the same 1,043 training femurs and the same folds. Tested on the local data (CSV byte-identical to `splits/folds_v1.csv`) and on a copy of Kaggle's exact folder layout (same data True, same folds True, all files found True).
- Run `S00_env_check_r2` on Kaggle to confirm with the real Kaggle files. Use Save Version for every run from now on.
