# kaggle/: running experiments on Kaggle with "Save Version"

## Files
| File | What it is |
|---|---|
| `launcher.ipynb` | The only notebook you need. Import it into Kaggle once, then change `RUN` for each run. |
| `runs.json` | The list of runs: script, arguments, extra packages, expected time. New runs are added here, not in the notebook. |
| `run.py` | Starts one run from `runs.json`, writes `run_meta.json` (git commit, command, runtime, exit code). |
| `env_check.py` | Step 0 (`S00_env_check`): checks GPU, packages, internet and data, and confirms the Kaggle data and folds match `splits/folds_v1.csv`. |

## One-time set-up (about 5 minutes)
1. On Kaggle: **Create → New Notebook**, then **File → Import Notebook** and upload `kaggle/launcher.ipynb` from this repository.
2. **Add Input** (right panel) → search for and attach the dataset **utopianstar/dataset**.
3. **Settings** (right panel):
   - **Accelerator:** GPU (T4 x2 or P100).
   - **Internet:** On. This needs a phone-verified Kaggle account.
4. Rename the notebook, for example `hipfx-launcher`.

## Every run (step by step)
1. In the settings cell, set `RUN` to a name from `runs.json` (first time: `S00_env_check`). Keep `BRANCH = "main"` (before the setup pull request is merged, use `claude/project-thread-kges8j`), or use a tag such as `A-v1-frozen` for final runs.
2. Click **Save Version** → choose **Save & Run All (Commit)** → **Save**. Kaggle runs the whole notebook in the background (up to 12 hours) and you can close the browser.
3. When the version shows *Complete*, open it → **Output** → download `results_<RUN>.zip`.
4. Put the zip in `results/_inbox/` in your local folder and tell Claude, or run `python scripts/register_run.py results/_inbox/results_<RUN>.zip --version <n>`. It files the run into `results/<RUN>/`, adds a notes stub and a line to `results/EXPERIMENTS.md`. Then the same files go to the repository.

## Runs available now
| RUN | Step | GPU time | What it does |
|---|---|---|---|
| `S00_env_check` | 0 | ~4 min | Hardware and package report, dataset location and counts, val-empty check, split identity check, GPU speed test (done 2026-11-26: found Kaggle's val/ folder holds 104 training femurs) |
| `S00_env_check_r2` | 0 | ~4 min | Same check after pooling val/ back into train: data, folds and file list must now be identical to `splits/folds_v1.csv` |
| `S01_yolov8s_seg_1024` | 1 | 2.9 h (measured) | Baseline 1: the paper's final model, YOLOv8-small-seg at 1024 px, retrained on our 5 folds (100 epochs, Ultralytics defaults, batch 8). Two folds run at a time, one per T4. Writes out-of-fold and test predictions, `metrics.json` and `summary.md` (done 2026-11-26, register line 3) |
| `S02_yolov8s_seg_640` | 1 | 1.35 h (measured) | Baseline 2: the paper's best-Dice setting, YOLOv8-small-seg at 640 px (batch 16), same protocol (done 2026-11-26, register line 4) |
| `S03_yolo11s_seg_1024` | 2 | ~3 h | Baseline 3: YOLO11-small-seg at 1024 px with exactly S01's protocol, so it differs from S01 only in the architecture |
| `S00_net_check` | 2 | ~15 min | Pre-flight of the 7 Step 2 networks (S04–S10) on the T4s: downloads their ImageNet weights, finds the largest micro-batch that fits (gradients are accumulated to 16), times a step, estimates minutes per fold. Trains nothing. Run it before S04–S10 |
| `S04_densenet121` | 2 | ~1–1.5 h (to be measured) | Baseline 4: DenseNet121 classifier (the article's best classifier), `scripts/train_net.py`, 768 × 512 px, 60 epochs |
| `S05_efficientnet_b4` | 2 | ~1.5 h (to be measured) | Baseline 5: EfficientNet-B4 classifier (an article baseline) |
| `S06_convnext_tiny` | 2 | ~1–1.5 h (to be measured) | Baseline 6 = A0: ConvNeXt-T classifier, the backbone of M0 and of both papers |
| `S07_unet_r34` | 2 | ~1.5 h (to be measured) | Baseline 7: U-Net (ResNet34), 6-class semantic segmentation |
| `S08_deeplabv3p_r34` | 2 | ~1.5 h (to be measured) | Baseline 8: DeepLabV3+ (ResNet34) |
| `S09_segformer_b2` | 2 | ~1.5 h (to be measured) | Baseline 9: SegFormer (MiT-B2) |
| `S10_m0_convnext_unet` | 2 | ~2 h (to be measured) | Core model M0 (= A1 = R0): S06's ConvNeXt-T + U-Net decoder (binary mask) + the same 6-class head |

**About Kaggle's val/ folder:** the Kaggle copy moved 104 of the 1,043 training femurs into `val/`, without keeping both femurs of an x-ray together. All code pools `val/` back into train (`src/femurseg/paths.py`), so Kaggle and the local copy give the same 1,043 training femurs and the same 5 folds.

Run names: **S** = shared (baselines S1–S9, core model M0 = run S10), **A** = Paper A (SLG-Net), **R** = Paper B (RefFrac). S01–S03 use `scripts/train_yolo_seg.py`; S04–S10 use `scripts/train_net.py` (protocol: `Analysis/05_Step2_Baselines_Protocol.md`). Until the Step 2 pull request is merged, launch S00_net_check with `BRANCH = "claude/project-thread-kges8j"`.

## Good habits that save quota
- Kaggle's free GPU quota is commonly reported as about 30 h per week, with 12 h per session. Your exact remaining quota is shown in your Kaggle account settings.
- One Save Version run trains all 5 folds, two at a time on the two T4s. A time guard does not start a fold that could
  push the session past 11 hours; finished folds are kept, and skipped folds are listed in `metrics.json`. They are
  then run in a continuation run (same script with `--folds`, added to `runs.json` under a new name such as `S01_..._b`).
- Big files (weights `last.pt`, predicted masks) stay in the version's Output under `work/<RUN>/`; only
  `results/<RUN>/` is zipped and committed.
- Use fold 0 for development. Run all 5 folds only for frozen configurations.
- Never tune anything on the test set. Fixed baselines (S01–S10) are frozen by definition, so they predict the test
  set; their fracture threshold is fitted on the out-of-fold predictions only.
- Two sessions can run at the same time on one account (S01 and S02 did). Kaggle allows one account per person, so do
  not spread runs over a second account of your own.
