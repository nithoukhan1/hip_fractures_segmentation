# kaggle/: running experiments on Kaggle with "Save Version"

## Files
| File | What it is |
|---|---|
| `launcher.ipynb` | The only notebook you need. Import it into Kaggle once, then change `RUN` for each run. |
| `runs.json` | The list of runs: script, arguments, extra packages, expected time. New runs are added here, not in the notebook. |
| `run.py` | Starts one run from `runs.json`, or a *parallel group* of runs (one per GPU, see below), and writes `run_meta.json` (git commit, command, runtime, exit code). It first checks that the dataset is attached (see step 5 below). |
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
4. Put the zip in `results/_inbox/` in your local folder and tell Claude, or run `python scripts/register_run.py results/_inbox/results_<RUN>.zip --version <n>`. It files the run into `results/<RUN>/`, adds a notes stub and a line to `results/EXPERIMENTS.md`. The zip of a parallel group is filed as separate runs, each with its own line. Then the same files go to the repository.
5. **If the log says `NO DATASET FOUND`**, the notebook has no dataset attached (this happened to S09's first launch on 2026-11-27). The run stops within seconds, before installing anything, and the log shows what `/kaggle/input` holds. Attach **utopianstar/dataset** (right panel → **Input** → **Add Input**) and Save Version again with the same `RUN`. The zip of such a run holds only `run_meta.json`; `register_run.py` refuses it, so rename it (for example `..._failed_no_data.zip`) and keep the name free for the real run. Every notebook you launch from needs the dataset attached, so check its Input panel first.

## Runs available now
| RUN | Step | GPU time | What it does |
|---|---|---|---|
| `S00_env_check` | 0 | ~4 min | Hardware and package report, dataset location and counts, val-empty check, split identity check, GPU speed test (done 2026-11-26: found Kaggle's val/ folder holds 104 training femurs) |
| `S00_env_check_r2` | 0 | ~4 min | Same check after pooling val/ back into train: data, folds and file list must now be identical to `splits/folds_v1.csv` |
| `S01_yolov8s_seg_1024` | 1 | 2.9 h (measured) | Baseline 1: the paper's final model, YOLOv8-small-seg at 1024 px, retrained on our 5 folds (100 epochs, Ultralytics defaults, batch 8). Two folds run at a time, one per T4. Writes out-of-fold and test predictions, `metrics.json` and `summary.md` (done 2026-11-26, register line 3) |
| `S02_yolov8s_seg_640` | 1 | 1.35 h (measured) | Baseline 2: the paper's best-Dice setting, YOLOv8-small-seg at 640 px (batch 16), same protocol (done 2026-11-26, register line 4) |
| `S03_yolo11s_seg_1024` | 2 | 2.96 h (measured) | Baseline 3: YOLO11-small-seg at 1024 px with exactly S01's protocol, so it differs from S01 only in the architecture (done 2026-11-27, register line 6) |
| `S00_net_check` | 2 | 3.5 min (measured) | Pre-flight of the 7 Step 2 networks (S04–S10) on the T4s: downloads their ImageNet weights, finds the largest micro-batch that fits (gradients are accumulated to 16), times a step, estimates minutes per fold. Trains nothing (done 2026-11-26, register line 5: all 7 fit; S05 and S09 use micro-batch 8 × 2) |
| `S04_densenet121` | 2 | ~1.6 h (S00_net_check) | Baseline 4: DenseNet121 classifier (the article's best classifier), `scripts/train_net.py`, 768 × 512 px, 60 epochs |
| `S05_efficientnet_b4` | 2 | ~2.1 h (S00_net_check) | Baseline 5: EfficientNet-B4 classifier (an article baseline) |
| `S06_convnext_tiny` | 2 | 1.46 h (measured) | Baseline 6 = A0: ConvNeXt-T classifier, the backbone of M0 and of both papers (done 2026-11-27, register line 7) |
| `S07_unet_r34` | 2 | ~1.7 h (S00_net_check) | Baseline 7: U-Net (ResNet34), 6-class semantic segmentation |
| `S08_deeplabv3p_r34` | 2 | ~1.5 h (S00_net_check) | Baseline 8: DeepLabV3+ (ResNet34) |
| `S09_segformer_b2` | 2 | ~3.3 h (S00_net_check) | Baseline 9: SegFormer (MiT-B2) |
| `S10_m0_convnext_unet` | 2 | 2.33 h (measured) | Core model M0 (= A1 = R0): S06's ConvNeXt-T + U-Net decoder (binary mask) + the same 6-class head (done 2026-11-27, register line 8) |
| `A_dev_round1` | 3 | ~55 min (estimate) | Paper A, fold-0 development round 1 (`Analysis/06_Step3_PaperA_Protocol.md` §4), a parallel group: `A6_dev_base` (SLG-Net A6 with the base settings) on GPU 0 and `A6_dev_no_tf` (question D1: no teacher forcing) on GPU 1. Fold 0 only, no test-set predictions, with the oracle test |
| `A_dev_round2` | 3 | ~55 min (estimate) | Round 2, a parallel group: `A6_dev_detach` (D2: the decision losses do not reach the mask) on GPU 0 and `A6_dev_scale32` (D3: region pooling at 1/32 only) on GPU 1 |
| `R00_reference_bank` | 4 | ~10 min (estimate) | Paper B, step B-1 (`Analysis/07_Step4_PaperB_Reference_Bank_Protocol.md`). Trains nothing and uses one T4: encodes every femur, as released and mirrored, with the frozen ImageNet weights of M0's backbone; picks each femur's healthy references from the bank of each fold model; writes the pair index and the reference table that every R run will read, the checks and a figure |

**About Kaggle's val/ folder:** the Kaggle copy moved 104 of the 1,043 training femurs into `val/`, without keeping both femurs of an x-ray together. All code pools `val/` back into train (`src/femurseg/paths.py`), so Kaggle and the local copy give the same 1,043 training femurs and the same 5 folds.

Run names: **S** = shared (baselines S1–S9, core model M0 = run S10), **A** = Paper A (SLG-Net), **R** = Paper B (RefFrac). S01–S03 use `scripts/train_yolo_seg.py`; S04–S10 and the A runs use `scripts/train_net.py` (protocols: `Analysis/05_Step2_Baselines_Protocol.md` and `Analysis/06_Step3_PaperA_Protocol.md`); R00 uses `scripts/build_reference_bank.py` (`Analysis/07_Step4_PaperB_Reference_Bank_Protocol.md`). `A6_dev_*` are fold-0 development runs of A6; the final runs will be named after their variant (A2–A6).

## Two runs in one session (parallel groups)
A development run trains one fold, so it needs only one of the session's two T4s. A *parallel group* in `runs.json` (for example `A_dev_round1`) starts two such runs at the same time, one per GPU:
1. Set `RUN` to the group's name (`A_dev_round1`), exactly like a single run, and Save Version.
2. Each run gets one GPU and half of the 4 CPU cores. The image cache is built once: the second run waits for the first, then reads the same files.
3. The log shows both runs, each line prefixed by its name (`[A6_dev_base] ...`). Each run's own log is also saved as `controller_log.txt` in its results folder.
4. Download the one zip, `results_A_dev_round1.zip`. It holds both runs; `register_run.py` files them as `results/A6_dev_base/` and `results/A6_dev_no_tf/`, with one register line each.
5. If one run fails, the other still finishes and is filed; the group's `group_meta.json` lists each run's exit code.

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
