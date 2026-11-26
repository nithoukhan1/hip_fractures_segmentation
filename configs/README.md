# configs/

One YAML file per experiment, named `<run-id>.yaml`, for example `B1_yolov8s_seg_1024.yaml` or `A6_slgnet_full.yaml`.
A config fully defines a run: model, input size, augmentations, losses, optimiser, epochs, folds to run, and the split file (`splits/folds_v1.csv`).
The training scripts copy the config into the run's results folder, so every result can be traced back to its exact settings and git commit.

Configs are added from Step 1 (baselines) onwards. See `Analysis/03_Idea_Ranking_and_Research_Plan.md`, Section 7.
