# S00_net_check: network pre-flight

Input 768 x 512, effective batch 16, 60 epochs. Cache built in 46 s. Estimated run time = 3 rounds of folds (5 folds, 2 at a time) x minutes per fold.

| Network | OK | Params (M) | Build + download (s) | Micro-batch x accumulation | Peak GPU memory (GB) | s per micro-step | Est. min per fold | Est. min per run |
|---|---|---|---|---|---|---|---|---|
| `cls:densenet121.tv_in1k` | yes | 6.96 | 4.2 | 16 x 1 | 8.48 | 0.4813 | 28.6 | 86 |
| `cls:efficientnet_b4.ra2_in1k` | yes | 17.56 | 5.2 | 8 x 2 | 7.48 | 0.3387 | 39.8 | 119 |
| `cls:convnext_tiny.fb_in22k_ft_in1k` | yes | 27.82 | 4.7 | 16 x 1 | 8.3 | 0.508 | 30.1 | 90 |
| `seg:Unet:resnet34` | yes | 24.44 | 3.9 | 16 x 1 | 8.38 | 0.5252 | 31.5 | 94 |
| `seg:DeepLabV3Plus:resnet34` | yes | 22.44 | 2.4 | 16 x 1 | 6.25 | 0.447 | 27.0 | 81 |
| `seg:Segformer:mit_b2` | yes | 24.72 | 4.5 | 8 x 2 | 10.58 | 0.5401 | 63.3 | 190 |
| `m0:convnext_tiny.fb_in22k_ft_in1k` | yes | 31.93 | 3.5 | 16 x 1 | 11.19 | 0.9139 | 53.8 | 161 |
