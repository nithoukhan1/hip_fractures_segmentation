# Report 05: Step 2 protocol, baselines S04–S09 and the core model M0 (S10)

**Date:** 2026-11-26 · **Code:** `scripts/train_net.py` with `src/femurseg/nets.py`, `torchdata.py`, `evaluation.py`, `runner.py` · **Runs:** `kaggle/runs.json`
**Why this report:** it fixes, before any of these runs exists, how every non-YOLO baseline and M0 is trained and scored, so that the methods section of both papers can be written from it and no choice is made after seeing results.

---

## 1. The seven runs

| Run | Network (`--net`) | Type | Why it is there |
|---|---|---|---|
| `S04_densenet121` | `cls:densenet121.tv_in1k` | Image classifier, 6 classes | The article's best classifier (test AUC .993, 6-class accuracy 83.8%, Table 3) |
| `S05_efficientnet_b4` | `cls:efficientnet_b4.ra2_in1k` | Image classifier | An article baseline (AUC .990, accuracy 79.0%) |
| `S06_convnext_tiny` | `cls:convnext_tiny.fb_in22k_ft_in1k` | Image classifier | **= A0.** Same backbone as M0 and both papers' models: separates the backbone from what the papers add |
| `S07_unet_r34` | `seg:Unet:resnet34` | Semantic segmentation, 6 classes per pixel | Standard segmentation baseline for the Dice comparison with YOLO (RQ1) |
| `S08_deeplabv3p_r34` | `seg:DeepLabV3Plus:resnet34` | Semantic segmentation | Second standard family (atrous context) |
| `S09_segformer_b2` | `seg:Segformer:mit_b2` | Semantic segmentation | Transformer segmenter |
| `S10_m0_convnext_unet` | `m0:convnext_tiny.fb_in22k_ft_in1k` | U-Net with the ConvNeXt-T encoder: binary fracture mask + 6-class head | **The core model M0 = A1 (Paper A) = R0 (Paper B)** |

**M0 in one sentence:** M0 is S06 plus a U-Net decoder. Its 6-class head is the S06 classifier's own head (global average pooling, LayerNorm, dropout, linear) on the same encoder's last feature map. `tests/test_nets.py` checks that, with the same weights, M0 and S06 give identical class logits. So **S06 vs S10 answers one question only: does learning the fracture mask as an extra task help the classification?** Paper A then changes how M0 decides (region pooling, geometry, hierarchy) and Paper B what it compares with (a healthy reference).

Weights: ImageNet weights from timm (`densenet121.tv_in1k` = the original torchvision weights; `efficientnet_b4.ra2_in1k`; `convnext_tiny.fb_in22k_ft_in1k` = the authors' ImageNet-22k pre-training fine-tuned on 1k) and from segmentation_models_pytorch 0.6.0 (`resnet34` and `mit_b2` ImageNet encoders). Decoders and heads start random.

## 2. One recipe for every network (no per-model tuning)

| Item | Setting | Reason |
|---|---|---|
| Input | 768 × 512 px: the 1024 × 1536 femur crop at exactly half size (area interpolation); grey copied to 3 channels; ImageNet normalisation | Report 03 §6; keeps the 2:3 shape; fits a T4 |
| Augmentation (training only) | Horizontal flip; affine (scale 0.9–1.1, rotation ±10°, shift ±5%); brightness/contrast ±0.2; gamma 80–120; Gaussian noise or blur (p 0.3). No vertical flip | Left and right femurs, small positioning differences, old CR images |
| Optimiser | AdamW, lr 2 × 10⁻⁴, weight decay 0.05 (none on biases and norms), gradient clipping 1.0 | Standard fine-tuning settings for all of these families |
| Schedule | 2 warm-up epochs, then cosine decay to 0; **60 epochs, final weights** | Fixed length: the held-out fold never picks an epoch (Report 04 §3.4, rule 1) |
| Batch | 16 images. If 16 do not fit in a T4, the pre-flight halves the micro-batch and accumulates gradients up to 16 | Same optimisation for every model |
| Precision | Mixed (fp16) | Speed on T4 |
| Losses | Classifiers: cross-entropy. Segmenters: pixel cross-entropy + Dice over the 5 fracture classes. M0: cross-entropy + (binary cross-entropy + Dice) of the fracture mask, weight 1 | nnU-Net-style CE + Dice for masks |
| Seeds | 2026 + fold | Recorded per fold |

The time per fold and the batch that fits are measured by the pre-flight run `S00_net_check` on Kaggle's T4s before training starts.

## 3. From network outputs to a decision (same rule as S01–S03)

Every model writes, for each femur, a fracture **score**, five class confidences, a predicted class and (if it has one) a mask. The same code (`src/femurseg/evaluation.py`) scores every model, YOLO included; it re-scores the filed S01 and S02 to exactly their published `metrics.json`.

| Model type | Fracture score | Class if a fracture is called | Mask |
|---|---|---|---|
| Classifier (S04–S06) | 1 − P(no fracture) | most probable fracture class | none (no Dice) |
| Segmenter (S07–S09) | mean of the 500 highest pixel fracture probabilities (1 − P(background)), a robust maximum | class with the largest share of the predicted region | pixels with fracture probability ≥ 0.5 |
| M0 (S10) | 1 − P(no fracture) from its head | most probable fracture class | sigmoid ≥ 0.5 |

- A femur is called fractured when its score ≥ the threshold **fitted on the pooled out-of-fold predictions (Youden; ties → higher sensitivity)**, applied unchanged to the test set. Results at the plain threshold 0.5 are reported too.
- Masks are scored at **full resolution** (1024 × 1536) against the released masks: the probability map is resized with bilinear interpolation, then thresholded. A missed fracture counts as Dice 0. NSD is reported at 2, 10 and 20 px.
- **Test set:** all 5 fold models predict it. We report mean ± SD over the 5 models and the 5-model ensemble (mean class probabilities and score; mask = mean probability map ≥ 0.5).

## 4. What gets compared, and how

| Analysis | Runs | Question |
|---|---|---|
| C02 | S01 vs S03 | Does a newer YOLO (YOLO11) change anything? |
| C03 | S04, S05, S06 vs S01 | Do image classifiers grade better than YOLO? Does our retraining reproduce the article's classifier numbers? |
| C04 | S07, S08, S09, S10 vs S01 | Which segmenter outlines fractures best (Dice, NSD@10, HD95, per location; RQ1)? |
| C05 | S06 vs S10 | Does the mask as an extra task help the classifier (Paper A, A0 → A1)? |

Each analysis = `scripts/run_stats.py` (cluster-bootstrap intervals, DeLong, McNemar, randomisation and Wilcoxon tests with Holm) + `scripts/run_breakdown.py` (per class, folds, errors, article) + `report.md`. Primary endpoints stay those fixed in Report 04: macro-F1 and Dice on fractured femurs (missed = 0). Classifiers have no Dice, so they enter only the classification comparisons.

## 5. Order of the Kaggle runs

1. `S00_net_check` (about 15 min, trains nothing): weights download, memory, speed. Its numbers set the time estimates in `kaggle/runs.json`.
2. `S06_convnext_tiny` and `S10_m0_convnext_unet` first (the two runs both papers depend on), in two sessions if the quota allows.
3. Then `S07`, `S04`, `S08`, `S09`, `S05`.

Each run trains 5 folds, two at a time on the two T4s, with the time guard of `src/femurseg/runner.py`. Outputs: `results/<RUN>/` (zipped, filed through `results/_inbox/`) and `work/<RUN>/` (weights, probability maps; kept in the Kaggle version output).

## 6. Known limitations, stated now

- One recipe for all networks, so no baseline gets tuned settings. This is the usual and fairer choice, but a baseline could do better with its own tuning. The recipe is fixed before any result.
- The YOLO baselines see 1024 px, the others 768 × 512. This is the standard input of each family; the test of resolution belongs to the sensitivity runs (Report 04, A-sens at 1152 × 768).
- The segmenters' fracture score is a design choice (top-500 mean). It is stated here, before the results, and the same for all three.
