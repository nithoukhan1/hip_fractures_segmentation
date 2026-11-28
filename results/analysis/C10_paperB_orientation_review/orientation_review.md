# Orientation review: review of Paper B's orientation rule (Analysis/07 section 8)

Made by `scripts/check_orientation.py` from the released images of all 1253 femurs (96 x 64 thumbnails; no label used) in 37 s.

| Test | Result |
|---|---|
| Pixel mirror check: the partner (other hip) correlates better mirrored | 796 of 810 femurs (98.3%) |
| Same check with the frozen ImageNet features (R00_reference_bank, version 1) | 473 of 810 (58.4%) |
| Pelvis side rule: the two femurs of an x-ray get opposite sides | 402 of 405 x-rays (99.3%) |
| Variant, reported only (whole image) | 383 of 405 (94.6%) |
| Variant, reported only (upper third) | 404 of 405 (99.8%) |
| Random pairs from different x-ray groups: the side rule agrees with the thumbnails | 19832 of 20000 (99.2%) |
| Pelvis on the left (all / train / test) | 49.7% / 49.6% / 50.5% |

- X-rays whose two femurs got the same side: 00031_00022.png + 00032_00022.png, 00415_00273.png + 00416_00273.png, 00456_00298.png + 00457_00298.png.
- Relative brightness difference of the two upper halves: median 0.584, 5th percentile 0.264; 13 femurs below 0.05 (side ambiguous).
