# FracAtlas external test set: landmarks and cases

The rules are in `Analysis/11_New_Direction_Metrics_FracAtlas_Method_Plan.md`, section 8.

| File | What it holds |
|---|---|
| `landmarks.csv` | One row per included hip. It gives the pelvic midline x of the image and the centre of the femoral head (side R = the image's left), as fractions of the image width and height. They were read by eye, without the fracture outlines. `order_group` says whether the image is one of the 22 fractured images or one of the 60 healthy images |
| `excluded.csv` | Images or sides left out while screening, each with its reason |
| `cases.csv` | Written by `scripts/make_fracatlas_crops.py`. One row per hip: label (fractured or healthy), number of fracture polygons, mask pixels, crop box in the original image, scale, and the share of the crop outside the X-ray |

The crops themselves (`FracAtlas_crops\images` and `FracAtlas_crops\masks` on the PC, 1024 × 1536 PNG) are made from the FracAtlas data and are not committed. To make them again, run:

    python scripts/make_fracatlas_crops.py --fracatlas "FracAtlas (FigShare)/FracAtlas" --out FracAtlas_crops

Totals: 148 hips from 82 images, 23 fractured and 125 healthy.
