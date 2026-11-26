"""Published results of the base article, for side-by-side tables.

González et al. Classification and segmentation of hip fractures in x-rays highlighting fracture regions for
interpretable diagnosis. Insights into Imaging 2025;16:86. doi:10.1186/s13244-025-01958-y.

Typed from the article: Table 1 (test counts), Table 2 (YOLOv8-seg grid on the test set, mean (SD) of six training
runs), Table 3 (final models and classification baselines on the test set) and Fig. 3 (confusion matrix of the final
model). tests/test_reference.py checks that the confusion matrix reproduces every number the article derives from it.

Class order everywhere: NF, 2AB (neck Garden I-II), 2CD (neck Garden III-IV), 3AB (pertrochanteric Evans I-II),
3CDE (pertrochanteric Evans III-V), 4 (subtrochanteric), i.e. our ``cls6`` 0..5.
"""
from __future__ import annotations

CITATION = "González et al., Insights Imaging 2025;16:86 (doi 10.1186/s13244-025-01958-y)"
CLASS_ORDER = ("NF", "2AB", "2CD", "3AB", "3CDE", "4")

# Table 1, test column (femurs)
TEST_COUNTS = {"NF": 65, "2AB": 16, "2CD": 41, "3AB": 34, "3CDE": 42, "4": 12}

# Table 2: test set, mean and SD of six training runs per setting. Accuracy is the 6-class accuracy (fraction).
# The article does not say how missed fractures or no-fracture femurs enter its Dice ("average Dice between
# annotated fracture regions and the regions automatically detected").
GRID = {
    ("nano", 320): {"auc": (0.978, 0.005), "accuracy_6class": (0.8111, 0.0217), "dice": (0.771, 0.023)},
    ("nano", 640): {"auc": (0.983, 0.006), "accuracy_6class": (0.8254, 0.0115), "dice": (0.778, 0.016)},
    ("nano", 1024): {"auc": (0.985, 0.009), "accuracy_6class": (0.8317, 0.0123), "dice": (0.777, 0.014)},
    ("nano", 1536): {"auc": (0.980, 0.006), "accuracy_6class": (0.8055, 0.0416), "dice": (0.767, 0.018)},
    ("small", 320): {"auc": (0.979, 0.009), "accuracy_6class": (0.7968, 0.0169), "dice": (0.777, 0.009)},
    ("small", 640): {"auc": (0.985, 0.006), "accuracy_6class": (0.8198, 0.0297), "dice": (0.780, 0.011)},
    ("small", 1024): {"auc": (0.986, 0.008), "accuracy_6class": (0.8440, 0.0251), "dice": (0.761, 0.008)},
    ("small", 1536): {"auc": (0.989, 0.003), "accuracy_6class": (0.8190, 0.0120), "dice": (0.755, 0.010)},
    ("medium", 320): {"auc": (0.980, 0.005), "accuracy_6class": (0.8068, 0.0157), "dice": (0.778, 0.014)},
    ("medium", 640): {"auc": (0.985, 0.005), "accuracy_6class": (0.8176, 0.0232), "dice": (0.774, 0.015)},
    ("medium", 1024): {"auc": (0.985, 0.003), "accuracy_6class": (0.8319, 0.0189), "dice": (0.761, 0.012)},
    ("medium", 1536): {"auc": (0.983, 0.005), "accuracy_6class": (0.8313, 0.0157), "dice": (0.765, 0.006)},
}

# Final selected model (small, 1024): of six runs, the one with the highest validation accuracy (Results, Table 3).
FINAL = {
    "auc": 0.981, "auc_ci": (0.965, 0.997),
    "fracture_accuracy": 0.962, "fracture_accuracy_ci": (0.926, 0.983),
    "fracture_sensitivity": 0.9517, "fracture_specificity": 0.9846, "fracture_ppv": 0.9928, "fracture_npv": 0.9014,
    "accuracy_6class": 0.8619, "accuracy_6class_ci": (0.8077, 0.9055),
    "dice": 0.77, "dice_ci": (0.56, 0.98),
    "errors": {"detection": 8, "location": 6, "grade": 15},  # text: 15 within-location, 6 location; rest = detection
}

# Fig. 3: confusion matrix of the final model on the 210 test femurs (rows = true class, columns = predicted)
FINAL_CONFUSION = (
    (64, 0, 1, 0, 0, 0),
    (3, 8, 3, 2, 0, 0),
    (0, 2, 39, 0, 0, 0),
    (1, 0, 0, 30, 3, 0),
    (1, 0, 1, 7, 31, 2),
    (2, 0, 0, 0, 1, 9),
)

# Table 3: other models on the test set. Accuracy = 6-class accuracy (fraction) with its 95% CI.
TABLE3 = {
    "VGG16": {"auc": 0.986, "auc_ci": (0.975, 0.998), "accuracy_6class": 0.7429, "accuracy_ci": (0.6782, 0.8005)},
    "DenseNet121": {"auc": 0.993, "auc_ci": (0.985, 1.000), "accuracy_6class": 0.8381, "accuracy_ci": (0.7812, 0.8852)},
    "EfficientNetB4": {"auc": 0.990, "auc_ci": (0.981, 0.998), "accuracy_6class": 0.7905, "accuracy_ci": (0.7291, 0.8434)},
    "InceptionV3": {"auc": 0.985, "auc_ci": (0.972, 0.998), "accuracy_6class": 0.7762, "accuracy_ci": (0.7137, 0.8307)},
    "ResNet50V2": {"auc": 0.982, "auc_ci": (0.967, 0.997), "accuracy_6class": 0.8000, "accuracy_ci": (0.7394, 0.8519)},
    "YOLOv8-seg nano 320": {"auc": 0.982, "auc_ci": (0.970, 0.995), "accuracy_6class": 0.8333,
                            "accuracy_ci": (0.7759, 0.8811), "dice": 0.78, "dice_ci": (0.57, 0.98)},
    "YOLOv8-seg small 1024": {"auc": 0.981, "auc_ci": (0.965, 0.997), "accuracy_6class": 0.8619,
                              "accuracy_ci": (0.8077, 0.9055), "dice": 0.77, "dice_ci": (0.56, 0.98)},
}


def grid(size: str, imgsz: int) -> dict:
    """Table 2 entry for a model size ('nano', 'small', 'medium') and image size (320, 640, 1024, 1536)."""
    return GRID[(size, imgsz)]
