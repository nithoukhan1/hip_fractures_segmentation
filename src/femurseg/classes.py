"""Class definitions for the HUSJA-UA femoral fracture dataset (Gonzalez et al., Insights Imaging 2025).

The label codes come from the dataset's ``data.yml``. The digit is the anatomical location
(2 = femoral neck, 3 = pertrochanteric, 4 = subtrochanteric) and the letters are the grades
(Garden A-D = I-IV for the neck, Evans A-E = I-V for pertrochanteric fractures). So the
released labels already follow a location -> grade hierarchy.

Index conventions used everywhere in this repository:
  * ``cls6``    0..5, the 6-class index used for training. It equals the pixel value in the
                released mask PNGs (mask value = YOLO class id + 1, 0 = background).
  * ``yolo_id`` -1 for "no fracture" (no label file), else 0..4 as in data.yml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FemurClass:
    cls6: int
    yolo_id: int
    code: str
    name: str
    location: str  # none | neck | pertrochanteric | subtrochanteric
    grade: str     # none | low | high | ungraded  (low = undisplaced/stable, high = displaced/unstable)


CLASSES: tuple[FemurClass, ...] = (
    FemurClass(0, -1, "NF", "No fracture", "none", "none"),
    FemurClass(1, 0, "2AB", "Neck Garden I-II", "neck", "low"),
    FemurClass(2, 1, "2CD", "Neck Garden III-IV", "neck", "high"),
    FemurClass(3, 2, "3AB", "Pertroch Evans I-II", "pertrochanteric", "low"),
    FemurClass(4, 3, "3CDE", "Pertroch Evans III-V", "pertrochanteric", "high"),
    FemurClass(5, 4, "4", "Subtrochanteric", "subtrochanteric", "ungraded"),
)

LOCATIONS: tuple[str, ...] = ("neck", "pertrochanteric", "subtrochanteric")
BY_CLS6 = {c.cls6: c for c in CLASSES}
BY_YOLO = {c.yolo_id: c for c in CLASSES}
CLASS_NAMES = [c.name for c in CLASSES]

# Counts in the official release (paper Table 1, verified by our EDA). Used as a sanity check.
EXPECTED_COUNTS = {
    "train": {0: 363, 1: 57, 2: 254, 3: 134, 4: 199, 5: 36},
    "test": {0: 65, 1: 16, 2: 41, 3: 34, 4: 42, 5: 12},
}


def yolo_to_cls6(yolo_id: int) -> int:
    """Map a YOLO label id (-1 for no fracture) to the 6-class index."""
    if yolo_id not in BY_YOLO:
        raise ValueError(f"unknown YOLO class id {yolo_id}")
    return yolo_id + 1
