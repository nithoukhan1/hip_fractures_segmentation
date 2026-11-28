"""Unit tests of scripts/eval_fracatlas.py (the zero-shot FracAtlas test), on CPU without downloads.

A synthetic crop folder (6 hips of 3 images, 1536 x 1024) and random fold weights of small networks: the script must
run end to end, write per_hip.csv, metrics.json and summary.md, and its metrics must equal a direct computation.

    python -B tests/test_eval_fracatlas.py
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import eval_fracatlas as EF  # noqa: E402

H, W = 1536, 1024


def make_crops(root: Path) -> Path:
    import cv2

    crops = root / "crops"
    (crops / "images").mkdir(parents=True)
    (crops / "masks").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i, (img_id, side, label) in enumerate([("a.jpg", "R", "fractured"), ("a.jpg", "L", "healthy"),
                                               ("b.jpg", "R", "healthy"), ("b.jpg", "L", "fractured"),
                                               ("c.jpg", "R", "healthy"), ("c.jpg", "L", "excluded_shaft")]):
        hip = f"{Path(img_id).stem}_{side}"
        img = (rng.random((H, W)) * 255).astype(np.uint8)
        msk = np.zeros((H, W), np.uint8)
        if label == "fractured":
            msk[400 + 50 * i:700 + 50 * i, 300:600] = 255
        if label != "excluded_shaft":
            cv2.imwrite(str(crops / "images" / f"{hip}.png"), img)
            cv2.imwrite(str(crops / "masks" / f"{hip}.png"), msk)
        rows.append({"hip": hip, "image_id": img_id, "side": side, "label": label, "group": "x", "polygons": 0,
                     "mask_pixels": int((msk > 0).sum())})
    with open(crops / "cases.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return crops


def make_run(results: Path, weights: Path, run: str, net: str, size_hw=(96, 64)) -> None:
    import torch

    from femurseg.nets import build_net, parse_net

    d = results / run
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"net": net, "size_hw": list(size_hw), "drop_rate": 0.2, "topk": 50,
                                               "no_amp": True}))
    (d / "metrics.json").write_text(json.dumps({"thresholds": {"oof_youden": 0.5}}))
    (weights / run).mkdir(parents=True)
    for k in range(EF.N_FOLDS):
        torch.manual_seed(k)
        model = build_net(parse_net(net), pretrained=False)
        torch.save(model.state_dict(), weights / run / f"fold{k}.pt")


def test_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crops = make_crops(root)
        results, weights, out = root / "results", root / "weights", root / "out"
        make_run(results, weights, "M_check", "m0:convnext_atto")
        make_run(results, weights, "C_check", "cls:resnet18")
        rc = EF.main(["M_check", "C_check", "--crops", str(crops), "--weights", str(weights), "--results",
                      str(results), "--out", str(out), "--gpus", "cpu", "--batch", "2", "--n-boot", "50"])
        assert rc == 0
        rep = json.loads((out / "metrics.json").read_text())
        assert rep["done"] == ["M_check", "C_check"] and rep["n_hips"] == 5, rep
        with open(out / "M_check" / "per_hip.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5 and {r["hip"] for r in rows} == {"a_R", "a_L", "b_R", "b_L", "c_R"}
        for r in rows:  # ensemble score = mean of the fold scores; called = score >= the run's threshold
            fs = [float(r[f"score_fold{k}"]) for k in range(EF.N_FOLDS)]
            assert abs(float(r["score"]) - np.mean(fs)) < 1e-6
            assert int(r["called"]) == int(float(r["score"]) >= 0.5)
        fr = [r for r in rows if r["label"] == "fractured"]
        assert all(r["dice_top"] != "" and r["pointing_hit"] in ("0", "1") for r in fr)
        assert len(list((out / "M_check" / "masks").glob("*.png"))) == 5
        m = rep["runs"]["M_check"]
        y = np.array([int(r["y"]) for r in rows])
        called = np.array([int(r["called"]) for r in rows])
        assert m["n"] == 5 and m["n_fractured"] == 2 and m["tp"] == int((called & y).sum())
        dice = np.mean([float(r["dice_top"]) * int(r["called"]) for r in fr])
        assert abs(m["dice"]["estimate"] - dice) < 1e-9
        assert "dice" not in rep["runs"]["C_check"]  # a classifier has no mask
        summary = (out / "summary.md").read_text()
        assert "M_check" in summary and "C_check" in summary
        # --summarise rebuilds the same tables from the CSVs
        assert EF.main(["--summarise", str(out), "--n-boot", "50"]) == 0
        again = json.loads((out / "metrics.json").read_text())
        assert again["runs"]["M_check"]["tp"] == m["tp"]


def test_summarise_rows() -> None:
    rows = [{"hip": f"h{i}", "image_id": f"i{i // 2}", "y": y, "score": s, "called": int(s >= 0.5), "threshold": 0.5,
             "dice_top": d, "iou_top": d / 2 if d != "" else "", "hd95_top": 10.0, "pointing_hit": 1 if y else ""}
            for i, (y, s, d) in enumerate([(1, 0.9, 0.8), (1, 0.4, 0.6), (0, 0.6, ""), (0, 0.1, ""), (0, 0.2, "")])]
    for r in rows:
        if r["y"] == 0:
            del r["dice_top"], r["iou_top"], r["hd95_top"]
    m = EF.summarise_rows(rows, n_boot=50)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 2, 1)
    assert abs(m["sensitivity"] - 0.5) < 1e-12 and abs(m["specificity"] - 2 / 3) < 1e-12
    assert abs(m["precision"] - 0.5) < 1e-12 and abs(m["f1"] - 0.5) < 1e-12
    assert abs(m["dice"]["estimate"] - 0.4) < 1e-12        # the missed fracture counts 0: (0.8 + 0) / 2
    assert abs(m["auc"]["auc"] - 5 / 6) < 1e-12 and m["pointing_game"] == {"hits": 2, "n": 2}


def test_patient_groups() -> None:
    g = EF.patient_groups(["IMG0000010.jpg", "IMG0000012.jpg", "IMG0000015.jpg", "IMG0000016.jpg", "IMG0000100.jpg"])
    assert g == {"IMG0000010.jpg": "IMG0000010.jpg", "IMG0000012.jpg": "IMG0000010.jpg",
                 "IMG0000015.jpg": "IMG0000015.jpg", "IMG0000016.jpg": "IMG0000015.jpg",
                 "IMG0000100.jpg": "IMG0000100.jpg"}


if __name__ == "__main__":
    test_patient_groups()
    print("ok  patient_groups")
    test_summarise_rows()
    print("ok  summarise_rows")
    test_end_to_end()
    print("ok  end to end")
