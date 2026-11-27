"""Paper B (RefFrac), step B-1: the pair index and the table of healthy reference femurs.

Protocol: Analysis/07_Step4_PaperB_Reference_Bank_Protocol.md. The rules, in short:
  * fold model k has one bank: the healthy femurs (cls6 0) of its four training folds (290 or 291 femurs). It serves
    every femur that model sees: its training femurs, its held-out fold and the test set;
  * a femur never gets a reference from its own group of the fold file (its x-ray, its partner and the duplicate
    x-ray 00262 of x-ray 00261); for held-out and test femurs this removes nothing;
  * similarity = cosine of frozen ImageNet features after subtracting the mean feature of the fold model's training
    femurs (all femur crops look alike, so the raw features share a large common part that would squeeze every
    cosine towards 1). A bank femur is compared as released and mirrored left-right, and is used in the better
    orientation; each bank femur counts once; ties go to the earlier file of the fold file;
  * kept per fold model and femur: the TOP_K nearest healthy femurs, one random healthy femur (run R2, swap test a)
    and one random fractured femur of the training folds (swap test b), both drawn with a seed made from the femur's
    file name and the fold model, in the better orientation too.
The partner (run R7) is the other femur of the same x-ray, always mirrored, and used only when it is healthy.

Pure numpy and the standard library: the features come from scripts/build_reference_bank.py.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from .data import FemurRecord

TOP_K = 5      # k = 1 (run R3) and k = 3 (RefFrac) use the first 1 and 3; 5 leaves room for a sensitivity check
N_FOLDS = 5
DUPLICATE_COS = 0.995  # a reference this similar to its femur is listed in the summary (check 3)


def pair_type(rec: FemurRecord) -> str:
    """'single', or '<healthy|fractured>, <healthy|fractured> partner' for a femur of a two-femur x-ray."""
    if not rec.pair_file:
        return "single"
    return f"{'fractured' if rec.cls6 > 0 else 'healthy'}, {'fractured' if rec.pair_cls6 > 0 else 'healthy'} partner"


def bank_records(records: list[FemurRecord], fold_model: int, healthy: bool = True) -> list[FemurRecord]:
    """The healthy (or, for swap test b, the fractured) femurs of the four training folds of ``fold_model``."""
    return [r for r in records if r.split == "train" and r.fold != fold_model and (r.cls6 == 0) == healthy]


def unit(x) -> np.ndarray:
    """Rows scaled to length 1."""
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def center_of(records: list[FemurRecord], feats: dict, fold_model: int | None = None) -> np.ndarray:
    """(D,) mean feature, over both orientations, of the training femurs of ``fold_model`` (of all 1,043 training
    femurs if None). Images only, no labels; held-out and test femurs never enter it."""
    rs = [r for r in records if r.split == "train" and (fold_model is None or r.fold != fold_model)]
    return np.mean(np.stack([np.asarray(feats[r.key], dtype=np.float64) for r in rs]), axis=(0, 1))


def similarities(target_feats, bank_feats, center=0.0) -> tuple[np.ndarray, np.ndarray]:
    """Cosine, after subtracting ``center``, of every target (as released) with every bank femur in its better
    orientation.

    target_feats: (N, D), the targets as released. bank_feats: (M, 2, D), each bank femur as released [:, 0] and
    mirrored [:, 1]. Returns (cos (N, M), mirrored (N, M) bool); a tie keeps the orientation as released.
    """
    c = np.asarray(center, dtype=np.float64)
    s = np.einsum("nd,mod->nmo", unit(np.asarray(target_feats) - c), unit(np.asarray(bank_feats) - c))
    mirrored = s[:, :, 1] > s[:, :, 0]
    return np.where(mirrored, s[:, :, 1], s[:, :, 0]), mirrored


def nearest(cos: np.ndarray, target_groups: list[str], bank_groups: list[str], k: int) -> np.ndarray:
    """(N, k) bank indices of the k most similar bank femurs outside each target's own group, most similar first."""
    masked = np.where(np.asarray(target_groups)[:, None] == np.asarray(bank_groups)[None, :], -np.inf, cos)
    order = np.argsort(-masked, axis=1, kind="stable")[:, :k]
    if np.isinf(np.take_along_axis(masked, order, axis=1)).any():
        raise ValueError(f"a femur has fewer than {k} bank femurs outside its own group")
    return order


def seeded_pick(target: FemurRecord, fold_model: int, kind: str, candidates: list[FemurRecord]) -> FemurRecord:
    """One femur of ``candidates`` outside the target's group, chosen by md5(file | fold model | kind): the same
    femur on every machine and in every run."""
    pool = [c for c in candidates if c.group != target.group]
    if not pool:
        raise ValueError(f"no {kind} candidate outside the group of {target.file}")
    h = int(hashlib.md5(f"{target.file}|{fold_model}|{kind}".encode()).hexdigest()[:12], 16)
    return pool[h % len(pool)]


def _cos2(t: np.ndarray, ref2: np.ndarray, center=0.0) -> np.ndarray:
    """(2,) cosines, after subtracting ``center``, of one target (D,) with one reference [as released, mirrored]."""
    c = np.asarray(center, dtype=np.float64)
    return unit(np.asarray(ref2) - c) @ unit(np.asarray(t) - c)


def _better(t: np.ndarray, ref2: np.ndarray, center=0.0) -> tuple[bool, float]:
    """(mirrored, cosine) of one reference (2, D) against one target (D,), in its better orientation."""
    s = _cos2(t, ref2, center)
    return bool(s[1] > s[0]), float(max(s[0], s[1]))


def reference_table(records: list[FemurRecord], feats: dict, k: int = TOP_K) -> list[dict]:
    """One row per fold model and femur (5 x 1,253 rows on the real data). ``feats`` maps a femur's (split, file) to
    its (2, D) features: [as released, mirrored]."""
    allf = np.stack([np.asarray(feats[r.key])[0] for r in records])
    groups = [r.group for r in records]
    rows = []
    for m in range(N_FOLDS):
        healthy, fractured = bank_records(records, m, True), bank_records(records, m, False)
        mu = center_of(records, feats, m)
        cos, mirrored = similarities(allf, np.stack([feats[b.key] for b in healthy]), mu)
        order = nearest(cos, groups, [b.group for b in healthy], k)
        for i, r in enumerate(records):
            role = "test" if r.split == "test" else ("held_out" if r.fold == m else "train")
            row = {"fold_model": m, "file": r.file, "split": r.split, "fold": r.fold, "group": r.group,
                   "cls6": r.cls6, "role": role}
            for j, b in enumerate(order[i], 1):
                row.update({f"ref{j}": healthy[b].file, f"ref{j}_mirrored": int(mirrored[i, b]),
                            f"ref{j}_cos": round(float(cos[i, b]), 6)})
            for kind, pool in (("random_healthy", healthy), ("random_fractured", fractured)):
                c = seeded_pick(r, m, kind, pool)
                mir, cs = _better(allf[i], np.asarray(feats[c.key]), mu)
                row.update({kind: c.file, f"{kind}_mirrored": int(mir), f"{kind}_cos": round(cs, 6)})
            rows.append(row)
    return rows


def pair_rows(records: list[FemurRecord], feats: dict | None = None) -> list[dict]:
    """The pair index: one row per femur with its partner (the other femur of its x-ray), the pair type, whether
    the partner may serve as a reference (it is healthy), and, given features, the cosine of the femur with its
    partner mirrored and as released (check 1), centred on the mean of all training femurs."""
    by_key = {r.key: r for r in records}
    mu = center_of(records, feats) if feats is not None else None
    rows = []
    for r in records:
        p = by_key.get((r.split, r.pair_file)) if r.pair_file else None
        if r.pair_file and p is None:
            raise ValueError(f"{r.file}: partner {r.pair_file} is not in the fold file")
        row = {"file": r.file, "split": r.split, "fold": r.fold, "group": r.group, "cls6": r.cls6,
               "partner": r.pair_file, "partner_cls6": p.cls6 if p else "", "pair_type": pair_type(r),
               "partner_usable": int(p is not None and p.cls6 == 0)}
        if feats is not None:
            row.update(cos_partner_mirrored="", cos_partner_as_released="")
            if p is not None:
                s = _cos2(np.asarray(feats[r.key])[0], feats[p.key], mu)  # [as released, mirrored]
                row.update(cos_partner_mirrored=round(float(s[1]), 6), cos_partner_as_released=round(float(s[0]), 6))
        rows.append(row)
    return rows


def _summary(x) -> dict:
    a = np.asarray(x, dtype=float)
    if not len(a):
        return {"n": 0}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(med), "q1": float(q1), "q3": float(q3)}


def sanity(records: list[FemurRecord], feats: dict, table: list[dict], pairs: list[dict]) -> dict:
    """The checks of the protocol (§4): 1 mirror check, 2 retrieval vs chance, 3 duplicates, 4 held-out vs test
    similarity, 5 partner rank, 6 who is in the bank. Raises if a reference is pixel-identical to its femur."""
    by_file = {r.file: r for r in records}
    out: dict = {}
    two = [p for p in pairs if p["partner"]]
    closer = sum(p["cos_partner_mirrored"] > p["cos_partner_as_released"] for p in two)
    out["1_mirror_check"] = {"femurs_with_partner": len(two), "mirrored_partner_closer": closer,
                             "share": closer / len(two) if two else float("nan"),
                             "rule": ">= 0.80 use the orientation rule as written; 0.60-0.80 use it and state it as "
                                     "a limitation; < 0.60 review the orientation rule before B-2"}
    per_model = []
    for m in range(N_FOLDS):
        held = [t for t in table if t["fold_model"] == m and t["role"] == "held_out"]
        test = [t for t in table if t["fold_model"] == m and t["role"] == "test"]
        top3 = [np.mean([t[f"ref{j}_cos"] for j in (1, 2, 3)]) for t in held]
        rnd = [t["random_healthy_cos"] for t in held]
        per_model.append({"fold_model": m, "held_out_top3_mean_cos": float(np.mean(top3)),
                          "held_out_random_cos": float(np.mean(rnd)),
                          "held_out_top1": _summary([t["ref1_cos"] for t in held]),
                          "test_top1": _summary([t["ref1_cos"] for t in test])})
    out["2_retrieval_vs_chance"] = {"per_fold_model": [{k: v for k, v in p.items() if "top1" not in k}
                                                        for p in per_model],
                                    "top3_minus_random_mean": float(np.mean([p["held_out_top3_mean_cos"]
                                                                             - p["held_out_random_cos"]
                                                                             for p in per_model]))}
    same_pixels, close = [], []
    for t in table:
        for col in [f"ref{j}" for j in range(1, TOP_K + 1) if f"ref{j}" in t] + ["random_healthy", "random_fractured"]:
            md5 = by_file[t["file"]].pixel_md5
            if md5 and by_file[t[col]].pixel_md5 == md5:
                same_pixels.append((t["fold_model"], t["file"], t[col]))
            if t[f"{col}_cos"] >= DUPLICATE_COS:
                close.append({"fold_model": t["fold_model"], "file": t["file"], "reference": t[col],
                              "cos": t[f"{col}_cos"]})
    if same_pixels:
        raise ValueError(f"references with the same pixels as their femur: {same_pixels[:5]}")
    out["3_duplicates"] = {"pixel_identical_references": 0, f"references_with_cos_ge_{DUPLICATE_COS}": close}
    out["4_held_out_vs_test_top1"] = [{"fold_model": p["fold_model"], "held_out": p["held_out_top1"],
                                       "test": p["test_top1"]} for p in per_model]
    ranks = []
    pair_by_file = {p["file"]: p for p in pairs}
    for m in range(N_FOLDS):
        healthy = bank_records(records, m, True)
        bank = np.stack([feats[b.key] for b in healthy])
        mu = center_of(records, feats, m)
        for t in table:
            p = pair_by_file[t["file"]]
            if t["fold_model"] != m or t["role"] == "train" or not p["partner_usable"]:
                continue
            target = np.asarray(feats[by_file[t["file"]].key])[0]
            cos, _ = similarities(target[None], bank, mu)
            _, own = _better(target, np.asarray(feats[by_file[p["partner"]].key]), mu)
            ranks.append(1 + int((cos[0] > own).sum()))
    out["5_partner_rank"] = {"femurs": len(ranks), "median_rank": float(np.median(ranks)) if ranks else float("nan"),
                             "share_rank_1": float(np.mean(np.asarray(ranks) == 1)) if ranks else float("nan"),
                             "share_top5": float(np.mean(np.asarray(ranks) <= 5)) if ranks else float("nan"),
                             "about": "rank the own healthy partner (in its better orientation) would have among the "
                                      "fold model's bank, for held-out and test femurs with a healthy partner; test "
                                      "femurs count once per fold model"}
    out["6_bank"] = [{"fold_model": m, "healthy_femurs": len(bank_records(records, m, True)),
                      "fractured_femurs": len(bank_records(records, m, False)),
                      "healthy_by_pair_type": dict(Counter(pair_type(b) for b in bank_records(records, m, True)))}
                     for m in range(N_FOLDS)]
    return out


def counts(records: list[FemurRecord]) -> dict:
    """Pair counts of the protocol's data table (§2), per split."""
    out = {}
    for split in ("train", "test"):
        rs = [r for r in records if r.split == split]
        out[split] = {"femurs": len(rs), "healthy": sum(r.cls6 == 0 for r in rs),
                      "fractured": sum(r.cls6 > 0 for r in rs), "by_pair_type": dict(Counter(map(pair_type, rs)))}
    return out
