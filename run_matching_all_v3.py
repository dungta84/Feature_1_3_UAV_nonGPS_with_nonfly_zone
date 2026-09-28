# -*- coding: utf-8 -*-
# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
"""
run_matching_all_v3.py

Candidate selection by vertex voting, built on three properties taken from
the paper.

(A) The descriptor is only used where it is defined as in the paper.
    Theorem 5.4 assumes U(x*) inside Omega. For an aerial crop this means the
    disk of radius R around a vertex must lie inside the crop; vertices closer
    than R to the crop border have a truncated descriptor and are not used.

(B) Distance that can only grow under refinement.
    d_N(v, u) = sum_i |w_i(v) - w_i(u)| = || R_N (g_v - g_u) ||_{L^1}
    (cell areas in px^2, no normalization). If P' refines P then
    R_P = R_P R_P' and R_P is an L^1 contraction (Lemma 5.3), hence
    d_P <= d_P'. With a fixed tolerance eps, the vertex candidate set
        C_N(v) = { reference vertices u : d_N(v, u) <= eps }
    can only shrink when the partition is refined (Corollary 5.5 in
    tolerance form).

(C) Object candidates by majority vote.
    A reference object o is a candidate for an aerial object q if at least a
    fraction RHO of the valid vertices of q have a vertex candidate in o:
        vote_N(q, o) = #{ v in q : C_N(v) meets o } / #{ valid v in q } >= RHO.
    Each C_N(v) shrinks under refinement, so vote_N(q, o) is non-increasing
    and the object candidate set also shrinks. This is guaranteed along k
    (k -> k+1 halves every sector, nested partitions); along n the ring radii
    change, so there the decrease is observed, not proved.

eps is fixed once for all (n, k): the smallest value for which every true
object stays a candidate at every resolution (so that a smaller candidate set
never comes from dropping the correct object). It is calibrated on the test
set itself, since no separate calibration set exists; sweep_eps.csv shows
other values.

The script also prints the mean distance from a query to its true object and
to the other objects. Under refinement both grow towards their limits
||g_v - g_u||_{L^1} (Theorem 5.4); the size of that growth bounds how much any
fixed-tolerance rule can reduce the candidate set on a given data set.

Outputs in results_v3/: per_query.csv, per_test.csv, summary_nk.csv,
sweep_eps.csv, separation.csv.
"""
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from shapely.affinity import translate
from shapely.geometry import Polygon

# ===================== CONFIGURATION (same for every n, k) =====================

REFERENCE_DIR = os.path.join("data", "references")
AERIAL_FEATURE_DIR = os.path.join("data", "aerials", "features_grid")
TEST_META_CSV = os.path.join("data", "aerials", "test_meta.csv")
OUT_DIR = "results_v3"

N_LIST = [3, 4, 5]
K_LIST = [3, 4, 5]

PX_PER_METER = 2.445496
R_MAX_M = 6.011
R_PX = R_MAX_M * PX_PER_METER
DISK_AREA_PX = np.pi * R_PX ** 2

RHO = 0.5                   # majority of the valid vertices of the query
MIN_VALID_VERTICES = 3      # queries with fewer valid vertices are skipped
GT_MIN_IOU = 0.50
ALPHA_SWEEP = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
# [COCH-V] vertices where the shape vector is used:
#   "coch_extreme": extreme (convex) vertices of the COCH of each building (paper);
#   "all": every vertex of the COCH polygon, staircase corners included (earlier runs).
# The polygon itself is always rebuilt from all its vertices.
VERTEX_MODE = "coch_extreme"


# ===================== LOADING =====================

def _descriptor_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.match(r"^r[\d.]+_d\d+$", c)]

    def key(c):
        m = re.match(r"^r([\d.]+)_d(\d+)$", c)
        return (float(m.group(1)), int(m.group(2)))

    return sorted(cols, key=key)


def _read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _objects(df: pd.DataFrame) -> Dict[int, Dict]:
    cols = _descriptor_cols(df)
    objs = {}
    for oid, g in df.groupby("object_id"):
        g = g.sort_values("vertex_id")
        xy = g[["x", "y"]].to_numpy(dtype=np.float64)
        W = np.nan_to_num(g[cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if VERTEX_MODE in ("coch_extreme", "och_extreme"):
            if "vertex_type" not in g.columns:
                raise ValueError("VERTEX_MODE='coch_extreme' needs the vertex_type column; rerun run_examples.py")
            use = (g["vertex_type"].astype(str) == "convex").to_numpy()
        else:
            use = np.ones(len(g), dtype=bool)
        objs[int(oid)] = {"xy": xy, "W": W, "poly": poly, "use": use}
    return objs


def load_meta() -> pd.DataFrame:
    return pd.read_csv(TEST_META_CSV, encoding="utf-8-sig").set_index("file_name")


def ground_truth_object(q_poly: Polygon, offset: Tuple[float, float], ref: Dict[int, Dict]) -> Tuple[int, float]:
    shifted = translate(q_poly, xoff=offset[0], yoff=offset[1])
    best_oid, best_iou = -1, 0.0
    for oid, o in ref.items():
        try:
            inter = shifted.intersection(o["poly"]).area
            union = shifted.union(o["poly"]).area
        except Exception:
            continue
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_oid, best_iou = oid, iou
    return best_oid, best_iou


# ===================== PER (n, k) =====================

def collect(n: int, k: int, meta: pd.DataFrame) -> List[Dict]:
    """For every query object: nearest descriptor distance of each valid vertex to each reference object."""
    ref = _objects(_read(os.path.join(REFERENCE_DIR, f"reference_n{n}_k{k}.csv")))
    aer = _read(os.path.join(AERIAL_FEATURE_DIR, f"aerial_n{n}_k{k}.csv"))
    queries = []
    for afile, g in aer.groupby("aerial_file"):
        afile = str(afile)
        if afile not in meta.index:
            continue
        w, h = float(meta.loc[afile, "crop_w"]), float(meta.loc[afile, "crop_h"])
        off = (float(meta.loc[afile, "crop_x1"]), float(meta.loc[afile, "crop_y1"]))
        for qid, q in _objects(g).items():
            gt, iou = ground_truth_object(q["poly"], off, ref)
            if gt < 0 or iou < GT_MIN_IOU:
                continue
            x, y = q["xy"][:, 0], q["xy"][:, 1]
            valid = (x >= R_PX) & (y >= R_PX) & (x <= w - 1 - R_PX) & (y <= h - 1 - R_PX)   # (A)
            valid &= q["use"]                                                                  # [COCH-V]
            if valid.sum() < MIN_VALID_VERTICES:
                continue
            dmin, offs = {}, {}
            for oid, o in ref.items():
                oW, oxy = o["W"][o["use"]], o["xy"][o["use"]]                               # [COCH-V]
                d = cdist(q["W"][valid], oW, metric="cityblock")                             # (B)
                j = np.argmin(d, axis=1)
                dmin[oid] = d[np.arange(d.shape[0]), j]
                offs[oid] = np.median(oxy[j] - q["xy"][valid], axis=0)
            queries.append({"n": n, "k": k, "aerial_file": afile, "query_object": qid,
                            "gt_object": gt, "gt_iou": iou, "valid_vertices": int(valid.sum()),
                            "dmin": dmin, "offset": offs, "true_offset": off})
    return queries


def votes(q: Dict, eps: float) -> Dict[int, float]:
    return {o: float(np.mean(d <= eps)) for o, d in q["dmin"].items()}                        # (C)


def min_eps_keeping_gt(q: Dict) -> float:
    """Smallest eps with vote(q, true object) >= RHO."""
    d = np.sort(q["dmin"][q["gt_object"]])
    m = int(np.ceil(RHO * len(d)))
    return float(d[m - 1])


# ===================== MAIN =====================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = load_meta()
    data = {(n, k): collect(n, k, meta) for n in N_LIST for k in K_LIST}
    for (n, k), qs in data.items():
        print(f"[OK] n={n}, k={k}: {len(qs)} queries")

    if not any(data.values()):
        print("No query matches a reference building (IoU >= %.2f with >= %d valid vertices): "
              "nothing to evaluate. Check the building polygons (masks_overlay.png)." % (GT_MIN_IOU, MIN_VALID_VERTICES))
        return
    eps = max(min_eps_keeping_gt(q) for qs in data.values() for q in qs)
    print(f"vertex mode: {VERTEX_MODE}")
    pd.DataFrame([{"vertex_mode": VERTEX_MODE, "eps_px2": eps, "eps_over_disk": eps / DISK_AREA_PX,
                   "R_px": R_PX, "rho": RHO}]).to_csv(os.path.join(OUT_DIR, "run_info.csv"), index=False)
    print(f"\neps = {eps:.2f} px^2 = {eps / DISK_AREA_PX:.4f} x disk area, RHO = {RHO}")

    rows = []
    for (n, k), qs in data.items():
        for q in qs:
            v = votes(q, eps)
            cands = sorted([o for o, s in v.items() if s >= RHO], key=lambda o: -v[o])
            top1 = max(v, key=lambda o: (v[o], -np.median(q["dmin"][o])))
            rows.append({
                "n": n, "k": k, "aerial_file": q["aerial_file"], "query_object": q["query_object"],
                "gt_object": q["gt_object"], "gt_iou": round(q["gt_iou"], 4),
                "valid_vertices": q["valid_vertices"],
                "vote_gt": round(v[q["gt_object"]], 3),
                "num_candidates": len(cands), "gt_in_candidates": int(q["gt_object"] in cands),
                "candidates": ",".join(map(str, cands)),
                "top1_object": top1, "top1_correct": int(top1 == q["gt_object"]),
                "offset_x": float(q["offset"][top1][0]), "offset_y": float(q["offset"][top1][1]),
                "true_x": q["true_offset"][0], "true_y": q["true_offset"][1],
            })
    pq = pd.DataFrame(rows).sort_values(["n", "k", "aerial_file", "query_object"])
    pq.to_csv(os.path.join(OUT_DIR, "per_query.csv"), index=False, encoding="utf-8-sig")

    tests = []
    for (n, k, afile), g in pq.groupby(["n", "k", "aerial_file"]):
        ex, ey = float(np.median(g.offset_x)), float(np.median(g.offset_y))
        tx, ty = float(g.true_x.iloc[0]), float(g.true_y.iloc[0])
        err = float(np.hypot(ex - tx, ey - ty))
        tests.append({"n": n, "k": k, "aerial_file": afile, "queries": len(g),
                      "candidates": " | ".join(g.candidates), "mean_num_candidates": round(g.num_candidates.mean(), 3),
                      "est_x": ex, "est_y": ey, "true_x": tx, "true_y": ty,
                      "error_px": round(err, 2), "error_m": round(err / PX_PER_METER, 2)})
    pd.DataFrame(tests).to_csv(os.path.join(OUT_DIR, "per_test.csv"), index=False, encoding="utf-8-sig")

    sm = pq.groupby(["n", "k"]).agg(queries=("num_candidates", "size"),
                                    mean_candidates=("num_candidates", "mean"),
                                    recall=("gt_in_candidates", "mean"),
                                    top1_acc=("top1_correct", "mean")).reset_index().round(3)
    sm.to_csv(os.path.join(OUT_DIR, "summary_nk.csv"), index=False, encoding="utf-8-sig")

    sweep = []
    for alpha in ALPHA_SWEEP:
        e = alpha * DISK_AREA_PX
        for (n, k), qs in data.items():
            vs = [votes(q, e) for q in qs]
            sweep.append({"alpha": alpha, "eps_px2": round(e, 2), "n": n, "k": k,
                          "mean_candidates": round(float(np.mean([sum(s >= RHO for s in v.values()) for v in vs])), 3),
                          "recall": round(float(np.mean([v[q["gt_object"]] >= RHO for v, q in zip(vs, qs)])), 3)})
    pd.DataFrame(sweep).to_csv(os.path.join(OUT_DIR, "sweep_eps.csv"), index=False, encoding="utf-8-sig")

    sep = []
    for (n, k), qs in data.items():
        d_true = [np.median(q["dmin"][q["gt_object"]]) for q in qs]
        d_other = [np.median(q["dmin"][o]) for q in qs for o in q["dmin"] if o != q["gt_object"]]
        sep.append({"n": n, "k": k, "median_d_true": round(float(np.median(d_true)), 3),
                    "median_d_other": round(float(np.median(d_other)), 3),
                    "ratio_other_true": round(float(np.median(d_other) / np.median(d_true)), 3)})
    sp = pd.DataFrame(sep)
    sp.to_csv(os.path.join(OUT_DIR, "separation.csv"), index=False, encoding="utf-8-sig")

    viol = 0
    for n in N_LIST:
        for k0, k1 in zip(K_LIST[:-1], K_LIST[1:]):
            a = pq[(pq.n == n) & (pq.k == k0)].set_index(["aerial_file", "query_object"]).num_candidates
            b = pq[(pq.n == n) & (pq.k == k1)].set_index(["aerial_file", "query_object"]).num_candidates
            viol += int((b > a.reindex(b.index)).sum())
    print(f"queries whose candidate count grows from k to k+1 (should be 0): {viol}")

    print("\nMean number of candidates")
    print(sm.pivot(index="n", columns="k", values="mean_candidates").to_string())
    print("\nRecall (true object among candidates)")
    print(sm.pivot(index="n", columns="k", values="recall").to_string())
    print("\nTop-1 accuracy")
    print(sm.pivot(index="n", columns="k", values="top1_acc").to_string())
    print("\nMedian distance to the true object / to the other objects (px^2)")
    print(sp.pivot(index="n", columns="k", values="median_d_true").to_string())
    print(sp.pivot(index="n", columns="k", values="median_d_other").to_string())
    print(f"\nResults written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
