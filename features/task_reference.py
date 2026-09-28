# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
# features/task_reference.py
import os
import re
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from shapely.ops import nearest_points

from .core import (
    mask_override,
    och_vertex_types,
    ReferenceAnalyzer,
    load_nofly_csv_as_polygons,
    hull_vertices_xy_from_polygon
)


# =========================
# Low-level helpers
# =========================
def _build_mask_from_thresh(image_bgr: np.ndarray, thresh: int) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    t = int(np.clip(thresh, 0, 255))

    mask1 = cv2.inRange(
        hsv,
        np.array([0, t, t], dtype=np.uint8),
        np.array([40, 255, 255], dtype=np.uint8)
    )
    mask2 = cv2.inRange(
        hsv,
        np.array([165, t, t], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8)
    )
    roof_mask = cv2.bitwise_or(mask1, mask2)
    roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return roof_mask


def _prepare_reference_analyzer(
    reference_image: str,
    px_per_meter: float,
    use_oc_hull: bool,
    roof_thresh: int
) -> ReferenceAnalyzer:
    img = cv2.imread(reference_image)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh reference: {reference_image}")

    mask = mask_override(reference_image)          # [EXT-MASK] external mask, if any
    if mask is None:
        mask = _build_mask_from_thresh(img, roof_thresh)

    ref = ReferenceAnalyzer(
        image_path=reference_image,
        px_per_meter=px_per_meter,
        use_oc_hull=use_oc_hull
    )
    ref.original_image = img.copy()
    ref.image = img.copy()
    ref.image_shape = img.shape[:2]
    ref.binary_image = mask
    ref.roof_mask = mask
    ref.roof_thresh = int(roof_thresh)
    ref.sync_alias()
    return ref


def replace_polygons_by_nfz(
    building_polys: List[Polygon],
    nofly_polys: List[Polygon],
    drop_if_intersects: bool = True
) -> Tuple[List[Polygon], List[Polygon], Dict[str, int]]:
    """
    Returns:
      - merged_polygons_used_for_compute (kept_buildings + nfz)
      - dropped_buildings (for visualization)
      - stats
    """
    blds = [p for p in building_polys if p is not None and (not p.is_empty) and p.area > 0]
    nfzs = [p for p in nofly_polys if p is not None and (not p.is_empty) and p.area > 0]

    stats = {
        "building_before": len(blds),
        "nfz_count": len(nfzs),
        "dropped_buildings": 0,
        "kept_buildings": 0,
        "final_count": 0,
        "drop_if_intersects": int(drop_if_intersects),
    }

    if len(nfzs) == 0:
        stats["kept_buildings"] = len(blds)
        stats["final_count"] = len(blds)
        return blds.copy(), [], stats

    prepared_nfz = [prep(z) for z in nfzs]
    kept_blds = []
    dropped_blds = []

    for b in blds:
        remove = False
        for i in range(len(nfzs)):
            if drop_if_intersects:
                if prepared_nfz[i].intersects(b):
                    remove = True
                    break
            else:
                z = nfzs[i]
                if b.within(z) or b.covered_by(z):
                    remove = True
                    break
        if remove:
            dropped_blds.append(b)
            stats["dropped_buildings"] += 1
        else:
            kept_blds.append(b)

    merged = kept_blds + nfzs
    stats["kept_buildings"] = len(kept_blds)
    stats["final_count"] = len(merged)
    return merged, dropped_blds, stats


def compute_global_rmax_with_witness(
    polygons: List[Polygon],
    px_per_meter: float,
    shrink_factor: float = 0.95,
    min_rmax_m: float = 0.1,
    fallback_rmax_m: float = 5.0,
    use_convex_hull: bool = False,
    eps_zero: float = 1e-9,
    extreme_only: bool = False,
) -> Dict:
    # [COCH-V] extreme_only=True: the distance is taken from the extreme (convex) vertices of
    # each polygon only, the vertices where the shape vector is used; the disk U(x*) of
    # radius R then meets only the polygon of x*.
    polys = []
    for p in polygons:
        if p is None or p.is_empty or p.area <= 0:
            continue
        q = p.convex_hull if use_convex_hull else p
        if q is not None and (not q.is_empty) and q.area > 0:
            polys.append(q)

    if len(polys) < 2:
        rmax_m = float(max(min_rmax_m, fallback_rmax_m))
        return {
            "min_dist_px": float("nan"),
            "min_dist_m": float("nan"),
            "rmax_px": float(rmax_m * px_per_meter),
            "rmax_m": float(rmax_m),
            "witness_pt": None,
            "nearest_on_other": None,
            "witness_pair": None,
        }

    min_dist_px = np.inf
    witness_pt = None
    nearest_on_other = None
    witness_pair = None

    for i, pi in enumerate(polys):
        verts_i = hull_vertices_xy_from_polygon(pi)
        if extreme_only:
            vt = och_vertex_types(verts_i)
            verts_i = [v for v, t in zip(verts_i, vt) if t == "convex"]
        for (vx, vy) in verts_i:
            pt = Point(vx, vy)
            for j, pj in enumerate(polys):
                if i == j:
                    continue
                d = pt.distance(pj)
                if d <= eps_zero:
                    continue
                if d < min_dist_px:
                    min_dist_px = d
                    p_on_v, p_on_j = nearest_points(pt, pj)
                    witness_pt = (float(p_on_v.x), float(p_on_v.y))
                    nearest_on_other = (float(p_on_j.x), float(p_on_j.y))
                    witness_pair = (i, j)

    if (not np.isfinite(min_dist_px)) or (min_dist_px <= 0):
        rmax_m = float(max(min_rmax_m, fallback_rmax_m))
        return {
            "min_dist_px": float("nan"),
            "min_dist_m": float("nan"),
            "rmax_px": float(rmax_m * px_per_meter),
            "rmax_m": float(rmax_m),
            "witness_pt": None,
            "nearest_on_other": None,
            "witness_pair": None,
        }

    rmax_px = float(min_dist_px * shrink_factor)
    rmax_m = float(max(min_rmax_m, rmax_px / px_per_meter))
    return {
        "min_dist_px": float(min_dist_px),
        "min_dist_m": float(min_dist_px / px_per_meter),
        "rmax_px": float(rmax_px),
        "rmax_m": float(rmax_m),
        "witness_pt": witness_pt,
        "nearest_on_other": nearest_on_other,
        "witness_pair": witness_pair,
    }


# =========================
# CSV schema normalization
# =========================
def _parse_r_d_col(col: str):
    m = re.match(r"^r([0-9]+(?:\.[0-9]+)?)_d([0-9]+)$", str(col))
    if not m:
        return None
    return float(m.group(1)), int(m.group(2))


def _format_radius_token(v: float) -> str:
    return f"{float(v):.5f}".rstrip("0").rstrip(".")


def normalize_feature_table_schema(df: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    required = ["object_id", "vertex_id", "x", "y"]
    for c in required:
        if c not in out.columns:
            raise ValueError(f"Thiếu cột bắt buộc: {c}")

    # tìm descriptor cols
    rd_cols = []
    for c in out.columns:
        p = _parse_r_d_col(c)
        if p is not None:
            rd_cols.append((c, p[0], p[1]))

    if len(rd_cols) == 0:
        raise ValueError("Không tìm thấy cột descriptor dạng r*_d*")

    # rename descriptor về token radius ổn định
    rename_map = {}
    for old, r, d in rd_cols:
        rename_map[old] = f"r{_format_radius_token(r)}_d{int(d)}"
    out = out.rename(columns=rename_map)

    # parse lại và sort
    rd_cols2 = []
    for c in out.columns:
        p = _parse_r_d_col(c)
        if p is not None:
            rd_cols2.append((c, p[0], p[1]))
    rd_sorted = [t[0] for t in sorted(rd_cols2, key=lambda z: (z[1], z[2]))]

    # vertex_total
    if "vertex_total" not in out.columns:
        out["vertex_total"] = out[rd_sorted].sum(axis=1)

    # type cast
    out["object_id"] = out["object_id"].astype(int)
    out["vertex_id"] = out["vertex_id"].astype(int)
    out["x"] = out["x"].astype(float)
    out["y"] = out["y"].astype(float)
    for c in rd_sorted:
        out[c] = out[c].astype(float)
    out["vertex_total"] = out["vertex_total"].astype(float)

    # reorder
    cols = ["object_id", "vertex_id", "x", "y"] + rd_sorted + ["vertex_total"]
    if "vertex_type" in out.columns:                   # [COCH-V] keep the COCH vertex type
        cols.append("vertex_type")
    out = out[cols]

    # row order stable
    out = out.sort_values(["object_id", "vertex_id"], kind="mergesort").reset_index(drop=True)

    # round
    float_cols = ["x", "y"] + rd_sorted + ["vertex_total"]
    out[float_cols] = out[float_cols].round(decimals)

    return out


def save_feature_csv_normalized(df: pd.DataFrame, out_csv: str, decimals: int = 6):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    norm = normalize_feature_table_schema(df, decimals=decimals)
    norm.to_csv(out_csv, index=False, encoding="utf-8")


# =========================
# Main task function
# =========================
def run_reference_task(
    reference_image: str,
    out_csv: str,
    nofly_csv: Optional[str] = None,
    px_per_meter: float = 2.445496,
    roof_thresh: int = 18,
    min_area: int = 2850,
    border_margin: int = 0,
    object_polygon_mode: str = "och",
    n: int = 3,
    k: int = 3,
    max_radius_m: Optional[float] = None,
    sector_num_points: int = 96,
    save_overview: bool = False,
    overview_path: Optional[str] = None,
    # rmax params
    use_oc_hull: bool = True,
    drop_if_intersects: bool = True,
    shrink_factor: float = 0.95,
    min_rmax_m: float = 0.1,
    fallback_rmax_m: float = 5.0,
    use_convex_hull_for_rmax: bool = False,
    # [EX] True: the no-fly zone is an obstacle as in the paper (buildings that meet it are
    # removed and it bounds the radius, but its area is NOT added to the shape vector);
    # False: original behaviour (its polygon is added to the map and its area counted).
    nfz_as_obstacle: bool = False,
    # csv
    csv_decimals: int = 6,
    # logs
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Chạy trích xuất đặc trưng cho reference map và xuất CSV chuẩn schema:
      object_id, vertex_id, x, y, r..._d..., vertex_total

    Trả về dict metadata:
      {
        "out_csv": ...,
        "overview_path": ... or None,
        "global_rmax_m": ...,
        "rinfo": ...,
        "replace_stats": ...,
        "num_polygons_after_replace": ...
      }
    """
    if not os.path.isfile(reference_image):
        raise FileNotFoundError(f"Không tìm thấy reference image: {reference_image}")

    if px_per_meter is None or px_per_meter <= 0:
        raise ValueError("px_per_meter phải > 0")

    # 1) Prepare analyzer (load image + threshold mask)
    ref = _prepare_reference_analyzer(
        reference_image=reference_image,
        px_per_meter=px_per_meter,
        use_oc_hull=use_oc_hull,
        roof_thresh=roof_thresh
    )

    # 2) Extract polygons (compatible API cũ/mới)
    extracted_ok = False
    last_ex = None
    try:
        # API cũ
        ref.extract_polygons(
            min_area=int(min_area),
            border_margin=int(border_margin),
            use_convex_hull=(str(object_polygon_mode).lower() in ("och", "convex", "hull", "true"))
        )
        extracted_ok = True
    except TypeError as ex:
        last_ex = ex

    if not extracted_ok:
        try:
            # API mới
            ref.extract_polygons(
                min_area=int(min_area),
                border_margin=int(border_margin),
                object_polygon_mode=object_polygon_mode
            )
            extracted_ok = True
        except Exception as ex2:
            raise RuntimeError(
                f"extract_polygons thất bại ở cả API cũ/mới. last_old={last_ex}, new={ex2}"
            )

    if verbose:
        print(f"[REF] extracted polygons: {len(ref.polygons)}")

    # 3) NFZ replace (nếu có)
    nfz_polys: List[Polygon] = []
    replace_stats = {
        "building_before": len(ref.polygons),
        "nfz_count": 0,
        "dropped_buildings": 0,
        "kept_buildings": len(ref.polygons),
        "final_count": len(ref.polygons),
        "drop_if_intersects": int(drop_if_intersects),
    }
    dropped_buildings: List[Polygon] = []

    if nofly_csv is not None and str(nofly_csv).strip() != "" and os.path.isfile(nofly_csv):
        nfz_polys = load_nofly_csv_as_polygons(nofly_csv)
        ref.polygons, dropped_buildings, replace_stats = replace_polygons_by_nfz(
            building_polys=ref.polygons,
            nofly_polys=nfz_polys,
            drop_if_intersects=drop_if_intersects
        )
        if verbose:
            print(f"[REF] NFZ loaded: {len(nfz_polys)}")
            print(f"[REF] replace stats: {replace_stats}")
    else:
        if verbose:
            print("[REF] nofly_csv not found/empty -> skip NFZ replace")

    # 4) Determine max_radius_m
    #    Nếu user truyền max_radius_m thì dùng trực tiếp.
    #    Nếu None hoặc <=0 thì tự estimate global rmax từ polygons hiện tại.
    rinfo = None
    if max_radius_m is None or float(max_radius_m) <= 0:
        rinfo = compute_global_rmax_with_witness(
            polygons=ref.polygons,
            px_per_meter=px_per_meter,
            shrink_factor=shrink_factor,
            min_rmax_m=min_rmax_m,
            fallback_rmax_m=fallback_rmax_m,
            use_convex_hull=use_convex_hull_for_rmax,
            eps_zero=1e-9
        )
        max_radius_m = float(rinfo["rmax_m"])
        if verbose:
            print(f"[REF] auto global_rmax_m = {max_radius_m:.6f}")
    else:
        max_radius_m = float(max_radius_m)
        if verbose:
            print(f"[REF] use provided max_radius_m = {max_radius_m:.6f}")

    # [EX] NFZ as obstacle: keep only the buildings for the shape vector
    if nfz_as_obstacle and len(nfz_polys) > 0:
        ref.polygons = list(ref.polygons[: int(replace_stats.get("kept_buildings", len(ref.polygons)))])
        if verbose:
            print(f"[REF] NFZ as obstacle: {len(ref.polygons)} building polygons used for the features")

    # 5) Compute features
    ref.compute_features(
        n=int(n),
        k=int(k),
        max_radius_m=float(max_radius_m),
        sector_num_points=int(sector_num_points)
    )

    # 6) Export CSV with normalized schema
    ft = ref.get_feature_table()
    if isinstance(ft, pd.DataFrame):
        df = ft.copy()
    else:
        df = pd.DataFrame(ft)

    save_feature_csv_normalized(df, out_csv, decimals=int(csv_decimals))

    # 7) Optional overview
    final_overview = None
    if save_overview:
        final_overview = overview_path
        if final_overview is None or str(final_overview).strip() == "":
            base = os.path.splitext(os.path.basename(out_csv))[0]
            final_overview = os.path.join(os.path.dirname(out_csv), f"{base}_overview.png")
        os.makedirs(os.path.dirname(final_overview), exist_ok=True)

        try:
            # Ưu tiên plot_overview của analyzer nếu có
            ref.plot_overview(out_path=final_overview, show_vertices=True)
        except Exception:
            # fallback đơn giản: không làm fail task nếu không vẽ được
            final_overview = None
            if verbose:
                print("[WARN] Không thể lưu overview (bỏ qua).")

    result = {
        "out_csv": out_csv,
        "overview_path": final_overview,
        "global_rmax_m": float(max_radius_m),
        "rinfo": rinfo,
        "replace_stats": replace_stats,
        "num_polygons_after_replace": len(ref.polygons),
        "num_dropped_buildings": len(dropped_buildings),
        "n": int(n),
        "k": int(k),
    }

    if verbose:
        print(f"[REF] saved csv: {out_csv}")
        if final_overview:
            print(f"[REF] saved overview: {final_overview}")

    return result
