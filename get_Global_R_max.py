# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
# get_Global_R_max.py
import os
from typing import List, Dict, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from shapely.ops import nearest_points

from features.core import (
    ReferenceAnalyzer,
    load_nofly_csv_as_polygons,
    hull_vertices_xy_from_polygon
)


def _build_mask_from_thresh(image_bgr, thresh: int):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    t = int(np.clip(thresh, 0, 255))

    mask1 = cv2.inRange(hsv, np.array([0, t, t], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array([165, t, t], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    roof_mask = cv2.bitwise_or(mask1, mask2)

    roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return roof_mask


def _prepare_ref_like_old_pipeline(
    image_path: str,
    px_per_meter: float,
    use_oc_hull: bool,
    thresh: int
) -> ReferenceAnalyzer:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    mask = _build_mask_from_thresh(img, thresh)

    ref = ReferenceAnalyzer(
        image_path=image_path,
        px_per_meter=px_per_meter,
        use_oc_hull=use_oc_hull
    )
    ref.original_image = img.copy()
    ref.image = img.copy()
    ref.image_shape = img.shape[:2]
    ref.binary_image = mask
    ref.roof_mask = mask
    ref.roof_thresh = int(thresh)
    ref.sync_alias()
    return ref


def replace_polygons_by_nfz(
    building_polys: List[Polygon],
    nofly_polys: List[Polygon],
    drop_if_intersects: bool = True
) -> Tuple[List[Polygon], List[Polygon], Dict[str, int]]:
    """
    Returns:
      - merged_polygons_used_for_rmax
      - dropped_buildings (để vẽ mờ)
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
    eps_zero: float = 1e-9
) -> Dict:
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


def _label_polygon(ax, poly: Polygon, text: str, color: str, zorder: int = 9):
    if poly is None or poly.is_empty:
        return
    c = poly.centroid
    ax.text(
        c.x, c.y, text,
        color=color, fontsize=8, ha="center", va="center", zorder=zorder,
        bbox=dict(facecolor="black", alpha=1, edgecolor="none", pad=1.5)
    )


def _plot_result(
    image_bgr,
    polygons_used: List[Polygon],
    nfz_polys: List[Polygon],
    dropped_buildings: List[Polygon],
    rinfo: Dict,
    px_per_meter: float,
    out_path: str
):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img_rgb)
    ax.axis("off")
    ax.set_title(f"REFMAP with no-fly zone")
    #ax.set_title(f"GLOBAL_R_MAX={float(rinfo.get('rmax_m', np.nan)):.4f} m")

    # 1) Building bị drop: tô mờ + ID D1, D2, ...
    for idx, p in enumerate(dropped_buildings, start=1):
        if p is None or p.is_empty:
            continue
        x, y = p.exterior.xy
        ax.fill(x, y, color="gray", alpha=0.18, zorder=1)
        ax.plot(x, y, color="lightgray", linewidth=1.0, alpha=0.80, zorder=1)
        _label_polygon(ax, p, f"D{idx}", color="lightgray", zorder=5)
        _label_polygon(ax, p, f"{idx}", color="lightgray", zorder=5)

    # 2) Polygon dùng để tính: viền cyan + ID B1, B2, ...
    for idx, p in enumerate(polygons_used, start=1):
        if p is None or p.is_empty:
            continue
        x, y = p.exterior.xy
        ax.plot(x, y, color="cyan", linewidth=1.5, zorder=3)
        _label_polygon(ax, p, f"{idx}", color="yellow", zorder=11)

    # 3) NFZ
    nfz_idx = 0
    for p in nfz_polys:
        nfz_idx += 1 
        if p is None or p.is_empty:
            continue
        x, y = p.exterior.xy
        ax.fill(x, y, color="magenta", alpha=0.18, zorder=2)
        ax.plot(x, y, color="magenta", linewidth=1.8, zorder=4)
        _label_polygon(ax, p, f"NFZ-{nfz_idx}", color="yellow", zorder=11)

    # 4) Circle tại witness vertex x*
    wp = rinfo.get("witness_pt", None)
    np2 = rinfo.get("nearest_on_other", None)

    if wp is not None:
        cx, cy = wp
        r_px = float(rinfo["rmax_m"]) * float(px_per_meter)

        ax.add_patch(plt.Circle((cx, cy), r_px, fill=False, color="yellow", linewidth=2.2, zorder=6))
        ax.scatter([cx], [cy], c="red", s=36, zorder=7)
        ax.text(
            cx + 5, cy - 5, r"$x^*$",
            color="red", fontsize=11, zorder=8,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2)
        )

        if np2 is not None:
            qx, qy = np2
            ax.plot([cx, qx], [cy, qy], "--", color="lime", linewidth=1.3, zorder=7)

    # 5) Legend có thêm R và x*
    Rm = float(rinfo.get("rmax_m", np.nan))
    legend_items = [
        #Patch(facecolor="gray", edgecolor="lightgray", alpha=0.45, label="Dropped buildings (D*)"),
        Patch(facecolor="magenta", edgecolor="magenta", alpha=0.25, label="NFZ"),
        #Line2D([0], [0], color="cyan", lw=2, label="Polygons used "),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=7, label=r"$x^*$ "), # (witness vertex)"),
        Line2D([0], [0], color="yellow", lw=2, label=rf"$R={Rm:.4f}\,\mathrm{{m}}$"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, frameon=True, framealpha=0.95)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[RMAX] Saved: {out_path}")
    plt.show()
    plt.close(fig)


def main():
    reference_image = os.path.join("references", "refmap.png")
    nofly_csv = os.path.join("data", "references", "non_fly_zone.csv")
    out_plot = os.path.join("data", "references", "global_rmax_preview.png")

    # Khớp code bạn đưa
    px_per_meter = 2.445496
    fixed_ref_thresh = 18
    fixed_min_area_och = 2850
    border_margin = 0

    use_oc_hull = True
    use_convex_hull_extract = True
    drop_if_intersects = True

    shrink_factor = 0.98
    min_rmax_m = 0.1
    fallback_rmax_m = 5.0
    use_convex_hull_for_rmax = False

    # 1) prepare ref
    ref = _prepare_ref_like_old_pipeline(
        image_path=reference_image,
        px_per_meter=px_per_meter,
        use_oc_hull=use_oc_hull,
        thresh=fixed_ref_thresh
    )

    # Tương thích API cũ/mới
    try:
        ref.extract_polygons(
            min_area=fixed_min_area_och,
            border_margin=border_margin,
            use_convex_hull=use_convex_hull_extract
        )
    except TypeError:
        mode = "och" if use_convex_hull_extract else "raw"
        ref.extract_polygons(
            min_area=fixed_min_area_och,
            border_margin=border_margin,
            object_polygon_mode=mode
        )

    print(f"[INFO] extracted_building_polygons = {len(ref.polygons)}")

    # 2) load NFZ
    nfz_polys = load_nofly_csv_as_polygons(nofly_csv)
    print(f"[INFO] loaded_nfz_polygons = {len(nfz_polys)}")

    # 3) replace + dropped
    ref.polygons, dropped_buildings, rep_stats = replace_polygons_by_nfz(
        building_polys=ref.polygons,
        nofly_polys=nfz_polys,
        drop_if_intersects=drop_if_intersects
    )
    print(f"[INFO] REPLACE STATS = {rep_stats}")

    # 4) compute rmax + witness
    rinfo = compute_global_rmax_with_witness(
        polygons=ref.polygons,
        px_per_meter=px_per_meter,
        shrink_factor=shrink_factor,
        min_rmax_m=min_rmax_m,
        fallback_rmax_m=fallback_rmax_m,
        use_convex_hull=use_convex_hull_for_rmax,
        eps_zero=1e-9
    )

    print("\n===== GLOBAL R_MAX =====")
    print(f"min_dist_px = {rinfo.get('min_dist_px')}")
    print(f"min_dist_m  = {rinfo.get('min_dist_m')}")
    print(f"rmax_px     = {rinfo.get('rmax_px')}")
    print(f"rmax_m      = {rinfo.get('rmax_m')}")
    print(f"witness_pt  = {rinfo.get('witness_pt')}")
    print(f"nearest_pt  = {rinfo.get('nearest_on_other')}")
    print(f"pair(i,j)   = {rinfo.get('witness_pair')}")

    # 5) plot
    _plot_result(
        image_bgr=ref.original_image,
        polygons_used=ref.polygons,
        nfz_polys=nfz_polys,
        dropped_buildings=dropped_buildings,
        rinfo=rinfo,
        px_per_meter=px_per_meter,
        out_path=out_plot
    )


if __name__ == "__main__":
    main()
