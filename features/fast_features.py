# -*- coding: utf-8 -*-
# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
"""
features/fast_features.py

Faster drop-in replacement for core.compute_sector_features_from_points_global.
Same inputs, same outputs, same sector polygons (generate_sector_polygon with
the same number of arc points), so the numbers agree with the original up to
floating-point rounding. Changes:

1. All sector polygons of one vertex are built at once from precomputed
   templates (numpy + shapely.polygons) instead of one Python call each.
2. The intersections sector x polygon are computed in one vectorized GEOS call
   (shapely.intersection on arrays) instead of a Python double loop.
3. A polygon is skipped only if its true distance to the vertex exceeds the
   radius (shapely.distance). The original skipped a polygon when all of its
   VERTICES were farther than the radius, which drops polygons whose edge
   crosses the disk while all corners lie outside it; that case is now counted.

As in the original, the area inside ring j and sector i is
    raw[r_j][i] - raw[r_{j-1}][i],  raw[r][i] = area(sector(r, i) ∩ polygons),
clipped at 0, and the polygons' areas are summed (polygons are assumed
disjoint, as in the original).
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree


def _meters_to_pixels(m: float, ppm: float) -> float:
    return float(m) * float(ppm)


def _build_radii_m(n: int, max_radius_m: float) -> Tuple[float, ...]:
    return tuple((i * max_radius_m) / n for i in range(1, n + 1))


def _sector_templates(radii_px: List[float], num_dirs: int, num_points: int) -> np.ndarray:
    """Coordinates of every sector polygon centred at the origin, shape (R*S, num_points+2, 2)."""
    step = 360.0 / num_dirs
    out = []
    for rp in radii_px:
        for i in range(num_dirs):
            ang = np.radians(np.linspace(i * step, (i + 1) * step, int(num_points)))
            pts = np.vstack([[0.0, 0.0], np.c_[rp * np.cos(ang), rp * np.sin(ang)], [0.0, 0.0]])
            out.append(pts)
    return np.asarray(out)


def compute_sector_features_fast(
    input_points_xy: List[Tuple[float, float]],
    all_polygons: List[Polygon],
    n: int = 3,
    k: int = 3,
    max_radius_m: float = 100.0,
    circle_radii_m: Optional[Tuple[float, ...]] = None,
    px_per_meter: float = 2.445496,
    sector_num_points: int = 140,
) -> Dict:
    num_dirs = 2 ** int(k)
    if circle_radii_m is None:
        circle_radii_m = _build_radii_m(int(n), float(max_radius_m))
    circle_radii_m = tuple(circle_radii_m)
    radii_px = [_meters_to_pixels(r, px_per_meter) for r in circle_radii_m]
    n_r = len(circle_radii_m)
    n_v = len(input_points_xy)

    raw_all = np.zeros((n_v, n_r, num_dirs), dtype=np.float64)
    polys = [p for p in all_polygons if p is not None and not p.is_empty and p.area > 0]

    if polys and n_v:
        poly_arr = np.asarray(polys, dtype=object)
        tree = STRtree(poly_arr)
        tmpl = _sector_templates(radii_px, num_dirs, sector_num_points)     # (R*S, P, 2)
        r_max = max(radii_px)
        for v, (x, y) in enumerate(input_points_xy):
            center = shapely.points(x, y)
            cand = tree.query(center, predicate="dwithin", distance=r_max)
            if cand.size == 0:
                continue
            cp = poly_arr[cand]
            dist = shapely.distance(center, cp)
            sectors = shapely.polygons(tmpl + np.array([x, y]))             # (R*S,)
            sectors = shapely.make_valid(sectors)
            # a polygon only contributes to radii not smaller than its distance
            inter = shapely.intersection(sectors[:, None], cp[None, :])     # (R*S, C)
            area = shapely.area(inter)
            rad_of_sector = np.repeat(np.asarray(radii_px), num_dirs)
            area[rad_of_sector[:, None] < dist[None, :]] = 0.0
            raw_all[v] = area.sum(axis=1).reshape(n_r, num_dirs)

    raw = {rm: raw_all[:, j, :].astype(np.float32) for j, rm in enumerate(circle_radii_m)}
    feat, prev = {}, None
    for rm in circle_radii_m:
        feat[rm] = raw[rm].copy() if prev is None else np.maximum(0.0, raw[rm] - raw[prev])
        prev = rm
    vertex_total = np.zeros(n_v, dtype=np.float32)
    for rm in circle_radii_m:
        vertex_total += feat[rm].sum(axis=1)
    aerial_dictionary = {vid: [float(v) for rm in circle_radii_m for v in feat[rm][vid, :]]
                         for vid in range(n_v)}
    return {"raw": raw, "feat": feat, "vertex_total": vertex_total,
            "aerial_dictionary": aerial_dictionary, "circle_radii_m": circle_radii_m,
            "num_dirs": num_dirs}
