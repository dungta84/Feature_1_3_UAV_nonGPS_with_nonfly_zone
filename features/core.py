# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
# features/core.py
from typing import Tuple, Dict, List, Optional
import os
import numpy as np
import pandas as pd
import cv2
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.prepared import prep

from OrthoHullLib import OrthoHullLib
from features.fast_features import compute_sector_features_fast

# [FAST] True: vectorized feature computation (same results, much faster); False: original loop
USE_FAST_FEATURES = True


# =========================
# Utility functions
# =========================

def oc_hull_from_polygon(raw_polygon):
    if raw_polygon is None or raw_polygon.is_empty:
        return Polygon()

    geom = raw_polygon
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return Polygon()

    if isinstance(geom, MultiPolygon):
        geom = unary_union(geom)
        if geom.is_empty:
            return Polygon()
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)

    if not isinstance(geom, Polygon):
        geom = geom.convex_hull
        if geom.is_empty or not isinstance(geom, Polygon):
            return Polygon()

    coords = list(geom.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        return Polygon()

    pts_list = [Point(float(x), float(y)) for x, y in coords]
    lib = OrthoHullLib()
    oc_coords = lib.get_ortho_hull(pts_list, close_ring=True)

    if not oc_coords or len(oc_coords) < 4:
        return geom.convex_hull

    oc_poly = Polygon(oc_coords)
    if not oc_poly.is_valid:
        oc_poly = oc_poly.buffer(0)
    if oc_poly.is_empty:
        return geom.convex_hull

    return oc_poly


# [COCH-V] Vertex types of an orthogonal (x, y)-polygon such as the COCH (connected
# orthogonal convex hull) of a building, computed by OGraham.py.
# By An, Huyen and Le (Appl. Math. Comput. 397 (2021) 125889), the convex vertices of the
# connected orthogonal convex hull are its extreme points, and they belong to the input
# point set; the reflex vertices are the corners of the staircases between them.
# The shape vector is used at the extreme (convex) vertices only.
def och_vertex_types(coords, tol: float = 1e-9) -> List[str]:
    """'convex' (extreme point of the COCH), 'reflex' (staircase corner) or 'flat'."""
    p = np.asarray(coords, dtype=float)
    m = len(p)
    if m < 3:
        return ["flat"] * m
    x, y = p[:, 0], p[:, 1]
    area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))   # >0: counter-clockwise in (x, y)
    e_in = p - np.roll(p, 1, axis=0)
    e_out = np.roll(p, -1, axis=0) - p
    cross = e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0]
    s = np.sign(area2) if area2 != 0 else 1.0
    return ["flat" if abs(c) <= tol else ("convex" if c * s > 0 else "reflex") for c in cross]


# [EXT-MASK] Building masks from another segmentation method. When the environment
# variable FEATURE_MASK_DIR is set, the roof mask of an image <name>.png is read from
# FEATURE_MASK_DIR/<name>_mask.png (white = building) instead of the HSV threshold.
# The variable is inherited by the worker processes of run_examples.py.
MASK_DIR_ENV = "FEATURE_MASK_DIR"


def mask_override(image_path: str) -> Optional[np.ndarray]:
    mdir = os.environ.get(MASK_DIR_ENV, "").strip()
    if not mdir:
        return None
    stem = os.path.splitext(os.path.basename(image_path))[0]
    mp = os.path.join(mdir, f"{stem}_mask.png")
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"{MASK_DIR_ENV} is set but the mask is missing: {mp}")
    return ((m > 127).astype(np.uint8) * 255)


def meters_to_pixels(meters: float, px_per_meter: float) -> float:
    return float(meters) * float(px_per_meter)


def hull_vertices_xy_from_polygon(poly: Polygon):
    if poly is None or poly.is_empty or poly.area <= 0:
        return []
    coords = list(poly.exterior.coords)
    return [(float(x), float(y)) for x, y in coords[:-1]]


def generate_sector_polygon(
    center: Point,
    radius_pixels: float,
    start_angle: float,
    end_angle: float,
    num_points: int = 120
) -> Polygon:
    pts = [(center.x, center.y)]
    for angle in np.linspace(start_angle, end_angle, int(num_points)):
        x = center.x + radius_pixels * np.cos(np.radians(angle))
        y = center.y + radius_pixels * np.sin(np.radians(angle))
        pts.append((x, y))
    pts.append((center.x, center.y))

    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def build_radii_m(n: int, max_radius_m: float = 100.0) -> Tuple[float, ...]:
    if n < 1:
        raise ValueError("n phải >= 1")
    if max_radius_m <= 0:
        raise ValueError("max_radius_m phải > 0")
    return tuple((i * max_radius_m) / n for i in range(1, n + 1))


def _geometry_area(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    if isinstance(geom, Polygon):
        return float(geom.area)
    if hasattr(geom, "geoms"):
        return float(sum(g.area for g in geom.geoms if isinstance(g, Polygon)))
    return float(getattr(geom, "area", 0.0) or 0.0)


def _min_vertex_distance_sq(center_xy: Tuple[float, float], poly_vertices: List[Tuple[float, float]]) -> float:
    cx, cy = center_xy
    d2_min = np.inf
    for x, y in poly_vertices:
        dx = x - cx
        dy = y - cy
        d2 = dx * dx + dy * dy
        if d2 < d2_min:
            d2_min = d2
    return float(d2_min)


def _query_candidate_indices(tree: STRtree, geom) -> np.ndarray:
    q = tree.query(geom)

    if isinstance(q, np.ndarray):
        if q.dtype.kind in ("i", "u"):
            return q.astype(np.int64, copy=False)
        try:
            return q.astype(np.int64)
        except Exception:
            return np.array([], dtype=np.int64)

    if isinstance(q, (list, tuple)):
        try:
            return np.array(q, dtype=np.int64)
        except Exception:
            return np.array([], dtype=np.int64)

    return np.array([], dtype=np.int64)


# =========================
# No-fly helpers
# =========================

def circle_to_polygon_180(center_x: float, center_y: float, radius: float) -> Polygon:
    """
    Xấp xỉ hình tròn thành polygon 180 đỉnh.
    """
    if radius is None or float(radius) <= 0:
        return Polygon()

    angles = np.linspace(0.0, 360.0, 180, endpoint=False)
    pts = []
    cx, cy, r = float(center_x), float(center_y), float(radius)
    for a in angles:
        rad = np.radians(a)
        x = cx + r * np.cos(rad)
        y = cy + r * np.sin(rad)
        pts.append((x, y))

    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return Polygon()
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def load_nofly_csv_as_polygons(csv_path: str) -> List[Polygon]:
    """
    Đọc CSV no-fly-zone -> list Polygon.
    Hỗ trợ:
      - circle  -> polygon 180 đỉnh
      - polygon -> polygon thường
    Safe cho case rỗng/không tồn tại/lỗi format: trả [].
    """
    if not csv_path or (not os.path.isfile(csv_path)):
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Không đọc được no-fly CSV '{csv_path}': {e}. -> coi như không cấm bay")
        return []

    if df is None or df.empty:
        return []

    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"zone_id", "zone_type", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] no-fly CSV thiếu cột {missing}. -> coi như không cấm bay")
        return []

    if "enabled" not in df.columns:
        df["enabled"] = 1
    if "radius" not in df.columns:
        df["radius"] = np.nan
    if "vertex_order" not in df.columns:
        df["vertex_order"] = np.nan

    try:
        df = df[df["enabled"].fillna(1).astype(int) == 1].copy()
    except Exception:
        df = df.copy()

    if df.empty:
        return []

    out: List[Polygon] = []

    for _, g in df.groupby("zone_id"):
        ztype = str(g["zone_type"].iloc[0]).strip().lower()

        if ztype == "circle":
            row = g.iloc[0]
            try:
                cx = float(row["x"])
                cy = float(row["y"])
                r = float(row["radius"]) if pd.notna(row["radius"]) else 0.0
            except Exception:
                continue

            poly = circle_to_polygon_180(cx, cy, r)
            if not poly.is_empty and poly.area > 0:
                out.append(poly)

        elif ztype == "polygon":
            gg = g.copy()
            if gg["vertex_order"].notna().any():
                gg["vertex_order"] = gg["vertex_order"].fillna(1e9).astype(float)
                gg = gg.sort_values("vertex_order")

            pts = []
            for x, y in zip(gg["x"], gg["y"]):
                try:
                    pts.append((float(x), float(y)))
                except Exception:
                    pass

            if len(pts) < 3:
                continue

            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda ge: ge.area)
            if poly.area > 0:
                out.append(poly)

    return out


# =========================
# Feature computation
# =========================

def compute_sector_features_from_points_global(
    input_points_xy: List[Tuple[float, float]],
    all_polygons: List[Polygon],
    n: int = 3,
    k: int = 3,
    max_radius_m: float = 100.0,
    circle_radii_m: Optional[Tuple[float, ...]] = None,
    px_per_meter: float = 2.445496,
    sector_num_points: int = 140,
):
    num_dirs = 2 ** int(k)
    step = 360.0 / num_dirs

    if circle_radii_m is None:
        circle_radii_m = build_radii_m(n=n, max_radius_m=max_radius_m)
    else:
        circle_radii_m = tuple(circle_radii_m)

    radii_pixels = [meters_to_pixels(rm, px_per_meter) for rm in circle_radii_m]

    n_vertices = len(input_points_xy)
    raw = {rm: np.zeros((n_vertices, num_dirs), dtype=np.float32) for rm in circle_radii_m}

    if len(all_polygons) == 0 or n_vertices == 0:
        feat = {rm: raw[rm].copy() for rm in circle_radii_m}
        vertex_total = np.zeros(n_vertices, dtype=np.float32)
        aerial_dictionary = {vid: [0.0] * (len(circle_radii_m) * num_dirs) for vid in range(n_vertices)}
        return {
            "raw": raw,
            "feat": feat,
            "vertex_total": vertex_total,
            "aerial_dictionary": aerial_dictionary,
            "circle_radii_m": circle_radii_m,
            "num_dirs": num_dirs
        }

    valid_polys = []
    for p in all_polygons:
        if p is None or p.is_empty or p.area <= 0:
            continue
        valid_polys.append(p)

    if len(valid_polys) == 0:
        feat = {rm: raw[rm].copy() for rm in circle_radii_m}
        vertex_total = np.zeros(n_vertices, dtype=np.float32)
        aerial_dictionary = {vid: [0.0] * (len(circle_radii_m) * num_dirs) for vid in range(n_vertices)}
        return {
            "raw": raw,
            "feat": feat,
            "vertex_total": vertex_total,
            "aerial_dictionary": aerial_dictionary,
            "circle_radii_m": circle_radii_m,
            "num_dirs": num_dirs
        }

    tree = STRtree(valid_polys)
    poly_vertices_cache = [hull_vertices_xy_from_polygon(p) for p in valid_polys]
    prepared_polys = [prep(p) for p in valid_polys]

    cand_cache: Dict[Tuple[int, int], np.ndarray] = {}

    for vid, (x, y) in enumerate(input_points_xy):
        center = Point(x, y)

        for ridx, (rm, rp) in enumerate(zip(circle_radii_m, radii_pixels)):
            rp2 = rp * rp

            key = (vid, ridx)
            if key not in cand_cache:
                disk = center.buffer(rp, resolution=24)
                cand_idx = _query_candidate_indices(tree, disk)
                cand_cache[key] = cand_idx
            else:
                cand_idx = cand_cache[key]

            if cand_idx.size == 0:
                continue

            for i in range(num_dirs):
                sa = i * step
                ea = (i + 1) * step
                sector = generate_sector_polygon(center, rp, sa, ea, num_points=sector_num_points)

                area_sum = 0.0
                for pidx in cand_idx:
                    pidx = int(pidx)
                    if pidx < 0 or pidx >= len(valid_polys):
                        continue

                    poly = valid_polys[pidx]
                    verts = poly_vertices_cache[pidx]
                    if len(verts) > 0:
                        d2_min = _min_vertex_distance_sq((x, y), verts)
                        if d2_min > rp2:
                            continue

                    if not prepared_polys[pidx].intersects(sector):
                        continue

                    inter = sector.intersection(poly)
                    if inter.is_empty:
                        continue

                    area_sum += _geometry_area(inter)

                raw[rm][vid, i] = area_sum

    feat = {}
    prev_rm = None
    for rm in circle_radii_m:
        if prev_rm is None:
            feat[rm] = raw[rm].copy()
        else:
            feat[rm] = np.maximum(0.0, raw[rm] - raw[prev_rm])
        prev_rm = rm

    vertex_total = np.zeros(n_vertices, dtype=np.float32)
    for rm in circle_radii_m:
        vertex_total += feat[rm].sum(axis=1)

    aerial_dictionary = {}
    for vid in range(n_vertices):
        flat = []
        for rm in circle_radii_m:
            flat.extend(feat[rm][vid, :].astype(float).tolist())
        aerial_dictionary[vid] = flat

    return {
        "raw": raw,
        "feat": feat,
        "vertex_total": vertex_total,
        "aerial_dictionary": aerial_dictionary,
        "circle_radii_m": circle_radii_m,
        "num_dirs": num_dirs
    }


# =========================
# Core classes
# =========================

class BaseSectorAnalyzer:
    class VertexData:
        def __init__(self, coordinates: Tuple[float, float]):
            self.coordinates = coordinates
            self.intersections: Dict[float, List[float]] = {}
            self.total_sum: float = 0.0

    class ObjectData:
        def __init__(self):
            self.vertices: Dict[int, "BaseSectorAnalyzer.VertexData"] = {}
            self.total_intersection_sum: float = 0.0

        def add_vertex(self, vertex_id: int, coordinates: Tuple[float, float]) -> None:
            self.vertices[vertex_id] = BaseSectorAnalyzer.VertexData(coordinates)

    def __init__(self, image_path: str, px_per_meter, use_oc_hull: bool = True):
        self.image_path = image_path
        self.px_per_meter = float(px_per_meter)
        self.use_oc_hull = bool(use_oc_hull)

        self.image = None
        self.binary_image = None
        self.image_shape = None
        self.original_image = None
        self.roof_mask = None
        self.roof_thresh = None

        self.polygons: List[Polygon] = []
        self.extra_polygons: List[Polygon] = []  # no-fly-zone / object phụ
        self.multipolygon: MultiPolygon = MultiPolygon()
        self.contours: List[np.ndarray] = []

        self.objects: Dict[int, BaseSectorAnalyzer.ObjectData] = {}
        self.feature_sums: Dict[int, float] = {}
        self.feature_dict: Dict[int, Dict[int, List[float]]] = {}

        self.vertex_coordinates: Dict[int, List[Tuple[float, float]]] = {}
        self.geometries: Dict[int, Polygon] = {}
        self.bound: Dict[int, Dict[str, Dict[str, float]]] = {}

        self._feature_df = pd.DataFrame()
        self._last_circle_radii_m: Optional[Tuple[float, ...]] = None
        self._last_num_dirs: Optional[int] = None
        self._last_n = None
        self._last_k = None

    # ---------- segmentation ----------
    @staticmethod
    def build_mask_legacy_hsv(image_bgr: np.ndarray, thresh: int = 6) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        t = int(np.clip(thresh, 0, 255))

        mask1 = cv2.inRange(
            hsv, np.array([0, t, t], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8)
        )
        mask2 = cv2.inRange(
            hsv, np.array([165, t, t], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)
        )
        roof_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return roof_mask

    def load_image(self, mask_mode: str = "legacy_hsv", legacy_thresh: int = 6) -> None:
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {self.image_path}")

        self.original_image = img.copy()
        self.image_shape = img.shape[:2]

        mask = mask_override(self.image_path)          # [EXT-MASK] external mask, if any
        if mask is None:
            mask = self.build_mask_legacy_hsv(img, legacy_thresh)

        self.roof_mask = mask
        self.binary_image = (mask > 0).astype(np.uint8) * 255
        self.image = self.binary_image.copy()
        self.roof_thresh = int(legacy_thresh)

    def sync_alias(self):
        if self.image is None and self.binary_image is not None:
            self.image = self.binary_image.copy()
        if self.binary_image is None and self.image is not None:
            self.binary_image = self.image.copy()
        if self.roof_mask is None and self.binary_image is not None:
            self.roof_mask = self.binary_image.copy()

    # ---------- polygons ----------
    def extract_polygons(
        self,
        min_area: int = 1000,
        border_margin: int = 0,
        object_polygon_mode: str = "och",
    ) -> None:
        """
        object_polygon_mode:
          - 'och'    : building = COCH (OrthoHullLib / OGraham.py)
          - 'convex' : convex hull
          - 'raw'    : contour thô
        """
        if self.binary_image is None:
            raise ValueError("Chưa có binary_image. Hãy load_image hoặc set binary_image trước.")

        binary255 = (self.binary_image > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = self.image_shape if self.image_shape is not None else binary255.shape[:2]

        self.contours = contours
        self.polygons = []
        self.objects = {}
        self.vertex_coordinates = {}
        self.geometries = {}
        self.bound = {}

        oid = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < float(min_area):
                continue

            pts = cnt.reshape(-1, 2)
            if len(pts) < 3:
                continue

            raw_poly = Polygon([(float(x), float(y)) for x, y in pts])
            if not raw_poly.is_valid:
                raw_poly = raw_poly.buffer(0)
            if raw_poly.is_empty:
                continue

            minx, miny, maxx, maxy = raw_poly.bounds
            if border_margin > 0:
                if minx <= border_margin or miny <= border_margin or maxx >= (w - border_margin) or maxy >= (h - border_margin):
                    continue

            if object_polygon_mode in ("och", "coch"):
                poly = oc_hull_from_polygon(raw_poly)
            elif object_polygon_mode == "convex":
                poly = raw_poly.convex_hull
            else:
                poly = raw_poly

            if poly is None or poly.is_empty:
                continue
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < float(min_area):
                continue

            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)

            self.polygons.append(poly)

            obj = BaseSectorAnalyzer.ObjectData()
            coords = list(poly.exterior.coords)[:-1]
            for vid, (x, y) in enumerate(coords, start=1):
                obj.add_vertex(vid, (float(x), float(y)))
            self.objects[oid] = obj
            self.vertex_coordinates[oid] = [(float(x), float(y)) for x, y in coords]
            self.geometries[oid] = poly
            self.bound[oid] = {
                "bbox": {
                    "minx": float(poly.bounds[0]),
                    "miny": float(poly.bounds[1]),
                    "maxx": float(poly.bounds[2]),
                    "maxy": float(poly.bounds[3]),
                }
            }
            oid += 1

        if len(self.polygons) > 0:
            self.multipolygon = MultiPolygon(self.polygons)
        else:
            self.multipolygon = MultiPolygon()

    def set_extra_polygons(self, extra_polygons: Optional[List[Polygon]]) -> None:
        """
        Set polygon phụ (ví dụ no-fly-zone).
        None / [] => vùng cấm bay rỗng (không cấm bay).
        """
        self.extra_polygons = []
        if not extra_polygons:
            return

        for p in extra_polygons:
            if p is None or p.is_empty:
                continue
            pp = p
            if not pp.is_valid:
                pp = pp.buffer(0)
            if pp.is_empty:
                continue
            if isinstance(pp, MultiPolygon):
                for g in pp.geoms:
                    if not g.is_empty and g.area > 0:
                        self.extra_polygons.append(g)
            elif isinstance(pp, Polygon):
                if pp.area > 0:
                    self.extra_polygons.append(pp)

    # ---------- features ----------
    def compute_features(
        self,
        n: int = 3,
        k: int = 3,
        max_radius_m: float = 100.0,
        circle_radii_m: Optional[Tuple[float, ...]] = None,
        sector_num_points: int = 140,
        extra_polygons: Optional[List[Polygon]] = None,
    ) -> None:
        # NFZ rỗng -> list rỗng
        if extra_polygons is None:
            extra_polygons = []

        if circle_radii_m is None:
            circle_radii_m = build_radii_m(n=n, max_radius_m=max_radius_m)
        else:
            circle_radii_m = tuple(circle_radii_m)

        self._last_circle_radii_m = circle_radii_m
        self._last_num_dirs = 2 ** int(k)
        self._last_n = int(n)
        self._last_k = int(k)

        all_polys = list(self.polygons)

        # extra from arg
        for p in extra_polygons:
            if p is None or p.is_empty:
                continue
            pp = p if p.is_valid else p.buffer(0)
            if pp.is_empty:
                continue
            if isinstance(pp, MultiPolygon):
                all_polys.extend([g for g in pp.geoms if not g.is_empty and g.area > 0])
            else:
                if pp.area > 0:
                    all_polys.append(pp)

        # extra from self
        for p in self.extra_polygons:
            if p is not None and (not p.is_empty) and p.area > 0:
                all_polys.append(p)

        records = []
        self.feature_dict = {}
        self.feature_sums = {}

        for oid in sorted(self.objects.keys()):
            coords = self.vertex_coordinates.get(oid, [])
            if len(coords) == 0:
                continue

            # [FAST] vectorized version, same sectors and outputs (features/fast_features.py)
            _compute = compute_sector_features_fast if USE_FAST_FEATURES else compute_sector_features_from_points_global
            out = _compute(
                input_points_xy=coords,
                all_polygons=all_polys,
                n=n,
                k=k,
                max_radius_m=max_radius_m,
                circle_radii_m=circle_radii_m,
                px_per_meter=self.px_per_meter,
                sector_num_points=sector_num_points
            )

            feat = out["feat"]
            vertex_total = out["vertex_total"]
            vtypes = och_vertex_types(coords)          # [COCH-V]

            self.feature_dict[oid] = {}
            obj_sum = 0.0

            for vid in range(len(coords)):
                row_feat = []
                for rm in circle_radii_m:
                    row_feat.extend(feat[rm][vid, :].astype(float).tolist())

                self.feature_dict[oid][vid + 1] = row_feat
                vtotal = float(vertex_total[vid])
                obj_sum += vtotal

                rec = {
                    "object_id": int(oid),
                    "vertex_id": int(vid + 1),
                    "x": float(coords[vid][0]),
                    "y": float(coords[vid][1]),
                    "vertex_total": vtotal,
                    "vertex_type": vtypes[vid],        # [COCH-V]
                }

                nd = 2 ** int(k)
                col_idx = 0
                for rm in circle_radii_m:
                    rname = int(round(float(rm)))
                    for d in range(nd):
                        rec[f"r{rname}_d{d}"] = float(row_feat[col_idx])
                        col_idx += 1

                records.append(rec)

            self.feature_sums[oid] = float(obj_sum)

        if len(records) == 0:
            self._feature_df = pd.DataFrame(
                columns=["object_id", "vertex_id", "x", "y", "vertex_total"]
            )
        else:
            df = pd.DataFrame(records)
            df = df.sort_values(["object_id", "vertex_id"]).reset_index(drop=True)
            self._feature_df = df

    def get_feature_table(self) -> pd.DataFrame:
        return self._feature_df.copy()

    def export_feature_csv(self, out_csv: str) -> None:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self._feature_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    def get_all_polygons_for_feature(self) -> List[Polygon]:
        out = list(self.polygons)
        out.extend(self.extra_polygons)
        return out
    #
    def plot_vertex_sectors(
        self,
        building_id: int,
        vertex_id: int,
        out_path: str = "sector_demo.png",
        n: int = 3,
        k: int = 3,
        max_radius_m: float = 100.0,
        circle_radii_m: Optional[Tuple[float, ...]] = None,
        sector_num_points: int = 140
    ):
        if building_id not in self.objects:
            raise ValueError(f"Không có building_id={building_id}")
        if vertex_id not in self.objects[building_id].vertices:
            raise ValueError(f"Không có vertex_id={vertex_id} trong building_id={building_id}")

        vertex = self.objects[building_id].vertices[vertex_id]
        center = Point(vertex.coordinates[0], vertex.coordinates[1])

        num_dirs = 2 ** int(k)
        step = 360.0 / num_dirs
        if circle_radii_m is None:
            circle_radii_m = build_radii_m(n=n, max_radius_m=max_radius_m)

        fig, ax = plt.subplots(figsize=(8, 8))

        for oid, poly in enumerate(self.polygons, start=1):
            if poly is None or poly.is_empty:
                continue

            x, y = poly.exterior.xy
            xc, yc = poly.convex_hull.exterior.xy

            if oid == building_id:
                ax.plot(xc, yc, color="blue", linestyle="--", linewidth=1.2, label=f"Convex Hull {oid}")
                ax.plot(x, y, color="black", linewidth=2.2, label=f"Building {oid} (target)")
                ax.fill(x, y, color="gray", alpha=0.45)
            else:
                ax.plot(x, y, color="gray", linewidth=1.2, alpha=0.9)
                ax.fill(x, y, color="cyan", alpha=0.20)

        cmap = plt.cm.get_cmap("tab10", len(circle_radii_m))

        for ri, r in enumerate(circle_radii_m):
            rp = meters_to_pixels(r, self.px_per_meter)
            color = cmap(ri)

            for i in range(num_dirs):
                sa = i * step
                ea = (i + 1) * step
                sec = generate_sector_polygon(center, rp, sa, ea, num_points=sector_num_points)

                sx, sy = sec.exterior.xy
                ax.plot(sx, sy, color=color, alpha=0.80, linewidth=0.8)

                for poly in self.polygons:
                    if poly is None or poly.is_empty:
                        continue
                    inter = sec.intersection(poly)
                    if inter.is_empty:
                        continue

                    if isinstance(inter, Polygon):
                        geoms = [inter]
                    else:
                        geoms = [g for g in getattr(inter, "geoms", []) if isinstance(g, Polygon)]

                    for g in geoms:
                        gx, gy = g.exterior.xy
                        ax.fill(gx, gy, color=color, alpha=0.28)

        ax.scatter([center.x], [center.y], color="red", s=65, zorder=10, label=f"Vertex {vertex_id}")
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_title(
            f"Building {building_id} - Vertex {vertex_id}\n"
            f"R_max={max_radius_m}m, n={len(circle_radii_m)}, k={k}, sectors={len(circle_radii_m)*(2**k)}"
        )
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_path, dpi=220)
        plt.show()
        plt.close(fig)


class Aerial(BaseSectorAnalyzer):
    pass


class ReferenceAnalyzer(BaseSectorAnalyzer):
    def estimate_rmax_from_vertex_to_other_polygons(
        self,
        shrink_factor: float = 0.95,
        min_rmax_m: float = 0.1,
        fallback_rmax_m: float = 10.0,
        use_convex_hull: bool = True
    ) -> Dict[str, float]:
        """
        R_max dựa trên:
        - Với mỗi đỉnh của mỗi polygon i,
        - tính khoảng cách từ đỉnh đó tới mọi polygon j != i bằng Shapely:
            d = Point(vx, vy).distance(polygon_j)
        - lấy khoảng cách nhỏ nhất toàn cục min_dist_px
        - R_max = shrink_factor * min_dist_px / px_per_meter

        Trả về:
        {
            "min_dist_px": ...,
            "min_dist_m": ...,
            "rmax_px": ...,
            "rmax_m": ...
        }
        """
        if not (0.0 < shrink_factor < 1.0):
            raise ValueError("shrink_factor phải nằm trong (0,1).")
        if self.px_per_meter is None or self.px_per_meter <= 0:
            raise ValueError("px_per_meter phải > 0.")

        # Chuẩn bị danh sách polygon hợp lệ
        polys: List[Polygon] = []
        for p in self.polygons:
            if p is None or p.is_empty or p.area <= 0:
                continue
            pp = p.convex_hull if use_convex_hull else p
            if pp is not None and (not pp.is_empty) and pp.area > 0:
                polys.append(pp)

        # Không đủ 2 polygon thì fallback
        if len(polys) < 2:
            rmax_m = float(max(min_rmax_m, fallback_rmax_m))
            return {
                "min_dist_px": float("nan"),
                "min_dist_m": float("nan"),
                "rmax_px": float(rmax_m * self.px_per_meter),
                "rmax_m": float(rmax_m),
            }

        # Cache vertices của từng polygon
        vertices_cache: List[List[Tuple[float, float]]] = [hull_vertices_xy_from_polygon(p) for p in polys]

        min_dist_px = np.inf

        # Duyệt từng đỉnh của từng polygon
        for i, verts_i in enumerate(vertices_cache):
            if len(verts_i) == 0:
                continue

            for (vx, vy) in verts_i:
                pt = Point(vx, vy)

                # khoảng cách nhỏ nhất từ đỉnh này đến các polygon khác
                d_vertex_min = np.inf
                for j, pj in enumerate(polys):
                    if j == i:
                        continue
                    d = pt.distance(pj)  # <-- shapely point-to-polygon distance
                    if d < d_vertex_min:
                        d_vertex_min = d

                if d_vertex_min < min_dist_px:
                    min_dist_px = d_vertex_min

        # Nếu dữ liệu lỗi hoặc có overlap/chạm => min_dist_px có thể 0
        if (not np.isfinite(min_dist_px)) or (min_dist_px <= 0):
            rmax_m = float(max(min_rmax_m, fallback_rmax_m))
            return {
                "min_dist_px": float(min_dist_px if np.isfinite(min_dist_px) else np.nan),
                "min_dist_m": float((min_dist_px / self.px_per_meter) if np.isfinite(min_dist_px) else np.nan),
                "rmax_px": float(rmax_m * self.px_per_meter),
                "rmax_m": float(rmax_m),
            }

        rmax_px = float(min_dist_px * shrink_factor)
        rmax_m = float(rmax_px / self.px_per_meter)
        rmax_m = float(max(min_rmax_m, rmax_m))

        return {
            "min_dist_px": float(min_dist_px),
            "min_dist_m": float(min_dist_px / self.px_per_meter),
            "rmax_px": float(rmax_m * self.px_per_meter),
            "rmax_m": float(rmax_m),
        }

    pass
