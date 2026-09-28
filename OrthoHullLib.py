# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
# OrthoHullLib.py
"""
Connected orthogonal convex hull (COCH) of a building contour.

The hull is computed by o_graham() of OGraham.py, the modified Graham scan of
An, Huyen and Le (Appl. Math. Comput. 397 (2021) 125889). The COCH is an
(x, y)-polygon: its convex vertices are the extreme points of the COCH, which
belong to the input point set, and its reflex vertices are the corners of the
staircases between them.

The ring returned by o_graham() may repeat a point or contain a straight-angle
point; clean_ring() removes them so that shapely gets a simple polygon.
The public API (OrthoHullLib().get_ortho_hull) is the one used by features/core.py.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
from shapely import Point

from OGraham import o_graham

PointLike = Union[Point, Tuple[float, float], List[float]]


def _xy(p: PointLike) -> Tuple[float, float]:
    if isinstance(p, Point) or (hasattr(p, "x") and hasattr(p, "y")):
        return float(p.x), float(p.y)
    return float(p[0]), float(p[1])


def clean_ring(ring: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    """Open ring without repeated points, straight-angle points or zero-width spikes."""
    r = [(float(p[0]), float(p[1])) for p in ring]
    if len(r) > 1 and r[0] == r[-1]:
        r = r[:-1]
    changed = True
    while changed and len(r) >= 3:
        changed = False
        out: List[Tuple[float, float]] = []
        for p in r:
            if out and out[-1] == p:
                changed = True
                continue
            out.append(p)
        if len(out) > 1 and out[0] == out[-1]:
            out.pop()
            changed = True
        r = out
        m = len(r)
        for i in range(m):
            a, b, c = r[i - 1], r[i], r[(i + 1) % m]
            if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) == 0:
                r = r[:i] + r[i + 1:]           # collinear: straight point or spike tip
                changed = True
                break
    return r


class OrthoHullLib:
    __slots__ = ()

    def get_ortho_hull(self, points: Sequence[PointLike], close_ring: bool = True,
                       return_points: bool = False):
        """COCH of the points (O-Graham) as a list of (x, y); closed if close_ring."""
        pts = [_xy(p) for p in points]
        if len(pts) < 3:
            ring = pts
        else:
            ring = clean_ring(o_graham(np.asarray(pts, dtype=float)).tolist())
        if close_ring and len(ring) > 1 and ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        if return_points:
            return [Point(x, y) for x, y in ring]
        return ring

    # name kept for compatibility with older scripts
    def oc_hull_from_points(self, points, close_ring: bool = True, return_points: bool = False):
        return self.get_ortho_hull(points, close_ring=close_ring, return_points=return_points)
