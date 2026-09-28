# -*- coding: utf-8 -*-
# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
"""
plot_reference_maps.py

Reference-map figures of Examples 4.2 and 4.3 (Figures 3 and 6 of the paper):
the satellite image with the building polygons used in the experiments
(numbered as in the feature files), the pair of polygons that fixes the radius
R (R = 0.95 x their distance, as in get_Global_R_max.py) and, for Example 4.3,
the no-fly zone.

Usage (from the project root):  python plot_reference_maps.py
Output: results_examples/<example>/reference_map.pdf / .png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon as MplPolygon
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from plot_localization import load_nofly, load_polygons, NFZ_COL, NFZ_EDGE
from features.core import och_vertex_types

OUT_ROOT = os.environ.get("RESULTS_ROOT", "results_examples")
REFERENCE_IMAGE = os.path.join("references", "refmap.png")
PX_PER_METER = 2.445496
EDGE = "#00d5e6"


def witness(polys, nfz, extreme_only=True):
    """Smallest distance from a polygon vertex to another polygon (get_Global_R_max rule);
    extreme_only: only the extreme (convex) vertices of the COCH, where the descriptor is used."""
    items = [(f"#{o}", p) for o, p in polys.items()] + [("NFZ", z) for z in nfz]
    best = (np.inf, None, None, None, None)
    for i, (ni, pi) in enumerate(items):
        verts = list(pi.exterior.coords)[:-1]
        if extreme_only:
            verts = [v for v, t in zip(verts, och_vertex_types(verts)) if t == "convex"]
        for vx, vy in verts:
            pt = Point(vx, vy)
            for j, (nj, pj) in enumerate(items):
                if i == j:
                    continue
                d = pt.distance(pj)
                if 1e-9 < d < best[0]:
                    a, b = nearest_points(pt, pj)
                    best = (d, (a.x, a.y), (b.x, b.y), ni, nj)
    return best


def main():
    img = cv2.cvtColor(cv2.imread(REFERENCE_IMAGE), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    for ex in ("ex4_2", "ex4_3"):
        base = os.path.join(OUT_ROOT, ex)
        if not os.path.isfile(os.path.join(base, "radius.json")):
            print(f"{base}: no results, skipped")
            continue
        R = json.load(open(os.path.join(base, "radius.json")))
        polys = load_polygons(os.path.join(base, "references", "reference_n3_k3.csv"))
        nfz = load_nofly() if ex == "ex4_3" else []
        d, a, b, na, nb = witness(polys, nfz, R.get("descriptor_vertices", "extreme") == "extreme")
        fig, ax = plt.subplots(figsize=(11, 6.6))
        ax.imshow(img)
        for o, p in polys.items():
            ax.add_patch(MplPolygon(np.asarray(p.exterior.coords), closed=True, fc="none", ec=EDGE, lw=1.8))
            c = p.representative_point()
            ax.text(c.x, c.y, str(o), color="#ffffff", fontsize=10, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#0b0b0b", ec="none", alpha=0.8))
        for z in nfz:
            ax.add_patch(MplPolygon(np.asarray(z.exterior.coords), closed=True, fc=NFZ_COL, alpha=0.45,
                                    ec="none"))
            ax.add_patch(MplPolygon(np.asarray(z.exterior.coords), closed=True, fc="none",
                                    ec=NFZ_EDGE, lw=2.5))
            ax.text(z.centroid.x, z.centroid.y, "no-fly zone", color="#ffffff", fontsize=11, ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc=NFZ_COL, ec="none"))
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#ffd400", lw=2.5)
        ax.add_patch(Circle(a, R["R_px"], fc="none", ec="#ffd400", lw=2))
        ax.plot(*a, "o", color="#d03b3b", ms=5)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_xticks([]); ax.set_yticks([])
        bar = 50 * PX_PER_METER
        ax.plot([15, 15 + bar], [H - 18, H - 18], color="#ffffff", lw=4)
        ax.text(15 + bar / 2, H - 26, "50 m", color="#ffffff", ha="center", va="bottom", fontsize=11)
        handles = [Patch(fc="none", ec=EDGE, lw=1.8, label=f"building polygon ({len(polys)})"),
                   Line2D([], [], color="#ffd400", lw=2.5,
                          label=f"closest pair {na}-{nb}: {d / PX_PER_METER:.2f} m, R = 0.95 x = {R['R_m']:.2f} m")]
        if nfz:
            handles.append(Patch(fc=(0.922, 0.408, 0.204, 0.45), ec=NFZ_EDGE, lw=2.5, label="no-fly zone (obstacle)"))
        ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)
        fig.tight_layout()
        out = os.path.join(base, "reference_map")
        fig.savefig(out + ".pdf")
        fig.savefig(out + ".png", dpi=200)
        plt.close(fig)
        print(f"{out}.pdf  closest pair {na}-{nb} {d / PX_PER_METER:.3f} m")


if __name__ == "__main__":
    main()
