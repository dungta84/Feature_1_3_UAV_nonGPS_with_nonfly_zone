# -*- coding: utf-8 -*-
# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
"""
plot_descriptor.py

Figure of the descriptor construction (Figure 4 of the paper), drawn as a
vector figure: (a) an aerial crop with the extracted building polygons and the
selected vertex x*; (b) the disk U(x*) of radius R split into n rings and 2^k
sectors, with the building area in every cell (the components w_i of W(x*)).
The cell values are read from the feature CSV written by run_examples.py, so
the figure shows the numbers used in the experiments.

Usage (from the project root):
    python plot_descriptor.py                        # ex4_2, test_04, (n, k) = (5, 5), building 4
    python plot_descriptor.py ex4_2 test_04.png 5 5 4   # last argument: reference building
Output: results_examples/<example>/descriptor_<test>_n{n}_k{k}.pdf / .png
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Polygon as MplPolygon, Rectangle, Wedge
from shapely.geometry import Polygon

OUT_ROOT = os.environ.get("RESULTS_ROOT", "results_examples")
PX_PER_METER = 2.445496
INK, INK2 = "#0b0b0b", "#52514e"
EDGE = "#d03b3b"
BLUE = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])


def main(example="ex4_2", test="test_04.png", n=5, k=5, building=4):
    """building: reference building number to show (its aerial polygon is found in
    matching/per_query.csv); None: the largest building of the aerial image."""
    base = os.path.join(OUT_ROOT, example)
    R_px = json.load(open(os.path.join(base, "radius.json")))["R_px"]
    aer = pd.read_csv(os.path.join(base, "aerials", "features_grid", f"aerial_n{n}_k{k}.csv"))
    aer.columns = [c.strip().lower() for c in aer.columns]
    g = aer[aer.aerial_file == test]
    img = cv2.cvtColor(cv2.imread(os.path.join(base, "aerials", test)), cv2.COLOR_BGR2RGB)
    Hc, Wc = img.shape[:2]
    cols = [c for c in g.columns if re.match(r"^r[\d.]+_d\d+$", c)]
    cols = sorted(cols, key=lambda c: (float(re.match(r"^r([\d.]+)_d(\d+)$", c).group(1)),
                                       int(re.match(r"^r([\d.]+)_d(\d+)$", c).group(2))))
    s = 2 ** k

    polys = {int(o): gg.sort_values("vertex_id") for o, gg in g.groupby("object_id")}
    # the largest building of the crop, and on it the valid vertex with the most occupied cells
    main_obj = None
    if building is not None:
        pq = pd.read_csv(os.path.join(base, "matching", "per_query.csv"), encoding="utf-8-sig")
        hit = pq[(pq.n == n) & (pq.k == k) & (pq.aerial_file == test) & (pq.gt_object == int(building))]
        if len(hit) == 0:
            print(f"building {building} is not a query of {test} at (n, k) = ({n}, {k}); "
                  f"the largest building of the image is shown")
        else:
            main_obj = int(hit.query_object.iloc[0])
    if main_obj is None:
        main_obj = max(polys, key=lambda o: Polygon(polys[o][["x", "y"]].to_numpy()).area)
    gg = polys[main_obj]
    xy = gg[["x", "y"]].to_numpy(float)
    valid = (xy[:, 0] >= R_px) & (xy[:, 1] >= R_px) & (xy[:, 0] <= Wc - 1 - R_px) & (xy[:, 1] <= Hc - 1 - R_px)
    # x* is an extreme (convex) vertex of the COCH, where the descriptor is used
    if "vertex_type" in gg.columns:
        ext = (gg["vertex_type"].astype(str) == "convex").to_numpy()
    else:
        ext = np.ones(len(gg), dtype=bool)
    valid &= ext
    W = gg[cols].to_numpy(float)
    # among them, a clear outer corner of the COCH: both edges at x* at least 0.4 R long and
    # the building covering about a quarter of the disk (not a one-pixel step of a staircase)
    e_prev = np.hypot(*(xy - np.roll(xy, 1, axis=0)).T)
    e_next = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
    frac = W.sum(axis=1) / (np.pi * R_px ** 2)
    clear = valid & (np.minimum(e_prev, e_next) >= 0.4 * R_px)
    pool = clear if clear.any() else valid
    score = np.where(pool, -np.abs(frac - 0.25), -1e9)
    vi = int(np.argmax(score))
    x0, y0 = xy[vi]
    w = W[vi].reshape(n, s) / PX_PER_METER ** 2                     # m^2 per cell

    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 0.05], wspace=0.08,
                          left=0.03, right=0.94, top=0.9, bottom=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img, alpha=0.55)
    for o, pg in polys.items():
        p = pg[["x", "y"]].to_numpy(float)
        ax.add_patch(MplPolygon(p, closed=True, fc="none", ec=EDGE if o == main_obj else "#2a78d6",
                                lw=2.0 if o == main_obj else 1.2))
    ax.plot(xy[ext, 0], xy[ext, 1], "o", ms=2.2, color=EDGE, zorder=4)
    ax.add_patch(Circle((x0, y0), R_px, fc="none", ec=INK, lw=1.2))
    ax.plot(x0, y0, "o", color=INK, ms=5)
    zoom = 1.9 * R_px
    ax.add_patch(Rectangle((x0 - zoom, y0 - zoom), 2 * zoom, 2 * zoom, fc="none", ec=INK, lw=0.8, ls="--"))
    ax.annotate(r"$x^*$", (x0, y0), xytext=(x0 + 1.4 * R_px, y0 - 1.4 * R_px), fontsize=13,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.set_xlim(0, Wc); ax.set_ylim(Hc, 0)
    ax.set_xticks([]); ax.set_yticks([])
    bar = 20 * PX_PER_METER
    ax.plot([8, 8 + bar], [Hc - 10, Hc - 10], color=INK, lw=3)
    ax.text(8 + bar / 2, Hc - 14, "20 m", ha="center", va="bottom", fontsize=10)
    ax.set_title("(a) Aerial image, COCH of the buildings, extreme vertices (dots) and $x^*$", loc="left", fontsize=11)

    bx = fig.add_subplot(gs[0, 1])
    vmax = max(float(w.max()), 1e-9)
    norm = Normalize(0, vmax)
    dr = R_px / n
    for j in range(n):
        for i in range(s):
            a0, a1 = 360.0 * i / s, 360.0 * (i + 1) / s
            # same convention as features/core.py: angle from +x towards +y in image
            # coordinates (y down); the axis is inverted, so Wedge uses the angles as they are
            bx.add_patch(Wedge((x0, y0), (j + 1) * dr, a0, a1, width=dr, fc=BLUE(norm(w[j, i])),
                               ec="#9a9994", lw=0.5))
    p = gg[["x", "y"]].to_numpy(float)
    for o, pg in polys.items():
        bx.add_patch(MplPolygon(pg[["x", "y"]].to_numpy(float), closed=True, fc="none",
                                ec=EDGE if o == main_obj else "#2a78d6", lw=2.2))
    bx.plot(xy[ext, 0], xy[ext, 1], "o", ms=4, color=EDGE, zorder=4)
    bx.plot(xy[~ext, 0], xy[~ext, 1], "s", ms=3.5, mfc="white", mec=EDGE, mew=1.0, zorder=4)
    bx.plot(x0, y0, "o", color=INK, ms=6, zorder=5)
    bx.set_xlim(x0 - zoom, x0 + zoom); bx.set_ylim(y0 + zoom, y0 - zoom)
    bx.set_aspect("equal")
    bx.set_xticks([]); bx.set_yticks([])
    for sp in bx.spines.values():
        sp.set_linestyle("--")
    bx.set_title(f"(b) $U(x^*)$: {n} rings $\\times$ {s} sectors ($N$ = {n * s}), R = {R_px / PX_PER_METER:.2f} m",
                 loc="left", fontsize=11)
    cax = fig.add_subplot(gs[0, 2])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=BLUE), cax=cax)
    cb.set_label(r"building area in the cell, $w_i(x^*)$ [m$^2$]", color=INK2)
    out = os.path.join(base, f"descriptor_{test[:-4]}_n{n}_k{k}")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(f"{out}.pdf  vertex {vi + 1} of object {main_obj}, occupied cells {(w > 0.01).sum()} / {n * s}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 4:
        main(a[0], a[1], int(a[2]), int(a[3]), int(a[4]) if len(a) >= 5 else None)
    else:
        main()
