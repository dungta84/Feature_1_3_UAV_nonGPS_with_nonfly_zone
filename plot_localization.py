# -*- coding: utf-8 -*-
# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
"""
plot_localization.py

Localization map for every test crop of Examples 4.2 and 4.3 (results of
run_examples.py), in the layout: (a) reference map with the image footprint,
the coarse candidates and the best match; (b) close-up on the footprint;
(c) zoom on the UAV position with the error.

The UAV position is the centre of the aerial crop. TRUE = centre of the crop
at its true offset (crop_x1, crop_y1 in test_meta.csv); ESTIMATED = centre of
the crop moved by the translation estimated from the best-matched building
(median of the vertex correspondences, run_matching_all_v3.py).

Usage (from the project root):
    python plot_localization.py                 # both examples, (n, k) = (5, 5)
    python plot_localization.py ex4_2 3 3       # one example, chosen (n, k)
Output: results_examples/<example>/localization/n{n}_k{k}/loc_<test>.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle
from shapely.geometry import Polygon, box

OUT_ROOT = os.environ.get("RESULTS_ROOT", "results_examples")
REFERENCE_IMAGE = os.path.join("references", "refmap.png")
NOFLY_CSV = os.path.join("data", "references", "non_fly_zone.csv")
PX_PER_METER = 2.445496

# colours (text in ink tones, identity carried by the marks)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
REF_EDGE, REF_FILL = "#9a9994", "#ffffff"
FOOT_FILL = "#f4f1ea"
CAND_FILL, CAND_EDGE = "#b7d3f6", "#256abf"
BEST_FILL, BEST_EDGE = "#1c3557", "#d03b3b"
EST_COL, TRUE_COL = "#d03b3b", "#2a78d6"
NFZ_COL = "#eb6834"
NFZ_EDGE = "#b8441a"      # darker edge of the no-fly zone


def load_polygons(ref_csv):
    df = pd.read_csv(ref_csv)
    polys = {}
    for oid, g in df.groupby("object_id"):
        xy = g.sort_values("vertex_id")[["x", "y"]].to_numpy(float)
        p = Polygon(xy)
        polys[int(oid)] = p if p.is_valid else p.buffer(0)
    return polys


def load_nofly():
    if not os.path.isfile(NOFLY_CSV):
        return []
    df = pd.read_csv(NOFLY_CSV, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    out = []
    for _, g in df[df.get("enabled", 1) == 1].groupby("zone_id"):
        if str(g.zone_type.iloc[0]).lower() == "polygon":
            out.append(Polygon(g.sort_values("vertex_order")[["x", "y"]].to_numpy(float)))
        else:
            r = g.iloc[0]
            out.append(Polygon([(r.x + r.radius * np.cos(a), r.y + r.radius * np.sin(a))
                                for a in np.linspace(0, 2 * np.pi, 180, endpoint=False)]))
    return out


def draw_poly(ax, poly, fc, ec, lw=0.8, z=2, hatch=None, alpha=1.0):
    geoms = getattr(poly, "geoms", [poly])
    for g in geoms:
        ax.add_patch(MplPolygon(np.asarray(g.exterior.coords), closed=True, fc=fc, ec=ec,
                                lw=lw, zorder=z, hatch=hatch, alpha=alpha))


def plot_one(example, n, k, test, meta_row, per_test_row, per_query, polys, nfz, img_shape, out_png, paper=False):
    x1, y1 = float(meta_row.crop_x1), float(meta_row.crop_y1)
    w, h = float(meta_row.crop_w), float(meta_row.crop_h)
    est_x, est_y = float(per_test_row.est_x), float(per_test_row.est_y)
    true_c = np.array([x1 + w / 2, y1 + h / 2])
    est_c = np.array([est_x + w / 2, est_y + h / 2])
    err_m = float(np.hypot(*(est_c - true_c))) / PX_PER_METER
    foot = box(est_x, est_y, est_x + w, est_y + h)

    q = per_query[per_query.aerial_file == test]
    cands = sorted({int(c) for s in q.candidates.astype(str) for c in s.split(",") if c.strip()})
    best = int(q.sort_values("valid_vertices", ascending=False).top1_object.iloc[0])
    gt = int(q.sort_values("valid_vertices", ascending=False).gt_object.iloc[0])
    inside = [o for o, p in polys.items() if p.intersection(foot).area >= 0.5 * p.area]

    H, W = img_shape
    # paper=True: no title lines (the caption describes the figure), tighter layout, larger text
    fig = plt.figure(figsize=(13, 6.6) if paper else (15.5, 9.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], height_ratios=[1.0, 1.25, 1.1],
                          left=0.06, right=0.99, top=0.96 if paper else 0.88, bottom=0.08,
                          wspace=0.12, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    for z in nfz:
        draw_poly(ax, z, NFZ_COL, "none", lw=0, z=1, alpha=0.45)
        draw_poly(ax, z, "none", NFZ_EDGE, lw=2.0, z=1)
    for o, p in polys.items():
        if o == best:
            draw_poly(ax, p, BEST_FILL, BEST_EDGE, lw=1.8, z=4)
        elif o in cands:
            draw_poly(ax, p, CAND_FILL, CAND_EDGE, lw=1.1, z=3)
        else:
            draw_poly(ax, p, REF_FILL, REF_EDGE, lw=0.8, z=2)
        c = p.representative_point()
        ax.text(c.x, c.y, str(o), ha="center", va="center", fontsize=9,
                color="#ffffff" if o == best else INK2, zorder=6, fontweight="bold" if o in cands else None)
    ax.add_patch(Rectangle((est_x, est_y), w, h, fc=FOOT_FILL, ec=INK, lw=1.8, zorder=1.5, alpha=0.9))
    ax.plot(*true_c, "o", color=TRUE_COL, ms=7, zorder=8)
    ax.plot(*est_c, "+", color=EST_COL, ms=16, mew=3, zorder=9)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("reference map x [px]", color=INK2)
    ax.set_ylabel("reference map y [px]", color=INK2)
    ax.tick_params(colors=INK2)
    for s in ax.spines.values():
        s.set_color(GRID)
    bar = 50 * PX_PER_METER
    ax.plot([20, 20 + bar], [H - 25, H - 25], color=INK, lw=4)
    ax.text(20 + bar / 2, H - 35, "50 m", ha="center", va="bottom", fontsize=10)
    ax.set_title("(a) Reference map", loc="left", fontsize=11)

    handles = [Patch(fc=REF_FILL, ec=REF_EDGE, label=f"Reference building ({len(polys)})"),
               Patch(fc=CAND_FILL, ec=CAND_EDGE, label=f"Coarse candidate ({len(cands)})"),
               Patch(fc=BEST_FILL, ec=BEST_EDGE, label="Best match (after coarse step)"),
               Patch(fc=FOOT_FILL, ec=INK, label="Image footprint at the estimated position"),
               Line2D([], [], marker="+", ls="none", color=EST_COL, ms=12, mew=3, label="UAV estimated"),
               Line2D([], [], marker="o", ls="none", color=TRUE_COL, ms=7, label="UAV true (crop centre)")]
    if nfz:
        handles.append(Patch(fc=(0.922, 0.408, 0.204, 0.45), ec=NFZ_EDGE, lw=2.0, label="No-fly zone (obstacle)"))
    lax = fig.add_subplot(gs[0, 1])
    lax.axis("off")
    lax.legend(handles=handles, loc="upper left", fontsize=10, frameon=True, edgecolor=GRID)

    bx = fig.add_subplot(gs[1, 1])
    pad = 40
    for z in nfz:
        draw_poly(bx, z, NFZ_COL, "none", lw=0, z=1, alpha=0.45)
        draw_poly(bx, z, "none", NFZ_EDGE, lw=2.0, z=1)
    for o, p in polys.items():
        fc, ec, lw = (BEST_FILL, BEST_EDGE, 1.8) if o == best else ((CAND_FILL, CAND_EDGE, 1.1) if o in cands else (REF_FILL, REF_EDGE, 0.8))
        draw_poly(bx, p, fc, ec, lw=lw, z=3)
        if o in cands or o in inside:
            c = p.representative_point()
            if est_x - pad < c.x < est_x + w + pad and est_y - pad < c.y < est_y + h + pad:
                bx.text(c.x, c.y, f"#{o}", ha="center", va="center", fontsize=9, zorder=7,
                        color="#ffffff" if o == best else INK,
                        bbox=None if o == best else dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=CAND_EDGE, lw=0.8))
    bx.add_patch(Rectangle((est_x, est_y), w, h, fc=FOOT_FILL, ec=INK, lw=1.8, zorder=1.5))
    bx.plot(*true_c, "o", color=TRUE_COL, ms=7, zorder=8)
    bx.plot(*est_c, "+", color=EST_COL, ms=14, mew=3, zorder=9)
    bx.set_xlim(est_x - pad, est_x + w + pad)
    bx.set_ylim(est_y + h + pad, est_y - pad)
    bx.set_aspect("equal")
    bx.set_xticks([]); bx.set_yticks([])
    bx.set_title("(b) Close-up on the image footprint", loc="left", fontsize=11)

    cx = fig.add_subplot(gs[2, 1])
    half_m = max(8.0, err_m * 1.3)
    half = half_m * PX_PER_METER
    step = 2.0 if half_m <= 10 else (10.0 if half_m <= 60 else 20.0)   # at most about 5 rings
    for r in np.arange(step, half_m + 0.1, step):
        cx.add_patch(plt.Circle(true_c, r * PX_PER_METER, fill=False, ls=":", ec=INK2, lw=0.8))
        cx.text(true_c[0], true_c[1] - r * PX_PER_METER, f"{r:.0f} m", fontsize=7, color=INK2,
                ha="center", va="bottom")
    cx.plot([true_c[0], est_c[0]], [true_c[1], est_c[1]], color=INK, lw=1)
    cx.plot(*true_c, "o", color=TRUE_COL, ms=8)
    cx.plot(*est_c, "+", color=EST_COL, ms=16, mew=3)
    cx.set_xlim(true_c[0] - half, true_c[0] + half)
    cx.set_ylim(true_c[1] + half, true_c[1] - half)
    cx.set_aspect("equal")
    cx.set_xticks([]); cx.set_yticks([])
    cx.set_title(f"(c) UAV position: error = {err_m:.2f} m", loc="left",
                 fontsize=11, color=EST_COL if err_m > 1 else INK)
    cx.set_xlabel(f"rings every {step:.0f} m around the true position", fontsize=9, color=INK2)

    title = "Example 4.3 (with no-fly zone)" if example == "ex4_3" else "Example 4.2 (no no-fly zone)"
    if not paper:
        fig.text(0.05, 0.955, f"UAV localization on the reference map: {title}, {test}, (n, k) = ({n}, {k})",
             fontsize=15, fontweight="bold", color=INK)
        fig.text(0.05, 0.925, f"true building #{gt}   best match #{best}   coarse candidates: "
             f"{', '.join('#' + str(c) for c in cands)}   buildings in footprint: {len(inside)}   "
                 f"R = {float(per_test_row.R_m):.2f} m   error = {err_m:.2f} m", fontsize=11, color=INK2)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    if out_png.endswith(".png"):
        fig.savefig(out_png[:-4] + ".pdf")
    plt.close(fig)
    return err_m, best, gt, cands


def run(example, n, k):
    base = os.path.join(OUT_ROOT, example)
    meta = pd.read_csv(os.path.join(base, "aerials", "test_meta.csv"), encoding="utf-8-sig").set_index("file_name")
    pt = pd.read_csv(os.path.join(base, "matching", "per_test.csv"), encoding="utf-8-sig")
    pq = pd.read_csv(os.path.join(base, "matching", "per_query.csv"), encoding="utf-8-sig")
    R_m = float(pd.read_json(os.path.join(base, "radius.json"), typ="series")["R_m"])
    polys = load_polygons(os.path.join(base, "references", f"reference_n{n}_k{k}.csv"))
    nfz = load_nofly() if example == "ex4_3" else []
    import cv2
    img = cv2.imread(REFERENCE_IMAGE)
    shape = img.shape[:2] if img is not None else (768, 1358)
    pt = pt[(pt.n == n) & (pt.k == k)]
    pq = pq[(pq.n == n) & (pq.k == k)]
    rows = []
    for _, r in pt.iterrows():
        r = r.copy()
        r["R_m"] = R_m
        out = os.path.join(base, "localization", f"n{n}_k{k}", f"loc_{r.aerial_file[:-4]}.png")
        err, best, gt, cands = plot_one(example, n, k, r.aerial_file, meta.loc[r.aerial_file], r, pq,
                                        polys, nfz, shape, out)
        rows.append({"test": r.aerial_file, "true_building": gt, "best_match": best,
                     "num_candidates": len(cands), "candidates": " ".join(map(str, cands)),
                     "error_m": round(err, 2)})
        print(f"[{example}] {out}  error {err:.2f} m", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(base, "localization", f"n{n}_k{k}", "summary.csv"),
                              index=False, encoding="utf-8-sig")


def run_paper(example, test, n, k):
    """Compact version of one localization map for the paper (no title, larger text)."""
    plt.rcParams.update({"font.size": 12})
    base = os.path.join(OUT_ROOT, example)
    meta = pd.read_csv(os.path.join(base, "aerials", "test_meta.csv"), encoding="utf-8-sig").set_index("file_name")
    pt = pd.read_csv(os.path.join(base, "matching", "per_test.csv"), encoding="utf-8-sig")
    pq = pd.read_csv(os.path.join(base, "matching", "per_query.csv"), encoding="utf-8-sig")
    R_m = float(pd.read_json(os.path.join(base, "radius.json"), typ="series")["R_m"])
    polys = load_polygons(os.path.join(base, "references", f"reference_n{n}_k{k}.csv"))
    nfz = load_nofly() if example == "ex4_3" else []
    r = pt[(pt.n == n) & (pt.k == k) & (pt.aerial_file == test)].iloc[0].copy()
    r["R_m"] = R_m
    out = os.path.join(base, "localization", f"paper_{test[:-4]}_n{n}_k{k}.png")
    plot_one(example, n, k, test, meta.loc[test], r, pq[(pq.n == n) & (pq.k == k)], polys, nfz,
             (768, 1358), out, paper=True)
    print(out[:-4] + ".pdf")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "paper":          # python plot_localization.py paper ex4_2 test_04.png 5 5
        run_paper(args[1], args[2], int(args[3]), int(args[4]))
    elif len(args) >= 3:
        run(args[0], int(args[1]), int(args[2]))
    else:
        for ex in (args or ["ex4_2", "ex4_3"]):
            run(ex, 5, 5)
