# -*- coding: utf-8 -*-
# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
"""
run_examples.py

Recomputes Examples 4.2 and 4.3 of the paper from the reference map, with the
same candidate rule for both (run_matching_all_v3.py: L1 distance between shape
vectors, majority vote of the valid vertices, one tolerance for every (n, k)).

  ex4_2 : no no-fly zone.
  ex4_3 : the no-fly zone of data/references/non_fly_zone.csv is an obstacle
          (paper definition): buildings that meet it are removed, aerial crops
          that meet it are not used, it bounds the radius R, and its area is
          not added to the shape vector.

For each example the radius R is computed automatically, as in
get_Global_R_max.py: 0.95 x the smallest distance from a polygon vertex to
another polygon (with the no-fly zone included for ex4_3).

Outputs, for each example, in results_examples/<example>/:
  radius.json                          R and the pair of polygons that fixes it
  references/reference_n{n}_k{k}.csv   reference features
  aerials/test_*.png, test_meta.csv    aerial crops
  aerials/features_grid/aerial_n{n}_k{k}.csv
  test_overview.png
  matching/summary_nk.csv, per_query.csv, per_test.csv, sweep_eps.csv, separation.csv
  log.txt
Options (run from the project root):
  python run_examples.py                       HSV roof threshold, COCH extreme vertices
  python run_examples.py --vertices all        every COCH vertex (staircase corners included)
  python run_examples.py --radius-m 5.83       fixed R instead of the automatic value
  python run_examples.py ex4_2                 one example only
"""
import argparse
import json
import shutil
import os
import sys
import time

# The scripts print Vietnamese text. On Windows, when the output goes to a file,
# Python may use cp1252 and fail on those characters; force UTF-8 here (this also
# runs in the worker processes, which import this module).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from concurrent.futures import ProcessPoolExecutor, as_completed

from features.task_reference import (
    _prepare_reference_analyzer, compute_global_rmax_with_witness, replace_polygons_by_nfz,
    run_reference_task,
)
from features.core import load_nofly_csv_as_polygons
from features.task_aerials import run_aerials_task
from pipelines.gen_test_data import run_gen_test_data

REFERENCE_IMAGE = os.path.join("references", "refmap.png")
NOFLY_CSV = os.path.join("data", "references", "non_fly_zone.csv")
OUT_ROOT = "results_examples"

PX_PER_METER = 2.445496
REF_THRESH, REF_MIN_AREA = 18, 2850        # as in run_reference.py
AER_THRESH, AER_MIN_AREA = 16, 1250        # as in run_aerials.py (building detection inside a crop)
# The test crops are cut around the SAME buildings as the reference map and the
# radius R (threshold REF_THRESH, minimum area REF_MIN_AREA): one crop per building,
# 12 buildings. Earlier the crops were chosen with AER_THRESH / AER_MIN_AREA, which
# gave 25 objects (small roofs that are not in the reference map).
SECTOR_POINTS = 96
N_LIST = [3, 4, 5]
K_LIST = [3, 4, 5]

# [COCH-V] set from the command line in main()
VERTICES = "extreme"        # "extreme" (COCH extreme vertices) | "all"
RADIUS_M = None             # None: automatic R (0.95 x smallest vertex-to-polygon distance)

EXAMPLES = {
    "ex4_2": {"nofly_csv": None},
    "ex4_3": {"nofly_csv": NOFLY_CSV},
}


def log(msg, fh):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def compute_radius(nofly_csv):
    ref = _prepare_reference_analyzer(REFERENCE_IMAGE, PX_PER_METER, True, REF_THRESH)
    ref.extract_polygons(min_area=REF_MIN_AREA, border_margin=0, object_polygon_mode="och")
    polys = list(ref.polygons)
    stats = {"buildings": len(polys)}
    if nofly_csv:
        nfz = load_nofly_csv_as_polygons(nofly_csv)
        polys, dropped, st = replace_polygons_by_nfz(polys, nfz, drop_if_intersects=True)
        stats.update(st)
    info = compute_global_rmax_with_witness(polys, PX_PER_METER, extreme_only=(VERTICES == "extreme"))
    return float(info["rmax_m"]), info, stats


def job_reference(args):
    n, k, out_csv, nofly_csv, r_m = args
    run_reference_task(reference_image=REFERENCE_IMAGE, out_csv=out_csv, nofly_csv=nofly_csv,
                       px_per_meter=PX_PER_METER, roof_thresh=REF_THRESH, min_area=REF_MIN_AREA,
                       border_margin=0, object_polygon_mode="och", n=n, k=k, max_radius_m=r_m,
                       sector_num_points=SECTOR_POINTS, save_overview=False,
                       nfz_as_obstacle=True, verbose=False)
    return n, k


def job_aerial(args):
    n, k, aerial_dir, out_csv, r_m = args
    df = run_aerials_task(aerial_dir=aerial_dir, out_csv=out_csv, px_per_meter=PX_PER_METER,
                          roof_thresh=AER_THRESH, min_area=AER_MIN_AREA, border_margin=0,
                          object_polygon_mode="och", n=n, k=k, max_radius_m=r_m,
                          sector_num_points=SECTOR_POINTS, image_glob="test_*.png", max_workers=1,
                          csv_decimals=6, verbose=False)
    return n, k, len(df)


def run_example(name, cfg, workers):
    out = os.path.join(OUT_ROOT, name)
    if os.path.isdir(out):
        shutil.rmtree(out)                  # no stale crops or CSV files from an earlier run
    for sub in ("references", "aerials", os.path.join("aerials", "features_grid"), "matching"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    fh = open(os.path.join(out, "log.txt"), "w", encoding="utf-8")
    t0 = time.time()
    nofly = cfg["nofly_csv"]
    log(f"[{name}] descriptor vertices: {VERTICES}", fh)

    r_m, info, stats = compute_radius(nofly)
    r_auto = r_m
    if RADIUS_M:
        log(f"[{name}] automatic R = {r_m:.3f} m replaced by --radius-m {RADIUS_M} m", fh)
        r_m = float(RADIUS_M)
    if r_m < 2.0:
        log(f"[{name}] WARNING: R = {r_m:.3f} m is small: two building polygons almost touch; "
            f"see radius.json (witness_pair) and masks_overlay.png", fh)
    json.dump({"R_m": r_m, "R_px": r_m * PX_PER_METER, "min_dist_m": info["min_dist_m"],
               "witness_pair": info["witness_pair"], "polygon_stats": stats,
               "segmentation": "hsv", "descriptor_vertices": VERTICES, "R_auto_m": r_auto},
              open(os.path.join(out, "radius.json"), "w"), indent=2)
    log(f"[{name}] R = {r_m:.3f} m (min distance {info['min_dist_m']:.3f} m, x 0.95); {stats}", fh)

    jobs = [(n, k, os.path.join(out, "references", f"reference_n{n}_k{k}.csv"), nofly, r_m)
            for n in N_LIST for k in K_LIST]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(job_reference, j) for j in jobs]):
            n, k = f.result()
            log(f"[{name}] reference n={n} k={k} done ({time.time() - t0:.0f} s)", fh)

    aerial_dir = os.path.join(out, "aerials")
    meta = run_gen_test_data(reference_image=REFERENCE_IMAGE, out_dir=aerial_dir, border_margin=0,
                             object_polygon_mode="och", fixed_thresh=REF_THRESH,
                             fixed_min_area_och=REF_MIN_AREA, global_extra_ratio=0.05,
                             sort_by="object_id",
                             test_overview_path=os.path.join(out, "test_overview.png"),
                             nofly_csv=nofly if nofly else "", enable_nofly_filter=bool(nofly),
                             nofly_intersection_eps=0.0)
    log(f"[{name}] aerial crops: {len(meta)}", fh)

    jobs = [(n, k, aerial_dir, os.path.join(aerial_dir, "features_grid", f"aerial_n{n}_k{k}.csv"), r_m)
            for n in N_LIST for k in K_LIST]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(job_aerial, j) for j in jobs]):
            n, k, rows = f.result()
            log(f"[{name}] aerial n={n} k={k}: {rows} rows ({time.time() - t0:.0f} s)", fh)

    import run_matching_all_v3 as v3
    v3.REFERENCE_DIR = os.path.join(out, "references")
    v3.AERIAL_FEATURE_DIR = os.path.join(aerial_dir, "features_grid")
    v3.TEST_META_CSV = os.path.join(aerial_dir, "test_meta.csv")
    v3.OUT_DIR = os.path.join(out, "matching")
    v3.R_MAX_M = r_m
    v3.R_PX = r_m * PX_PER_METER
    v3.DISK_AREA_PX = 3.141592653589793 * v3.R_PX ** 2
    v3.VERTEX_MODE = "coch_extreme" if VERTICES == "extreme" else "all"
    v3.main()
    log(f"[{name}] finished in {time.time() - t0:.0f} s", fh)
    fh.close()


def main():
    global VERTICES, OUT_ROOT, RADIUS_M
    ap = argparse.ArgumentParser()
    ap.add_argument("examples", nargs="*", default=list(EXAMPLES))
    ap.add_argument("--vertices", choices=["extreme", "all"], default="extreme")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--radius-m", type=float, default=None,
                    help="fixed R in metres instead of the automatic value (for comparison only)")
    a = ap.parse_args()
    VERTICES, RADIUS_M = a.vertices, a.radius_m
    OUT_ROOT = a.out_root or "results_examples"
    workers = max(1, (os.cpu_count() or 4) - 1)
    for name in a.examples:
        run_example(name, EXAMPLES[name], workers)


if __name__ == "__main__":
    main()
