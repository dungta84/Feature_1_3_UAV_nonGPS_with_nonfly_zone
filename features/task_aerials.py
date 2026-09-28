# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
# features/task_aerials.py
from typing import Optional, List, Tuple
import os
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from features.core import Aerial


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

    # descriptor cols
    rd_cols = []
    for c in out.columns:
        p = _parse_r_d_col(c)
        if p is not None:
            rd_cols.append((c, p[0], p[1]))
    if len(rd_cols) == 0:
        raise ValueError("Không tìm thấy cột descriptor dạng r*_d*")

    # rename ổn định
    rename_map = {}
    for old, r, d in rd_cols:
        rename_map[old] = f"r{_format_radius_token(r)}_d{int(d)}"
    out = out.rename(columns=rename_map)

    # parse lại + sort
    rd_cols2 = []
    for c in out.columns:
        p = _parse_r_d_col(c)
        if p is not None:
            rd_cols2.append((c, p[0], p[1]))
    rd_sorted = [t[0] for t in sorted(rd_cols2, key=lambda z: (z[1], z[2]))]

    if "vertex_total" not in out.columns:
        out["vertex_total"] = out[rd_sorted].sum(axis=1)

    out["object_id"] = out["object_id"].astype(int)
    out["vertex_id"] = out["vertex_id"].astype(int)
    out["x"] = out["x"].astype(float)
    out["y"] = out["y"].astype(float)
    for c in rd_sorted:
        out[c] = out[c].astype(float)
    out["vertex_total"] = out["vertex_total"].astype(float)

    # giữ metadata nếu có
    meta_cols = []
    for c in ["aerial_file", "aerial_path", "global_object_id"]:
        if c in out.columns:
            meta_cols.append(c)

    final_cols = meta_cols + ["object_id", "vertex_id", "x", "y"] + rd_sorted + ["vertex_total"]
    if "vertex_type" in out.columns:                   # [COCH-V] keep the COCH vertex type
        final_cols.append("vertex_type")
    out = out[final_cols]

    sort_cols = []
    if "aerial_file" in out.columns:
        sort_cols.append("aerial_file")
    sort_cols += ["object_id", "vertex_id"]

    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    float_cols = ["x", "y"] + rd_sorted + ["vertex_total"]
    out[float_cols] = out[float_cols].round(decimals)

    return out


def save_feature_csv_normalized(df: pd.DataFrame, out_csv: str, decimals: int = 6):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    norm = normalize_feature_table_schema(df, decimals=decimals)
    norm.to_csv(out_csv, index=False, encoding="utf-8")


# =========================
# Per-image worker
# =========================
def _process_one_aerial(
    img_path: str,
    px_per_meter: float,
    roof_thresh: int,
    min_area: int,
    border_margin: int,
    object_polygon_mode: str,
    n: int,
    k: int,
    max_radius_m: float,
    sector_num_points: int,
) -> pd.DataFrame:
    ar = Aerial(
        image_path=img_path,
        px_per_meter=px_per_meter,
        use_oc_hull=(str(object_polygon_mode).lower() == "och"),
    )
    ar.load_image(mask_mode="legacy_hsv", legacy_thresh=roof_thresh)
    ar.sync_alias()

    # tương thích API extract_polygons cũ/mới
    try:
        ar.extract_polygons(
            min_area=min_area,
            border_margin=border_margin,
            object_polygon_mode=object_polygon_mode
        )
    except TypeError:
        ar.extract_polygons(
            min_area=min_area,
            border_margin=border_margin,
            use_convex_hull=(str(object_polygon_mode).lower() in ("och", "convex", "hull", "true"))
        )

    if len(ar.polygons) == 0:
        return pd.DataFrame()

    ar.compute_features(
        n=int(n),
        k=int(k),
        max_radius_m=float(max_radius_m),  # fixed 6.011 từ caller
        sector_num_points=int(sector_num_points),
    )

    df = ar.get_feature_table()
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if df.empty:
        return pd.DataFrame()

    # metadata
    df.insert(0, "aerial_file", os.path.basename(img_path))
    df.insert(1, "aerial_path", img_path)
    df["global_object_id"] = [f"{os.path.basename(img_path)}::obj{int(x)}" for x in df["object_id"].values]
    return df


# =========================
# Main task
# =========================
def run_aerials_task(
    aerial_dir: str = r"data\aerials",
    out_csv: str = r"data\aerials\aerial_features.csv",
    px_per_meter: float = 8.0,
    roof_thresh: int = 22,
    min_area: int = 1250,
    border_margin: int = 0,
    object_polygon_mode: str = "och",   # "och" | "convex" | "raw"
    n: int = 5,
    k: int = 5,
    max_radius_m: float = 6.011,        # FIXED theo yêu cầu
    sector_num_points: int = 96,
    image_glob: str = "*.png",
    max_workers: int = 1,
    csv_decimals: int = 6,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Trích feature từ toàn bộ ảnh aerial crop trong aerial_dir.
    Output CSV schema chuẩn:
      [meta...], object_id, vertex_id, x, y, r..._d..., vertex_total
    """
    if not os.path.isdir(aerial_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục aerial: {aerial_dir}")

    # [EX] the radius is now passed by the caller (it was forced to 6.011 here)
    max_radius_m = float(max_radius_m)

    paths = sorted(glob.glob(os.path.join(aerial_dir, image_glob)))
    # loại preview/meta
    paths = [
        p for p in paths
        if not os.path.basename(p).lower().endswith("_overview.png")
    ]

    if len(paths) == 0:
        if verbose:
            print(f"[AERIAL] Không có ảnh nào khớp pattern: {os.path.join(aerial_dir, image_glob)}")
        df_empty = pd.DataFrame()
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df_empty.to_csv(out_csv, index=False, encoding="utf-8")
        return df_empty

    if verbose:
        print(f"[AERIAL] images={len(paths)} | n={n}, k={k} | Rmax={max_radius_m}")

    rows: List[pd.DataFrame] = []

    if max_workers is None or max_workers <= 1:
        # tuần tự
        for idx, img_path in enumerate(paths, start=1):
            try:
                df = _process_one_aerial(
                    img_path=img_path,
                    px_per_meter=px_per_meter,
                    roof_thresh=roof_thresh,
                    min_area=min_area,
                    border_margin=border_margin,
                    object_polygon_mode=object_polygon_mode,
                    n=n,
                    k=k,
                    max_radius_m=max_radius_m,
                    sector_num_points=sector_num_points,
                )
                if df.empty:
                    if verbose:
                        print(f"[AERIAL][{idx}/{len(paths)}] Skip: {os.path.basename(img_path)}")
                    continue
                rows.append(df)
                if verbose:
                    print(f"[AERIAL][{idx}/{len(paths)}] OK: {os.path.basename(img_path)} | rows={len(df)}")
            except Exception as e:
                if verbose:
                    print(f"[AERIAL][{idx}/{len(paths)}] ERROR {img_path}: {e}")
    else:
        # song song theo ảnh
        workers = int(max_workers)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_map = {}
            for p in paths:
                fut = ex.submit(
                    _process_one_aerial,
                    p,
                    px_per_meter,
                    roof_thresh,
                    min_area,
                    border_margin,
                    object_polygon_mode,
                    n,
                    k,
                    max_radius_m,
                    sector_num_points,
                )
                fut_map[fut] = p

            done_idx = 0
            for fut in as_completed(fut_map):
                done_idx += 1
                p = fut_map[fut]
                try:
                    df = fut.result()
                    if df is not None and (not df.empty):
                        rows.append(df)
                        if verbose:
                            print(f"[AERIAL][{done_idx}/{len(paths)}] OK: {os.path.basename(p)} | rows={len(df)}")
                    else:
                        if verbose:
                            print(f"[AERIAL][{done_idx}/{len(paths)}] Skip: {os.path.basename(p)}")
                except Exception as e:
                    if verbose:
                        print(f"[AERIAL][{done_idx}/{len(paths)}] ERROR {p}: {e}")

    if len(rows) == 0:
        out_df = pd.DataFrame()
    else:
        out_df = pd.concat(rows, ignore_index=True)
        out_df = normalize_feature_table_schema(out_df, decimals=csv_decimals)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    if verbose:
        print(f"[AERIAL] Saved: {out_csv} | rows={len(out_df)}")

    return out_df


if __name__ == "__main__":
    run_aerials_task()
