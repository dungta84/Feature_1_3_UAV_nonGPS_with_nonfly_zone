# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
# gen_test_data.py
from typing import Dict, Optional, List
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from shapely.geometry import Polygon, box

from features.core import Aerial, oc_hull_from_polygon, load_nofly_csv_as_polygons


def choose_thresholds_for_image(
    image_path: str,
    use_oc_hull: bool = True,
    fixed_thresh: Optional[int] = None,
    fixed_min_area_och: Optional[float] = None
) -> Dict[str, object]:
    """
    Trả về:
    {
      "image_bgr": np.ndarray,
      "roof_mask": np.ndarray,
      "roof_thresh": int,
      "min_area_och_thresh": float
    }
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernel = np.ones((5, 5), np.uint8)

    init_t = int(fixed_thresh) if fixed_thresh is not None else 0
    init_t = int(np.clip(init_t, 0, 255))

    init_area = float(fixed_min_area_och) if fixed_min_area_och is not None else 1250.0
    init_area = max(0.0, init_area)

    state = {"thresh": init_t, "min_area_och": init_area}

    def build_mask(thresh: int):
        t = int(np.clip(thresh, 0, 255))
        mask1 = cv2.inRange(hsv, np.array([0, t, t], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
        mask2 = cv2.inRange(hsv, np.array([165, t, t], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        roof_mask_local = cv2.bitwise_or(mask1, mask2)
        roof_mask_local = cv2.morphologyEx(roof_mask_local, cv2.MORPH_OPEN, kernel, iterations=2)
        roof_mask_local = cv2.morphologyEx(roof_mask_local, cv2.MORPH_CLOSE, kernel, iterations=2)
        return roof_mask_local

    def draw_overlay(ax, roof_mask_local):
        ax.clear()
        ax.imshow(img_rgb)
        ax.set_title(f"Contour + COCH | thresh={state['thresh']}, min_area={state['min_area_och']:.0f}")
        ax.axis("off")

        contours, _ = cv2.findContours(roof_mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        idx = 1
        for cnt in contours:
            pts = cnt[:, 0, :]
            if len(pts) < 3:
                continue

            poly = Polygon([(float(x), float(y)) for x, y in pts])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue

            och = oc_hull_from_polygon(poly) if use_oc_hull else poly
            if och is None or och.is_empty:
                continue
            if float(och.area) < float(state["min_area_och"]):
                continue

            xs = pts[:, 0]
            ys = pts[:, 1]
            ax.plot(xs, ys, color="lime", linewidth=1.0)

            hx, hy = och.exterior.xy
            ax.plot(hx, hy, color="cyan", linewidth=2.0)

            cx, cy = och.centroid.x, och.centroid.y
            ax.text(cx, cy, str(idx), color="yellow", fontsize=11, ha="center", va="center")
            idx += 1

    # fixed mode
    if fixed_thresh is not None and fixed_min_area_och is not None:
        roof_mask = build_mask(init_t)
        return {
            "image_bgr": img,
            "roof_mask": roof_mask,
            "roof_thresh": int(init_t),
            "min_area_och_thresh": float(init_area),
        }

    # interactive mode
    fig, ax = plt.subplots(figsize=(9, 8))
    plt.subplots_adjust(bottom=0.22)

    init_mask = build_mask(state["thresh"])
    draw_overlay(ax, init_mask)

    ax_slider_t = plt.axes([0.18, 0.10, 0.64, 0.03])
    slider_t = Slider(ax_slider_t, "Ngưỡng S,V", 0, 255, valinit=state["thresh"], valstep=1)

    h, w = img.shape[:2]
    area_max = float(h * w) / 100.0
    ax_slider_a = plt.axes([0.18, 0.05, 0.64, 0.03])
    slider_a = Slider(ax_slider_a, "Min area COCH", 0, area_max, valinit=state["min_area_och"], valstep=10)

    def update(_):
        state["thresh"] = int(slider_t.val)
        state["min_area_och"] = float(slider_a.val)
        draw_overlay(ax, build_mask(state["thresh"]))
        fig.canvas.draw_idle()

    slider_t.on_changed(update)
    slider_a.on_changed(update)
    plt.show()

    roof_mask = build_mask(state["thresh"])
    return {
        "image_bgr": img,
        "roof_mask": roof_mask,
        "roof_thresh": int(state["thresh"]),
        "min_area_och_thresh": float(state["min_area_och"]),
    }


def clamp_window(cx: float, cy: float, w: int, h: int, W: int, H: int):
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = x1 + w
    y2 = y1 + h

    if x1 < 0:
        x2 += -x1
        x1 = 0
    if y1 < 0:
        y2 += -y1
        y1 = 0
    if x2 > W:
        d = x2 - W
        x1 -= d
        x2 = W
    if y2 > H:
        d = y2 - H
        y1 -= d
        y2 = H

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    return x1, y1, x2, y2


def _build_nfz_union(nofly_polygons: List[Polygon]):
    if not nofly_polygons:
        return None
    try:
        from shapely.ops import unary_union
        u = unary_union(nofly_polygons)
        if u.is_empty:
            return None
        return u
    except Exception:
        return None


def export_uniform_test_crops_from_analyzer(
    analyzer: Aerial,
    out_dir: str = "aerials",
    prefix: str = "test",
    ext: str = ".png",
    global_extra_ratio: float = 0.02,
    sort_by: str = "object_id",
    save_meta_csv: bool = True,
    # no-fly filter
    nofly_polygons: Optional[List[Polygon]] = None,
    nofly_intersection_eps: float = 0.0,
) -> pd.DataFrame:
    if analyzer.original_image is None:
        raise ValueError("Analyzer chưa có original_image.")
    if len(analyzer.polygons) == 0:
        raise ValueError("Analyzer không có polygon nào để crop.")

    os.makedirs(out_dir, exist_ok=True)
    img = analyzer.original_image
    H, W = img.shape[:2]

    nfz_union = _build_nfz_union(nofly_polygons or [])

    items = [(oid, analyzer.polygons[oid - 1]) for oid in sorted(analyzer.objects.keys())]
    if sort_by == "area_desc":
        items = sorted(items, key=lambda x: float(x[1].area), reverse=True)

    oid_max, poly_max = max(items, key=lambda x: float(x[1].area))
    minx, miny, maxx, maxy = poly_max.bounds
    bw = maxx - minx
    bh = maxy - miny

    std_w = max(1, int(np.ceil(bw * (1.0 + global_extra_ratio))))
    std_h = max(1, int(np.ceil(bh * (1.0 + global_extra_ratio))))

    rows = []
    skipped_nfz = 0

    total = len(items)
    num_digits = max(2, len(str(total)))

    for _, (oid, poly) in enumerate(items, start=1):
        cx, cy = poly.centroid.x, poly.centroid.y
        x1, y1, x2, y2 = clamp_window(cx, cy, std_w, std_h, W, H)

        # Skip crop if intersects NFZ
        if nfz_union is not None:
            rect = box(float(x1), float(y1), float(x2), float(y2))
            try:
                inter_area = float(rect.intersection(nfz_union).area)
            except Exception:
                inter_area = 0.0
            if inter_area > float(nofly_intersection_eps):
                skipped_nfz += 1
                continue

        crop = img[y1:y2, x1:x2].copy()
        ch, cw = crop.shape[:2]

        if (cw != std_w) or (ch != std_h):
            crop = cv2.resize(crop, (std_w, std_h), interpolation=cv2.INTER_AREA)
            ch, cw = std_h, std_w

        out_index = len(rows) + 1
        filename = f"{prefix}_{out_index:0{num_digits}d}{ext}"
        out_path = os.path.join(out_dir, filename)

        ok = cv2.imwrite(out_path, crop)
        if not ok:
            print(f"[WARN] Không lưu được: {out_path}")
            continue

        rows.append({
            "index": out_index,
            "source_object_id": oid,
            "file_name": filename,
            "file_path": out_path,
            "crop_x1": x1, "crop_y1": y1, "crop_x2": x2, "crop_y2": y2,
            "crop_w": int(cw), "crop_h": int(ch),
            "std_w": int(std_w), "std_h": int(std_h),
            "poly_area": float(poly.area),
            "max_area_object_id": int(oid_max),
            "global_extra_ratio": float(global_extra_ratio),
            "roof_thresh_used": analyzer.roof_thresh,
            "skipped_by_nofly": 0,
        })

    df_meta = pd.DataFrame(rows)

    if save_meta_csv and len(df_meta) > 0:
        meta_path = os.path.join(out_dir, f"{prefix}_meta.csv")
        df_meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
        print(f"Đã lưu metadata: {meta_path}")

    print(f"Đã tạo {len(df_meta)} ảnh test trong '{out_dir}'")
    print(f"Kích thước chuẩn: {std_w}x{std_h}")
    print(f"Số crop bị loại do no-fly-zone: {skipped_nfz}")
    return df_meta


def plot_test_overview_with_ids(
    analyzer: Aerial,
    out_path: str = os.path.join("data", "test_overview.png"),
    title: str = "Test Overview",
    show_mask: bool = True,
    nofly_polygons: Optional[List[Polygon]] = None,
) -> None:
    if analyzer.original_image is None:
        raise ValueError("Analyzer chưa có original_image để vẽ overview.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img_rgb = cv2.cvtColor(analyzer.original_image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(img_rgb)

    if show_mask and analyzer.binary_image is not None:
        ax.imshow(analyzer.binary_image, alpha=0.25, cmap="Reds")

    # vẽ no-fly-zone
    if nofly_polygons:
        for i, p in enumerate(nofly_polygons, start=1):
            if p is None or p.is_empty:
                continue
            x, y = p.exterior.xy
            ax.fill(x, y, color="magenta", alpha=0.20, zorder=2)
            ax.plot(x, y, color="magenta", linewidth=2.0, zorder=3)
            c = p.centroid
            ax.text(
                c.x, c.y, f"NFZ-{i}",
                color="white", fontsize=8, ha="center", va="center",
                bbox=dict(facecolor="magenta", alpha=0.5, edgecolor="none", pad=1.2),
                zorder=5
            )

    ax.set_title(title)
    ax.axis("off")

    for oid, poly in enumerate(analyzer.polygons, start=1):
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.plot(x, y, color="cyan", linewidth=2.0, zorder=4)

        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(
            cx, cy, str(oid),
            color="yellow", fontsize=10, ha="center", va="center",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5),
            zorder=6
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.show()
    plt.close(fig)

    print(f"Đã lưu test_overview: {out_path}")


def run_gen_test_data(
    reference_image: str = r"references\refmap.png",
    out_dir: str = "aerials",
    border_margin: int = 0,
    object_polygon_mode: str = "och",  # "och" | "convex" | "raw"
    fixed_thresh: Optional[int] = 22,
    fixed_min_area_och: Optional[float] = 1250,
    global_extra_ratio: float = 0.05,
    sort_by: str = "object_id",
    test_overview_path: str = os.path.join("data", "test_overview.png"),
    # NFZ
    nofly_csv: str = os.path.join("data", "references", "non_fly_zone.csv"),
    enable_nofly_filter: bool = True,
    nofly_intersection_eps: float = 0.0,
) -> pd.DataFrame:
    if object_polygon_mode not in ("och", "convex", "raw"):
        raise ValueError("object_polygon_mode phải là 'och' | 'convex' | 'raw'")

    if not os.path.isfile(reference_image):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {reference_image}")

    # 1) Threshold chọn mask
    pick = choose_thresholds_for_image(
        image_path=reference_image,
        use_oc_hull=(object_polygon_mode == "och"),
        fixed_thresh=fixed_thresh,
        fixed_min_area_och=fixed_min_area_och,
    )
    print(f"roof_thresh={pick['roof_thresh']}, min_area={pick['min_area_och_thresh']}")

    # 2) Analyzer cho refmap để lấy object polygons
    aer = Aerial(reference_image, px_per_meter=10.0, use_oc_hull=(object_polygon_mode == "och"))
    aer.original_image = pick["image_bgr"].copy()
    aer.image_shape = aer.original_image.shape[:2]
    aer.binary_image = pick["roof_mask"]
    aer.roof_mask = pick["roof_mask"]
    aer.roof_thresh = int(pick["roof_thresh"])
    aer.sync_alias()

    aer.extract_polygons(
        min_area=int(pick["min_area_och_thresh"]),
        border_margin=border_margin,
        object_polygon_mode=object_polygon_mode
    )

    print(f"Số object hợp lệ sau lọc: {len(aer.polygons)}")

    # 3) Load no-fly
    nfz_polys = []
    if enable_nofly_filter:
        nfz_polys = load_nofly_csv_as_polygons(nofly_csv)
    print(f"Số no-fly polygons: {len(nfz_polys)}")

    # 4) Vẽ overview
    plot_test_overview_with_ids(
        analyzer=aer,
        out_path=test_overview_path,
        title=f"Test Overview (th={pick['roof_thresh']}, min_area={int(pick['min_area_och_thresh'])})",
        show_mask=True,
        nofly_polygons=nfz_polys
    )

    # 5) Export crop + skip NFZ intersection
    df_meta = export_uniform_test_crops_from_analyzer(
        analyzer=aer,
        out_dir=out_dir,
        prefix="test",
        ext=".png",
        global_extra_ratio=global_extra_ratio,
        sort_by=sort_by,
        save_meta_csv=True,
        nofly_polygons=nfz_polys,
        nofly_intersection_eps=nofly_intersection_eps,
    )

    return df_meta


if __name__ == "__main__":
    REFERENCE_IMAGE = r"references\refmap.png"
    OUT_DIR = "aerials"

    run_gen_test_data(
        reference_image=REFERENCE_IMAGE,
        out_dir=OUT_DIR,
        border_margin=0,
        object_polygon_mode="och",
        fixed_thresh=22,
        fixed_min_area_och=1250,
        global_extra_ratio=0.05,
        sort_by="object_id",
        test_overview_path=os.path.join("data", "test_overview.png"),
        nofly_csv=os.path.join("data", "references", "non_fly_zone.csv"),
        enable_nofly_filter=True,
        nofly_intersection_eps=0.0,
    )

    print("✅ Hoàn tất tạo test data (đã lọc no-fly-zone).")
