# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
# tools/preview_no_fly_zone.py
from typing import List, Optional
import os
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from features.core import load_nofly_csv_as_polygons


def preview_no_fly_zone(
    reference_image: str = r"references\refmap.png",
    nofly_csv: str = r"data\references\non_fly_zone.csv",
    out_path: Optional[str] = r"data\references\nofly_preview.png",
    show: bool = True,
) -> List[Polygon]:
    if not os.path.isfile(reference_image):
        raise FileNotFoundError(f"Không tìm thấy ảnh reference: {reference_image}")

    img = cv2.imread(reference_image)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {reference_image}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    nfz_polys = load_nofly_csv_as_polygons(nofly_csv)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(img_rgb)
    ax.set_title(f"No-Fly Preview | count={len(nfz_polys)}")
    ax.axis("off")

    for i, p in enumerate(nfz_polys, start=1):
        if p is None or p.is_empty:
            continue
        x, y = p.exterior.xy
        ax.fill(x, y, color="magenta", alpha=0.25, zorder=2)
        ax.plot(x, y, color="magenta", linewidth=2.0, zorder=3)

        c = p.centroid
        ax.text(
            c.x, c.y, f"NFZ-{i}",
            color="white", fontsize=9, ha="center", va="center",
            bbox=dict(facecolor="magenta", alpha=0.55, edgecolor="none", pad=1.5),
            zorder=4
        )

    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"[PREVIEW] Saved: {out_path}")

    if show:
        plt.show()

    plt.close(fig)
    return nfz_polys


if __name__ == "__main__":
    preview_no_fly_zone()
