# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
# tools/setup_no_fly_zone.py
import os
import math
from typing import List, Dict, Tuple, Optional

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


class NoFlyZoneEditor:
    def __init__(
        self,
        image_path: str,
        out_csv: str = "data/references/non_fly_zone.csv"
    ):
        self.image_path = image_path
        self.out_csv = out_csv

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
        self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self.mode = "circle"  # "circle" | "polygon"
        self.zones: List[Dict] = []

        # circle temp state
        self.circle_center: Optional[Tuple[float, float]] = None

        # polygon temp state
        self.poly_points: List[Tuple[float, float]] = []

        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.redraw()

    def _next_zone_id(self) -> int:
        if len(self.zones) == 0:
            return 1
        return max(z["zone_id"] for z in self.zones) + 1

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.img_rgb)
        self.ax.axis("off")

        # draw saved zones
        for z in self.zones:
            if z["zone_type"] == "circle":
                cx, cy = z["center"]
                r = z["radius"]
                patch = Circle((cx, cy), r, fill=True, alpha=0.25, color="red", ec="red", lw=2)
                self.ax.add_patch(patch)
                self.ax.text(cx, cy, f"Z{z['zone_id']}", color="yellow", fontsize=10, ha="center", va="center")
            else:
                pts = z["points"]
                patch = MplPolygon(pts, closed=True, fill=True, alpha=0.25, color="orange", ec="orange", lw=2)
                self.ax.add_patch(patch)
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                self.ax.text(cx, cy, f"Z{z['zone_id']}", color="yellow", fontsize=10, ha="center", va="center")

        # draw temporary circle
        if self.mode == "circle" and self.circle_center is not None:
            cx, cy = self.circle_center
            self.ax.scatter([cx], [cy], c="cyan", s=30)
            self.ax.text(cx + 5, cy - 5, "center", color="cyan", fontsize=9)

        # draw temporary polygon
        if self.mode == "polygon" and len(self.poly_points) > 0:
            xs = [p[0] for p in self.poly_points]
            ys = [p[1] for p in self.poly_points]
            self.ax.plot(xs, ys, color="lime", lw=2, marker="o")
            for i, (x, y) in enumerate(self.poly_points):
                self.ax.text(x + 3, y - 3, str(i), color="lime", fontsize=8)

        title = (
            f"No-Fly Zone Editor | Mode: {self.mode.upper()} | "
            "Keys: [1]circle [2]polygon [Enter]close poly [u]undo [d]del last zone [s]save [q]quit"
        )
        self.ax.set_title(title, fontsize=10)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)

        if self.mode == "circle":
            if self.circle_center is None:
                self.circle_center = (x, y)
            else:
                cx, cy = self.circle_center
                r = math.hypot(x - cx, y - cy)
                if r > 1.0:
                    self.zones.append({
                        "zone_id": self._next_zone_id(),
                        "zone_type": "circle",
                        "center": (cx, cy),
                        "radius": r,
                        "enabled": 1
                    })
                self.circle_center = None

        elif self.mode == "polygon":
            self.poly_points.append((x, y))

        self.redraw()

    def on_key(self, event):
        key = str(event.key).lower()

        if key == "1":
            self.mode = "circle"
            self.circle_center = None
            self.poly_points = []
        elif key == "2":
            self.mode = "polygon"
            self.circle_center = None
            self.poly_points = []
        elif key == "enter":
            if self.mode == "polygon" and len(self.poly_points) >= 3:
                poly = Polygon(self.poly_points)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty and poly.area > 0:
                    pts = list(poly.exterior.coords)[:-1]
                    self.zones.append({
                        "zone_id": self._next_zone_id(),
                        "zone_type": "polygon",
                        "points": [(float(px), float(py)) for px, py in pts],
                        "enabled": 1
                    })
                self.poly_points = []
        elif key == "u":
            if self.mode == "polygon" and len(self.poly_points) > 0:
                self.poly_points.pop()
        elif key == "d":
            if len(self.zones) > 0:
                self.zones.pop()
        elif key == "s":
            self.save_csv()
        elif key == "q":
            plt.close(self.fig)
            return

        self.redraw()

    def save_csv(self):
        rows = []
        for z in self.zones:
            zid = int(z["zone_id"])
            if z["zone_type"] == "circle":
                cx, cy = z["center"]
                rows.append({
                    "zone_id": zid,
                    "zone_type": "circle",
                    "x": float(cx),
                    "y": float(cy),
                    "radius": float(z["radius"]),
                    "vertex_order": "",
                    "enabled": int(z.get("enabled", 1))
                })
            else:
                pts = z["points"]
                for i, (x, y) in enumerate(pts):
                    rows.append({
                        "zone_id": zid,
                        "zone_type": "polygon",
                        "x": float(x),
                        "y": float(y),
                        "radius": "",
                        "vertex_order": int(i),
                        "enabled": int(z.get("enabled", 1))
                    })

        df = pd.DataFrame(rows, columns=[
            "zone_id", "zone_type", "x", "y", "radius", "vertex_order", "enabled"
        ])

        out_dir = os.path.dirname(self.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(self.out_csv, index=False, encoding="utf-8-sig")
        print(f"✅ Đã lưu no-fly zones: {self.out_csv}")
        print(f"   Số zone: {len(self.zones)}")

    def run(self):
        plt.show()


def run_setup_no_fly_zone(
    reference_image: str = r"references\refmap.png",
    out_csv: str = r"data\references\non_fly_zone.csv"
):
    editor = NoFlyZoneEditor(image_path=reference_image, out_csv=out_csv)
    editor.run()


if __name__ == "__main__":
    run_setup_no_fly_zone()
