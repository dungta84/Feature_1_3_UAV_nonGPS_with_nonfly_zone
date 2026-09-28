# -*- coding: utf-8 -*-
# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
"""
get_px_per_meter.py
- Ảnh cố định: references/refmap.png
- Click 2 điểm đầu-cuối thanh scale bar
- Nhập độ dài thực (m) từ bàn phím (ví dụ 90)
- Trả ra px_per_meter
"""

import math
import cv2

IMAGE_PATH = "references/refmap.png"

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    img_draw = param["img_draw"]

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 2:
            clicked_points.append((x, y))
            cv2.circle(img_draw, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                img_draw,
                f"P{len(clicked_points)}({x},{y})",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )
            if len(clicked_points) == 2:
                cv2.line(img_draw, clicked_points[0], clicked_points[1], (255, 0, 0), 2)


def compute_distance_px(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] Không đọc được ảnh: {IMAGE_PATH}")
        return

    img_draw = img.copy()

    guide = "Click 2 diem scale bar | R: reset | Enter: tinh | ESC: thoat"
    cv2.namedWindow("get_px_per_meter", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("get_px_per_meter", mouse_callback, {"img_draw": img_draw})

    while True:
        show = img_draw.copy()
        cv2.putText(show, guide, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2, cv2.LINE_AA)
        cv2.imshow("get_px_per_meter", show)

        key = cv2.waitKey(20) & 0xFF

        # ESC
        if key == 27:
            print("[INFO] Thoát.")
            break

        # R hoặc r: reset
        if key in [ord("r"), ord("R")]:
            clicked_points.clear()
            img_draw = img.copy()
            cv2.setMouseCallback("get_px_per_meter", mouse_callback, {"img_draw": img_draw})
            print("[INFO] Đã reset điểm.")

        # Enter
        if key in [13, 10]:
            if len(clicked_points) != 2:
                print("[WARN] Cần click đủ 2 điểm trước khi tính.")
                continue

            raw = input("Nhập độ dài thực của scale bar (m), ví dụ 90: ").strip()
            try:
                meters = float(raw)
            except ValueError:
                print("[ERROR] Giá trị mét không hợp lệ.")
                continue

            if meters <= 0:
                print("[ERROR] Giá trị mét phải > 0.")
                continue

            p1, p2 = clicked_points
            dist_px = compute_distance_px(p1, p2)

            if dist_px <= 0:
                print("[ERROR] Khoảng cách pixel không hợp lệ.")
                continue

            px_per_meter = dist_px / meters
            meter_per_px = meters / dist_px

            print("\n========== KẾT QUẢ ==========")
            print(f"Ảnh: {IMAGE_PATH}")
            print(f"P1 = {p1}")
            print(f"P2 = {p2}")
            print(f"Độ dài scale bar (px): {dist_px:.4f}")
            print(f"Độ dài thực (m): {meters:.4f}")
            print(f"px_per_meter = {px_per_meter:.6f}")
            print(f"meter_per_px = {meter_per_px:.6f}")
            print("=============================\n")

            cv2.putText(
                img_draw,
                f"px_per_meter={px_per_meter:.6f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                img_draw,
                f"meter_per_px={meter_per_px:.6f}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
