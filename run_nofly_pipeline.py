# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
# run_setup_nofly.py
import os

from tools.setup_no_fly_zone import run_setup_no_fly_zone
from tools.preview_no_fly_zone import preview_no_fly_zone


def main():
    """
    Chỉ làm 2 việc:
    1) Mở UI để vẽ/sửa No-Fly Zone và lưu CSV
    2) Mở preview ngay sau khi lưu để kiểm tra
    """

    reference_image = os.path.join("references", "refmap.png")
    nofly_csv = os.path.join("data", "references", "non_fly_zone.csv")
    preview_png = os.path.join("data", "references", "nofly_preview.png")

    print("=" * 72)
    print("SETUP NO-FLY ZONE")
    print("=" * 72)
    print("Phím tắt thường dùng trong UI:")
    print("- [1]: Circle mode")
    print("- [2]: Polygon mode")
    print("- Polygon: click nhiều điểm, [Enter] để chốt")
    print("- [u]: undo điểm cuối polygon")
    print("- [d]: xóa zone cuối")
    print("- [s]: lưu CSV")
    print("- [q]: thoát")
    print("")

    run_setup_no_fly_zone(
        reference_image=reference_image,
        out_csv=nofly_csv
    )

    print("\n" + "=" * 72)
    print("PREVIEW NO-FLY ZONE")
    print("=" * 72)

    preview_no_fly_zone(
        reference_image=reference_image,
        nofly_csv=nofly_csv,
        out_path=preview_png,
        show=True
    )

    print("\n✅ Xong setup NFZ")
    print(f"- CSV: {nofly_csv}")
    print(f"- Preview: {preview_png}")


if __name__ == "__main__":
    main()
