"""工具：交互选取怪物条大致区域，find_monster_zone 自动精确定位。

用法：
    uv run python -m src.cannotmax.tools.select_crop_ratio images/tmp/original_screenshot.png
"""
import sys
import cv2
import numpy as np

from ..core.roi_selector import ROISelector
from ..utils.find_monster_zone import find_monster_zone


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "images/tmp/original_screenshot.png"
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # 1. 用户拖框选大致区域
    selector = ROISelector()
    roi = selector.select_roi(img, example_image_path="images/eg.png")
    if roi is None:
        print("未选择区域")
        sys.exit(0)

    (px1, py1), (px2, py2) = roi
    cropped = img[py1:py2, px1:px2]
    print(f"用户选区像素: [{px1},{py1}] → [{px2},{py2}]")

    # 2. find_monster_zone 自动精确定位
    d_avatar, d_nums = find_monster_zone(cropped)
    if d_avatar is None:
        print("find_monster_zone 检测失败，请重新框选")
        sys.exit(1)

    ch, cw = cropped.shape[:2]
    avatar_px = np.round(d_avatar * [cw, ch, cw, ch]).astype(int)
    ax_min = avatar_px[:, 0].min()
    ay_min = avatar_px[:, 1].min()
    ax_max = avatar_px[:, 2].max()
    ay_max = avatar_px[:, 3].max()

    # 3. 换算回全图坐标
    gx1 = px1 + ax_min
    gy1 = py1 + ay_min
    gx2 = px1 + ax_max
    gy2 = py1 + ay_max

    print(f"\n全局像素坐标: [({gx1:.0f}, {gy1:.0f}), ({gx2:.0f}, {gy2:.0f})]")
    print(f"\n新的 DEFAULT_CROP_RATIO:")
    print(f"DEFAULT_CROP_RATIO: tuple[tuple[float, float], tuple[float, float]] = (")
    print(f"    ({gx1 / w:.4f}, {gy1 / h:.4f}),")
    print(f"    ({gx2 / w:.4f}, {gy2 / h:.4f}),")
    print(f")")


if __name__ == "__main__":
    main()
