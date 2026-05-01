"""工具：从截图中交互选取怪物条区域，输出新的 DEFAULT_CROP_RATIO。

用法：
    uv run python -m src.cannotmax.tools.select_crop_ratio images/tmp/original_screenshot.png
"""
import sys
import cv2

from ..core.roi_selector import ROISelector


def main():
    if len(sys.argv) < 2:
        img_path = "images/tmp/pc_original_screenshot.png"
    else:
        img_path = sys.argv[1]

    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    selector = ROISelector()
    roi = selector.select_roi(img, example_image_path="images/eg.png")

    if roi is None:
        print("未选择区域")
        sys.exit(0)

    (px1, py1), (px2, py2) = roi
    x1, y1 = px1 / w, py1 / h
    x2, y2 = px2 / w, py2 / h

    print(f"\n像素坐标: [({px1}, {py1}), ({px2}, {py2})]")
    print(f"\n新的 DEFAULT_CROP_RATIO:")
    print(f"DEFAULT_CROP_RATIO: tuple[tuple[float, float], tuple[float, float]] = (")
    print(f"    ({x1:.4f}, {y1:.4f}),")
    print(f"    ({x2:.4f}, {y2:.4f}),")
    print(f")")


if __name__ == "__main__":
    main()
