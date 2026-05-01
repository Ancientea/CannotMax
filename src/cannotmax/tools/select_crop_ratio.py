"""工具：交互选取怪物条大致区域，find_monster_zone 自动精确定位。

用法：
    uv run python -m src.cannotmax.tools.select_crop_ratio images/tmp/original_screenshot.png
    ENTER 确认选择  |  ESC 重选  |  Q 退出
"""
import sys
import cv2
import numpy as np

from ..utils.find_monster_zone import find_monster_zone


def select_region(image):
    """交互拖框选区域，返回 [(x1,y1),(x2,y2)]，按 Q 返回 None。"""
    img = image.copy()
    roi_box = []
    drawing = False
    win_name = "Select Region (ENTER=confirm, ESC=retry, Q=quit)"

    def on_mouse(event, x, y, flags, param):
        nonlocal roi_box, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_box = [(x, y)]
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            tmp = img.copy()
            cv2.rectangle(tmp, roi_box[0], (x, y), (0, 255, 0), 2)
            cv2.imshow(win_name, tmp)
        elif event == cv2.EVENT_LBUTTONUP:
            roi_box.append((x, y))
            drawing = False

    cv2.putText(img, "Drag to select | ENTER:confirm | ESC:retry | Q:quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        cv2.imshow(win_name, img)
        key = cv2.waitKey(0)
        if key in (13, 32) and len(roi_box) == 2:  # ENTER or SPACE
            break
        elif key in (27, ord('r'), ord('R')):  # ESC or R
            roi_box = []
            img = image.copy()
            cv2.putText(img, "Drag to select | ENTER:confirm | ESC:retry | Q:quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif key in (ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    x1 = min(roi_box[0][0], roi_box[1][0])
    y1 = min(roi_box[0][1], roi_box[1][1])
    x2 = max(roi_box[0][0], roi_box[1][0])
    y2 = max(roi_box[0][1], roi_box[1][1])
    return [(x1, y1), (x2, y2)]


def normalize_coords(coords, ref_w, ref_h):
    """将像素坐标转为相对于 ref_w x ref_h 的相对坐标。"""
    result = []
    for x1, y1, x2, y2 in coords:
        result.append((
            round(float(x1) / ref_w, 4),
            round(float(y1) / ref_h, 4),
            round(float(x2) / ref_w, 4),
            round(float(y2) / ref_h, 4),
        ))
    return result


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "images/tmp/original_screenshot.png"
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # 1. 用户拖框选大致区域
    roi = select_region(img)
    if roi is None:
        print("已取消")
        sys.exit(0)

    (px1, py1), (px2, py2) = roi
    cropped = img[py1:py2, px1:px2]
    ch, cw = cropped.shape[:2]
    print(f"用户选区: [{px1},{py1}] → [{px2},{py2}]  ({cw}x{ch})")

    # 2. find_monster_zone 自动精确定位
    d_avatar, d_nums = find_monster_zone(cropped)
    if d_avatar is None:
        print("find_monster_zone 检测失败，请重新框选")
        sys.exit(1)

    avatar_px = np.round(d_avatar * [cw, ch, cw, ch]).astype(int)
    nums_px   = np.round(d_nums   * [cw, ch, cw, ch]).astype(int)

    ax_min = avatar_px[:, 0].min()
    ay_min = avatar_px[:, 1].min()
    ax_max = avatar_px[:, 2].max()
    ay_max = avatar_px[:, 3].max()
    bar_w = ax_max - ax_min
    bar_h = ay_max - ay_min

    # 3. 裁切比（全图坐标）
    gx1 = px1 + ax_min
    gy1 = py1 + ay_min
    gx2 = px1 + ax_max
    gy2 = py1 + ay_max

    # 4. 头像和数字区域（相对怪物条 bar_w x bar_h）
    avatar_rel = normalize_coords(
        [(ax1 - ax_min, ay1 - ay_min, ax2 - ax_min, ay2 - ay_min)
         for ax1, ay1, ax2, ay2 in avatar_px],
        bar_w, bar_h,
    )
    nums_rel = normalize_coords(
        [(nx1 - ax_min, ny1 - ay_min, nx2 - ax_min, ny2 - ay_min)
         for nx1, ny1, nx2, ny2 in nums_px],
        bar_w, bar_h,
    )

    print(f"\n=== 1. 裁剪比例 (DEFAULT_CROP_RATIO) ===")
    print(f"DEFAULT_CROP_RATIO: tuple[tuple[float, float], tuple[float, float]] = (")
    print(f"    ({gx1 / w:.4f}, {gy1 / h:.4f}),")
    print(f"    ({gx2 / w:.4f}, {gy2 / h:.4f}),")
    print(f")")
    print(f"  对应像素: [({gx1:.0f}, {gy1:.0f}), ({gx2:.0f}, {gy2:.0f})]")

    print(f"\n=== 2. 头像区域 (DEFAULT_AVATAR_REGIONS) ===")
    print(f"  怪物条尺寸: {bar_w:.0f}x{bar_h:.0f}  (resize 到 975x119)")
    print(f"DEFAULT_AVATAR_REGIONS = (")
    for i, (x1, y1, x2, y2) in enumerate(avatar_rel):
        print(f"    ({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}),  # region {i}")
    print(f")")

    print(f"\n=== 3. 数字区域 (DEFAULT_NUMBER_REGIONS) ===")
    print(f"DEFAULT_NUMBER_REGIONS = (")
    for i, (x1, y1, x2, y2) in enumerate(nums_rel):
        print(f"    ({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}),  # region {i}")
    print(f")")


if __name__ == "__main__":
    main()
