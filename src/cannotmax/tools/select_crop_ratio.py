"""工具：交互选取怪物条大致区域，find_monster_zone 自动精确定位。

用法：
    uv run python -m src.cannotmax.tools.select_crop_ratio images/tmp/original_screenshot.png
    ENTER 确认  |  ESC 重选  |  Q 退出
"""

import sys

import cv2
import numpy as np

from cannotmax.config.paths import TMP_IMAGES_DIR
from cannotmax.utils.find_monster_zone import find_monster_zone


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

    cv2.putText(
        img,
        "Drag to select | ENTER:confirm | ESC:retry | Q:quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        cv2.imshow(win_name, img)
        key = cv2.waitKey(0)
        if key in (13, 32) and len(roi_box) == 2:
            break
        elif key in (27, ord("r"), ord("R")):
            roi_box = []
            img = image.copy()
            cv2.putText(
                img,
                "Drag to select | ENTER:confirm | ESC:retry | Q:quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        elif key in (ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    x1 = min(roi_box[0][0], roi_box[1][0])
    y1 = min(roi_box[0][1], roi_box[1][1])
    x2 = max(roi_box[0][0], roi_box[1][0])
    y2 = max(roi_box[0][1], roi_box[1][1])
    return [(x1, y1), (x2, y2)]


def confirm_regions(image, avatar_px, nums_px, bar_bbox, global_offset):
    """展示检测到的区域，让用户确认或重试。"""
    px1, py1 = global_offset
    display = image.copy()

    # 画裁剪边界
    gx1, gy1, gx2, gy2 = bar_bbox
    cv2.rectangle(display, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (255, 255, 0), 2)

    # 画头像区域 (紫色)
    for ax1, ay1, ax2, ay2 in avatar_px:
        gax1 = int(px1 + ax1)
        gay1 = int(py1 + ay1)
        gax2 = int(px1 + ax2)
        gay2 = int(py1 + ay2)
        cv2.rectangle(display, (gax1, gay1), (gax2, gay2), (225, 0, 225), 2)

    # 画数字区域 (青色)
    for nx1, ny1, nx2, ny2 in nums_px:
        gnx1 = int(px1 + nx1)
        gny1 = int(py1 + ny1)
        gnx2 = int(px1 + nx2)
        gny2 = int(py1 + ny2)
        cv2.rectangle(display, (gnx1, gny1), (gnx2, gny2), (225, 225, 0), 2)

    # 区域索引标注
    for i, (ax1, ay1, ax2, ay2) in enumerate(avatar_px):
        cx = int(px1 + (ax1 + ax2) / 2)
        cy = int(py1 + ay1 - 5)
        cv2.putText(
            display, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

    win_name = "Detected Regions (ENTER=accept, ESC=retry)"
    cv2.putText(
        display,
        "Yellow=bar | Purple=avatar | Cyan=number | ENTER=accept | ESC=retry",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    while True:
        cv2.imshow(win_name, display)
        key = cv2.waitKey(0)
        if key in (13, 32):
            cv2.destroyAllWindows()
            return True
        elif key in (27, ord("r"), ord("R")):
            cv2.destroyAllWindows()
            return False


def normalize_coords(coords, ref_w, ref_h):
    result = []
    for x1, y1, x2, y2 in coords:
        result.append(
            (
                round(float(x1) / ref_w, 4),
                round(float(y1) / ref_h, 4),
                round(float(x2) / ref_w, 4),
                round(float(y2) / ref_h, 4),
            )
        )
    return result


def main():
    img_path = (
        sys.argv[1] if len(sys.argv) > 1 else TMP_IMAGES_DIR / "original_screenshot.png"
    )
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    while True:
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
            print("find_monster_zone 检测失败，请重新框选\n")
            continue

        avatar_px = np.round(d_avatar * [cw, ch, cw, ch]).astype(int)
        nums_px = np.round(d_nums * [cw, ch, cw, ch]).astype(int)

        ax_min = avatar_px[:, 0].min()
        ay_min = avatar_px[:, 1].min()
        ax_max = avatar_px[:, 2].max()
        ay_max = avatar_px[:, 3].max()
        bar_w = ax_max - ax_min
        bar_h = ay_max - ay_min

        # 全图坐标
        gx1 = px1 + ax_min
        gy1 = py1 + ay_min
        gx2 = px1 + ax_max
        gy2 = py1 + ay_max

        # 3. 展示区域让用户确认
        ok = confirm_regions(img, avatar_px, nums_px, (gx1, gy1, gx2, gy2), (px1, py1))
        if not ok:
            print("重新选择...\n")
            continue

        # 4. 计算相对坐标
        avatar_rel = normalize_coords(
            [
                (ax1 - ax_min, ay1 - ay_min, ax2 - ax_min, ay2 - ay_min)
                for ax1, ay1, ax2, ay2 in avatar_px
            ],
            bar_w,
            bar_h,
        )
        nums_rel = normalize_coords(
            [
                (nx1 - ax_min, ny1 - ay_min, nx2 - ax_min, ny2 - ay_min)
                for nx1, ny1, nx2, ny2 in nums_px
            ],
            bar_w,
            bar_h,
        )
        break

    print("\n=== 1. 裁剪比例 (DEFAULT_CROP_RATIO) ===")
    print("DEFAULT_CROP_RATIO: tuple[tuple[float, float], tuple[float, float]] = (")
    print(f"    ({gx1 / w:.4f}, {gy1 / h:.4f}),")
    print(f"    ({gx2 / w:.4f}, {gy2 / h:.4f}),")
    print(")")
    print(f"  对应像素: [({gx1:.0f}, {gy1:.0f}), ({gx2:.0f}, {gy2:.0f})]")

    print("\n=== 2. 头像区域 (DEFAULT_AVATAR_REGIONS) ===")
    print(f"  怪物条尺寸: {bar_w:.0f}x{bar_h:.0f}  (resize 到 975x119)")
    print("DEFAULT_AVATAR_REGIONS = (")
    for i, (x1, y1, x2, y2) in enumerate(avatar_rel):
        print(f"    ({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}),  # region {i}")
    print(")")

    print("\n=== 3. 数字区域 (DEFAULT_NUMBER_REGIONS) ===")
    print("DEFAULT_NUMBER_REGIONS = (")
    for i, (x1, y1, x2, y2) in enumerate(nums_rel):
        print(f"    ({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}),  # region {i}")
    print(")")


if __name__ == "__main__":
    main()
