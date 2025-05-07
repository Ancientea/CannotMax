import logging
import os
import subprocess
import sys
import cv2
import numpy as np
from PIL import ImageGrab
from rapidocr import RapidOCR
import find_monster_zone

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rapidocr_eng = RapidOCR()

# 是否启用debug模式
intelligent_workers_debug = True

# 定义全局变量
MONSTER_COUNT = 56  # 设置怪物数量

# 鼠标交互全局变量
drawing = False
roi_box = []

# 预定义相对坐标
relative_regions_nums = [
    (0.0300, 0.7, 0.1400, 1),
    (0.1600, 0.7, 0.2700, 1),
    (0.2900, 0.7, 0.4000, 1),
    (0.6100, 0.7, 0.7200, 1),
    (0.7300, 0.7, 0.8400, 1),
    (0.8600, 0.7, 0.9700, 1),
]
relative_regions = [
    (0.0000, 0.1, 0.1200, 0.77),
    (0.1200, 0.1, 0.2400, 0.77),
    (0.2400, 0.1, 0.3600, 0.77),
    (0.6400, 0.1, 0.7600, 0.77),
    (0.7600, 0.1, 0.8800, 0.77),
    (0.8800, 0.1, 1.0000, 0.77),
]


def save_number_image(number, processed, mon_id):
    """保存数字图片到对应文件夹
    Args:
        number: 识别出的数字
        processed: 处理后的图片
        mon_id: 怪物ID
    """
    if number and mon_id != 0:
        # 创建数字对应的文件夹
        num_folder = os.path.join("images", "nums", str(number))
        if not os.path.exists(num_folder):
            os.makedirs(num_folder)

        # 获取文件夹中已有的图片数量
        existing_files = [f for f in os.listdir(num_folder) if f.endswith(".png")]
        next_index = len(existing_files) + 1

        # 保存图片，命名为 id_序号.png
        save_path = os.path.join(num_folder, f"{mon_id}_{next_index}.png")
        cv2.imwrite(save_path, processed)


def mouse_callback(event, x, y, flags, param):
    global roi_box, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_box = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = param.copy()
        cv2.rectangle(img_copy, roi_box[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_box.append((x, y))
        drawing = False


def select_roi():
    """改进的交互式区域选择"""
    global roi_box  # 声明为全局变量
    while True:
        # 获取初始截图
        screenshot = np.array(ImageGrab.grab())
        img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # 添加操作提示
        cv2.putText(img, "Drag to select area | ENTER:confirm | ESC:retry",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 添加示例图片
        example_img = cv2.imread("images/eg.png")
        # 显示示例图片在单独的窗口中
        cv2.imshow("example", example_img)

        # 显示窗口
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", 1280, 720)
        cv2.setMouseCallback("Select ROI", mouse_callback, img)
        cv2.imshow("Select ROI", img)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 13 and len(roi_box) == 2:  # Enter确认
            # 标准化坐标 (x1,y1)为左上角，(x2,y2)为右下角
            x1, y1 = min(roi_box[0][0], roi_box[1][0]), min(roi_box[0][1], roi_box[1][1])
            x2, y2 = max(roi_box[0][0], roi_box[1][0]), max(roi_box[0][1], roi_box[1][1])
            return [(x1, y1), (x2, y2)]
        elif key == 27:  # ESC重试
            roi_box = []
            continue


def add_black_border(img: cv2.typing.MatLike, border_size=3):
    return cv2.copyMakeBorder(
        img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],  # BGR格式的黑色
    )


def crop_to_min_bounding_rect(image: cv2.typing.MatLike):
    """裁剪图像到包含所有轮廓的最小外接矩形"""
    # 转为灰度图（如果传入的是二值图，这个操作不会有问题）
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # 寻找轮廓
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 如果没有找到轮廓就直接返回原图
    if not contours:
        return image
    # 合并所有轮廓点并获取外接矩形
    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    # 裁剪图片并返回
    return image[y : y + h, x : x + w]


def preprocess(img: cv2.typing.MatLike):
    """彩色图像二值化处理，增强数字可见性"""
    # 检查图像是否为彩色
    if len(img.shape) == 2:
        # 如果是灰度图像，转换为三通道
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 创建较宽松的亮色阈值范围（包括浅灰、白色等亮色）
    # BGR格式
    lower_bright = np.array([180, 180, 180])
    upper_bright = np.array([255, 255, 255])

    # 基于颜色范围创建掩码
    bright_mask = cv2.inRange(img, lower_bright, upper_bright)

    # 进行形态学操作，增强文本可见性
    # 创建一个小的椭圆形核
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    # 膨胀操作，使文字更粗
    # dilated = cv2.dilate(bright_mask, kernel, iterations=1)
    # 闭操作，填充文字内的小空隙
    # closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    # closed = dilated
    closed = bright_mask

    # 去除细小噪声：过滤不够大的连通区域
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1:
            # 用黑色填充宽度小于等于1的区域
            cv2.drawContours(closed, [contour], -1, 0, thickness=cv2.FILLED)
        if h <= 13:
            # 用黑色填充高度小于等于13的区域
            cv2.drawContours(closed, [contour], -1, 0, thickness=cv2.FILLED)

    return closed


def find_best_match(target: cv2.typing.MatLike, ref_images: dict[int, cv2.typing.MatLike]):
    """
    模板匹配找到最佳匹配的参考图像
    :param target: 目标图像
    :param ref_images: 参考图像字典 {id: image}
    :return: (最佳匹配的id, 最小差异值)
    """
    confidence = float("-inf")
    best_id = -1

    # 确保目标图像是RGB格式
    if len(target.shape) == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

    for img_id, ref_img in ref_images.items():
        try:
            # 模板匹配
            match_algorithm = cv2.TM_CCOEFF_NORMED
            res = cv2.matchTemplate(target, ref_img, match_algorithm)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > confidence:
                confidence = max_val
                best_id = img_id
        except Exception as e:
            logger.exception(f"处理参考图像 {img_id} 时出错:", e)
            continue

    return best_id, confidence


def do_num_ocr(img: cv2.typing.MatLike):
    result = rapidocr_eng(img, use_det=False, use_cls=False, use_rec=True)
    print(f"OCR: text: '{result.txts[0]}', score: {result.scores[0]}")
    number = "".join([c for c in result.txts[0] if c.isdigit()])
    confidence = result.scores[0]
    return number, confidence


def process_regions(main_roi, screenshot: cv2.typing.MatLike | None = None,matched_threshold = 0.5,ocr_threshold = 0.95):
    """处理主区域中的所有区域（优化特征匹配）
    Args:
        main_roi: 主要感兴趣区域的坐标
        screenshot: 可选的预先捕获的截图
    Returns:
        区域处理结果的列表
    """
    results = []
    (x1, y1), (x2, y2) = main_roi

    # 如果没有提供screenshot，则获取最新截图（仅截取主区域）
    if screenshot is None:
        screenshot = np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2)))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # 手动框选的截图需先识别目标区域
        cv2.imwrite(f"images/tmp/zone1.png", screenshot)
        d_avatar, d_nums = find_monster_zone.cutFrame(screenshot)
        height, width, _ = screenshot.shape
        divisors = np.array([width, height, width, height])
        avatar = np.round(d_avatar * divisors).astype("int")
        x_min, x_max, y_min, y_max = width, 0, height, 0
        for x1, y1, x2, y2 in avatar:
            x_min = min(x_min, min(x1, x2))
            x_max = max(x_max, max(x1, x2))
            y_min = min(y_min, min(y1, y2))
            y_max = max(y_max, max(y1, y2))
        screenshot = screenshot[y_min:y_max, x_min:x_max]
    else:
        # 从当前screenshot中提取主区域
        screenshot = screenshot[y1:y2, x1:x2]

    # 转换到标准1920*1080下目标区域
    screenshot = cv2.resize(screenshot, (969, 119))
    main_height = screenshot.shape[0]
    main_width = screenshot.shape[1]

    if intelligent_workers_debug:  # 如果处于debug模式
        # 存储模板图像用于debug
        cv2.imwrite(f"images/tmp/zone.png", screenshot)

    # 遍历所有区域
    for idx, rel in enumerate(relative_regions):
        try:
            # ================== 模板匹配部分 ==================
            # 计算模板匹配的子区域坐标
            rx1 = int(rel[0] * main_width)
            ry1 = int(rel[1] * main_height)
            rx2 = int(rel[2] * main_width)
            ry2 = int(rel[3] * main_height)
            # 提取模板匹配用的子区域
            sub_roi = screenshot[ry1:ry2, rx1:rx2]

            # 图像匹配
            matched_id, confidence = find_best_match(sub_roi, ref_images)
            print(f"target: {idx} confidence: {confidence}")
            if matched_id != 0 and confidence < matched_threshold:
                raise ValueError(f"模板匹配置信度过低: {confidence}")
        except Exception as e:
            print(f"区域 {idx} 匹配失败: {str(e)}")
            results.append({"region_id": idx, "error": str(e)})
            return results

        try:
            # ================== OCR数字识别部分 ==================
            rel_num = relative_regions_nums[idx]
            rx1_num = int(rel_num[0] * main_width)
            ry1_num = int(rel_num[1] * main_height)
            rx2_num = int(rel_num[2] * main_width)
            ry2_num = int(rel_num[3] * main_height)

            # 提取OCR识别用的子区域
            sub_roi_num = screenshot[ry1_num:ry2_num, rx1_num:rx2_num]
            processed = preprocess(sub_roi_num)  # 二值化预处理
            processed = crop_to_min_bounding_rect(processed)  # 去除多余黑框
            processed = add_black_border(processed, border_size=3)  # 加上3像素黑框

            # OCR识别（保留优化后的处理逻辑）
            number, ocr_confidence = do_num_ocr(processed)
            if number != "" and ocr_confidence < ocr_threshold:
                raise ValueError(f"OCR置信度过低: {ocr_confidence}")

            if intelligent_workers_debug:  # 如果处于debug模式
                # 存储模板图像用于debug
                cv2.imwrite(f"images/tmp/target_{idx}.png", sub_roi)

                # 存储OCR图像用于debug
                cv2.imwrite(f"images/tmp/number_{idx}.png", processed)

                # 保存有数字的图片到images/nums中的对应文件夹
                if number:
                    save_number_image(number, processed, matched_id)

            if number == "" and matched_id != 0:
                raise ValueError("发现有怪物但无数量异常数据！")
            if matched_id == 0 and number != "":
                raise ValueError("发现无怪物但有数量异常数据！")

            results.append(
                {
                    "region_id": idx,
                    "matched_id": matched_id,
                    "number": number if number else "N/A",
                    "confidence": round(confidence, 2),
                }
            )
        except Exception as e:
            print(f"区域 {idx} OCR识别失败: {str(e)}")
            results.append(
                {"region_id": idx, "matched_id": matched_id, "number": "N/A", "error": str(e)}
            )
    return results


def load_ref_images(ref_dir="images"):
    """加载参考图片库"""
    ref_images = {}
    for i in range(MONSTER_COUNT + 1):
        path = os.path.join(ref_dir, f"{i}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR_BGR)
            # 裁切模板匹配图像比例
            img = img[
                int(img.shape[0] * 0.16) : int(img.shape[0] * 0.80),  # 高度取靠上部分
                int(img.shape[1] * 0.18) : int(img.shape[1] * 0.82),  # 宽度与高度一致
            ]
            # 调整参考图像大小以匹配目标图像
            ref_resized = cv2.resize(img, (80, 80))
            ref_resized = ref_resized[0:70, :]

            if intelligent_workers_debug:  # 如果处于debug模式
                # 存储模板图像用于debug
                if not os.path.exists("images/tmp"):
                    os.makedirs("images/tmp")
                cv2.imwrite(f"images/tmp/xref_{i}.png", ref_resized)

            if img is not None:
                ref_images[i] = ref_resized
    return ref_images


ref_images = load_ref_images()  # 直接加载图片储存在全局变量，避免反复加载

if __name__ == "__main__":
    print("请用鼠标拖拽选择主区域...")
    main_roi = select_roi()
    results = process_regions(main_roi)
    # 输出结果
    print("\n识别结果：")
    for res in results:
        if "error" in res:
            print(f"区域{res['region_id']}: 错误 - {res['error']}")
        else:
            if res["matched_id"] != 0:
                print(
                    f"区域{res['region_id']} => 匹配ID:{res['matched_id']} 数字:{res['number']} 置信度:{res['confidence']}"
                )
