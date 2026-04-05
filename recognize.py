import cv2
import numpy as np
import os
from PIL import ImageGrab

# 导入 RapidOCR
try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    try:
        from rapidocr_openvino import RapidOCR
    except ImportError:
        print("请先安装 RapidOCR: pip install rapidocr_onnxruntime")
        RapidOCR = None

# ROI保持不变
relative_regions = [
    (0.0, 0.0, 0.131, 1),
    (0.1462, 0.0, 0.2762, 1),
    (0.2923, 0.0, 0.4214, 1),
    (0.5786, 0.0, 0.7087, 1),
    (0.7248, 0.0, 0.8538, 1),
    (0.8679, 0.0, 1, 1)
]

# 鼠标交互全局变量
drawing = False
roi_box = []


def get_rapidocr_engine(prefer_gpu=True):
    """rapidocr引擎初始化"""
    try:
        if prefer_gpu:
            import torch
            if torch.cuda.is_available():
                from rapidocr_paddle import RapidOCR as RapidOCRPaddle
                return RapidOCRPaddle(
                    params={
                        "Global.with_torch": True,
                        "EngineConfig.torch.use_cuda": True,
                        "EngineConfig.torch.gpu_id": 0,
                    }
                )
    except ImportError:
        pass
    # 默认使用 ONNXRuntime 引擎
    if RapidOCR is not None:
        return RapidOCR()
    return None


# 初始化全局OCR引擎实例
rapidocr_eng = get_rapidocr_engine()


def load_ref_images(ref_dir="images"):
    """加载参考图片库 (0~26 范围)"""
    ref_images = {}
    for i in range(27):
        path = os.path.join(ref_dir, f"{i}.png")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                ref_images[i] = img
    return ref_images


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
    """交互式区域选择"""
    global roi_box
    while True:
        screenshot = np.array(ImageGrab.grab())
        img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        cv2.putText(img, "Drag to select area | ENTER:confirm | ESC:retry",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        example_img = cv2.imread("images/eg.png")
        if example_img is not None:
            cv2.imshow("example", example_img)

        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", 1280, 720)
        cv2.setMouseCallback("Select ROI", mouse_callback, img)
        cv2.imshow("Select ROI", img)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 13 and len(roi_box) == 2:
            x1, y1 = min(roi_box[0][0], roi_box[1][0]), min(roi_box[0][1], roi_box[1][1])
            x2, y2 = max(roi_box[0][0], roi_box[1][0]), max(roi_box[0][1], roi_box[1][1])
            return [(x1, y1), (x2, y2)]
        elif key == 27:
            roi_box = []
            continue


def preprocess(img):
    """增强预处理"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 细小噪声去除
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 5:  # 过滤掉过小的噪点连通域
            cv2.drawContours(thresh, [contour], -1, 0, thickness=cv2.FILLED)

    return thresh


def crop_to_min_bounding_rect(image):
    """裁剪图像到包含所有轮廓的最小外接矩形"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    return image[y: y + h, x: x + w]


def add_black_border(img, border_size=3):
    """添加黑边，提升OCR识别率"""
    return cv2.copyMakeBorder(
        img,
        top=border_size, bottom=border_size,
        left=border_size, right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )


def find_best_match(target, ref_images):
    """matchTemplate 进行目标匹配"""
    confidence = float('-inf')
    best_id = -1

    if len(target.shape) == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

    for img_id, ref_img in ref_images.items():
        try:
            # 动态缩放 ref 尺寸
            ref_resized = cv2.resize(ref_img, (target.shape[1], target.shape[0]))
            res = cv2.matchTemplate(target, ref_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > confidence:
                confidence = max_val
                best_id = img_id
        except Exception:
            continue

    return best_id, confidence


def do_num_ocr(img):
    """RapidOCR 封装调用"""
    if not rapidocr_eng:
        return "", 0.0
    try:
        result, elapse = rapidocr_eng(img, use_det=False, use_cls=False, use_rec=True)
        if result:
            # 兼容不同版本 RapidOCR 的返回结构
            if hasattr(result, 'txts') and result.txts:
                return result.txts[0], result.scores[0]
            elif isinstance(result, tuple) and result[0]:
                return result[0][0][0], result[0][0][1]
            elif isinstance(result, list) and len(result) > 0:
                return result[0][0], result[0][1]
    except Exception as e:
        print(f"OCR处理异常: {e}")
    return "", 0.0


def process_regions(main_roi, ref_images, screenshot=None):
    results = []
    (x1, y1), (x2, y2) = main_roi
    main_width = x2 - x1
    main_height = y2 - y1

    if screenshot is None:
        screenshot = np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2)))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    else:
        screenshot = screenshot[y1:y2, x1:x2]

    for idx, rel in enumerate(relative_regions):
        try:
            # 区域提取
            rx1 = int(rel[0] * main_width)
            ry1 = int(rel[1] * main_height)
            rx2 = int(rel[2] * main_width)
            ry2 = int(rel[3] * main_height)
            sub_roi = screenshot[ry1:ry2, rx1:rx2]

            # 匹配ID
            matched_id, confidence = find_best_match(sub_roi, ref_images)

            # 提取数字区域
            number_roi = sub_roi[-sub_roi.shape[0] // 4:, sub_roi.shape[1] // 3:]

            # 增强预处理
            processed = preprocess(number_roi)  # 基础二值化+去噪
            processed = crop_to_min_bounding_rect(processed)  # 裁掉多余空白
            processed = add_black_border(processed, 3)  # 加黑边防文字贴边

            # OCR数字识别
            number, ocr_conf = do_num_ocr(processed)

            # 格式化截取
            if number:
                number = number.replace('×', 'x').lower()
                x_pos = number.find('x')
                if x_pos != -1:
                    number = number[x_pos + 1:]
                number = ''.join(filter(str.isdigit, number))

            # 保存截取到的数字图
            #if number:
            #    save_path = f"images/nums/{number}.png"
            #    if not os.path.exists("images/nums"):
            #        os.makedirs("images/nums")
            #    cv2.imwrite(save_path, processed)

            results.append({
                "region_id": idx,
                "matched_id": matched_id,
                "number": number if number else "N/A",
                "confidence": round(confidence, 2)
            })
        except Exception as e:
            print(f"区域{idx}处理失败: {str(e)}")
            results.append({
                "region_id": idx,
                "error": str(e)
            })

    return results


if __name__ == "__main__":
    print("请用鼠标拖拽选择主区域...")
    main_roi = select_roi()
    ref_images = load_ref_images()

    results = process_regions(main_roi, ref_images)

    print("\n识别结果：")
    for res in results:
        if 'error' in res:
            print(f"区域{res['region_id']}: 错误 - {res['error']}")
        else:
            if res['matched_id'] != -1:  # 匹配有效时输出
                print(
                    f"区域{res['region_id']} => 匹配ID:{res['matched_id']} 数字:{res['number']} 置信度:{res['confidence']}")