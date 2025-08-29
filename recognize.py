# -*- coding: utf-8 -*-
"""
recognize.py
---------------------
围绕「怪物识别」的一站式流程：
- **快速选择 RapidOCR 引擎**（优先 Torch-GPU，失败回退 CPU/ONNX）；
- **WinRT 屏幕捕获**：从 `winrt_capture.WinRTScreenCapture` 统一获取 BGR 帧；
- **交互式主 ROI 选择** 与 **自动精细化**（基于 `find_monster_zone.cutFrame` 的候选区域反推）；
- **模板匹配**（怪物头像） + **数字 OCR**（数量），并输出每个区域的识别结果；
- 可选导出调试可视化：标准化主 ROI 上的 6 个头像/数字区域覆盖图、候选检测结果等。
"""
from __future__ import annotations
import logging
import os
from typing import Dict, Tuple, List

import cv2
import numpy as np
from rapidocr import RapidOCR

# 统一使用项目内的 WinRT 封装（事件注册）
from winrt_capture import WinRTScreenCapture
import find_monster_zone

# ---------------------------- 基本配置 ----------------------------
# 若填写了窗口标题，将优先锁定该窗口；否则按 monitor_index 抓整屏
WINDOW_NAME: str = "HQC"
CAPTURE_MONITOR_INDEX: int = 1

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 调试与常量
intelligent_workers_debug = True
MONSTER_COUNT = 78

# 数字区域相对坐标（以主 ROI 标准化到 969x119 为参考）
relative_regions_nums: List[Tuple[float, float, float, float]] = [
    (0.0300, 0.7, 0.1400, 1),
    (0.1600, 0.7, 0.2700, 1),
    (0.2900, 0.7, 0.4000, 1),
    (0.6100, 0.7, 0.7200, 1),
    (0.7300, 0.7, 0.8400, 1),
    (0.8600, 0.7, 0.9700, 1),
]

# 怪物头像相对坐标（以主 ROI 标准化到 969x119 为参考）
relative_regions: List[Tuple[float, float, float, float]] = [
    (0.0000, 0.1, 0.1200, 0.77),
    (0.1200, 0.1, 0.2400, 0.77),
    (0.2400, 0.1, 0.3600, 0.77),
    (0.6400, 0.1, 0.7600, 0.77),
    (0.7600, 0.1, 0.8800, 0.77),
    (0.8800, 0.1, 1.0000, 0.77),
]

# —— 新增：ROI 可视化开关与目录 ——
EXPORT_ROI_DEBUG = True
DEBUG_DIR = "images/tmp"
os.makedirs(DEBUG_DIR, exist_ok=True)


# ---------------- RapidOCR 引擎选择（仅保留这个版本） ----------------
def get_rapidocr_engine(prefer_gpu: bool = True) -> RapidOCR:
    """选择并创建 RapidOCR 引擎。

    优先尝试 **Torch-GPU**；初始化失败则回退到 **CPU(onnxruntime)**。
    该策略可在多数环境“即插即用”。
    """
    if prefer_gpu:
        try:
            # 注意：参数键与 RapidOCR 版本相关，这里与现有环境保持一致
            return RapidOCR(
                params={
                    "with_torch": True,
                    "Det.engine_type": "torch",
                    "Cls.engine_type": "torch",
                    "Rec.engine_type": "torch",
                    "EngineConfig.torch.use_cuda": True,
                    "EngineConfig.torch.gpu_id": 0,
                }
            )
        except Exception as e:
            logger.warning(f"RapidOCR Torch 初始化失败，回退 CPU：{e}")
    # 默认 CPU（onnxruntime）
    return RapidOCR()


# ---------------------- 模板匹配 / 图像处理 ----------------------
def find_best_match(
    target: cv2.typing.MatLike, ref_images: Dict[int, cv2.typing.MatLike]
) -> Tuple[int, float]:
    """在 `ref_images` 中寻找与 `target` 最匹配的模板。

    参数
    ----
    target: Mat
        目标子图。
    ref_images: Dict[int, Mat]
        模板库，键为 `id`，值为彩色模板图像。

    返回
    ----
    (best_id, confidence): Tuple[int, float]
        最佳匹配模板的 id 与相似度（`cv2.TM_CCOEFF_NORMED` 最大值）。
    """
    confidence = float("-inf")
    best_id = -1

    # 统一为 BGR 三通道以简化匹配
    if len(target.shape) == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

    for img_id, ref_img in ref_images.items():
        try:
            res = cv2.matchTemplate(target, ref_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > confidence:
                confidence = max_val
                best_id = img_id
        except Exception as e:
            logger.exception(f"处理参考图像 {img_id} 时出错: {e}")
            continue

    return best_id, confidence


def add_black_border(img: cv2.typing.MatLike, border_size: int = 3):
    """为图像四周添加黑色实线边框，利于 OCR 聚焦。"""
    return cv2.copyMakeBorder(
        img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],  # BGR 黑色
    )


def crop_to_min_bounding_rect(image: cv2.typing.MatLike):
    """裁剪到包含所有外轮廓的 *最小外接矩形*。"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    return image[y : y + h, x : x + w]


def preprocess(img: cv2.typing.MatLike):
    """简单二值化以增强数字可见性（对 **亮色** 字符更友好）。"""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    lower_bright = np.array([180, 180, 180], dtype=np.uint8)
    upper_bright = np.array([255, 255, 255], dtype=np.uint8)
    bright_mask = cv2.inRange(img, lower_bright, upper_bright)

    # 去除非常细小的噪点：过滤尺寸过小的外接矩形
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 13:
            cv2.drawContours(bright_mask, [contour], -1, 0, thickness=cv2.FILLED)

    return bright_mask


def load_ref_images(ref_dir: str = "images") -> Dict[int, cv2.typing.MatLike]:
    """加载模板库并做裁剪/缩放以匹配目标尺寸。

    - 裁剪模板靠上的主体区域；
    - 统一缩放到 80x80 后取 `0:70` 行，近似贴合识别区域大小；
    - 若开启 `intelligent_workers_debug`，导出中间模板到 `images/tmp` 便于核对。
    """
    ref_images: Dict[int, cv2.typing.MatLike] = {}
    os.makedirs("images/tmp", exist_ok=True)

    for i in range(MONSTER_COUNT + 1):
        path = os.path.join(ref_dir, f"{i}.png")
        if not os.path.exists(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # 裁剪模板匹配区域比例（靠上部份）
        img = img[
            int(img.shape[0] * 0.16) : int(img.shape[0] * 0.80),
            int(img.shape[1] * 0.18) : int(img.shape[1] * 0.82),
        ]
        # 调整为匹配目标图像的参考尺度
        ref_resized = cv2.resize(img, (80, 80))
        ref_resized = ref_resized[0:70, :]

        if intelligent_workers_debug:
            cv2.imwrite(f"images/tmp/xref_{i}.png", ref_resized)

        ref_images[i] = ref_resized

    return ref_images


# ----------------------------- 核心：识别类 -----------------------------
class RecognizeMonster:
    """怪物识别：ROI 选择 → 模板匹配 → 数字 OCR 的组合流程。"""

    def __init__(self) -> None:
        # 16:9 下怪物区域相对坐标（用于 ADB 图像截取时）
        self.roi_relative = [(0.2479, 0.8410), (0.7526, 0.9510)]

        # 手动主区域（屏幕坐标系；用于 WinRT 截屏后的裁剪）
        self.main_roi: List[Tuple[int, int]] = [(0, 0), (1919, 1079)]

        # 鼠标交互状态
        self.roi_box: List[Tuple[int, int]] = []
        self.drawing = False

        # OCR 引擎与参考模板
        self.rapidocr_eng = get_rapidocr_engine()
        self.ref_images = load_ref_images()

        # 初始化 WinRT 截屏：优先按窗口标题锁定，否则按显示器索引
        target_window = WINDOW_NAME.strip() if isinstance(WINDOW_NAME, str) else ""
        self._capture = WinRTScreenCapture(
            window_name=target_window if target_window else None,
            monitor_index=CAPTURE_MONITOR_INDEX if not target_window else None,
            capture_cursor=False,
            draw_border=None,
        )
        self._capture.start()

    # ------------------------------- ROI 交互 -------------------------------
    def mouse_callback(self, event, x: int, y: int, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_box = [(x, y)]
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img_copy = param.copy()
            cv2.rectangle(img_copy, self.roi_box[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Select ROI", img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_box.append((x, y))
            self.drawing = False

    def select_roi(self) -> List[Tuple[int, int]]:
        """交互式选择主 ROI（基于 WinRT 最新帧作为底图）。

        - **ENTER**：确认；
        - **ESC**：重选。
        """
        while True:
            # 等首帧
            if not self._capture.wait_first_frame(timeout_sec=3.0):
                logger.error("WinRT 捕获没有首帧，请检查窗口标题或显示器索引")
                raise RuntimeError("WinRT 捕获没有首帧")

            # 获取当前画面（BGR）
            screenshot = self._capture.snapshot()
            img = screenshot.copy()
            cv2.putText(
                img,
                "Drag to select area \n ENTER: confirm \n ESC: retry",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Select ROI", 1280, 720)
            cv2.setMouseCallback("Select ROI", self.mouse_callback, img)
            cv2.imshow("Select ROI", img)

            # 示例图（如存在）
            try:
                example_img = cv2.imread("images/eg.png", cv2.IMREAD_COLOR)
                if example_img is not None:
                    cv2.imshow("example", example_img)
            except Exception:
                pass

            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key == 13 and len(self.roi_box) == 2:  # Enter 确认
                x1 = min(self.roi_box[0][0], self.roi_box[1][0])
                y1 = min(self.roi_box[0][1], self.roi_box[1][1])
                x2 = max(self.roi_box[0][0], self.roi_box[1][0])
                y2 = max(self.roi_box[0][1], self.roi_box[1][1])
                logger.info(f"选择区域: {[(x1, y1), (x2, y2)]}")
                self.main_roi = [(x1, y1), (x2, y2)]
                return [(x1, y1), (x2, y2)]
            elif key == 27:  # ESC 重试
                self.roi_box = []
                continue

    def update_capture_target(self, window_name: str | None = None, monitor_index: int | None = None) -> bool:
        """切换 WinRT 截屏目标。

        - 二选一：`window_name` 或 `monitor_index`（`None` 表示不使用）。
        - 返回是否在 3 秒内拿到首帧。
        """
        try:
            if window_name:
                # 按窗口标题锁定
                self._capture.recreate(window_name=window_name, monitor_index=None)
            else:
                # 按显示器索引锁定（1 起始）
                self._capture.recreate(window_name=None, monitor_index=monitor_index)

            ok = self._capture.wait_first_frame(timeout_sec=3.0)
            if not ok:
                logger.error("切换目标后未在 3 秒内收到首帧，请检查窗口是否可见或显示器索引是否正确。")
            return ok
        except Exception as e:
            logger.exception(f"切换截屏目标失败: {e}")
            return False

    # ---------------------- 基于 main_roi 的手动截屏 ----------------------
    def get_manual_screenshot(self) -> np.ndarray:
        """基于当前 `main_roi` 从最新帧裁剪得到主区域截图。"""
        logger.info(f"获取区域 {self.main_roi} 的屏幕截图")
        (x1, y1), (x2, y2) = self.main_roi
        full = self._capture.snapshot()  # BGR
        h, w = full.shape[:2]

        # 坐标裁剪到有效范围内
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        screenshot = full[y1c:y2c, x1c:x2c]

        # —— 新增：保存整帧 + 主 ROI 可视化 ——
        if EXPORT_ROI_DEBUG:
            full_vis = full.copy()
            cv2.rectangle(full_vis, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)  # 绿色主 ROI
            cv2.imwrite(os.path.join(DEBUG_DIR, "main_roi_overlay.png"), full_vis)

        # 若使用手动框选，尝试自动细化怪物区域
        try:
            os.makedirs("images/tmp", exist_ok=True)
            cv2.imwrite("images/tmp/zone1.png", screenshot)
            d_avatar, d_nums = find_monster_zone.cutFrame(screenshot)
            height, width, _ = screenshot.shape
            divisors = np.array([width, height, width, height])

            avatar = np.round(d_avatar * divisors).astype("int")

            # —— 新增：保存 zone1.png 上的候选头像/数字区域 ——
            if EXPORT_ROI_DEBUG:
                vis_zone1 = screenshot.copy()
                # 头像区域（洋红）
                for (ax1, ay1, ax2, ay2) in avatar:
                    cv2.rectangle(vis_zone1, (ax1, ay1), (ax2, ay2), (255, 0, 255), 2)
                # 数字区域（使用 d_nums，同样反归一化）
                nums_abs = np.round(d_nums * divisors).astype("int")
                for (nx1, ny1, nx2, ny2) in nums_abs:
                    cv2.rectangle(vis_zone1, (nx1, ny1), (nx2, ny2), (0, 255, 255), 2)
                cv2.imwrite(os.path.join(DEBUG_DIR, "zone1_detected.png"), vis_zone1)

            x_min, x_max, y_min, y_max = width, 0, height, 0
            for ax1, ay1, ax2, ay2 in avatar:
                x_min = min(x_min, min(ax1, ax2))
                x_max = max(x_max, ax2)
                y_min = min(y_min, min(ay1, ay2))
                y_max = max(y_max, ay2)

            logger.info(f"识别到目标区域：{[(x_min, y_min), (x_max, y_max)]}")
            self.main_roi = [(x1 + x_min, y1 + y_min), (x1 + x_max, y1 + y_max)]
            screenshot = screenshot[y_min:y_max, x_min:x_max]
            logger.info(f"区域更新为: {self.main_roi}")
        except Exception as e:
            logger.error(f"区域识别失败，使用完整区域: {e}")

        return screenshot

    # --------- 处理主区域：模板匹配（头像） + 数字 OCR（数量） ---------
    def process_regions(
        self,
        image_adb: cv2.typing.MatLike | None = None,
        matched_threshold: float = 0.5,
        ocr_threshold: float = 0.95,
    ) -> List[Dict]:
        """处理主区域中的 6 个区域，输出每个区域的识别结果字典。

        使用方式
        --------
        - 若提供 `image_adb`：按 `roi_relative` 从该图像内裁剪主 ROI；
        - 否则：从 WinRT 最新帧中取 `main_roi` 并裁剪，随后 **标准化** 到 969x119；
        - 对每个头像区域做模板匹配得到 `matched_id` 与 `confidence`；
        - 对对应数字区域做 OCR 得到 `number` 与 `ocr_confidence`；
        - 对异常组合（有头像无数字/有数字无头像）给出错误提示。
        """
        results: List[Dict] = []

        if image_adb is None:
            screenshot = self.get_manual_screenshot()
        else:
            # 从 ADB 图像中裁剪 ROI
            h, w = image_adb.shape[:2]
            x1 = int(self.roi_relative[0][0] * w)
            y1 = int(self.roi_relative[0][1] * h)
            x2 = int(self.roi_relative[1][0] * w)
            y2 = int(self.roi_relative[1][1] * h)
            screenshot = image_adb[y1:y2, x1:x2]
            if screenshot.size == 0:
                raise ValueError("截图为空，请检查区域选择或截图方法。")

        # 转换到标准 969x119 的目标区域
        screenshot = cv2.resize(screenshot, (969, 119))
        main_height, main_width = screenshot.shape[:2]

        if intelligent_workers_debug:
            os.makedirs("images/tmp", exist_ok=True)
            cv2.imwrite("images/tmp/zone.png", screenshot)

        # —— 新增：在标准化主 ROI 上叠加 6 个头像/数字区域 ——
        if EXPORT_ROI_DEBUG:
            vis_std = screenshot.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 头像相对区域（洋红）
            for idx, rel in enumerate(relative_regions):
                rx1 = int(rel[0] * main_width)
                ry1 = int(rel[1] * main_height)
                rx2 = int(rel[2] * main_width)
                ry2 = int(rel[3] * main_height)
                cv2.rectangle(vis_std, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
                cv2.putText(vis_std, f"A{idx}", (rx1, max(12, ry1 - 4)), font, 0.45, (255, 0, 255), 1)
            # 数字相对区域（黄色）
            for idx, rel in enumerate(relative_regions_nums):
                rx1 = int(rel[0] * main_width)
                ry1 = int(rel[1] * main_height)
                rx2 = int(rel[2] * main_width)
                ry2 = int(rel[3] * main_height)
                cv2.rectangle(vis_std, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                cv2.putText(vis_std, f"N{idx}", (rx1, min(main_height - 5, ry2 + 12)), font, 0.45, (0, 255, 255), 1)
            cv2.imwrite(os.path.join(DEBUG_DIR, "standardized_roi_overlays.png"), vis_std)

        # 分别处理 6 个位置
        for idx, rel in enumerate(relative_regions):
            # ---------------- 模板匹配：头像 ----------------
            try:
                rx1 = int(rel[0] * main_width)
                ry1 = int(rel[1] * main_height)
                rx2 = int(rel[2] * main_width)
                ry2 = int(rel[3] * main_height)
                sub_roi = screenshot[ry1:ry2, rx1:rx2]

                matched_id, confidence = find_best_match(sub_roi, self.ref_images)
                logger.info(f"target: {idx} confidence: {confidence:.4f}")

                if matched_id != 0 and confidence < matched_threshold:
                    raise ValueError(f"模板匹配置信度过低: {confidence}")

            except Exception as e:
                logger.exception(f"区域 {idx} 匹配失败: {str(e)}")
                results.append({"region_id": idx, "matched_id": -1, "number": "N/A", "error": str(e)})
                continue

            # ---------------- OCR：数量 ----------------
            try:
                rel_num = relative_regions_nums[idx]
                rx1_num = int(rel_num[0] * main_width)
                ry1_num = int(rel_num[1] * main_height)
                rx2_num = int(rel_num[2] * main_width)
                ry2_num = int(rel_num[3] * main_height)
                sub_roi_num = screenshot[ry1_num:ry2_num, rx1_num:rx2_num]

                processed = preprocess(sub_roi_num)
                processed = crop_to_min_bounding_rect(processed)
                processed = add_black_border(processed, border_size=3)

                number, ocr_confidence = self.do_num_ocr(processed)

                if number != "" and ocr_confidence < ocr_threshold:
                    raise ValueError(f"OCR置信度过低: {ocr_confidence}")

                if intelligent_workers_debug:
                    cv2.imwrite(f"images/tmp/target_{idx}.png", sub_roi)
                    cv2.imwrite(f"images/tmp/number_{idx}.png", processed)

                # 结果自洽性检查
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
                logger.exception(f"区域 {idx} OCR识别失败: {str(e)}")
                results.append({"region_id": idx, "matched_id": matched_id, "number": "N/A", "error": str(e)})

        return results

    # --------------------------------- OCR 调用 ---------------------------------
    def do_num_ocr(self, img: cv2.typing.MatLike):
        """对单个数字区域执行 OCR，返回 `(text, score)`。"""
        result = self.rapidocr_eng(img, use_det=False, use_cls=False, use_rec=True)
        text = result.txts[0] if result.txts else ""
        score = float(result.scores[0]) if result.scores else 0.0
        logger.info(f"OCR: text:'{text}', score:{score}")
        if text != "" and not str(text).isdigit():
            raise ValueError(f"OCR识别结果不是数字: '{text}'")
        return text, score


# ------------------------------- 直接运行测试 -------------------------------
if __name__ == "__main__":
    print("请用鼠标拖拽选择主区域（来自 WinRT 捕获的实时画面）...")
    recognizer = RecognizeMonster()
    main_roi = recognizer.select_roi()
    results = recognizer.process_regions()

    print("\n识别结果：")
    for res in results:
        if "error" in res:
            print(f"区域 {res['region_id']}: 错误 - {res['error']}")
        else:
            if res["matched_id"] != 0:
                print(
                    f"区域 {res['region_id']} => 匹配ID: {res['matched_id']} 数字: {res['number']} 置信度: {res.get('confidence', 'NA')}"
                )
