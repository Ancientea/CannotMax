"""Monster recognition via template matching and OCR.

Identifies monsters in 6 battlefield regions using:
- Template matching against reference images
- OCR for monster count extraction (RapidOCR)
- Debug mode with intermediate image saving

Regions: 3 left + 3 right, each containing monster type + count

Usage:
    recognizer = RecognizeMonster()
    results = recognizer.process_regions(screenshot)
    # results: [{region_id, matched_id, number, confidence}, ...]
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from rapidocr import RapidOCR, EngineType

from ..config import (
    MONSTER_DATA, 
    MONSTER_IMAGES, 
    MONSTER_COUNT,
    get_relative_regions,
    get_relative_regions_nums,
)
from .roi_selector import ROISelector
from .screenshot_helper import ScreenshotHelper
from .connector import PcConnector

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 是否启用 debug 模式
intelligent_workers_debug = True
def get_rapidocr_engine(prefer_gpu=False):
    """Get RapidOCR engine with optional GPU support."""
    try:
        if prefer_gpu:
            import torch
            if torch.cuda.is_available():
                return RapidOCR(
                    params={
                        "Det.engine_type": EngineType.TORCH,
                        "Cls.engine_type": EngineType.TORCH,
                        "Rec.engine_type": EngineType.TORCH,
                        "EngineConfig.torch.use_cuda": True,
                        "EngineConfig.torch.gpu_id": 0,
                    }
                )
    except ImportError:
        logger.warning("torch 库未安装，使用 onnxruntime")
    return RapidOCR()


def load_ref_images():
    """Load reference monster images."""
    ref_images = {}
    for i in range(MONSTER_COUNT + 1):
        if i == 0:
            img = MONSTER_IMAGES.get("empty")
        else:
            img = MONSTER_IMAGES.get(MONSTER_DATA["原始名称"][i])
        
        if img is None:
            logger.error(f"无法加载参考图片 i={i}")
            continue
        
        # Crop and resize template
        img_crop = img[
            int(img.shape[0] * 0.16):int(img.shape[0] * 0.80),
            int(img.shape[1] * 0.18):int(img.shape[1] * 0.82),
        ]
        ref_resized = cv2.resize(img_crop, (74, 74))
        ref_resized = ref_resized[0:70, :]
        
        if intelligent_workers_debug:
            Path("images/tmp").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"images/tmp/xref_{i}.png", ref_resized)
        
        ref_images[i] = ref_resized
    return ref_images


def find_best_match(
    target: cv2.typing.MatLike, ref_images: dict[int, cv2.typing.MatLike]
):
    """Template matching to find best match."""
    confidence = float("-inf")
    best_id = -1
    
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
            logger.exception(f"处理参考图像 {img_id} 时出错:", e)
            continue
    
    return best_id, confidence


def preprocess(img: cv2.typing.MatLike):
    """Binarize image for OCR."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    lower_bright = np.array([180, 180, 180])
    upper_bright = np.array([255, 255, 255])
    bright_mask = cv2.inRange(img, lower_bright, upper_bright)
    
    # Remove noise
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 13:
            cv2.drawContours(bright_mask, [contour], -1, 0, thickness=cv2.FILLED)
    
    return bright_mask


def crop_to_min_bounding_rect(image: cv2.typing.MatLike):
    """Crop to minimum bounding rectangle of contours."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    
    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    return image[y:y+h, x:x+w]


def add_black_border(img: cv2.typing.MatLike, border_size=3):
    """Add black border to image."""
    return cv2.copyMakeBorder(
        img, top=border_size, bottom=border_size, left=border_size, right=border_size,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )


class RecognizeMonster:
    """Monster recognition via template matching and OCR."""
    
    ROI_RELATIVE = [(0.2464, 0.8410), (0.7542, 0.9510)]  # 16:9 下怪物区域相对坐标

    def __init__(
        self,
        method: str = "ADB",
        window_name: str | None = None,
        monitor_index: int | None = None,
    ):
        self.method = method
        self.main_roi = [(0, 0), (1919, 1079)]
        self.rapidocr_eng = get_rapidocr_engine()
        self.ref_images = load_ref_images()
        self._connector: PcConnector | None = None
        self._roi_selector = ROISelector()
        self._screenshot_helper = ScreenshotHelper(method=method, connector=None)
        # Load relative regions from config
        self.relative_regions = get_relative_regions()
        self.relative_regions_nums = get_relative_regions_nums()

        # Initialize connector for WIN mode
        if self.method == "WIN" and window_name is not None:
            try:
                logger.info("初始化 PcConnector...")
                self._connector = PcConnector(window_name=window_name)
                self._connector.connect()
                self._screenshot_helper._connector = self._connector
                
                # Get initial frame to set ROI
                frame = self._connector.capture_screenshot()
                if frame is not None:
                    h, w = frame.shape[:2]
                    self.main_roi = [(0, 0), (w - 1, h - 1)]
            except Exception as e:
                logger.exception("PcConnector init failed: %s", e)
                self._connector = None

    def select_roi(self, example_image_path: str = "images/eg.png"):
        """Interactive ROI selection."""
        if self.method == "WIN" and self._connector is not None:
            img = self._connector.capture_screenshot()
            if img is not None:
                return self._roi_selector.select_roi(img, example_image_path)
        elif self.method in ("WIN", "PIL"):
            from PIL import ImageGrab
            screenshot = np.array(ImageGrab.grab())
            img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            return self._roi_selector.select_roi(img, example_image_path)
        else:
            logger.error(f"当前模式 {self.method} 不支持交互式选择 ROI")
            return None

    def process_regions(
        self,
        image_adb: cv2.typing.MatLike | None = None,
        matched_threshold=0.5,
        ocr_threshold=0.95,
    ):
        """
        Process all 6 regions for monster recognition.
        
        Args:
            image_adb: ADB screenshot (required for ADB mode)
            matched_threshold: Template matching confidence threshold
            ocr_threshold: OCR confidence threshold
            
        Returns:
            list[dict]: Recognition results for each region
        """
        results = []
        
        # Get main screenshot
        if self.method == "ADB":
            if image_adb is None:
                raise ValueError("ADB 模式下必须提供 image_adb")
            logger.info("使用 ADB 图像")
            x1 = int(self.ROI_RELATIVE[0][0] * image_adb.shape[1])
            y1 = int(self.ROI_RELATIVE[0][1] * image_adb.shape[0])
            x2 = int(self.ROI_RELATIVE[1][0] * image_adb.shape[1])
            y2 = int(self.ROI_RELATIVE[1][1] * image_adb.shape[0])
            screenshot = image_adb[y1:y2, x1:x2]
        else:
            logger.info(f"使用 {self.method} 手动截图")
            ocr_threshold = 0.8
            screenshot = self._screenshot_helper.capture_screenshot(
                bbox=self.main_roi, auto_detect_zone=True
            )
        
        if screenshot is None or screenshot.size == 0:
            raise ValueError("截图为空")
        
        # Resize to standard 975x119
        screenshot = cv2.resize(screenshot, (975, 119))
        main_height = screenshot.shape[0]
        main_width = screenshot.shape[1]
        
        if intelligent_workers_debug:
            cv2.imwrite("images/tmp/zone.png", screenshot)
        
        # Process each of 6 regions
        for idx, rel in enumerate(self.relative_regions):
            try:
                # Template matching
                rx1 = int(rel[0] * main_width)
                ry1 = int(rel[1] * main_height)
                rx2 = int(rel[2] * main_width)
                ry2 = int(rel[3] * main_height)
                sub_roi = screenshot[ry1:ry2, rx1:rx2]
                
                matched_id, confidence = find_best_match(sub_roi, self.ref_images)
                logger.info(f"target: {idx} matched_id: {matched_id}, confidence: {confidence:.4f}")
                
                if matched_id != 0 and confidence < matched_threshold:
                    raise ValueError(f"模板匹配置信度过低：{confidence}")
                    
            except Exception as e:
                logger.exception(f"区域 {idx} 匹配失败：{str(e)}")
                results.append({
                    "region_id": idx, "matched_id": 0, "number": "N/A", "error": str(e)
                })
                continue
            
            try:
                # OCR
                rel_num = self.relative_regions_nums[idx]
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
                    raise ValueError(f"OCR 置信度过低：{ocr_confidence}")
                
                if intelligent_workers_debug:
                    cv2.imwrite(f"images/tmp/target_{idx}.png", sub_roi)
                    cv2.imwrite(f"images/tmp/number_{idx}.png", processed)
                
                if number == "" and matched_id != 0:
                    raise ValueError("发现有怪物但无数量异常数据！")
                if matched_id == 0 and number != "":
                    raise ValueError("发现无怪物但有数量异常数据！")
                
                results.append({
                    "region_id": idx,
                    "matched_id": matched_id,
                    "number": number if number else "N/A",
                    "confidence": round(confidence, 2),
                })
                
            except Exception as e:
                logger.exception(f"区域 {idx} OCR 识别失败：{str(e)}")
                results.append({
                    "region_id": idx,
                    "matched_id": matched_id,
                    "number": "N/A",
                    "error": str(e),
                })
        
        return results

    def do_num_ocr(self, img: cv2.typing.MatLike):
        """Perform OCR on number region."""
        result = self.rapidocr_eng(img, use_det=False, use_cls=False, use_rec=True)
        logger.info(f"OCR: text: '{result.txts[0]}', score: {result.scores[0]}")
        if result.txts[0] != "" and not result.txts[0].isdigit():
            raise ValueError(f"OCR 识别结果不是数字：'{result.txts[0]}'")
        return result.txts[0], result.scores[0]


if __name__ == "__main__":
    print("请用鼠标拖拽选择主区域...")
    recognizer = RecognizeMonster(method="PIL")
    main_roi = recognizer.select_roi()
    results = recognizer.process_regions()
    print("\n识别结果：")
    for res in results:
        if "error" in res:
            print(f"区域{res['region_id']}: 错误 - {res['error']}")
        else:
            if res["matched_id"] != 0:
                print(f"区域{res['region_id']} => 匹配 ID:{res['matched_id']} 数字:{res['number']} 置信度:{res['confidence']}")
