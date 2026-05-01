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
    DEBUG_MODE,
    DEFAULT_AVATAR_REGIONS,
    DEFAULT_NUMBER_REGIONS,
    PC_DEFAULT_AVATAR_REGIONS,
    PC_DEFAULT_NUMBER_REGIONS,
)
from ..config.constants import DEFAULT_CROP_RATIO, PC_DEFAULT_CROP_RATIO
from ..config.paths import TMP_IMAGES_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
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
            logger.error("无法加载参考图片 i=%d", i)
            continue
        
        # Crop and resize template
        img_crop = img[
            int(img.shape[0] * 0.16):int(img.shape[0] * 0.80),
            int(img.shape[1] * 0.18):int(img.shape[1] * 0.82),
        ]
        ref_resized = cv2.resize(img_crop, (74, 74))
        ref_resized = ref_resized[0:70, :]
        
        if DEBUG_MODE:
            TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"{TMP_IMAGES_DIR}/xref_{i}.png", ref_resized)
        
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
            logger.exception("处理参考图像 %d 时出错", img_id)
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


class ROINotSelectedError(Exception):
    """WIN 模式下用户未选择 ROI 时抛出"""
    pass


class RecognizeMonster:
    """Monster recognition via template matching and OCR."""

    def __init__(self, crop_ratio=None, avatar_regions=None, number_regions=None):
        """Initialize recognizer.
        
        Args:
            crop_ratio: 怪物条裁剪比例 [(x1,y1), (x2,y2)] 或 None（ADB/PC 用默认值）
            avatar_regions: 6个头像区域在 975x119 内的相对坐标，None 用默认值
            number_regions: 6个数字区域在 975x119 内的相对坐标，None 用默认值
        """
        self.ref_images = load_ref_images()
        self.ocr = get_rapidocr_engine()
        self.crop_ratio = crop_ratio
        self.avatar_regions = avatar_regions
        self.number_regions = number_regions
    
    def _resolve_crop_ratio(self, auto_fallback: bool, mode: str = "ADB") -> tuple:
        """解析裁剪比例。

        Args:
            auto_fallback: True 时 crop_ratio=None 回退到默认值
            mode: 捕获模式 (ADB/PC/WIN)，影响默认值选择

        Returns:
            ((x1, y1), (x2, y2)) 相对坐标

        Raises:
            ROINotSelectedError: WIN 模式且 crop_ratio=None 时
        """
        if self.crop_ratio is not None:
            return self.crop_ratio
        if auto_fallback:
            if mode == "PC":
                return PC_DEFAULT_CROP_RATIO
            return DEFAULT_CROP_RATIO
        raise ROINotSelectedError("请先选择怪物条范围")
    
    def _crop_by_ratio(self, screenshot: np.ndarray, ratio: tuple) -> np.ndarray:
        """按相对坐标裁切图像。

        Args:
            screenshot: BGR 图像
            ratio: ((x1, y1), (x2, y2)) 相对坐标

        Returns:
            裁切后的 BGR 图像
        """
        h, w = screenshot.shape[:2]
        (x1, y1), (x2, y2) = ratio
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        return screenshot[py1:py2, px1:px2]

    def process_regions(self, screenshot: np.ndarray, auto_fallback: bool = True, mode: str = "ADB") -> list[dict]:
        """Process full-screen screenshot to identify monsters.
        
        Args:
            screenshot: Full-screen BGR image (any resolution)
            auto_fallback: True 时 crop_ratio=None 回退默认值 (ADB/PC)；False 时抛异常 (WIN)
            mode: 捕获模式 (ADB/PC/WIN)，影响默认裁剪参数选择
        
        Returns:
            List of 6 recognition results
        """
        from ..utils import find_monster_zone
        
        avatar_regs = self.avatar_regions if self.avatar_regions is not None else (
            PC_DEFAULT_AVATAR_REGIONS if mode == "PC" else DEFAULT_AVATAR_REGIONS
        )
        number_regs = self.number_regions if self.number_regions is not None else (
            PC_DEFAULT_NUMBER_REGIONS if mode == "PC" else DEFAULT_NUMBER_REGIONS
        )
        
        # Save original screenshot for debugging
        if DEBUG_MODE:
            TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"{TMP_IMAGES_DIR}/original_screenshot.png", screenshot)
        
        # 1. Resolve crop ratio and crop
        ratio = self._resolve_crop_ratio(auto_fallback, mode)
        cropped = self._crop_by_ratio(screenshot, ratio)
        
        # 2. WIN mode: find_monster_zone secondary refinement
        if self.crop_ratio is not None and auto_fallback is False:
            try:
                d_avatar, d_nums = find_monster_zone.find_monster_zone(cropped)
                if d_avatar is not None:
                    h, w = cropped.shape[:2]
                    avatar_px = np.round(d_avatar * [w, h, w, h]).astype(int)
                    x_min = max(0, int(avatar_px[:, 0].min()))
                    y_min = max(0, int(avatar_px[:, 1].min()))
                    x_max = min(w, int(avatar_px[:, 2].max()))
                    y_max = min(h, int(avatar_px[:, 3].max()))
                    cropped = cropped[y_min:y_max, x_min:x_max]
            except Exception as e:
                logger.exception("Monster bar detection failed: %s", e)
                return []
        
        # 3. Resize to standard 975x119
        if cropped is None or cropped.size == 0:
            logger.error("Could not detect monster bar")
            return []
        
        try:
            monster_bar = cv2.resize(cropped, (975, 119))
        except Exception as e:
            logger.error("Crop failed: %s", e)
            return []
        
        if DEBUG_MODE:
            cv2.imwrite(f"{TMP_IMAGES_DIR}/monster_bar_975x119.png", monster_bar)
        
        # 4. Split into 6 regions using precise coordinates and recognize
        results = []
        for idx in range(6):
            ax1, ay1, ax2, ay2 = avatar_regs[idx]
            avatar_img = monster_bar[
                int(ay1 * 119):int(ay2 * 119),
                int(ax1 * 975):int(ax2 * 975),
            ]
            nx1, ny1, nx2, ny2 = number_regs[idx]
            num_img = monster_bar[
                int(ny1 * 119):int(ny2 * 119),
                int(nx1 * 975):int(nx2 * 975),
            ]
            result = self._recognize_region(avatar_img, num_img, idx)
            results.append(result)
        
        return results

    def _recognize_region(self, avatar_img: np.ndarray, num_img: np.ndarray, region_id: int,
                          matched_threshold=0.5, ocr_threshold=0.95) -> dict:
        """Recognize a single monster region (template matching + OCR).

        Args:
            avatar_img: Pre-cropped avatar sub-region from 975x119 bar
            num_img: Pre-cropped number sub-region from 975x119 bar
            region_id: Region index (0-5)
            matched_threshold: Template matching confidence threshold
            ocr_threshold: OCR confidence threshold

        Returns:
            Recognition result dict
        """
        # Template matching
        try:
            matched_id, confidence = find_best_match(avatar_img, self.ref_images)
            logger.info("target: %d matched_id: %d, confidence: %.4f", region_id, matched_id, confidence)
            
            if matched_id != 0 and confidence < matched_threshold:
                raise ValueError(f"模板匹配置信度过低：{confidence}")
        except Exception as e:
            logger.exception(f"区域 {region_id} 匹配失败：{str(e)}")
            return {
                "region_id": region_id, "matched_id": 0, "number": "N/A", "error": str(e)
            }
        
        # OCR on the number area
        try:
            processed = preprocess(num_img)
            processed = crop_to_min_bounding_rect(processed)
            processed = add_black_border(processed, border_size=3)
            
            number, ocr_confidence = self.do_num_ocr(processed)
            if number != "" and ocr_confidence < ocr_threshold:
                raise ValueError(f"OCR 置信度过低：{ocr_confidence}")
            
            if DEBUG_MODE:
                TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{TMP_IMAGES_DIR}/target_{region_id}.png", avatar_img)
                cv2.imwrite(f"{TMP_IMAGES_DIR}/number_{region_id}.png", processed)
            
            if number == "" and matched_id != 0:
                raise ValueError("发现有怪物但无数量异常数据！")
            if matched_id == 0 and number != "":
                raise ValueError("发现无怪物但有数量异常数据！")
            
            return {
                "region_id": region_id,
                "matched_id": matched_id,
                "number": number if number else "N/A",
                "confidence": round(confidence, 2),
            }
        except Exception as e:
            logger.exception(f"区域 {region_id} OCR 识别失败：{str(e)}")
            return {
                "region_id": region_id,
                "matched_id": matched_id,
                "number": "N/A",
                "error": str(e),
            }

    def do_num_ocr(self, img: cv2.typing.MatLike):
        """Perform OCR on number region."""
        result = self.ocr(img, use_det=False, use_cls=False, use_rec=True)
        logger.info("OCR: text: '%s', score: %s", result.txts[0], result.scores[0])
        if result.txts[0] != "" and not result.txts[0].isdigit():
            raise ValueError(f"OCR 识别结果不是数字：'{result.txts[0]}'")
        return result.txts[0], result.scores[0]


if __name__ == "__main__":
    # Example usage with test image
    test_img = f"{TMP_IMAGES_DIR}/zone1.png"
    if Path(test_img).exists():
        screenshot = cv2.imread(test_img)
        recognizer = RecognizeMonster()
        results = recognizer.process_regions(screenshot)
        print("识别结果：")
        for res in results:
            if "error" in res:
                print(f"区域{res['region_id']}: 错误 - {res['error']}")
            else:
                if res["matched_id"] != 0:
                    print(f"区域{res['region_id']} => 匹配 ID:{res['matched_id']} 数字:{res['number']} 置信度:{res['confidence']}")
    else:
        print(f"测试图片 {test_img} 不存在")
